"""
Memory Engine — Public API

remember(), recall(), consolidate(), forget(), stats()
"""

import json
import logging
import math
import re
import numpy as np
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from ..config import EngineConfig
from .memory import Memory, get_band, get_format
from .activation import compute_temperature  # New 7-stage equation
from .signals_enhanced import compute_da, compute_ne, compute_gaba, compute_usage
from .signals_enhanced import update_signals_batch  # for consolidation
from .pruning import compute_signal_uncertainty, compute_prune_probability, should_prune
from .protection import update_retrieval_importance, compute_compression_resistance
from .compression import should_compress, compress, COMPRESSION_THRESHOLDS
from .erosion import tokenize, score_token_importance, erode_tokens
from ..retrieval.vector import vector_recall
from ..retrieval.keyword import keyword_recall
from ..retrieval.associative import spreading_activation_recall, build_context
from ..retrieval.fusion import fuse_results, apply_lifecycle_modulation
from ..retrieval.temporal import detect_temporal_query, temporal_rerank, ensure_temporal_diversity
from ..retrieval.graph import graph_recall
from ..retrieval.temporal_decomposer import decompose as decompose_temporal, compute_temporal_hint
from ..storage.postgres import PostgresStore
from ..storage.entity_graph import EntityGraphStore
from ..embeddings.embed import embed, embed_query, get_model, get_embed_dim, assert_model_matches
from ..utils import extract_session_id, parse_context_date
from .entity_extraction import extract_entities, ENTITY_STOPWORDS

logger = logging.getLogger(__name__)

# Single source of truth for entity stopwords lives in entity_extraction.
_ENTITY_STOPWORDS = ENTITY_STOPWORDS


# Question words / verbs / vague terms that are never useful as entity seeds.
_CONCEPT_TERM_STOPWORDS = {
    "how", "many", "much", "what", "when", "where", "which", "who", "why",
    "did", "does", "do", "was", "were", "have", "has", "had", "is", "are",
    "the", "a", "an", "my", "me", "i", "we", "you", "they", "it", "this", "that",
    "and", "or", "but", "for", "with", "from", "to", "of", "in", "on", "at", "by",
    "all", "any", "some", "ago", "long", "pass", "passed", "take", "took", "get",
    "got", "between", "since", "more", "less", "than", "been", "being",
    "buy", "bought", "work", "worked", "make", "made", "go", "went", "see", "saw",
    "receive", "received", "meet", "met", "need", "needed", "days", "weeks",
    "months", "years", "day", "week", "month", "year", "time", "times",
    "last", "first", "next", "new", "old", "one", "two", "three",
    "about", "thing", "things", "item", "items",
} | ENTITY_STOPWORDS


def _extract_query_concept_terms(query: str) -> list:
    """
    Extract salient content words from a query to seed graph_recall when no
    proper-noun entities were found. Returns lowercased terms of length >= 4
    that survive a stopword filter. Used for common-noun questions like
    "how many model kits did I buy" where `extract_entities` returns [].
    """
    tokens = re.findall(r"[A-Za-z][A-Za-z'\-]*", query.lower())
    terms = []
    seen = set()
    for tok in tokens:
        if len(tok) < 4 or tok in _CONCEPT_TERM_STOPWORDS:
            continue
        if tok not in seen:
            seen.add(tok)
            terms.append(tok)
    return terms[:6]


class MemoryEngine:

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.store = PostgresStore(self.config.db_url)
        self._cycle_count = 0
        self._last_recall_paths: Optional[dict] = None

        # Initialize embedding model and verify config/runtime/schema agree.
        get_model(self.config.embed_model)
        assert_model_matches(self.config.embed_model, self.config.embed_dim)


    # ── REMEMBER ─────────────────────────────────────────────────

    def remember(
        self,
        content: str,
        source: str = "direct",
        context: str = "",
    ) -> dict:
        now = datetime.now(timezone.utc)

        # Embed
        content_embedding = embed(content, self.config.embed_model)

        # DA at ingest: no retrieval history yet, baseline only.
        da = compute_da(access_count=0, avg_outcome=0.0, user_rating=0.0, goal_alignment=0.0)

        # NE / GABA at ingest: cosine novelty / redundancy vs recent embeddings.
        recent_embeddings = self.store.get_recent_embeddings(limit=50)
        ne = compute_ne(content_embedding, recent_embeddings, age_days=0.001)
        gaba = compute_gaba(content_embedding, recent_embeddings, age_days=0.001)

        # Extract entities and update graph (before novelty gate — stats accumulate either way)
        # Only track named/numeric entities for graph — concepts are too noisy
        # Strip common prefixes that cause false positives in named entity extraction
        clean_content = re.sub(r'^\[?(?:Assistant|User|Session context)\]?:\s*', '', content)
        all_entities = extract_entities(clean_content)
        entities = []
        for n, t, c in all_entities:
            if t not in ("named", "numeric") or len(c) < 3:
                continue
            c_clean = c.rstrip('.,;:!?').strip()
            if t == "named":
                if c_clean.lower() not in ENTITY_STOPWORDS and len(c_clean) >= 2:
                    entities.append((n, t, c_clean))
                continue
            if t == "numeric":
                entities.append((n, t, c_clean))
        session_id = extract_session_id(context)
        entity_canonicals = [canonical for _, _, canonical in entities]
        entity_graph = EntityGraphStore(self.store.conn)

        # Novelty gate
        if ne < self.config.novelty_gate:
            logger.info(f"Gated out (NE={ne:.3f}): {content[:50]}...")
            mem = Memory(
                raw_content=content,
                temperature=0.1,
                region="neocortex",
                da_relevance=da,
                ne_novelty=ne,
                gaba_inhibition=gaba,
                usage_score=0.0,
                embedding=content_embedding,
                source=source,
                context=context,
                created_at=now,
                last_accessed=now,
                access_times=[now],
                access_count=0,
                knowledge_nugget=content[:100],
                compression_level="nugget",
                co_entities=entity_canonicals,
            )
            self.store.store(mem)

            # Gate-on-reject: memory is low-temp but entity stats still accumulate
            self._link_entities(entity_graph, mem.id, entities, session_id)

            return mem.to_dict()

        # Build spatiotemporal context
        hour = now.hour
        if hour < 6:
            tod = "night"
        elif hour < 12:
            tod = "morning"
        elif hour < 18:
            tod = "afternoon"
        else:
            tod = "evening"

        mem = Memory(
            raw_content=content,
            temperature=1.0,
            region="hippocampus",
            da_relevance=da,
            ne_novelty=ne,
            gaba_inhibition=gaba,
            usage_score=0.0,
            da_history=[da],
            ne_history=[ne],
            signal_variance=0.5,
            embedding=content_embedding,
            source=source,
            context=context,
            time_of_day=tod,
            day_of_week=now.strftime("%A").lower(),
            created_at=now,
            last_accessed=now,
            access_times=[now],
            access_count=0,
            compression_level="raw",
            tokens_original=len(content.split()),
            tokens_current=len(content.split()),
            co_entities=entity_canonicals,
        )

        # Initialize token-level erosion weights
        tokens = tokenize(content)
        mem.token_weights = score_token_importance(tokens, known_entities=entity_canonicals)

        self.store.store(mem)

        # Link entities to memory and update session tracking
        self._link_entities(entity_graph, mem.id, entities, session_id)

        logger.info(
            f"Remembered [{mem.band}] DA={da:.2f} NE={ne:.2f}: {content[:60]}..."
        )
        return mem.to_dict()

    # ── REMEMBER BATCH ──────────────────────────────────────────

    def remember_batch(
        self,
        items: List[Dict[str, str]],
        link_entities: bool = True,
        novelty_window: int = 200,
    ) -> List[dict]:
        """Batch-ingest multiple memories: one embedding call, one DB transaction.

        Each item: {"content": str, "source": str, "context": str,
                    "created_at": datetime (optional, for backdated imports)}
        link_entities: If True, extract and link entities (slower but better retrieval).
        novelty_window: Max recent embeddings to compare against for NE/GABA.
            Without this cap the inner loop is O(n²) and a 50k-item batch would
            take days. 200 matches the biological framing — novelty is a
            *local* recency signal, not a global corpus property.
        Returns list of memory dicts.
        """
        from ..embeddings.embed import embed_batch as _embed_batch
        from collections import deque

        if not items:
            return []

        default_now = datetime.now(timezone.utc)

        # ── Step 1: Batch embed all content via OpenAI API ──
        texts = [item["content"] for item in items]
        logger.info(f"Batch embedding {len(texts)} items...")
        embeddings = _embed_batch(texts, self.config.embed_model)
        logger.info(f"Embedding complete, building memories...")

        # ── Step 2: Build all Memory objects (CPU-only, no DB) ──
        memories = []
        entity_work = []  # (mem_id, entities, session_id) for deferred entity linking

        # Sliding-window novelty context. Seeded with stored recent embeddings;
        # new items push onto the deque, which drops the oldest when capped.
        # This keeps NE/GABA O(window × n) instead of O(n²) — essential for
        # batch imports of thousands of items (Claude chat history, etc).
        all_embeddings: deque = deque(
            self.store.get_recent_embeddings(limit=min(50, novelty_window)),
            maxlen=novelty_window,
        )

        for item, content_embedding in zip(items, embeddings):
            content = item["content"]
            source = item.get("source", "direct")
            context = item.get("context", "")

            # Per-item timestamp — defaults to now, but backdated imports pass
            # real historical datetimes so the memory is born with correct age.
            item_ts = item.get("created_at", default_now)
            if item_ts.tzinfo is None:
                item_ts = item_ts.replace(tzinfo=timezone.utc)
            hour = item_ts.hour
            tod = "night" if hour < 6 else "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"
            dow = item_ts.strftime("%A").lower()

            da = compute_da(access_count=0, avg_outcome=0.0, user_rating=0.0, goal_alignment=0.0)
            if all_embeddings:
                ne = compute_ne(content_embedding, list(all_embeddings), age_days=0.001)
                gaba = compute_gaba(content_embedding, list(all_embeddings), age_days=0.001)
            else:
                ne = 0.8
                gaba = 0.1
            all_embeddings.append(content_embedding)

            entity_canonicals = []
            entities = []
            if link_entities:
                clean_content = re.sub(r'^\[?(?:Assistant|User|Session context)\]?:\s*', '', content)
                all_entities = extract_entities(clean_content)
                for n, t, c in all_entities:
                    if t not in ("named", "numeric") or len(c) < 3:
                        continue
                    c_clean = c.rstrip('.,;:!?').strip()
                    if t == "named":
                        if c_clean.lower() not in ENTITY_STOPWORDS and len(c_clean) >= 2:
                            entities.append((n, t, c_clean))
                        continue
                    if t == "numeric":
                        entities.append((n, t, c_clean))
                entity_canonicals = [canonical for _, _, canonical in entities]

            session_id = extract_session_id(context)

            mem = Memory(
                raw_content=content, temperature=1.0, region="hippocampus",
                da_relevance=da, ne_novelty=ne, gaba_inhibition=gaba, usage_score=0.0,
                da_history=[da], ne_history=[ne], signal_variance=0.5,
                embedding=content_embedding, source=source, context=context,
                time_of_day=tod, day_of_week=dow,
                created_at=item_ts, last_accessed=item_ts, access_times=[item_ts], access_count=0,
                compression_level="raw",
                tokens_original=len(content.split()), tokens_current=len(content.split()),
                co_entities=entity_canonicals,
            )

            memories.append(mem)
            # Queue every memory for entity linking when link_entities is on,
            # even if no entities were extracted — _link_entities adds the
            # persona, so memories with no other entities still get the
            # universal owner edge.
            if link_entities:
                entity_work.append((mem.id, entities, session_id))

        # ── Step 3: Single DB transaction for all memory inserts ──
        logger.info(f"Inserting {len(memories)} memories into DB...")
        self.store.store_batch(memories)

        # ── Step 4: Batch entity graph updates (single transaction) ──
        if entity_work:
            logger.info(f"Linking entities for {len(entity_work)} memories...")
            entity_graph = EntityGraphStore(self.store.conn, autocommit=False)
            for mem_id, entities, session_id in entity_work:
                # Route through the same helper as remember() so persona
                # injection is consistent across both ingest paths.
                self._link_entities(entity_graph, mem_id, entities, session_id)
            self.store.conn.commit()

        logger.info(f"Batch ingested {len(memories)}/{len(items)} memories")
        return [m.to_dict() for m in memories]

    # ── RECALL ───────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        top_k: int = 10,
        min_temp: float = 0.0,
        reheat: bool = True,
        question_timestamp=None,
    ) -> List[dict]:
        query_emb = embed_query(query, self.config.embed_model)

        overfetch = top_k * self.config.overfetch_multiplier

        # Four parallel retrieval paths — each independently ablatable.
        vec_results = (
            vector_recall(query_emb, self.store, top_k=overfetch, min_temp=min_temp)
            if self.config.enable_vector_path else []
        )
        kw_results = (
            keyword_recall(query, self.store, top_k=overfetch, min_temp=min_temp)
            if self.config.enable_keyword_path else []
        )

        # Path 4: Entity graph walk — extract entities first so the spreading-
        # activation path below can use the same seeds.
        query_entity_tuples = extract_entities(query)
        query_entity_names = [canonical for _, _, canonical in query_entity_tuples]
        # If no proper-noun entities extracted (common for everyday-object
        # questions like "how many model kits did I buy"), fall back to
        # salient nouns so graph_recall's fuzzy LIKE can still match.
        if not query_entity_names:
            query_entity_names = _extract_query_concept_terms(query)
        graph_results = (
            graph_recall(
                query_entity_names, self.store, top_k=overfetch,
                fanout_norm_exponent=self.config.graph_fanout_norm_exponent,
            )
            if self.config.enable_graph_path else []
        )

        # Path 3 (real spreading activation): seed at query entities, propagate
        # through the entity graph with decay, return memories linked to the
        # most-activated nodes. Replaces the old "similarity to recent context"
        # implementation. Same query_entity_names used as seeds.
        # Spreading-activation path. The archive_trigger_threshold lets very
        # strongly activated nodes pull archived memories — déjà-vu. Without
        # it, archived memories are unreachable; with it, they only surface
        # when association is strong (not on weak hops or persona fallback).
        archive_trigger = (
            self.config.archive_trigger_threshold
            if self.config.archive_trigger_threshold > 0.0
            else None
        )
        assoc_results = (
            spreading_activation_recall(
                query_entity_names,
                self.store,
                top_k=overfetch,
                min_temp=min_temp,
                persona=self.config.persona,
                persona_seed_activation=self.config.persona_seed_activation,
                persona_fallback=self.config.persona_fallback_seed,
                archive_trigger_threshold=archive_trigger,
                decay=self.config.sa_decay,
                max_hops=self.config.sa_max_hops,
                fanout_norm_exponent=self.config.sa_fanout_norm_exponent,
            )
            if self.config.enable_associative_path else []
        )

        logger.debug(
            f"Recall paths for '{query[:60]}': "
            f"vec={len(vec_results)} kw={len(kw_results)} "
            f"assoc={len(assoc_results)} graph={len(graph_results)} "
            f"(entities={query_entity_names})"
        )

        # Diagnostic: stash per-path pre-fusion contribution so callers (the
        # bench debug dump) can see which path produced each result and at
        # what rank. Keyed by memory id.
        self._last_recall_paths = {
            "query": query,
            "query_entities": list(query_entity_names),
            "counts": {
                "vector": len(vec_results),
                "keyword": len(kw_results),
                "associative": len(assoc_results),
                "graph": len(graph_results),
            },
            "per_memory": {},
        }
        for path_name, path_results in (
            ("vector", vec_results),
            ("keyword", kw_results),
            ("associative", assoc_results),
            ("graph", graph_results),
        ):
            for rank, r in enumerate(path_results):
                mid = str(r.get("id"))
                slot = self._last_recall_paths["per_memory"].setdefault(
                    mid, {}
                )
                slot[path_name] = {
                    "rank": rank,
                    "score": float(r.get("score", r.get("similarity", 0.0)) or 0.0),
                }

        # Temporal decomposition: for temporal queries, also retrieve using
        # stripped event queries to improve recall on time-anchored questions
        if self.config.enable_temporal_decomposition:
            temporal_decomp = decompose_temporal(query, question_timestamp)
        else:
            temporal_decomp = {"is_temporal": False, "event_queries": [], "temporal_op": None, "temporal_params": {}}
        extra_vec = []
        extra_kw = []
        if temporal_decomp["is_temporal"]:
            seen_ids = {r["id"] for r in vec_results + kw_results}
            for eq in temporal_decomp["event_queries"]:
                if eq == query:
                    continue
                eq_emb = embed_query(eq, self.config.embed_model)
                for r in vector_recall(eq_emb, self.store, top_k=overfetch, min_temp=min_temp):
                    if r["id"] not in seen_ids:
                        extra_vec.append(r)
                        seen_ids.add(r["id"])
                for r in keyword_recall(eq, self.store, top_k=overfetch, min_temp=min_temp):
                    if r["id"] not in seen_ids:
                        extra_kw.append(r)
                        seen_ids.add(r["id"])

        # Fuse via RRF with session diversity and recency tiebreaker
        fused = fuse_results(
            vec_results + extra_vec, kw_results + extra_kw, assoc_results,
            vector_weight=self.config.vector_weight,
            keyword_weight=self.config.keyword_weight,
            associative_weight=self.config.associative_weight,
            rrf_k=self.config.rrf_k,
            recency_bonus=self.config.recency_bonus,
            session_decay=self.config.session_decay,
            top_k=top_k,
            query=query,
            graph_results=graph_results,
            archive_rrf_boost=getattr(self.config, "archive_rrf_boost", 1.0),
        )

        # Lifecycle modulation: multiply each fused_score by a bounded
        # function of (temperature, DA, NE). Hot goal-relevant memories tilt
        # upward; off-curve novelty tilts down. The modulation is bounded so
        # an overwhelming semantic match on a cold memory still surfaces.
        # This is the change that turns the lifecycle from a stored model
        # into a retrieval bias.
        if self.config.enable_lifecycle_modulation:
            fused = apply_lifecycle_modulation(
                fused,
                temp_lift=self.config.mod_temp_lift,
                da_lift=self.config.mod_da_lift,
                ne_penalty=self.config.mod_ne_penalty,
                mod_min=self.config.mod_min,
                mod_max=self.config.mod_max,
                archive_modulation_override=getattr(
                    self.config, "archive_modulation_override", 1.0
                ),
            )

        # Temporal re-ranking (Path 4) — only activates for temporal queries
        if self.config.enable_temporal_rerank:
            temporal_info = detect_temporal_query(query, question_timestamp)
            if temporal_info['is_temporal']:
                fused = temporal_rerank(
                    fused, temporal_info, question_timestamp,
                    temporal_weight=self.config.temporal_weight,
                    rrf_k=self.config.rrf_k,
                )
                fused = ensure_temporal_diversity(fused, temporal_info, top_k=top_k * 2)
        else:
            temporal_info = {"is_temporal": False}

        # Slice to top_k. RRF + lifecycle modulation already produced the
        # final ordering (modulation lifts hot/goal-relevant, dampens
        # off-curve novelty). Late-tier scaffolding (two-phase, temporal-
        # graph, contiguity) was deleted — the design has four retrieval
        # paths plus déjà-vu, all merged via fusion. No post-fusion appends.
        results = fused[:top_k]

        # Attach temporal hint for the answer model
        if temporal_decomp["is_temporal"] and temporal_decomp["temporal_op"]:
            ref_date = None
            if question_timestamp:
                try:
                    from ..retrieval.temporal import _parse_timestamp
                    ref_date = _parse_timestamp(question_timestamp)
                except Exception:
                    pass
            hint = compute_temporal_hint(
                temporal_decomp["temporal_op"],
                temporal_decomp["temporal_params"],
                results,
                ref_date,
            )
            if hint and results:
                results[0]["_temporal_hint"] = hint

        # Reheat accessed memories + update EWC.
        # All write-back is gated by config.recall_mutates_state. When False
        # (benchmarking mode), recall is read-only so the same query set
        # produces identical results across runs and is order-independent.
        top_k_ids = [r["id"] for r in results]
        all_ids = [r["id"] for r in fused]

        if reheat and self.config.recall_mutates_state:
            for result in results:
                mem = self.store.get(result["id"])
                if mem:
                    self._reheat(mem, result.get("similarity", result.get("fusion_score", 0.5)))
                    # Update result with reheated temp
                    result["temperature"] = mem.temperature

        if self.config.recall_mutates_state:
            # Log retrieval for EWC
            try:
                self.store.log_retrieval(query, query_emb, all_ids[:50], top_k_ids)
            except Exception as e:
                logger.warning(f"Failed to log retrieval: {e}")
                try:
                    self.store.conn.rollback()
                except Exception:
                    pass

            # Bounded importance update — only top_k hits and a small decay sample.
            self._update_retrieval_importance(top_k_ids)

        # Attach per-path diagnostic to each returned result. Harmless when
        # ignored; the bench debug dump reads it through.
        per_mem = self._last_recall_paths["per_memory"]
        for r in results:
            mid = str(r.get("id"))
            r["_path_debug"] = per_mem.get(mid, {})

        return results

    def _effective_reheat_amount(self, current_temp: float, match_strength: float) -> float:
        """Compute the reheat delta a memory WOULD receive given its match strength.

        Mirrors the formula in _reheat but is read-only — used for activation-
        based ranking before the slice. Same formula across both call sites
        guarantees the ranking matches the post-reheat temperature.
        """
        coldness_factor = 1.0 - current_temp
        similarity_factor = max(0.0, match_strength)
        return self.config.base_reheat + (
            self.config.coldness_reheat_factor * coldness_factor * similarity_factor
        )

    def _rank_by_effective_temperature(
        self,
        candidates: List[dict],
        top_k: int,
    ) -> List[dict]:
        """Rank a tier-union by post-reheat (effective) temperature.

        Each tier produces results with different score scales (RRF score,
        cosine similarity, binary). To rank fairly across tiers, we use the
        only common currency: the activation-equation temperature each memory
        WOULD have after this retrieval reheats it. Tiers contribute by
        providing a match_strength signal; activation does the ranking.

        Match strength source priority:
          1. similarity (vector path, two_phase)
          2. fusion_score (RRF output) — normalized to [0,1] via 1/(1+rank/k)
          3. 0.5 (default for tier hits without explicit score)
        """
        if not candidates:
            return []

        # Deduplicate by id — keep highest-match-strength version of any dupe
        best_by_id: Dict[str, dict] = {}
        for r in candidates:
            mid = r.get("id")
            if mid is None:
                continue
            ms = self._extract_match_strength(r)
            r["_match_strength"] = ms
            if mid not in best_by_id or ms > best_by_id[mid].get("_match_strength", 0.0):
                best_by_id[mid] = r

        # Fetch current temperature for each unique memory (one DB round-trip
        # per id; could be batched but the union is small — top_k * a few).
        for mid, r in best_by_id.items():
            mem = self.store.get(mid)
            current_temp = mem.temperature if mem else float(r.get("temperature", 0.0))
            ms = r["_match_strength"]
            r["_effective_temperature"] = current_temp + self._effective_reheat_amount(current_temp, ms)
            # Keep current temp visible for downstream consumers (and so the
            # answer model sees the un-mutated value when state is read-only).
            r.setdefault("temperature", current_temp)

        ranked = sorted(
            best_by_id.values(),
            key=lambda r: (r["_effective_temperature"], r["_match_strength"]),
            reverse=True,
        )
        return ranked[:top_k]

    @staticmethod
    def _extract_match_strength(r: dict) -> float:
        """Extract a [0,1] match strength signal from a result dict.

        Different tiers populate different fields. Order matters: we trust
        explicit similarity first, then RRF fusion_score (already on roughly
        the right scale), then default to a moderate value for binary tier hits.
        """
        if "similarity" in r and r["similarity"] is not None:
            return float(r["similarity"])
        if "fusion_score" in r and r["fusion_score"] is not None:
            # Fusion scores are usually small (~0.01-0.1 for RRF with k=60).
            # Map to [0,1] with a soft scale: 0.05 → ~0.5, 0.2 → ~0.9.
            fs = float(r["fusion_score"])
            return min(1.0, fs / (fs + 0.05))
        return 0.5

    def _reheat(self, mem: Memory, similarity: float):
        """Reheat proportional to coldness — cold memories spike harder."""
        coldness_factor = 1.0 - mem.temperature
        similarity_factor = max(0.0, similarity)
        reheat_amount = self.config.base_reheat + (
            self.config.coldness_reheat_factor * coldness_factor * similarity_factor
        )

        now = datetime.now(timezone.utc)
        mem.temperature = min(1.0, mem.temperature + reheat_amount)
        mem.last_accessed = now
        mem.access_count += 1
        mem.access_times = (mem.access_times + [now])[-100:]

        self.store.update(mem)

    def _update_retrieval_importance(self, top_k_ids: List[str]):
        """Update EWC importance for retrieved memories.

        Previously this scanned up to 500 memories per query — O(N) per recall
        with two failure modes: (a) latency and write contention scale with corpus
        size, and (b) the per-query decay made the importance landscape a function
        of query throughput rather than time. Now: only retrieved memories get
        positive updates here. Decay happens during consolidation, where it
        belongs and runs once per cycle rather than once per query.
        """
        if not top_k_ids:
            return
        for mem_id in top_k_ids:
            mem = self.store.get(mem_id)
            if mem is None:
                continue
            new_importance = update_retrieval_importance(
                True,
                mem.retrieval_importance,
                learning_rate=self.config.importance_learning_rate,
                decay_rate=self.config.importance_decay_rate,
            )
            self.store.update_fields(mem.id, {
                "retrieval_importance": new_importance,
                "retrieval_hits": mem.retrieval_hits + 1,
            })

    # ── CONSOLIDATE ──────────────────────────────────────────────

    def consolidate(self) -> dict:
        """
        The sleep cycle.

        1. Decay all temperatures via ACT-R power-law
        2. Update signal histories for MESU uncertainty tracking
        3. Deterministic compression (summary, nuggets, edges) with EWC protection
        4. Token erosion — differentially decay token weights
        5. Archive raw content for memories below archive threshold
        6. Prune probabilistically based on MESU uncertainty
        7. Migrate regions based on temperature
        """
        self._cycle_count += 1
        now = datetime.now(timezone.utc)

        stats = {
            "cycle": self._cycle_count,
            "processed": 0,
            "compressed": 0,
            "eroded": 0,
            "migrated": 0,
            "pruned": 0,
            "protected": 0,
        }

        memories = self.store.get_all_active()
        total_queries = self.store.get_total_query_count()

        # 0. Consolidate entity graph (strengthen co-accessed, fan-decay, prune)
        entity_graph = EntityGraphStore(self.store.conn)
        last_consol = None
        if memories:
            last_consol = next(
                (m.last_consolidated for m in memories if m.last_consolidated),
                None
            )
        pruned_edges = entity_graph.consolidate_entity_graph(last_consol)
        stats["edges_pruned"] = pruned_edges


        for mem in memories:
            # 1. Update GABA inhibition: increases with disuse
            cycles_since_access = self._cycles_since_access(mem, now)
            mem.gaba_inhibition = min(0.95, 1.0 - np.exp(-0.3 * cycles_since_access))

            # 1a. Decay retrieval importance once per cycle (was per-query;
            # see _update_retrieval_importance for rationale).
            mem.retrieval_importance = max(
                0.0,
                mem.retrieval_importance * (1.0 - self.config.importance_decay_rate),
            )

            # 1b. Recompute DA from retrieval history (no outcome/rating tracking yet).
            mem.da_relevance = compute_da(
                access_count=mem.access_count,
                avg_outcome=0.0,
                user_rating=0.0,
                goal_alignment=0.0,
            )

            # 1c. NE decays toward 0 with age (novelty fades; old things stop
            # being surprising). Full O(N²) cosine recompute is too expensive
            # for the base path — exponential decay is the biological analog.
            age_days = (now - mem.last_accessed).total_seconds() / 86400.0 if mem.last_accessed else 1.0
            age_days = max(0.0, age_days)
            decayed_ne = mem.ne_novelty * math.exp(-0.05 * age_days)
            # Clamp subnormals — pg `real` rejects values smaller than ~1e-37
            mem.ne_novelty = decayed_ne if decayed_ne > 1e-30 else 0.0

            # 2. Compute temperature using new 7-stage Dendric activation equation
            # Converts access times from datetime to days-ago float
            accesses_days_ago = [
                (now - t).total_seconds() / 86400.0 for t in mem.access_times
            ] if mem.access_times else [1.0]

            # Compute spreading activation from entity graph connectivity
            sa = entity_graph.compute_spreading_activation(mem.id)

            new_temp = compute_temperature(
                accesses_days_ago=accesses_days_ago,
                da_relevance=mem.da_relevance,
                ne_novelty=mem.ne_novelty,
                gaba_inhibition=mem.gaba_inhibition,
                spreading_activation=sa * self.config.spreading_activation_weight,
                noise=self.config.activation_use_noise,
                use_gane=self.config.activation_use_gane,
            )

            # 3. Update signal histories
            da_history = (mem.da_history + [mem.da_relevance])[-self.config.uncertainty_window:]
            ne_history = (mem.ne_history + [mem.ne_novelty])[-self.config.uncertainty_window:]

            usage = compute_usage(mem.access_count, mem.retrieval_hits, cycles_since_access)
            uncertainty = compute_signal_uncertainty(da_history, ne_history, [usage])

            # 4. Deterministic compression with EWC protection
            compressions = {}
            resistance = compute_compression_resistance(
                mem.retrieval_importance, uncertainty, new_temp
            )

            mem_dict = {
                "temperature": new_temp,
                "structured_summary": mem.structured_summary,
                "knowledge_nugget": mem.knowledge_nugget,
                "entity_edges": mem.entity_edges,
            }

            # Check if any compression level should fire
            needs_compression = any(
                should_compress(mem_dict, level, resistance)
                for level in ["structured_summary", "knowledge_nugget", "entity_edges"]
            )

            # Skip re-compression if temperature hasn't changed significantly
            if (mem.compression_temperature is not None
                    and abs(new_temp - mem.compression_temperature) < 0.05):
                needs_compression = False

            if needs_compression and mem.raw_content:
                token_scores = mem.token_weights or []
                compressed = compress(
                    mem.raw_content,
                    token_scores,
                    temperature=new_temp,
                )

                # Apply each level that should compress
                for level in ["structured_summary", "knowledge_nugget", "entity_edges"]:
                    if should_compress(mem_dict, level, resistance):
                        if level == "structured_summary":
                            compressions[level] = compressed['structured_summary']
                        elif level == "knowledge_nugget":
                            # Store nugget as the top nugget's text
                            nuggets = compressed['knowledge_nuggets']
                            compressions[level] = nuggets[0]['nugget'] if nuggets else ''
                        elif level == "entity_edges":
                            # Store edges as JSON string
                            compressions[level] = json.dumps(compressed['entity_edges'])
                        stats["compressed"] += 1
                    elif not should_compress(mem_dict, level, 0.0) and should_compress(
                        {**mem_dict, "temperature": new_temp}, level, 0.0
                    ):
                        stats["protected"] += 1

                # Store full nuggets list and compression metadata
                compressions["knowledge_nuggets"] = json.dumps(compressed['knowledge_nuggets'])
                compressions["last_compressed_at"] = now
                compressions["compression_temperature"] = new_temp

            # 5. Token erosion — differentially decay token weights
            if mem.token_weights:
                mem.token_weights = erode_tokens(
                    mem.token_weights,
                    base_decay=self.config.erosion_base_decay,
                )
                stats["eroded"] += 1
            elif mem.raw_content:
                # Backfill: initialize token weights for pre-existing memories
                tokens = tokenize(mem.raw_content)
                mem.token_weights = score_token_importance(
                    tokens, known_entities=list(mem.co_entities or [])
                )
                mem.token_weights = erode_tokens(
                    mem.token_weights,
                    base_decay=self.config.erosion_base_decay,
                )
                stats["eroded"] += 1

            # 6. Archive if below threshold
            new_region = mem.region
            if new_temp < COMPRESSION_THRESHOLDS["archive"] and mem.region != "archive":
                if mem.raw_content:
                    self.store.archive_raw(mem.id, mem.raw_content, {
                        "source": mem.source,
                        "context": mem.context,
                        "time_of_day": mem.time_of_day,
                        "day_of_week": mem.day_of_week,
                    })
                new_region = "archive"
                stats["migrated"] += 1
            elif new_temp < 0.65 and mem.region == "hippocampus":
                new_region = "neocortex"
                stats["migrated"] += 1

            # 7. MESU probabilistic pruning
            prune_prob = compute_prune_probability(
                new_temp, uncertainty, mem.retrieval_importance,
                mem.access_count, cycles_since_access,
                base_prune_rate=self.config.base_prune_rate,
            )

            if should_prune(prune_prob):
                self.store.delete(mem.id)
                stats["pruned"] += 1
                continue

            # 8. Persist compression entity edges to graph
            if compressions.get("entity_edges"):
                try:
                    edge_data = json.loads(compressions["entity_edges"])
                    entity_graph = EntityGraphStore(self.store.conn)
                    for edge in edge_data:
                        entity_graph.upsert_compression_edge(
                            source_name=edge['source'],
                            source_type=edge['source_type'],
                            target_name=edge['target'],
                            target_type=edge['target_type'],
                            predicate=edge['predicate'],
                            weight=edge['weight'],
                        )
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to persist compression edges: {e}")

            # 9. Update memory
            update = {
                "temperature": new_temp,
                "region": new_region,
                "da_history": da_history,
                "ne_history": ne_history,
                "signal_variance": uncertainty,
                "usage_score": usage,
                "last_consolidated": now,
                "retrieval_importance": mem.retrieval_importance,
            }
            if mem.token_weights is not None:
                update["token_weights"] = json.dumps(mem.token_weights)
            update.update(compressions)
            self.store.update_fields(mem.id, update)

            stats["processed"] += 1

        return stats

    def _cycles_since_access(self, mem: Memory, now: datetime = None) -> int:
        if now is None:
            now = datetime.now(timezone.utc)
        if not mem.last_accessed:
            return 100
        hours = (now - mem.last_accessed).total_seconds() / 3600
        return max(0, int(hours / 8))  # ~3 cycles per day

    def _with_persona(self, entities):
        """Append the configured persona to an entity list if not already present.

        First-person memory streams have an implicit owner. Linking every
        memory to a persona node makes that fact structural — spreading
        activation seeded at the persona reaches all memories in its stream,
        and queries that mention the persona by name (where memories say
        'I' / 'my') can resolve via the persona node.

        Persona is stored with a low salience (0.2 vs 0.5 for explicit entities)
        so the link is real but doesn't dominate per-memory entity weight.
        """
        persona = (self.config.persona or "").strip().lower()
        if not persona:
            return entities
        if any(c.lower() == persona for _, _, c in entities):
            return entities
        return list(entities) + [(self.config.persona, "named", persona)]

    def _link_entities(self, entity_graph, memory_id, entities, session_id):
        """Upsert entities, link to memory, and track session.

        Persona is special: it's the implicit owner, not a peer entity. It
        gets a memory→persona link (so spreading activation can fall back to
        it for persona-only queries) but NOT co-occurrence edges to other
        entities. Without this exception, every entity ends up edge-connected
        to the persona, and activation from any entity hops via persona to
        everything — destroying the specificity of spreading activation.
        """
        entities = self._with_persona(entities)
        persona_canonical = (self.config.persona or "").strip().lower()
        entity_ids = []
        non_persona_ids = []  # for co-occurrence edges (excludes persona)
        for name, etype, canonical in entities:
            eid = entity_graph.upsert_entity(canonical, etype, name)
            entity_graph.update_entity_session(eid, session_id)
            is_persona = canonical == persona_canonical
            salience = 0.2 if is_persona else 0.5
            entity_graph.insert_memory_entity(memory_id, eid, salience=salience)
            entity_ids.append(eid)
            if not is_persona:
                non_persona_ids.append(eid)
        # Co-occurrence edges between non-persona entities only.
        for i in range(len(non_persona_ids)):
            for j in range(i + 1, len(non_persona_ids)):
                entity_graph.upsert_entity_edge(non_persona_ids[i], non_persona_ids[j])

    def _best_content(self, mem: Memory) -> str:
        return (
            mem.raw_content
            or mem.structured_summary
            or mem.knowledge_nugget
            or mem.entity_edges
            or ""
        )

    # ── FORGET ───────────────────────────────────────────────────

    def forget(self, memory_id: str = None, below_temp: float = None) -> dict:
        if memory_id:
            self.store.delete(memory_id)
            return {"pruned": 1}

        if below_temp is not None:
            count = self.store.prune_below_temp(below_temp)
            return {"pruned": count}

        return {"pruned": 0}

    # ── STATS ────────────────────────────────────────────────────

    def stats(self) -> dict:
        return self.store.stats()

    def get_all(self, limit: int = 300) -> List[dict]:
        memories = self.store.get_all(limit=limit)
        return [m.to_dict() for m in memories]

    def get_memory(self, memory_id: str) -> Optional[dict]:
        mem = self.store.get(memory_id)
        if mem:
            return mem.to_dict()
        return None

    def close(self):
        self.store.close()
