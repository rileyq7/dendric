"""
Memory Engine — Public API

remember(), recall(), consolidate(), forget(), stats()
"""

import json
import logging
import re
import numpy as np
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from ..config import EngineConfig
from .memory import Memory, get_band, get_format
from .decay import compute_activation, activation_to_temperature
from .activation import compute_temperature  # New 7-stage equation
from .signals import compute_da, compute_ne, compute_usage
from .signals_enhanced import compute_da as compute_da_enhanced, compute_ne as compute_ne_enhanced, compute_gaba
from .pruning import compute_signal_uncertainty, compute_prune_probability, should_prune
from .protection import update_retrieval_importance, compute_compression_resistance
from .compression import should_compress, compress, COMPRESSION_THRESHOLDS
from .erosion import tokenize, score_token_importance, erode_tokens
from ..retrieval.vector import vector_recall
from ..retrieval.keyword import keyword_recall
from ..retrieval.associative import associative_recall, build_context
from ..retrieval.fusion import fuse_results
from ..retrieval.temporal import detect_temporal_query, temporal_rerank, ensure_temporal_diversity
from ..retrieval.graph import graph_recall
from ..retrieval.temporal_decomposer import decompose as decompose_temporal, compute_temporal_hint
from ..storage.postgres import PostgresStore
from ..storage.entity_graph import EntityGraphStore
from ..embeddings.embed import embed, embed_query, get_model, get_embed_dim
from ..utils import extract_session_id, parse_context_date
from .entity_extraction import extract_entities, extract_entities_gemma

logger = logging.getLogger(__name__)

# Common words misidentified as named entities (capitalized at sentence start,
# after [Assistant]:, etc.). Filter these out of entity graph tracking.
_ENTITY_STOPWORDS = {
    "here", "the", "this", "that", "it's", "i'm", "i'll", "i've", "i'd",
    "you", "you're", "you'll", "we", "we're", "they", "there", "these",
    "those", "what", "when", "where", "which", "who", "how", "why",
    "yes", "no", "not", "but", "and", "or", "so", "if", "then",
    "also", "just", "very", "really", "well", "sure", "great", "good",
    "some", "many", "much", "more", "most", "other", "another", "such",
    "like", "want", "need", "can", "will", "would", "could", "should",
    "let", "make", "get", "take", "give", "have", "had", "has", "was",
    "were", "been", "being", "are", "did", "does", "done",
    "user", "assistant", "session context", "user: i'm", "user: i",
    "here are", "one", "two", "three", "four", "five",
    "remember", "think", "know", "see", "look", "try", "keep",
    "first", "last", "new", "old", "next", "start", "end",
    "absolutely", "definitely", "certainly", "exactly", "actually",
    "however", "although", "while", "since", "because", "therefore",
    "sure", "sure,", "okay", "right", "thanks", "thank", "please",
    "maybe", "perhaps", "probably", "might", "today", "tomorrow",
    "based", "here's", "that's", "there's", "what's", "it",
}


class MemoryEngine:

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.store = PostgresStore(self.config.db_url)
        self._cycle_count = 0

        # Initialize embedding model
        get_model(self.config.embed_model)


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

        # Compute DA (relevance)
        da = compute_da(content, source, self.config.goals)

        # Compute NE (novelty)
        recent_embeddings = self.store.get_recent_embeddings(limit=50)
        ne = compute_ne(content_embedding, recent_embeddings)

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
                # Multi-word named entities (Dr. Patel, Austin Film Festival) are reliable
                if ' ' in c_clean and len(c_clean) > 5:
                    entities.append((n, t, c_clean))
                # Single-word named: skip — too many false positives from capitalized
                # sentence starts. The known model/dataset lists in extract_entities
                # already handle important single-word proper nouns.
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

        # High-salience boost
        if da > 0.5:
            mem.temperature = min(1.0, 1.0 + 0.05)

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
    ) -> List[dict]:
        """Batch-ingest multiple memories: one embedding call, one DB transaction.

        Each item: {"content": str, "source": str, "context": str}
        link_entities: If True, extract and link entities (slower but better retrieval).
        Returns list of memory dicts.
        """
        from ..embeddings.embed import embed_batch as _embed_batch

        if not items:
            return []

        now = datetime.now(timezone.utc)
        hour = now.hour
        tod = "night" if hour < 6 else "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"
        dow = now.strftime("%A").lower()

        # ── Step 1: Batch embed all content via OpenAI API ──
        texts = [item["content"] for item in items]
        logger.info(f"Batch embedding {len(texts)} items...")
        embeddings = _embed_batch(texts, self.config.embed_model)
        logger.info(f"Embedding complete, building memories...")

        # ── Step 2: Build all Memory objects (CPU-only, no DB) ──
        memories = []
        entity_work = []  # (mem_id, entities, session_id) for deferred entity linking

        for item, content_embedding in zip(items, embeddings):
            content = item["content"]
            source = item.get("source", "direct")
            context = item.get("context", "")

            da = compute_da(content, source, self.config.goals)

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
                        if ' ' in c_clean and len(c_clean) > 5:
                            entities.append((n, t, c_clean))
                        continue
                    if t == "numeric":
                        entities.append((n, t, c_clean))
                entity_canonicals = [canonical for _, _, canonical in entities]

            session_id = extract_session_id(context)

            mem = Memory(
                raw_content=content, temperature=1.0, region="hippocampus",
                da_relevance=da, ne_novelty=0.5, usage_score=0.0,
                da_history=[da], ne_history=[0.5], signal_variance=0.5,
                embedding=content_embedding, source=source, context=context,
                time_of_day=tod, day_of_week=dow,
                created_at=now, last_accessed=now, access_times=[now], access_count=0,
                compression_level="raw",
                tokens_original=len(content.split()), tokens_current=len(content.split()),
                co_entities=entity_canonicals,
            )

            memories.append(mem)
            if link_entities and entities:
                entity_work.append((mem.id, entities, session_id))

        # ── Step 3: Single DB transaction for all memory inserts ──
        logger.info(f"Inserting {len(memories)} memories into DB...")
        self.store.store_batch(memories)

        # ── Step 4: Batch entity graph updates (single transaction) ──
        if entity_work:
            logger.info(f"Linking entities for {len(entity_work)} memories...")
            entity_graph = EntityGraphStore(self.store.conn, autocommit=False)
            for mem_id, entities, session_id in entity_work:
                entity_ids = []
                for name, etype, canonical in entities:
                    eid = entity_graph.upsert_entity(canonical, etype, name)
                    entity_graph.update_entity_session(eid, session_id)
                    entity_graph.insert_memory_entity(mem_id, eid, salience=0.5)
                    entity_ids.append(eid)
                for i in range(len(entity_ids)):
                    for j in range(i + 1, len(entity_ids)):
                        entity_graph.upsert_entity_edge(entity_ids[i], entity_ids[j])
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

        # Four parallel retrieval paths
        vec_results = vector_recall(query_emb, self.store, top_k=overfetch, min_temp=min_temp)
        kw_results = keyword_recall(query, self.store, top_k=overfetch, min_temp=min_temp)

        ctx = build_context()
        assoc_results = associative_recall(ctx, self.store, top_k=5)

        # Path 4: Entity graph walk
        query_entity_tuples = extract_entities(query)
        query_entity_names = [canonical for _, _, canonical in query_entity_tuples]
        graph_results = graph_recall(
            query_entity_names, self.store, top_k=overfetch,
        )

        logger.debug(
            f"Recall paths for '{query[:60]}': "
            f"vec={len(vec_results)} kw={len(kw_results)} "
            f"assoc={len(assoc_results)} graph={len(graph_results)} "
            f"(entities={query_entity_names})"
        )

        # Temporal decomposition: for temporal queries, also retrieve using
        # stripped event queries to improve recall on time-anchored questions
        temporal_decomp = decompose_temporal(query, question_timestamp)
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
        )

        # Temporal re-ranking (Path 4) — only activates for temporal queries
        temporal_info = detect_temporal_query(query, question_timestamp)
        if temporal_info['is_temporal']:
            fused = temporal_rerank(
                fused, temporal_info, question_timestamp,
                temporal_weight=self.config.temporal_weight,
                rrf_k=self.config.rrf_k,
            )
            fused = ensure_temporal_diversity(fused, temporal_info, top_k=top_k * 2)

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

        # Reheat accessed memories + update EWC
        top_k_ids = [r["id"] for r in results]
        all_ids = [r["id"] for r in fused]

        if reheat:
            for result in results:
                mem = self.store.get(result["id"])
                if mem:
                    self._reheat(mem, result.get("similarity", result.get("fusion_score", 0.5)))
                    # Update result with reheated temp
                    result["temperature"] = mem.temperature

        # Log retrieval for EWC
        try:
            self.store.log_retrieval(query, query_emb, all_ids[:50], top_k_ids)
        except Exception as e:
            logger.warning(f"Failed to log retrieval: {e}")
            try:
                self.store.conn.rollback()
            except Exception:
                pass

        # Update retrieval importance for all memories
        self._update_retrieval_importance(top_k_ids)

        return results

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
        """Update EWC importance for all memories based on this retrieval."""
        all_mems = self.store.get_all(limit=500)
        for mem in all_mems:
            was_in_top_k = mem.id in top_k_ids
            new_importance = update_retrieval_importance(
                was_in_top_k,
                mem.retrieval_importance,
                learning_rate=self.config.importance_learning_rate,
                decay_rate=self.config.importance_decay_rate,
            )
            if was_in_top_k:
                mem.retrieval_hits += 1
            mem.retrieval_importance = new_importance
            self.store.update_fields(mem.id, {
                "retrieval_importance": new_importance,
                "retrieval_hits": mem.retrieval_hits,
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
                noise=True,  # Add stochasticity to consolidation
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
            }
            if mem.token_weights is not None:
                update["token_weights"] = json.dumps(mem.token_weights)
            update.update(compressions)
            self.store.update_fields(mem.id, update)

            stats["processed"] += 1

        # 9. Aggregate synthesis for high-fan-out entities
        agg_count = self._synthesize_aggregates()
        stats["aggregates_created"] = agg_count

        return stats

    def _cycles_since_access(self, mem: Memory, now: datetime = None) -> int:
        if now is None:
            now = datetime.now(timezone.utc)
        if not mem.last_accessed:
            return 100
        hours = (now - mem.last_accessed).total_seconds() / 3600
        return max(0, int(hours / 8))  # ~3 cycles per day

    def _link_entities(self, entity_graph, memory_id, entities, session_id):
        """Upsert entities, link to memory, and track session."""
        entity_ids = []
        for name, etype, canonical in entities:
            eid = entity_graph.upsert_entity(canonical, etype, name)
            entity_graph.update_entity_session(eid, session_id)
            entity_graph.insert_memory_entity(memory_id, eid, salience=0.5)
            entity_ids.append(eid)
        # Create co-occurrence edges between entities in same memory
        for i in range(len(entity_ids)):
            for j in range(i + 1, len(entity_ids)):
                entity_graph.upsert_entity_edge(entity_ids[i], entity_ids[j])

    def gemma_reextract_pass(
        self,
        max_memories: Optional[int] = None,
        only_unprocessed: bool = True,
        ollama_url: str = "http://127.0.0.1:11434",
        model: str = "gemma4:e2b",
    ) -> dict:
        """
        Offline re-extraction pass using Gemma 4 E2B.

        For each eligible memory:
        1. Re-extract entities + relations from raw_content via Gemma
        2. Wipe existing (noisy) entity links for that memory
        3. Insert clean Gemma entities and link them
        4. Persist relation triples as typed entity_edges

        Slow (~20s/memory) — intended to run between consolidation cycles
        offline. Result: cleaner entity graph, richer relations, better
        aggregates downstream.

        Args:
            max_memories: Cap on memories to process (None = all)
            only_unprocessed: Skip memories already re-extracted (gemma_reextracted=True)
            ollama_url: Ollama server endpoint
            model: Ollama model tag

        Returns: stats dict
        """
        import psycopg2.extras
        stats = {
            "processed": 0,
            "entities_added": 0,
            "triples_added": 0,
            "skipped_no_content": 0,
            "errors": 0,
        }

        # Select memories to process
        with self.store.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            where = "WHERE raw_content IS NOT NULL AND raw_content != ''"
            if only_unprocessed:
                where += " AND (gemma_reextracted IS NULL OR gemma_reextracted = FALSE)"
            where += " AND (source IS NULL OR source != 'aggregate')"
            limit = f"LIMIT {int(max_memories)}" if max_memories else ""
            cur.execute(f"""
                SELECT id, raw_content, context
                FROM memories
                {where}
                ORDER BY created_at
                {limit}
            """)
            rows = cur.fetchall()

        if not rows:
            logger.info("gemma_reextract_pass: no memories to process")
            return stats

        logger.info(f"gemma_reextract_pass: processing {len(rows)} memories with {model}")
        entity_graph = EntityGraphStore(self.store.conn)

        for i, row in enumerate(rows):
            mid = row["id"]
            content = row["raw_content"]
            if not content:
                stats["skipped_no_content"] += 1
                continue

            session_id = extract_session_id(row.get("context", ""))

            try:
                entities, triples = extract_entities_gemma(
                    content, ollama_url=ollama_url, model=model
                )
            except Exception as e:
                logger.warning(f"Gemma extraction failed for {mid}: {e}")
                stats["errors"] += 1
                continue

            # Wipe existing entity links for this memory (replace with Gemma's)
            entity_graph.delete_memory_entities(mid)

            # Insert clean Gemma entities + links
            entity_ids = []
            for name, etype, canonical in entities:
                eid = entity_graph.upsert_entity(canonical, etype, name)
                if session_id:
                    entity_graph.update_entity_session(eid, session_id)
                entity_graph.insert_memory_entity(mid, eid, salience=0.7)
                entity_ids.append(eid)
                stats["entities_added"] += 1

            # Co-occurrence edges among entities in this memory
            for a in range(len(entity_ids)):
                for b in range(a + 1, len(entity_ids)):
                    entity_graph.upsert_entity_edge(entity_ids[a], entity_ids[b])

            # Persist Gemma relation triples as typed/predicated edges
            for subj, rel, obj in triples:
                try:
                    entity_graph.upsert_compression_edge(
                        source_name=subj.lower().strip(),
                        source_type="named",
                        target_name=obj.lower().strip(),
                        target_type="named",
                        predicate=rel.lower().strip()[:64],
                        weight=1.0,
                    )
                    stats["triples_added"] += 1
                except Exception as e:
                    logger.debug(f"Edge insert failed: {e}")

            # Mark memory as re-extracted
            self.store.update_fields(mid, {"gemma_reextracted": True})
            stats["processed"] += 1

            if (i + 1) % 25 == 0:
                logger.info(
                    f"  gemma_reextract: {i+1}/{len(rows)}  "
                    f"entities={stats['entities_added']}  triples={stats['triples_added']}"
                )

        logger.info(
            f"gemma_reextract_pass DONE: processed={stats['processed']}  "
            f"entities={stats['entities_added']}  triples={stats['triples_added']}  "
            f"errors={stats['errors']}"
        )
        return stats

    def _synthesize_aggregates(self, min_sessions: int = 3) -> int:
        """Create or update aggregate memories for high-fan-out entities.

        For entities mentioned across many sessions, build a single retrievable
        memory that summarizes all instances — so aggregation queries can be
        answered within top_k=10 without needing to retrieve every mention.
        """
        import psycopg2.extras
        entity_graph = EntityGraphStore(self.store.conn)
        high_fan = entity_graph.get_high_fan_entities(min_sessions=min_sessions)

        if not high_fan:
            return 0

        count = 0
        for entity in high_fan:
            eid = entity["id"]
            canonical = entity["canonical_name"]
            etype = entity["entity_type"]
            session_ids = entity.get("session_ids") or []

            # Skip non-named entities, short names, and stopwords
            if etype not in ("named", "numeric"):
                continue
            if len(canonical) < 3 or canonical in _ENTITY_STOPWORDS:
                continue
            mention_count = entity.get("mention_count", 0)

            # Get all linked memory IDs
            memory_ids = entity_graph.get_memories_for_entity(eid)
            if not memory_ids:
                continue

            # Fetch memory content and dates
            instances = []
            with self.store.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, raw_content, knowledge_nugget, context, source
                    FROM memories
                    WHERE id = ANY(%s)
                    ORDER BY created_at
                """, (memory_ids,))
                rows = cur.fetchall()

            for row in rows:
                # Skip other aggregates
                if row.get("source") == "aggregate":
                    continue
                content = row.get("raw_content") or row.get("knowledge_nugget") or ""
                if not content:
                    continue
                ctx = row.get("context", "")
                dt = parse_context_date(ctx)
                date_str = dt.strftime("%Y/%m/%d") if dt else "unknown date"
                # Truncate to keep aggregate compact
                snippet = content[:150].replace("\n", " ").strip()
                instances.append(f"- ({date_str}) {snippet}")

            if not instances:
                continue

            # Build aggregate text
            n_instances = len(instances)
            n_sessions = len(session_ids)
            session_list = ", ".join(sorted(session_ids)[:10])
            instances_text = "\n".join(instances[:20])  # Cap at 20 to limit size

            aggregate_text = (
                f"[Aggregate: {canonical}] "
                f"Entity type: {etype}. "
                f"Mentioned {mention_count} times across {n_sessions} sessions "
                f"({session_list}).\n"
                f"Total distinct instances: {n_instances}.\n"
                f"Instances:\n{instances_text}"
            )

            # Check for existing aggregate
            existing_id = None
            with self.store.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT id FROM memories
                    WHERE source = 'aggregate' AND context LIKE %s
                    LIMIT 1
                """, (f"aggregate:{canonical}%",))
                row = cur.fetchone()
                if row:
                    existing_id = row["id"]

            aggregate_embedding = embed(aggregate_text, self.config.embed_model)

            if existing_id:
                # Update existing aggregate
                self.store.update_fields(existing_id, {
                    "raw_content": aggregate_text,
                    "embedding": aggregate_embedding,
                    "temperature": 0.8,
                })
            else:
                # Create new aggregate memory
                now = datetime.now(timezone.utc)
                mem = Memory(
                    raw_content=aggregate_text,
                    temperature=0.8,
                    region="neocortex",
                    da_relevance=0.5,
                    ne_novelty=0.5,
                    usage_score=0.0,
                    embedding=aggregate_embedding,
                    source="aggregate",
                    context=f"aggregate:{canonical}",
                    created_at=now,
                    last_accessed=now,
                    access_times=[now],
                    access_count=0,
                    compression_level="aggregate",
                    co_entities=[canonical],
                )
                self.store.store(mem)
                count += 1

            logger.info(
                f"{'Updated' if existing_id else 'Created'} aggregate for "
                f"'{canonical}': {n_instances} instances, {n_sessions} sessions"
            )

        return count

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
