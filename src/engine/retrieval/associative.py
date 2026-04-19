"""
Path 3: Spreading-Activation Associative Recall.

Replaces the prior "similarity to recent context" implementation with the
real mechanism: seed activation at query-touched entity nodes, propagate
through co-occurrence edges with decay over hops, and return memories
linked to the most-activated entities.

This is the path that makes "spreading activation through the associative
network" actually true rather than a euphemism for context-window cosine.
The graph is queried at retrieval time (not just at consolidation), and
detail-heavy questions find what compressed summaries lost.

Algorithm:
    1. Resolve query entity names → entity_ids in the graph
    2. activation[seed] = 1.0 for each seed
    3. For each hop in [1..max_hops]:
         For each entity with current activation:
           For each edge (entity ↔ neighbor):
             spread = current_act × edge_weight × decay^hop
             activation[neighbor] = max(activation[neighbor], spread)
       (max-aggregation over paths is more stable than sum at low hop counts)
    4. For each entity with activation > threshold:
         Pull its linked memories, score = activation[entity] × salience
    5. Aggregate per-memory scores (sum across all activated entities the
       memory is linked to)
    6. Return top_k by score
"""

import math
from collections import defaultdict
from typing import List, Dict, Any, Iterable, Optional

import psycopg2.extras

from ..storage.entity_graph import EntityGraphStore


def spreading_activation_recall(
    query_entities: Iterable[str],
    store,
    top_k: int = 10,
    max_hops: int = 2,
    decay: float = 0.5,
    activation_threshold: float = 0.05,
    min_temp: float = 0.0,
    archive_trigger_threshold: Optional[float] = None,
    persona: str = "",
    persona_seed_activation: float = 0.5,
    persona_fallback: bool = True,
) -> List[Dict[str, Any]]:
    """Spread activation from query entities and return the most-activated memories.

    Args:
        query_entities: Canonical names extracted from the query.
        store: PostgresStore.
        top_k: Number of memories to return.
        max_hops: How far to propagate (2 = seed → neighbor → neighbor-of-neighbor).
        decay: Per-hop multiplicative decay (0.5 = halve at each hop).
        activation_threshold: Entities below this activation don't contribute.
        min_temp: Filter active-region memories below this temperature.
        archive_trigger_threshold: If set, also pull archived memories linked
            to entities whose activation exceeds this threshold (déjà-vu trigger,
            wired in task C). None disables.
        persona: Canonical persona name. The persona seed gets a softer initial
            activation than other entities since "this is about me" is weaker
            evidence than "this is about Mango."
        persona_seed_activation: Initial activation for the persona seed (default
            0.5). Other entity seeds always start at 1.0.
        persona_fallback: When True and no query entities resolve to the graph,
            seed at the persona instead. This lets queries about the agent itself
            (e.g. "Has Jamie had X?") reach memories that say "I had X."

    Returns: list of memory dicts with `score` (total activation) and
        `retrieval_paths=['associative']`.
    """
    seeds = [s for s in (query_entities or []) if s]
    persona = (persona or "").strip().lower()

    eg = EntityGraphStore(store.conn)

    # Step 1: resolve seed entity names → ids, separating persona from others.
    # Persona is held back as a fallback rather than added alongside specific
    # entities — when a query has any other entity that resolves, the persona's
    # universal-owner contribution would just dilute the specific signal at
    # small top_k. Persona only seeds when nothing else does.
    non_persona_seeds: List = []  # (entity_id, initial_activation)
    persona_in_query = False
    for name in seeds:
        canonical = (name or "").strip().lower()
        if canonical == persona:
            persona_in_query = True
            continue  # held for fallback
        eid = eg.get_entity_by_name(canonical)
        if eid is not None:
            non_persona_seeds.append((eid, 1.0))

    seed_ids_with_act: List = list(non_persona_seeds)

    # Persona seed fires when EITHER:
    #   (a) the query mentions the persona by name and no specific entity
    #       resolved (e.g. "What does Jamie like?" — pure persona query), OR
    #   (b) persona_fallback is on and no query entities resolved at all
    #       (e.g. "Has Jamie ever been to Portugal?" with no "portugal" entity).
    needs_persona = (
        persona
        and not non_persona_seeds
        and (persona_in_query or persona_fallback)
    )
    if needs_persona:
        persona_eid = eg.get_entity_by_name(persona)
        if persona_eid is not None:
            seed_ids_with_act.append((persona_eid, persona_seed_activation))

    if not seed_ids_with_act:
        return []

    # Step 2 + 3: BFS with max-aggregation activation propagation
    activation: Dict = {eid: act for eid, act in seed_ids_with_act}
    frontier = [eid for eid, _ in seed_ids_with_act]
    for hop in range(1, max_hops + 1):
        next_frontier = []
        hop_factor = decay ** hop
        for eid in frontier:
            current_act = activation.get(eid, 0.0)
            if current_act < activation_threshold:
                continue
            for edge in eg.get_edges_for_entity(eid):
                neighbor = edge["entity_b"] if edge["entity_a"] == eid else edge["entity_a"]
                edge_weight = float(edge.get("weight", 1.0))
                spread = current_act * edge_weight * hop_factor
                if spread > activation.get(neighbor, 0.0):
                    activation[neighbor] = spread
                    next_frontier.append(neighbor)
        frontier = next_frontier
        if not frontier:
            break

    # Step 4 + 5: collect memories linked to activated entities, score by sum
    # Each entity's contribution is normalized by 1/sqrt(fan_out) so very-high-
    # degree nodes (the persona, which links to every memory) don't dominate.
    # sqrt is gentler than 1/fan; persona contribution stays meaningful but
    # doesn't swamp specific-entity contributions.
    activated_entities = [
        (eid, act) for eid, act in activation.items() if act >= activation_threshold
    ]
    if not activated_entities:
        return []

    memory_scores: Dict[str, float] = defaultdict(float)
    memory_paths: Dict[str, set] = defaultdict(set)
    for eid, act in activated_entities:
        linked = list(eg.get_memories_for_entity(eid))
        if not linked:
            continue
        # Fan-out normalization: divide each contribution by sqrt(num_linked).
        # A persona node linked to 500 memories contributes act/sqrt(500)≈act/22
        # per memory; an entity linked to 5 memories contributes act/sqrt(5)≈act/2.2.
        norm = 1.0 / math.sqrt(len(linked))
        per_memory_contrib = act * norm
        for mid in linked:
            memory_scores[str(mid)] += per_memory_contrib
            memory_paths[str(mid)].add(str(eid))

    if not memory_scores:
        return []

    # Step 6: fetch memory details in one round-trip, apply min_temp filter,
    # optionally pull archived memories for high-activation entities (déjà-vu).
    archive_triggered_ids: List[str] = []
    if archive_trigger_threshold is not None:
        archive_triggered_entities = [
            eid for eid, act in activated_entities if act >= archive_trigger_threshold
        ]
        if archive_triggered_entities:
            for eid in archive_triggered_entities:
                for mid in eg.get_memories_for_entity(eid):
                    archive_triggered_ids.append(str(mid))

    candidate_ids = list(memory_scores.keys())
    rows = _fetch_memories(
        store.conn, candidate_ids, min_temp, archive_triggered_ids
    )

    results: List[Dict[str, Any]] = []
    for row in rows:
        mid = str(row["id"])
        text = (
            row.get("raw_content")
            or row.get("structured_summary")
            or row.get("knowledge_nugget")
            or row.get("entity_edges")
            or ""
        )
        # Tag archived memories distinctly so path stats can show déjà-vu
        # firing separately from normal associative recall.
        is_archive = row.get("region") == "archive"
        paths = ["associative_archive"] if is_archive else ["associative"]
        results.append({
            "id": mid,
            "text": text,
            "content": text,
            "raw_content": text,
            "temperature": float(row.get("temperature", 0.0)),
            "region": row.get("region", "neocortex"),
            "context": row.get("context", ""),
            "created_at": row.get("created_at"),
            "compression_level": row.get("compression_level", "raw"),
            "score": memory_scores[mid],
            "similarity": min(1.0, memory_scores[mid] / max(1.0, len(seed_ids_with_act))),
            "retrieval_paths": paths,
            "_activated_entities": len(memory_paths[mid]),
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


def _fetch_memories(conn, ids: List[str], min_temp: float, archive_ids: List[str]):
    """Single round-trip fetch.

    Two queries on the same candidate set, split by region:
      - Active memories: region != 'archive', filtered by min_temp.
      - Archived memories: region = 'archive', no temp filter, but ONLY if the
        id is also in archive_ids (i.e., entity activation crossed the
        déjà-vu threshold). Without that gate, every spreading-activation
        recall would surface archived memories whether evoked or not, which
        defeats the point of archive being dormant-but-reachable.
    """
    if not ids:
        return []

    rows = []
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id, raw_content, structured_summary, knowledge_nugget,
                   entity_edges, temperature, region, context, created_at,
                   compression_level
            FROM memories
            WHERE id::text = ANY(%s)
              AND region != 'archive'
              AND temperature >= %s
        """, (ids, min_temp))
        rows.extend(cur.fetchall())

        # Archive query: only fires if there are archive-triggered ids, AND the
        # candidate must be in BOTH ids (had activation) and archive_ids
        # (entity crossed threshold).
        archive_candidates = list(set(archive_ids) & set(ids)) if archive_ids else []
        if archive_candidates:
            cur.execute("""
                SELECT id, raw_content, structured_summary, knowledge_nugget,
                       entity_edges, temperature, region, context, created_at,
                       compression_level
                FROM memories
                WHERE id::text = ANY(%s)
                  AND region = 'archive'
            """, (archive_candidates,))
            rows.extend(cur.fetchall())
    return rows


# ── Backwards-compatible shims ────────────────────────────────────────
# The engine still imports `associative_recall` and `build_context`.
# Keep those names alive but route the engine through the new function via
# its updated call site (engine.recall now passes query entities, not a
# context dict). build_context is no longer used by the new path but is
# kept for callers that might still rely on it.

def associative_recall(query_entities, store, top_k: int = 10, **kwargs):
    """Deprecated alias — calls spreading_activation_recall directly."""
    return spreading_activation_recall(query_entities, store, top_k=top_k, **kwargs)


def build_context(source: str = "", location: str = "") -> dict:
    """Legacy helper — kept for callers that still construct context dicts."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    hour = now.hour
    if hour < 6:
        tod = "night"
    elif hour < 12:
        tod = "morning"
    elif hour < 18:
        tod = "afternoon"
    else:
        tod = "evening"
    ctx = {"time_of_day": tod, "day_of_week": now.strftime("%A").lower()}
    if source:
        ctx["source"] = source
    if location:
        ctx["location"] = location
    return ctx
