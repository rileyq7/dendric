"""
Spreading activation for retrieval — the third path in three-path RRF fusion.

Computes ACT-R spreading activation for candidate memories based on entity overlap
with the query. This makes retrieval associative: memories connected via shared
entities get boosted even if they don't match the query text directly.
"""

import logging
from typing import List, Dict, Optional, Set, Tuple
import math

from ..core.entity_extraction import extract_entities
from ..storage.entity_graph import EntityGraphStore

logger = logging.getLogger(__name__)


def prepare_entity_cache(db_conn) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
    """
    Precompute entity cache for efficient SA computation.

    This should be called once per retrieval batch or retrieval session.

    Returns:
        (entity_cache, entity_fan_count)
        - entity_cache: {memory_id -> set(canonical_entity_names)}
        - entity_fan_count: {canonical_name -> fan_count}
    """
    entity_graph = EntityGraphStore(db_conn)

    # Get all memory-entity links
    with db_conn.cursor() as cur:
        cur.execute("""
            SELECT me.memory_id, e.canonical_name
            FROM memory_entities me
            JOIN entities e ON me.entity_id = e.id
        """)
        rows = cur.fetchall()

    entity_cache = {}
    for mem_id, entity_name in rows:
        mem_id_str = str(mem_id)
        if mem_id_str not in entity_cache:
            entity_cache[mem_id_str] = set()
        entity_cache[mem_id_str].add(entity_name)

    # Get fan counts
    entity_fan_count = entity_graph.get_entity_fan_counts()

    logger.info(f"Entity cache: {len(entity_cache)} memories, {len(entity_fan_count)} entities")
    return entity_cache, entity_fan_count


def compute_spreading_activation(
    memory_id: str,
    query_entities: List[str],
    entity_cache: Dict[str, Set[str]],
    entity_fan_count: Dict[str, int],
    W: float = 1.0,
    S_max: float = 2.0,
) -> float:
    """
    ACT-R spreading activation for a memory given query entities.

    SA = Σ_j (W / n_sources) * max(0, S_max - ln(fan_j))

    Where:
    - j iterates over entities in the query that also appear in this memory
    - W = total source activation (divided among query entities)
    - fan_j = how many memories contain entity j (more = weaker individual connections)
    - S_max = maximum associative strength

    Example:
    - Query for "Theo" (fan=5 memories)
    - Candidate memory contains "Theo"
    - S_ji = max(0, 2.0 - ln(5)) = 2.0 - 1.61 = 0.39
    - SA = (1.0 / 1) * 0.39 = 0.39

    Args:
        memory_id: UUID of memory to evaluate
        query_entities: List of canonical entity names from query
        entity_cache: {memory_id -> set(canonical_names)}
        entity_fan_count: {canonical_name -> fan_count}
        W: Source activation weight (default 1.0)
        S_max: Maximum associative strength (default 2.0, tuned for 0-1 scale)

    Returns:
        Spreading activation score (0.0+)
    """
    # Get entities for this memory
    mem_entities = entity_cache.get(memory_id, set())
    if not mem_entities:
        return 0.0

    # Find overlap
    query_set = set(query_entities)
    overlap = query_set.intersection(mem_entities)
    if not overlap:
        return 0.0

    # Divide activation across query entities
    n_sources = max(1, len(query_set))
    source_weight = W / n_sources

    total_SA = 0.0
    for entity_name in overlap:
        # Fan effect: more widespread entities produce weaker activation
        fan = entity_fan_count.get(entity_name, 1)

        # Associative strength
        # fan=1: S = 2.0 - 0 = 2.0 (maximum, rare entity)
        # fan=5: S = 2.0 - 1.61 = 0.39
        # fan=100: S = 2.0 - 4.61 = 0 (clamped, common entity)
        S_ji = max(0.0, S_max - math.log(max(1, fan)))

        total_SA += source_weight * S_ji

    return total_SA


def retrieve_with_spreading_activation(
    query: str,
    candidates: List[Dict],  # Results from vector + keyword retrieval
    db_conn,
    entity_cache: Optional[Dict[str, Set[str]]] = None,
    entity_fan_count: Optional[Dict[str, int]] = None,
    top_k: int = 10,
) -> List[Dict]:
    """
    Enhance retrieval candidates with spreading activation scores.

    Args:
        query: Query string
        candidates: Candidate memories from vector/keyword retrieval
                   Each should have: memory_id, vector_score, keyword_score
        db_conn: PostgreSQL connection
        entity_cache: Precomputed cache (or None to compute)
        entity_fan_count: Precomputed fan counts (or None to compute)
        top_k: Return top k results

    Returns:
        Candidates with spreading_activation and fusion scores
    """
    # Extract entities from query
    query_entities = extract_entities(query)
    query_entity_names = [canonical for _, _, canonical in query_entities]

    logger.debug(f"Query '{query}' extracted {len(query_entity_names)} entities")

    # Prepare caches if not provided
    if entity_cache is None or entity_fan_count is None:
        entity_cache, entity_fan_count = prepare_entity_cache(db_conn)

    # Compute SA for each candidate
    for candidate in candidates:
        mem_id = str(candidate["memory_id"])
        sa_score = compute_spreading_activation(
            mem_id,
            query_entity_names,
            entity_cache,
            entity_fan_count,
        )
        candidate["spreading_activation"] = sa_score

    # RRF fusion: reciprocal rank fusion of three paths
    # Rank each path independently, then combine
    candidates_sorted = rrf_fusion(candidates, top_k=top_k)

    return candidates_sorted[:top_k]


def rrf_fusion(
    candidates: List[Dict],
    k: int = 60,
    top_k: int = 10,
) -> List[Dict]:
    """
    Reciprocal rank fusion combining vector, keyword, and spreading activation scores.

    RRF formula: score = 1 / (k + rank)

    For each path:
    1. Sort by path score
    2. Assign ranks
    3. Compute RRF score
    4. Average across paths

    Args:
        candidates: List of {memory_id, vector_score, keyword_score, spreading_activation}
        k: RRF constant (typically 60)
        top_k: Return top k results

    Returns:
        Candidates sorted by fusion score
    """
    if not candidates:
        return []

    # Rank by each path
    by_vector = sorted(enumerate(candidates), key=lambda x: x[1].get("vector_score", 0), reverse=True)
    by_keyword = sorted(enumerate(candidates), key=lambda x: x[1].get("keyword_score", 0), reverse=True)
    by_sa = sorted(enumerate(candidates), key=lambda x: x[1].get("spreading_activation", 0), reverse=True)

    # Compute RRF for each candidate
    rrf_scores = {}
    for rank, (idx, _) in enumerate(by_vector, 1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank)

    for rank, (idx, _) in enumerate(by_keyword, 1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank)

    for rank, (idx, _) in enumerate(by_sa, 1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank)

    # Average and sort
    for idx, score in rrf_scores.items():
        candidates[idx]["fusion_score"] = score / 3.0

    result = sorted(candidates, key=lambda x: x.get("fusion_score", 0), reverse=True)

    logger.debug(f"RRF fusion: top result fusion_score={result[0].get('fusion_score', 0):.4f}")
    return result
