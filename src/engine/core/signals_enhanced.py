"""
Signal Computation for Dendric Activation

Computes DA (dopamine/relevance), NE (norepinephrine/novelty), and GABA (inhibition)
without LLM. These three signals feed into the 7-stage activation equation.

Reference:
- DA: retrieval frequency + goal alignment + outcome quality
- NE: dissimilarity to nearest neighbor + recency decay
- GABA: redundancy + staleness
"""

import math
from typing import List, Optional, Dict, Any


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    if not vec1 or not vec2:
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def compute_da(
    access_count: int,
    avg_outcome: float,
    user_rating: float,
    goal_alignment: float = 0.0,
    active_goal_weight: float = 0.4,
) -> float:
    """
    Compute dopamine (relevance/usefulness signal).

    Signals:
    - Retrieval frequency: accessed memories are useful
    - Goal alignment: relevant to active goals
    - User rating: explicit preference signal
    - Outcome quality: did it lead to good results

    Args:
        access_count: Total number of times retrieved (≥0)
        avg_outcome: Average outcome value ([-1, 1], or 0 for neutral)
        user_rating: Explicit user rating ([-1, 1], or 0 for unrated)
        goal_alignment: Cosine similarity to goal embeddings ([0, 1])
        active_goal_weight: Weight of goal alignment (0.4 typical)

    Returns:
        DA signal in [0, 1]
    """
    # Frequency component: saturates at high access counts
    freq = access_count / (access_count + 5)

    # Outcome component: normalize [-1, 1] to [0, 1]
    outcome_norm = (avg_outcome + 1) / 2

    # Rating component: normalize [-1, 1] to [0, 1]
    rating_norm = (user_rating + 1) / 2

    # Combine
    da = (
        0.25 * freq
        + active_goal_weight * goal_alignment
        + 0.2 * outcome_norm
        + 0.15 * rating_norm
    )

    return min(1.0, da)


def compute_ne(
    embedding: List[float],
    store_embeddings: List[List[float]],
    age_days: float = 1.0,
    recency_decay_rate: float = 0.05,
) -> float:
    """
    Compute norepinephrine (novelty/surprise signal).

    Measures dissimilarity to nearest neighbor, decayed by age.

    Args:
        embedding: This memory's embedding vector
        store_embeddings: List of other memories' embeddings (for NN search)
        age_days: Days since creation
        recency_decay_rate: Exponential decay constant (typical: 0.05)

    Returns:
        NE signal in [0, 1]
    """
    if not store_embeddings:
        return 1.0

    # Find nearest neighbor similarity
    similarities = [cosine_similarity(embedding, other) for other in store_embeddings]
    nearest_sim = max(similarities) if similarities else 0.0

    # Novelty: dissimilarity to nearest neighbor
    novelty = 1.0 - nearest_sim

    # Recency decay: novelty fades over time (old things stop being surprising)
    recency_decay = math.exp(-recency_decay_rate * age_days)

    ne = novelty * recency_decay
    return max(0.0, min(1.0, ne))


def compute_gaba(
    embedding: List[float],
    store_embeddings: List[List[float]],
    age_days: float = 1.0,
    stale_threshold: float = 30.0,
    recency_window: float = 10.0,
) -> float:
    """
    Compute GABA (inhibition/pruning signal).

    Combines redundancy (similarity to others) and staleness (time since access).

    Args:
        embedding: This memory's embedding vector
        store_embeddings: List of other memories' embeddings
        age_days: Days since last access
        stale_threshold: Days at which staleness becomes relevant (~30)
        recency_window: Sigmoid width for staleness curve (~10)

    Returns:
        GABA signal in [0, 1]
    """
    # Redundancy component: similar memories are more inhibited
    if store_embeddings:
        similarities = [cosine_similarity(embedding, other) for other in store_embeddings]
        max_sim = max(similarities) if similarities else 0.0
        # 0.8-1.0 similarity → 0-1 redundancy
        redundancy = max(0.0, (max_sim - 0.8) * 5)
    else:
        redundancy = 0.0

    # Staleness component: logistic curve centered at threshold
    staleness = 1.0 / (1.0 + math.exp(-(age_days - stale_threshold) / recency_window))

    # Combine
    gaba = 0.5 * min(1.0, redundancy) + 0.5 * staleness
    return min(1.0, gaba)


def compute_usage(
    access_count: int,
    retrieval_hits: int,
    cycles_since_last_access: int,
) -> float:
    """Usage score derived from access patterns."""
    freq = min(1.0, access_count / 20.0)
    relevance = min(1.0, retrieval_hits / 10.0)
    recency = max(0.0, 1.0 - (cycles_since_last_access / 30.0))
    return 0.4 * freq + 0.3 * relevance + 0.3 * recency


def update_signals_batch(
    memories: List[Dict[str, Any]],
    goal_embeddings: Optional[List[List[float]]] = None,
) -> None:
    """
    Update DA, NE, GABA for a batch of memories (during consolidation).

    Modifies memories in-place.

    Args:
        memories: List of memory dicts with 'embedding', 'accesses_days_ago', etc.
        goal_embeddings: Optional list of goal embeddings for DA computation
    """
    if not memories:
        return

    # Gather all embeddings for NE/GABA computation
    all_embeddings = [m.get("embedding") for m in memories if m.get("embedding")]

    for mem in memories:
        # Age of memory: use oldest access time, or 1 day if never accessed
        age_days = max(mem.get("accesses_days_ago", [1.0])) if mem.get("accesses_days_ago") else 1.0

        # DA: retrieval history + outcomes
        access_count = len(mem.get("accesses_days_ago", []))
        avg_outcome = mem.get("avg_outcome", 0.0)
        user_rating = mem.get("user_rating", 0.0)
        goal_align = (
            max(
                (cosine_similarity(mem.get("embedding", []), g) for g in goal_embeddings),
                default=0.0,
            )
            if goal_embeddings and mem.get("embedding")
            else 0.0
        )
        mem["da_relevance"] = compute_da(access_count, avg_outcome, user_rating, goal_align)

        # NE: dissimilarity to nearest neighbor, decayed by age
        other_embeddings = [
            m.get("embedding") for m in memories if m is not mem and m.get("embedding")
        ]
        mem["ne_novelty"] = compute_ne(mem.get("embedding", []), other_embeddings, age_days)

        # GABA: redundancy + staleness
        mem["gaba_inhibition"] = compute_gaba(mem.get("embedding", []), other_embeddings, age_days)
