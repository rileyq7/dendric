"""
Signal Scoring — Simplified to 3 signals.

DA (relevance): How goal-relevant is this memory?
NE (novelty): How novel was this when ingested?
usage: Derived from access patterns.
"""

from typing import List, Optional, Callable
import numpy as np


def compute_da(
    content: str,
    source: str,
    goals: List[str],
    existing_entities: Optional[List[str]] = None,
) -> float:
    score = 0.2

    lower = content.lower()
    for goal in goals:
        if goal.lower() in lower:
            score += 0.08

    source_weights = {
        "conversation": 0.15,
        "meeting": 0.15,
        "research": 0.10,
        "idea": 0.12,
        "planning": 0.08,
        "note": 0.05,
        "observation": 0.03,
        "direct": 0.05,
        "complaint": 0.04,
    }
    score += source_weights.get(source, 0.05)

    word_count = len(content.split())
    if word_count > 30:
        score += 0.05
    if word_count > 100:
        score += 0.05

    return min(1.0, score)


def compute_ne(
    content_embedding: List[float],
    recent_embeddings: List[List[float]],
    cosine_sim_fn: Optional[Callable] = None,
) -> float:
    if not recent_embeddings:
        return 0.8

    if cosine_sim_fn is None:
        cosine_sim_fn = _cosine_similarity

    max_sim = 0.0
    for existing in recent_embeddings:
        sim = cosine_sim_fn(content_embedding, existing)
        max_sim = max(max_sim, sim)

    novelty = max(0.1, min(1.0, 1.0 - max_sim))
    return novelty


def compute_usage(
    access_count: int,
    retrieval_hits: int,
    cycles_since_last_access: int,
) -> float:
    freq = min(1.0, access_count / 20.0)
    relevance = min(1.0, retrieval_hits / 10.0)
    recency = max(0.0, 1.0 - (cycles_since_last_access / 30.0))
    return 0.4 * freq + 0.3 * relevance + 0.3 * recency


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot = np.dot(a_arr, b_arr)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if norm == 0:
        return 0.0
    return float(dot / norm)
