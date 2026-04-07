"""
Reciprocal Rank Fusion (RRF) to merge results from all three retrieval paths.

Reference: Hindsight's TEMPR architecture uses RRF for multi-path fusion.

RRF score for memory m = sum(weight_path / (rrf_k + rank_in_path))
Memories appearing in multiple paths get boosted.
"""

import math
import re
from collections import Counter
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from ..utils import extract_session_id as _extract_session_id, parse_context_date as _parse_context_date


_AGGREGATION_PATTERN = re.compile(
    r"\b(?:how many|how much|total|in total|all the|combined|sum of|altogether)\b",
    re.IGNORECASE,
)


def _is_aggregation_query(query: str) -> bool:
    """Detect counting/aggregation queries that need broad recall, not session diversity."""
    return bool(_AGGREGATION_PATTERN.search(query))


def fuse_results(
    vector_results: List[Dict[str, Any]],
    keyword_results: List[Dict[str, Any]],
    associative_results: List[Dict[str, Any]],
    vector_weight: float = 1.0,
    keyword_weight: float = 1.2,
    associative_weight: float = 1.5,
    rrf_k: int = 60,
    recency_bonus: float = 0.002,
    session_decay: float = 0.85,
    top_k: Optional[int] = None,
    query: str = "",
    graph_results: Optional[List[Dict[str, Any]]] = None,
    graph_weight: float = 1.4,
) -> List[Dict[str, Any]]:
    scores = {}  # memory_id -> {score, data, paths}

    for rank, result in enumerate(vector_results):
        mid = result["id"]
        rrf_score = vector_weight / (rrf_k + rank + 1)
        # Similarity bonus: high-cosine vector hits get a boost proportional
        # to their actual similarity, so a 0.5-sim result beats a 0.2-sim one
        sim = result.get("similarity", 0.0)
        if sim > 0:
            rrf_score += vector_weight * sim * 0.02
        if mid not in scores:
            scores[mid] = {"score": 0.0, "data": result, "paths": []}
        scores[mid]["score"] += rrf_score
        scores[mid]["paths"].append("vector")

    for rank, result in enumerate(keyword_results):
        mid = result["id"]
        rrf_score = keyword_weight / (rrf_k + rank + 1)
        if mid not in scores:
            scores[mid] = {"score": 0.0, "data": result, "paths": []}
        scores[mid]["score"] += rrf_score
        scores[mid]["paths"].append("keyword")

    for rank, result in enumerate(associative_results):
        mid = result["id"]
        rrf_score = associative_weight / (rrf_k + rank + 1)
        if mid not in scores:
            scores[mid] = {"score": 0.0, "data": result, "paths": []}
        scores[mid]["score"] += rrf_score
        scores[mid]["paths"].append("associative")

    for rank, result in enumerate(graph_results or []):
        mid = result["id"]
        rrf_score = graph_weight / (rrf_k + rank + 1)
        if mid not in scores:
            scores[mid] = {"score": 0.0, "data": result, "paths": []}
        scores[mid]["score"] += rrf_score
        scores[mid]["paths"].append("graph")

    # Recency tiebreaker: small bonus so newer memories win when scores are close.
    # max_bonus is ~0.002 (vs RRF scores typically 0.01-0.06), so it only breaks ties.
    if recency_bonus > 0:
        for mid, entry in scores.items():
            ctx = entry["data"].get("context", "")
            dt = _parse_context_date(ctx)
            if dt is None:
                # Fall back to created_at if available
                ca = entry["data"].get("created_at")
                if isinstance(ca, str):
                    try:
                        dt = datetime.fromisoformat(ca)
                    except (ValueError, TypeError):
                        pass
                elif isinstance(ca, datetime):
                    dt = ca
            if dt:
                # Ensure both sides are tz-aware to avoid naive vs aware mismatch
                ref = datetime(2025, 1, 1, tzinfo=timezone.utc)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                age_days = max((ref - dt).total_seconds() / 86400, 0.01)
                entry["score"] += recency_bonus / (1 + math.log(1 + age_days))

    # Aggregate boost: for aggregation queries, boost pre-computed aggregate memories
    is_aggregation = _is_aggregation_query(query)
    if is_aggregation:
        for mid, entry in scores.items():
            if entry["data"].get("source") == "aggregate":
                entry["score"] *= 1.5

    # Session-penalized greedy selection: distributes results across sessions.
    # Skip penalty for aggregation/counting queries — they need broad recall
    # from the same sessions where items were discussed.
    use_session_penalty = top_k and session_decay < 1.0 and not is_aggregation
    if use_session_penalty:
        results = _session_penalized_select(scores, top_k, session_decay)
    else:
        ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        results = []
        for item in ranked:
            data = item["data"]
            data["fusion_score"] = round(item["score"], 6)
            data["retrieval_paths"] = item["paths"]
            results.append(data)

    return results


def _session_penalized_select(
    scores: Dict[str, Dict],
    top_k: int,
    session_decay: float,
) -> List[Dict[str, Any]]:
    """Three-phase session diversity enforcement.

    Phase 1: Take the #1 result (best match regardless of session)
    Phase 2: For each remaining slot, prefer the highest-ranked result from
             a session NOT yet represented, until min_sessions are covered
    Phase 3: Fill remaining slots by score with exponential session penalty

    This guarantees cross-session coverage while preserving relevance ordering.
    """
    # Build ranked list with session info
    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    for entry in ranked:
        entry["_session"] = _extract_session_id(entry["data"].get("context", ""))

    available_sessions = len(set(e["_session"] for e in ranked if e["_session"]))
    min_sessions = min(3, available_sessions)

    selected = []
    used_sessions: set = set()
    remaining = list(ranked)

    def _emit(entry):
        data = entry["data"]
        data["fusion_score"] = round(entry["score"], 6)
        data["retrieval_paths"] = entry["paths"]
        selected.append(data)
        used_sessions.add(entry["_session"])

    # Phase 1: Take #1 result
    if remaining:
        _emit(remaining.pop(0))

    # Phase 2: Fill from unseen sessions until min_sessions met
    while len(used_sessions) < min_sessions and remaining and len(selected) < top_k:
        for i, entry in enumerate(remaining):
            if entry["_session"] and entry["_session"] not in used_sessions:
                _emit(remaining.pop(i))
                break
        else:
            break  # no more unseen sessions

    # Phase 3: Fill rest with session-penalized scores
    session_counts: Counter = Counter()
    for s in selected:
        sess = _extract_session_id(s.get("context", ""))
        session_counts[sess] += 1

    while len(selected) < top_k and remaining:
        best_idx, best_penalized = 0, -1.0
        for i, entry in enumerate(remaining):
            sess = entry["_session"]
            penalized = entry["score"] * (session_decay ** session_counts.get(sess, 0))
            if penalized > best_penalized:
                best_penalized = penalized
                best_idx = i

        entry = remaining.pop(best_idx)
        session_counts[entry["_session"]] += 1
        _emit(entry)

    return selected
