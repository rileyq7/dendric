"""
Path 1: Vector Similarity via pgvector.

Good for: conceptual queries, broad topics, "tell me about X"
Bad for: specific numbers, exact names, factual recall
"""

from typing import List, Dict, Any


def vector_recall(
    query_embedding: List[float],
    store,
    top_k: int = 20,
    min_temp: float = 0.0,
) -> List[Dict[str, Any]]:
    return store.vector_search(query_embedding, top_k=top_k, min_temp=min_temp)
