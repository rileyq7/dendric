"""
Path 2: Keyword / Entity Match via tsvector + ILIKE.

Good for: specific facts, numbers, names, "94.4%", "CaMKII"
Bad for: vague conceptual queries
"""

from typing import List, Dict, Any


def keyword_recall(
    query_text: str,
    store,
    top_k: int = 20,
    min_temp: float = 0.0,
) -> List[Dict[str, Any]]:
    return store.keyword_search(query_text, top_k=top_k, min_temp=min_temp)
