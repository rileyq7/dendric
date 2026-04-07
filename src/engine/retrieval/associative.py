"""
Path 3: Associative Context Match.

Matches current context against encoding metadata.
The deja vu path -- surfaces archived memories that standard retrieval can't reach.

Good for: "what was I doing when...", spatiotemporal triggers
Only fires when: strong contextual overlap detected
"""

from typing import List, Dict, Any
from datetime import datetime, timezone


def associative_recall(
    current_context: dict,
    store,
    top_k: int = 5,
    min_overlap: int = 2,
) -> List[Dict[str, Any]]:
    return store.associative_search(
        current_context, top_k=top_k, min_overlap=min_overlap
    )


def build_context(source: str = "", location: str = "") -> dict:
    """Build a context dict from current environment."""
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

    dow = now.strftime("%A").lower()

    ctx = {
        "time_of_day": tod,
        "day_of_week": dow,
    }
    if source:
        ctx["source"] = source
    if location:
        ctx["location"] = location

    return ctx
