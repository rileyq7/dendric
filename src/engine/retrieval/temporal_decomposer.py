"""
Temporal Query Decomposer — strips temporal framing from queries to improve retrieval.

Problem: "How many weeks ago did I attend a bird watching workshop?" embeds poorly
against "I went to a great bird watching workshop yesterday!" because the temporal
framing dominates the embedding. This module extracts the event content for retrieval
and preserves the temporal operation for post-retrieval computation.
"""

import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List

try:
    import dateparser
except ImportError:
    dateparser = None


# Patterns ordered by specificity (most specific first)
DECOMPOSE_PATTERNS = [
    # "how many days/weeks/months ago did I/we [verb] X"
    (r"how\s+many\s+(\w+)\s+ago\s+(?:did|have|was)\s+(?:I|we)\s+(.+?)[\?\.]?$",
     "time_distance", ["unit", "event"]),

    # "how many days/weeks between X and Y"
    (r"how\s+many\s+(\w+)\s+(?:passed|elapsed|were there|between)\s+(?:between\s+)?(.+?)\s+and\s+(.+?)[\?\.]?$",
     "time_between", ["unit", "event_a", "event_b"]),

    # "how many days before/after X did Y happen"
    (r"how\s+many\s+(\w+)\s+(?:before|after)\s+(.+?)\s+(?:did|was|were)\s+(.+?)[\?\.]?$",
     "time_relative", ["unit", "event_a", "event_b"]),

    # "which happened first/earlier/later, X or Y"
    (r"which\s+(?:happened|came|occurred|was|did\s+\w+)\s+(?:first|earlier|later|more\s+recently)[,:]?\s*(.+?)\s+or\s+(.+?)[\?\.]?$",
     "ordering", ["event_a", "event_b"]),

    # "did X happen before/after Y"
    (r"(?:did|was|were)\s+(?:I|we|the)\s+(.+?)\s+(before|after)\s+(?:I|we|the|my)\s*(.+?)[\?\.]?$",
     "relative_order", ["event_a", "direction", "event_b"]),

    # "when did I/we [verb] X"
    (r"when\s+(?:did|was|were|have)\s+(?:I|we|the\s+last\s+time\s+I)\s+(.+?)[\?\.]?$",
     "when", ["event"]),

    # "what did I do/discuss N days/weeks ago" or "what ... last Tuesday"
    (r"what\s+(?:did|was)\s+(?:I|we)\s+(?:do|doing|discuss|talk\s+about|buy|cook|watch|visit|attend|play)\s+(.+?)[\?\.]?$",
     "event_at_time", ["temporal_expr"]),

    # "I mentioned X two months ago" / temporal qualifier at end
    (r"(.+?)\s+(\d+\s+(?:days?|weeks?|months?|years?)\s+ago)[\?\.]?$",
     "event_with_timeref", ["event", "time_expr"]),
]

# Temporal phrases to strip for cleaner event queries
TEMPORAL_STRIP_PATTERNS = [
    r"\b(?:how\s+many\s+\w+\s+ago)\b",
    r"\b(?:last|this\s+past)\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b(?:yesterday|today|recently|lately)\b",
    r"\b(?:\d+\s+(?:days?|weeks?|months?|years?)\s+ago)\b",
    r"\b(?:before|after)\s+(?:that|this)\b",
    r"\b(?:in|during|around)\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
    r"\b(?:two|three|four|five|six|seven|eight|nine|ten)\s+(?:days?|weeks?|months?|years?)\s+ago\b",
]

TEMPORAL_INDICATORS = {
    "ago", "before", "after", "last", "recent", "recently", "first",
    "earlier", "later", "when", "how many days", "how many weeks",
    "how many months", "how long", "yesterday", "this past",
}


def has_temporal_signal(query: str) -> bool:
    query_lower = query.lower()
    return any(indicator in query_lower for indicator in TEMPORAL_INDICATORS)


def decompose(query: str, reference_date: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Decompose a temporal query into event content and temporal operation.

    Returns:
        {
            "is_temporal": bool,
            "event_queries": [str],    # stripped queries for content retrieval
            "temporal_op": str | None,
            "temporal_params": dict,
            "date_range": tuple | None,
            "original_query": str,
        }
    """
    if reference_date is not None and isinstance(reference_date, str):
        reference_date = _parse_timestamp_str(reference_date) or datetime.utcnow()
    if reference_date is None:
        reference_date = datetime.utcnow()

    result = {
        "is_temporal": False,
        "event_queries": [query],
        "temporal_op": None,
        "temporal_params": {},
        "date_range": None,
        "original_query": query,
    }

    if not has_temporal_signal(query):
        return result

    query_stripped = query.strip().rstrip("?.")

    for pattern, op, param_names in DECOMPOSE_PATTERNS:
        match = re.match(pattern, query_stripped, re.IGNORECASE)
        if match:
            result["is_temporal"] = True
            result["temporal_op"] = op
            result["temporal_params"] = {
                name: match.group(i + 1).strip()
                for i, name in enumerate(param_names)
            }

            # Build event queries for content retrieval
            if op == "time_distance":
                result["event_queries"] = [result["temporal_params"]["event"]]
            elif op in ("time_between", "time_relative"):
                a = result["temporal_params"]["event_a"]
                b = result["temporal_params"]["event_b"]
                result["event_queries"] = [a, b, f"{a} {b}"]
            elif op == "ordering":
                a = result["temporal_params"]["event_a"]
                b = result["temporal_params"]["event_b"]
                result["event_queries"] = [a, b, f"{a} {b}"]
            elif op == "relative_order":
                a = result["temporal_params"]["event_a"]
                b = result["temporal_params"]["event_b"]
                result["event_queries"] = [a, b, f"{a} {b}"]
            elif op in ("when",):
                result["event_queries"] = [result["temporal_params"]["event"]]
            elif op == "event_at_time":
                temporal_expr = result["temporal_params"]["temporal_expr"]
                result["date_range"] = _resolve_date_range(temporal_expr, reference_date)
                result["event_queries"] = [_strip_temporal(temporal_expr)]
            elif op == "event_with_timeref":
                result["event_queries"] = [result["temporal_params"]["event"]]
                result["date_range"] = _resolve_date_range(
                    result["temporal_params"]["time_expr"], reference_date
                )
            break

    # Fallback: if no regex matched but temporal signal detected
    if not result["is_temporal"] and has_temporal_signal(query):
        stripped = _strip_temporal(query)
        if stripped and stripped != query.strip().rstrip("?."):
            result["is_temporal"] = True
            result["temporal_op"] = "general_temporal"
            result["event_queries"] = [stripped, query]

    return result


def _resolve_date_range(
    expr: str, reference_date: datetime
) -> Optional[Tuple[datetime, datetime]]:
    """Resolve a temporal expression to a (start, end) date range."""
    if dateparser is None:
        return None

    parsed = dateparser.parse(
        expr,
        settings={
            "RELATIVE_BASE": reference_date,
            "PREFER_DATES_FROM": "past",
        }
    )
    if parsed is None:
        return None

    expr_lower = expr.lower()
    if any(w in expr_lower for w in ["yesterday", "today", "monday", "tuesday",
            "wednesday", "thursday", "friday", "saturday", "sunday"]):
        delta = timedelta(days=1)
    elif "week" in expr_lower:
        delta = timedelta(days=3)
    elif "month" in expr_lower:
        delta = timedelta(days=15)
    else:
        delta = timedelta(days=3)

    return (parsed - delta, parsed + delta)


def _strip_temporal(query: str) -> str:
    """Remove temporal phrases to extract the content portion."""
    result = query
    for pattern in TEMPORAL_STRIP_PATTERNS:
        result = re.sub(pattern, " ", result, flags=re.IGNORECASE)
    return " ".join(result.split()).strip().rstrip("?.")


def compute_temporal_hint(
    temporal_op: str,
    temporal_params: dict,
    retrieved_memories: List[Dict[str, Any]],
    reference_date: Optional[datetime] = None,
) -> Optional[str]:
    """
    Post-retrieval: compute temporal answer from retrieved memory dates.
    Returns a hint string to prepend to context, or None.
    """
    if reference_date is None:
        reference_date = datetime.utcnow()

    # Extract dates from memories
    dated_memories = []
    for m in retrieved_memories:
        ctx = m.get("context", "")
        dt = _parse_context_date(ctx)
        if dt:
            dated_memories.append((dt, m))

    if not dated_memories:
        return None

    dated_memories.sort(key=lambda x: x[0])

    if temporal_op == "time_distance" and dated_memories:
        event_date = dated_memories[0][0]
        delta = reference_date - event_date
        unit = temporal_params.get("unit", "days")
        if unit in ("week", "weeks"):
            value = round(delta.days / 7)
        elif unit in ("month", "months"):
            value = round(delta.days / 30)
        elif unit in ("year", "years"):
            value = round(delta.days / 365)
        else:
            value = delta.days
        return (
            f"[Temporal computation: the event occurred approximately {value} {unit} ago "
            f"(event date: {event_date.strftime('%Y/%m/%d')}, "
            f"question date: {reference_date.strftime('%Y/%m/%d')})]"
        )

    if temporal_op in ("time_between", "time_relative") and len(dated_memories) >= 2:
        first_date = dated_memories[0][0]
        last_date = dated_memories[-1][0]
        delta = last_date - first_date
        unit = temporal_params.get("unit", "days")
        if unit in ("week", "weeks"):
            value = round(delta.days / 7)
        elif unit in ("month", "months"):
            value = round(delta.days / 30)
        else:
            value = delta.days
        return (
            f"[Temporal computation: approximately {value} {unit} between events "
            f"(first: {first_date.strftime('%Y/%m/%d')}, "
            f"second: {last_date.strftime('%Y/%m/%d')})]"
        )

    if temporal_op in ("ordering", "relative_order") and len(dated_memories) >= 2:
        first = dated_memories[0]
        last = dated_memories[-1]
        return (
            f"[Temporal computation: chronological order — "
            f"first ({first[0].strftime('%Y/%m/%d')}): ...{first[1].get('raw_content', '')[:80]}... "
            f"then ({last[0].strftime('%Y/%m/%d')}): ...{last[1].get('raw_content', '')[:80]}...]"
        )

    return None


def _parse_timestamp_str(ts: str) -> Optional[datetime]:
    """Parse a timestamp string like '2023/05/20 (Sat) 02:21'."""
    cleaned = re.sub(r"\s*\([^)]*\)\s*", " ", ts).strip()
    for fmt in ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M", "%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


def _parse_context_date(context: str) -> Optional[datetime]:
    """Parse date from context string like '2023/05/20 (Sat) 02:21 | session_0'."""
    if not context or "|" not in context:
        return None
    date_part = context.split("|")[0].strip()
    cleaned = re.sub(r"\s*\([^)]*\)\s*", " ", date_part).strip()
    for fmt in ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M", "%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None
