"""
Path 4: Temporal Retrieval — time-aware scoring for temporal reasoning queries.

Detects temporal signals in queries (e.g., "last week", "first conversation",
"before I switched jobs") and scores memories by temporal proximity/ordering.

Good for: "How many days ago...", "What did I say first...", "Before I moved..."
Bad for: Non-temporal factual queries
"""

import re
import math
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

try:
    from dateutil import parser as dateparser
except ImportError:
    dateparser = None


# ── Temporal signal words ───────────────────────────────────────────────────

TEMPORAL_KEYWORDS = {
    'recent', 'recently', 'lately', 'last', 'first', 'earlier',
    'before', 'after', 'during', 'since', 'until', 'when',
    'previous', 'previously', 'originally', 'initially',
    'changed', 'updated', 'switched', 'moved', 'started', 'stopped',
    'used to', 'no longer', 'now', 'currently',
    'yesterday', 'today', 'tomorrow',
    'week', 'month', 'year', 'ago',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'january', 'february', 'march', 'april', 'may', 'june',
    'july', 'august', 'september', 'october', 'november', 'december',
    'spring', 'summer', 'fall', 'autumn', 'winter',
    'beginning', 'end', 'start', 'middle',
    'once', 'twice', 'often', 'always', 'never',
    'days', 'weeks', 'months', 'years',
    'passed', 'elapsed', 'between', 'consecutive',
    'order', 'sequence', 'chronological',
}

RELATIVE_TIME_PATTERNS = [
    (r'last\s+(\d+)\s+days?', lambda m, now: (now - timedelta(days=int(m.group(1))), now)),
    (r'last\s+week', lambda m, now: (now - timedelta(weeks=1), now)),
    (r'last\s+month', lambda m, now: (now - timedelta(days=30), now)),
    (r'last\s+year', lambda m, now: (now - timedelta(days=365), now)),
    (r'(\d+)\s+days?\s+ago', lambda m, now: (now - timedelta(days=int(m.group(1))), now - timedelta(days=int(m.group(1))))),
    (r'(\d+)\s+weeks?\s+ago', lambda m, now: (now - timedelta(weeks=int(m.group(1))), now - timedelta(weeks=int(m.group(1))))),
    (r'(\d+)\s+months?\s+ago', lambda m, now: (now - timedelta(days=30*int(m.group(1))), now - timedelta(days=30*int(m.group(1))))),
    (r'yesterday', lambda m, now: (now - timedelta(days=1), now - timedelta(days=1))),
    (r'this\s+week', lambda m, now: (now - timedelta(days=now.weekday()), now)),
    (r'this\s+month', lambda m, now: (now.replace(day=1), now)),
]

ORDINAL_PATTERNS = [
    (r'first\s+(conversation|session|chat|time|mention)', 'first'),
    (r'second\s+(conversation|session|chat|time|mention)', 'second'),
    (r'third\s+(conversation|session|chat|time|mention)', 'third'),
    (r'last\s+(conversation|session|chat|time|mention)', 'last'),
    (r'most\s+recent\s+(conversation|session|chat|time|mention)', 'last'),
    (r'earliest\s+(conversation|session|chat|time|mention)', 'first'),
    (r'latest\s+(conversation|session|chat|time|mention)', 'last'),
]

UPDATE_PATTERNS = [
    r'before\s+(?:I|i)\s+(changed|switched|moved|started|stopped|quit|left|joined)',
    r'after\s+(?:I|i)\s+(changed|switched|moved|started|stopped|quit|left|joined)',
    r'(?:used\s+to|originally|initially|previously)\b',
    r'(?:no\s+longer|not\s+anymore|currently|now)\b',
]

# Patterns for "how many days/weeks/months" questions — need temporal spread
DURATION_PATTERNS = [
    r'how\s+many\s+days',
    r'how\s+many\s+weeks',
    r'how\s+many\s+months',
    r'how\s+many\s+years',
    r'how\s+long\s+(?:has|have|did|was|were)',
    r'days?\s+(?:passed|elapsed|between)',
    r'weeks?\s+(?:passed|elapsed|between)',
    r'months?\s+(?:passed|elapsed|between)',
]

# Ordering patterns — need multiple events in temporal order
ORDERING_PATTERNS = [
    r'(?:which|what)\s+(?:happened|came|occurred)\s+(?:first|last|earlier|later)',
    r'(?:order|sequence)\s+(?:from|of)',
    r'(?:first|last)\s+to\s+(?:first|last)',
    r'(?:before|after)\s+(?:the|my|I)',
    r'(?:happened|occurred)\s+(?:before|after|between)',
]


# ── Detection ──────────────────────────────────────────────────────────────

def _parse_timestamp(ts) -> Optional[datetime]:
    """Parse a timestamp from various formats."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, str):
        # LongMemEval format: "2023/05/20 (Sat) 02:21"
        cleaned = re.sub(r'\s*\([^)]*\)\s*', ' ', ts).strip()
        if dateparser:
            try:
                return dateparser.parse(cleaned)
            except (ValueError, TypeError):
                pass
        # Manual fallback for common formats
        for fmt in ['%Y/%m/%d %H:%M', '%Y-%m-%d %H:%M', '%Y/%m/%d', '%Y-%m-%d']:
            try:
                return datetime.strptime(cleaned, fmt)
            except ValueError:
                continue
    return None


def detect_temporal_query(query: str, question_timestamp=None) -> Dict[str, Any]:
    """
    Analyze a query for temporal signals.
    Returns a dict with temporal constraints for retrieval.
    """
    query_lower = query.lower()
    result = {
        'is_temporal': False,
        'time_range': None,
        'ordinal': None,
        'wants_change': False,
        'wants_duration': False,
        'wants_ordering': False,
        'recency_bias': False,
        'temporal_keywords': [],
    }

    # Check for temporal keywords
    found_keywords = [kw for kw in TEMPORAL_KEYWORDS if kw in query_lower]
    if not found_keywords:
        return result

    result['temporal_keywords'] = found_keywords

    # Parse reference timestamp
    now = _parse_timestamp(question_timestamp)
    if now is None:
        now = datetime.now()

    # Try relative time patterns
    for pattern, resolver in RELATIVE_TIME_PATTERNS:
        match = re.search(pattern, query_lower)
        if match:
            try:
                result['time_range'] = resolver(match, now)
                result['is_temporal'] = True
            except Exception:
                pass
            break

    # Try ordinal patterns
    for pattern, ordinal in ORDINAL_PATTERNS:
        if re.search(pattern, query_lower):
            result['ordinal'] = ordinal
            result['is_temporal'] = True
            break

    # Check for change/update queries
    for pattern in UPDATE_PATTERNS:
        if re.search(pattern, query_lower):
            result['wants_change'] = True
            result['is_temporal'] = True
            break

    # Check for duration queries ("how many days between X and Y")
    for pattern in DURATION_PATTERNS:
        if re.search(pattern, query_lower):
            result['wants_duration'] = True
            result['is_temporal'] = True
            break

    # Check for ordering queries ("which happened first")
    for pattern in ORDERING_PATTERNS:
        if re.search(pattern, query_lower):
            result['wants_ordering'] = True
            result['is_temporal'] = True
            break

    # Recency bias
    if any(kw in query_lower for kw in ['recent', 'recently', 'lately', 'last', 'latest']):
        result['recency_bias'] = True
        result['is_temporal'] = True

    return result


# ── Temporal Scoring ───────────────────────────────────────────────────────

def compute_temporal_score(
    mem_context: str,
    temporal_info: Dict[str, Any],
    question_timestamp=None,
) -> float:
    """
    Score a memory based on temporal relevance to the query.
    Returns 0.0-1.0 where 1.0 = perfect temporal match.

    Uses the memory's context field which contains the session date
    (format: "2023/05/20 (Sat) 02:21 | session_0").
    """
    if not temporal_info.get('is_temporal'):
        return 0.0

    # Extract date from context string (format: "date | session_N")
    mem_dt = _extract_date_from_context(mem_context)
    if mem_dt is None:
        return 0.0

    q_dt = _parse_timestamp(question_timestamp)
    if q_dt is None:
        q_dt = datetime.now()

    score = 0.0

    # Time range matching
    if temporal_info.get('time_range'):
        start, end = temporal_info['time_range']
        if start <= mem_dt <= end:
            score = 1.0
        else:
            if mem_dt < start:
                days_off = (start - mem_dt).total_seconds() / 86400
            else:
                days_off = (mem_dt - end).total_seconds() / 86400
            score = math.exp(-0.1 * days_off)

    # Recency bias: score based on how recent the memory is
    elif temporal_info.get('recency_bias'):
        days_ago = (q_dt - mem_dt).total_seconds() / 86400
        score = math.exp(-0.05 * max(0, days_ago))

    # Duration/ordering queries: we want temporal spread — moderate score for all
    elif temporal_info.get('wants_duration') or temporal_info.get('wants_ordering'):
        score = 0.5

    # Change detection: moderate score for all temporally relevant memories
    elif temporal_info.get('wants_change'):
        score = 0.5

    return min(1.0, max(0.0, score))


def compute_ordinal_scores(
    results: List[Dict[str, Any]],
    temporal_info: Dict[str, Any],
) -> Dict[str, float]:
    """
    For ordinal queries ("first conversation about X"), sort by timestamp
    and assign scores based on position.

    Returns dict of {memory_id: score}.
    """
    if not temporal_info.get('ordinal'):
        return {}

    # Extract timestamps from results
    timestamped = []
    for r in results:
        ctx = r.get('context', '')
        dt = _extract_date_from_context(ctx)
        if dt:
            timestamped.append((dt, r))

    if not timestamped:
        return {}

    timestamped.sort(key=lambda x: x[0])
    scores = {}

    ordinal = temporal_info['ordinal']
    if ordinal == 'first':
        for i, (ts, r) in enumerate(timestamped):
            scores[r['id']] = max(0, 1.0 - (i * 0.1))
    elif ordinal == 'second':
        for i, (ts, r) in enumerate(timestamped):
            scores[r['id']] = 1.0 if i == 1 else max(0, 0.5 - abs(i - 1) * 0.1)
    elif ordinal == 'third':
        for i, (ts, r) in enumerate(timestamped):
            scores[r['id']] = 1.0 if i == 2 else max(0, 0.5 - abs(i - 2) * 0.1)
    elif ordinal == 'last':
        for i, (ts, r) in enumerate(reversed(timestamped)):
            scores[r['id']] = max(0, 1.0 - (i * 0.1))

    return scores


def temporal_rerank(
    results: List[Dict[str, Any]],
    temporal_info: Dict[str, Any],
    question_timestamp=None,
    temporal_weight: float = 2.5,
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Re-rank retrieval results by adding a temporal RRF component.

    For each result, computes a temporal score and adds it as an additional
    RRF path with the specified weight.
    """
    if not temporal_info.get('is_temporal') or not results:
        return results

    # Compute temporal scores
    for r in results:
        r['temporal_score'] = compute_temporal_score(
            r.get('context', ''), temporal_info, question_timestamp
        )

    # Handle ordinal queries
    if temporal_info.get('ordinal'):
        ordinal_scores = compute_ordinal_scores(results, temporal_info)
        for r in results:
            if r['id'] in ordinal_scores:
                r['temporal_score'] = ordinal_scores[r['id']]

    # Sort by temporal score to get temporal ranks
    temporal_sorted = sorted(
        enumerate(results),
        key=lambda x: x[1].get('temporal_score', 0),
        reverse=True,
    )
    temporal_ranks = {}
    for rank, (orig_idx, r) in enumerate(temporal_sorted):
        temporal_ranks[r['id']] = rank

    # Add temporal RRF component to existing fusion score
    for r in results:
        temporal_rank = temporal_ranks.get(r['id'], len(results))
        temporal_rrf = temporal_weight / (rrf_k + temporal_rank + 1)
        # Reduce vector contribution for temporal queries
        original_score = r.get('fusion_score', 0)
        r['fusion_score'] = original_score * 0.7 + temporal_rrf
        r.setdefault('retrieval_paths', []).append('temporal')

    # Re-sort by updated fusion score
    results.sort(key=lambda r: r.get('fusion_score', 0), reverse=True)
    return results


def ensure_temporal_diversity(
    results: List[Dict[str, Any]],
    temporal_info: Dict[str, Any],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    For duration/ordering/change queries, ensure results span multiple
    time periods rather than clustering around one date.
    """
    needs_diversity = (
        temporal_info.get('wants_duration')
        or temporal_info.get('wants_ordering')
        or temporal_info.get('wants_change')
    )
    if not needs_diversity or len(results) < 4:
        return results[:top_k]

    # Extract timestamps and sort
    timestamped = []
    no_timestamp = []
    for r in results:
        dt = _extract_date_from_context(r.get('context', ''))
        if dt:
            timestamped.append((dt, r))
        else:
            no_timestamp.append(r)

    if len(timestamped) < 2:
        return results[:top_k]

    timestamped.sort(key=lambda x: x[0])

    # Split into time halves
    midpoint = len(timestamped) // 2
    early_half = timestamped[:midpoint]
    late_half = timestamped[midpoint:]

    # Score by fusion_score within each half
    early_sorted = sorted(early_half, key=lambda x: x[1].get('fusion_score', 0), reverse=True)
    late_sorted = sorted(late_half, key=lambda x: x[1].get('fusion_score', 0), reverse=True)

    # Interleave
    half_k = top_k // 2
    diverse_results = []
    diverse_results.extend(r for _, r in early_sorted[:half_k])
    diverse_results.extend(r for _, r in late_sorted[:half_k])

    # Fill remaining
    used_ids = {r['id'] for r in diverse_results}
    remaining = [r for _, r in timestamped if r['id'] not in used_ids]
    remaining.sort(key=lambda r: r.get('fusion_score', 0), reverse=True)
    diverse_results.extend(remaining[:top_k - len(diverse_results)])

    # Add non-timestamped if still room
    if len(diverse_results) < top_k:
        diverse_results.extend(no_timestamp[:top_k - len(diverse_results)])

    return diverse_results[:top_k]


# ── Helpers ────────────────────────────────────────────────────────────────

def _extract_date_from_context(context: str) -> Optional[datetime]:
    """Extract a date from a memory's context field.

    Expected formats:
    - "2023/05/20 (Sat) 02:21 | session_0"
    - "2023/05/20 (Sat) 02:21 | session_0_chunk_1"
    - "session_0" (no date)
    """
    if not context:
        return None

    # Try to extract date before the pipe separator
    parts = context.split('|')
    if len(parts) >= 2:
        date_part = parts[0].strip()
        return _parse_timestamp(date_part)

    # Maybe the whole context is a date
    return _parse_timestamp(context)
