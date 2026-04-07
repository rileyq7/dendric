"""
Shared utilities for the memory engine.
"""

import re
from datetime import datetime
from typing import Optional


def extract_session_id(context: str) -> str:
    """Extract session identifier from context string.

    Context format: "2023/05/20 (Sat) 02:21 | session_0_chunk_1"
    Returns the session part (e.g. "session_0") or "unknown".
    """
    if not context:
        return "unknown"
    parts = context.split("|")
    if len(parts) >= 2:
        session_part = parts[-1].strip()
        m = re.match(r"(session_\d+)", session_part)
        if m:
            return m.group(1)
        return session_part
    return "unknown"


def parse_context_date(context: str) -> Optional[datetime]:
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
