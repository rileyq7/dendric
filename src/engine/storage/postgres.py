"""
Postgres + pgvector storage backend.
"""

import json
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from ..core.memory import Memory
from .migrations import run_migrations

logger = logging.getLogger(__name__)

psycopg2.extras.register_uuid()


class PostgresStore:

    def __init__(self, db_url: str):
        self.db_url = db_url
        self.conn = psycopg2.connect(db_url)
        self.conn.autocommit = False
        register_vector(self.conn)
        run_migrations(self.conn)

    # ── CRUD ─────────────────────────────────────────────────────

    def store(self, mem: Memory) -> str:
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO memories (
                    id, raw_content, structured_summary, knowledge_nugget, entity_edges,
                    temperature, region,
                    da_relevance, ne_novelty, usage_score,
                    da_history, ne_history, signal_variance,
                    retrieval_hits, retrieval_importance,
                    embedding,
                    created_at, last_accessed, last_consolidated,
                    access_times, access_count,
                    source, context, location, time_of_day, day_of_week,
                    co_entities, preceding_memory_id,
                    compression_level, tokens_original, tokens_current,
                    token_weights,
                    knowledge_nuggets, last_compressed_at, compression_temperature,
                    metadata
                ) VALUES (
                    %(id)s, %(raw_content)s, %(structured_summary)s, %(knowledge_nugget)s, %(entity_edges)s,
                    %(temperature)s, %(region)s,
                    %(da_relevance)s, %(ne_novelty)s, %(usage_score)s,
                    %(da_history)s, %(ne_history)s, %(signal_variance)s,
                    %(retrieval_hits)s, %(retrieval_importance)s,
                    %(embedding)s,
                    %(created_at)s, %(last_accessed)s, %(last_consolidated)s,
                    %(access_times)s, %(access_count)s,
                    %(source)s, %(context)s, %(location)s, %(time_of_day)s, %(day_of_week)s,
                    %(co_entities)s, %(preceding_memory_id)s,
                    %(compression_level)s, %(tokens_original)s, %(tokens_current)s,
                    %(token_weights)s,
                    %(knowledge_nuggets)s, %(last_compressed_at)s, %(compression_temperature)s,
                    %(metadata)s
                )
                ON CONFLICT (id) DO UPDATE SET
                    raw_content = EXCLUDED.raw_content,
                    structured_summary = EXCLUDED.structured_summary,
                    knowledge_nugget = EXCLUDED.knowledge_nugget,
                    entity_edges = EXCLUDED.entity_edges,
                    temperature = EXCLUDED.temperature,
                    region = EXCLUDED.region,
                    da_relevance = EXCLUDED.da_relevance,
                    ne_novelty = EXCLUDED.ne_novelty,
                    usage_score = EXCLUDED.usage_score,
                    da_history = EXCLUDED.da_history,
                    ne_history = EXCLUDED.ne_history,
                    signal_variance = EXCLUDED.signal_variance,
                    retrieval_hits = EXCLUDED.retrieval_hits,
                    retrieval_importance = EXCLUDED.retrieval_importance,
                    embedding = EXCLUDED.embedding,
                    last_accessed = EXCLUDED.last_accessed,
                    last_consolidated = EXCLUDED.last_consolidated,
                    access_times = EXCLUDED.access_times,
                    access_count = EXCLUDED.access_count,
                    compression_level = EXCLUDED.compression_level,
                    tokens_original = EXCLUDED.tokens_original,
                    tokens_current = EXCLUDED.tokens_current,
                    token_weights = EXCLUDED.token_weights,
                    knowledge_nuggets = EXCLUDED.knowledge_nuggets,
                    last_compressed_at = EXCLUDED.last_compressed_at,
                    compression_temperature = EXCLUDED.compression_temperature,
                    metadata = EXCLUDED.metadata
            """, _mem_to_params(mem))
        self.conn.commit()
        return mem.id

    def store_batch(self, memories: List[Memory]):
        """Insert multiple memories in a single transaction (no per-item commit)."""
        with self.conn.cursor() as cur:
            for mem in memories:
                cur.execute("""
                    INSERT INTO memories (
                        id, raw_content, structured_summary, knowledge_nugget, entity_edges,
                        temperature, region,
                        da_relevance, ne_novelty, usage_score,
                        da_history, ne_history, signal_variance,
                        retrieval_hits, retrieval_importance,
                        embedding,
                        created_at, last_accessed, last_consolidated,
                        access_times, access_count,
                        source, context, location, time_of_day, day_of_week,
                        co_entities, preceding_memory_id,
                        compression_level, tokens_original, tokens_current,
                        token_weights,
                        knowledge_nuggets, last_compressed_at, compression_temperature,
                        metadata
                    ) VALUES (
                        %(id)s, %(raw_content)s, %(structured_summary)s, %(knowledge_nugget)s, %(entity_edges)s,
                        %(temperature)s, %(region)s,
                        %(da_relevance)s, %(ne_novelty)s, %(usage_score)s,
                        %(da_history)s, %(ne_history)s, %(signal_variance)s,
                        %(retrieval_hits)s, %(retrieval_importance)s,
                        %(embedding)s,
                        %(created_at)s, %(last_accessed)s, %(last_consolidated)s,
                        %(access_times)s, %(access_count)s,
                        %(source)s, %(context)s, %(location)s, %(time_of_day)s, %(day_of_week)s,
                        %(co_entities)s, %(preceding_memory_id)s,
                        %(compression_level)s, %(tokens_original)s, %(tokens_current)s,
                        %(token_weights)s,
                        %(knowledge_nuggets)s, %(last_compressed_at)s, %(compression_temperature)s,
                        %(metadata)s
                    )
                    ON CONFLICT (id) DO NOTHING
                """, _mem_to_params(mem))
        self.conn.commit()

    def get(self, memory_id: str) -> Optional[Memory]:
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM memories WHERE id = %s", (memory_id,))
            row = cur.fetchone()
        if row is None:
            return None
        return _row_to_mem(row)

    def update(self, mem: Memory):
        self.store(mem)

    def delete(self, memory_id: str):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM memories WHERE id = %s", (memory_id,))
        self.conn.commit()

    def get_all(self, limit: int = 300) -> List[Memory]:
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM memories ORDER BY temperature DESC LIMIT %s",
                (limit,)
            )
            rows = cur.fetchall()
        return [_row_to_mem(r) for r in rows]

    def get_all_active(self) -> List[Memory]:
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM memories ORDER BY temperature DESC")
            rows = cur.fetchall()
        return [_row_to_mem(r) for r in rows]

    # ── Vector similarity search ─────────────────────────────────

    def vector_search(
        self, query_embedding: List[float], top_k: int = 20, min_temp: float = 0.0
    ) -> List[Dict[str, Any]]:
        # Archive is dormant-but-reachable: NOT served by normal retrieval,
        # only by the associative déjà-vu trigger. So vector excludes it.
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT *, 1 - (embedding <=> %s::vector) as similarity
                FROM memories
                WHERE temperature >= %s
                  AND embedding IS NOT NULL
                  AND region != 'archive'
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, min_temp, query_embedding, top_k))
            rows = cur.fetchall()
        results = []
        for r in rows:
            mem = _row_to_mem(r)
            d = mem.to_dict()
            d["similarity"] = float(r["similarity"])
            results.append(d)
        return results

    # ── Keyword / FTS search ─────────────────────────────────────

    def keyword_search(
        self, query_text: str, top_k: int = 20, min_temp: float = 0.0
    ) -> List[Dict[str, Any]]:
        results = []
        seen = set()

        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Two-tier FTS: try strict AND first, fall back to OR if too few results.
            # AND is precise but fails when any query term is absent from the doc.
            # OR is broad but ts_rank sorts by term overlap so best matches come first.
            # Archive excluded — same rationale as vector_search.
            cur.execute("""
                SELECT *, ts_rank(content_tsv, plainto_tsquery('english', %s)) as score
                FROM memories
                WHERE content_tsv @@ plainto_tsquery('english', %s)
                  AND temperature >= %s
                  AND region != 'archive'
                ORDER BY ts_rank(content_tsv, plainto_tsquery('english', %s)) DESC
                LIMIT %s
            """, (query_text, query_text, min_temp, query_text, top_k))
            for r in cur.fetchall():
                mem = _row_to_mem(r)
                d = mem.to_dict()
                d["score"] = float(r["score"])
                results.append(d)
                seen.add(str(r["id"]))

            # OR fallback: if AND returned fewer than top_k, broaden with OR
            if len(results) < top_k:
                or_query = _build_or_tsquery(query_text)
                if or_query:
                    remaining = top_k - len(results)
                    cur.execute("""
                        SELECT *, ts_rank(content_tsv, to_tsquery('english', %s)) as score
                        FROM memories
                        WHERE content_tsv @@ to_tsquery('english', %s)
                          AND temperature >= %s
                          AND region != 'archive'
                          AND id::text != ALL(%s)
                        ORDER BY ts_rank(content_tsv, to_tsquery('english', %s)) DESC
                        LIMIT %s
                    """, (or_query, or_query, min_temp, list(seen), or_query, remaining))
                    for r in cur.fetchall():
                        if str(r["id"]) not in seen:
                            mem = _row_to_mem(r)
                            d = mem.to_dict()
                            d["score"] = float(r["score"])
                            results.append(d)
                            seen.add(str(r["id"]))

            # Substring match fallback on individual content words
            content_words = _extract_content_words(query_text)
            for word in content_words[:3]:
                pattern = f"%{word}%"
                cur.execute("""
                    SELECT *, 0.8 as score
                    FROM memories
                    WHERE raw_content ILIKE %s
                      AND temperature >= %s
                      AND region != 'archive'
                      AND id::text != ALL(%s)
                    LIMIT %s
                """, (pattern, min_temp, list(seen), top_k))
                for r in cur.fetchall():
                    if str(r["id"]) not in seen:
                        mem = _row_to_mem(r)
                        d = mem.to_dict()
                        d["score"] = float(r["score"])
                        results.append(d)
                        seen.add(str(r["id"]))

        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results[:top_k]

    # ── Associative context search ───────────────────────────────

    def associative_search(
        self, current_context: dict, top_k: int = 5, min_overlap: int = 2
    ) -> List[Dict[str, Any]]:
        # Build context matching as a subquery using a CTE with literal comparisons
        fields = {}
        if current_context.get("time_of_day"):
            fields["time_of_day"] = current_context["time_of_day"]
        if current_context.get("day_of_week"):
            fields["day_of_week"] = current_context["day_of_week"]
        if current_context.get("location"):
            fields["location"] = current_context["location"]
        if current_context.get("source"):
            fields["source"] = current_context["source"]

        if len(fields) < min_overlap:
            return []

        # Use a simple approach: fetch candidates matching ANY field, score in Python
        where_parts = []
        params = []
        for col, val in fields.items():
            where_parts.append(f"m.{col} = %s")
            params.append(val)

        where_clause = " OR ".join(where_parts)
        params.append(top_k * 3)  # fetch more to filter

        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT m.*, a.raw_content as archived_raw
                FROM memories m
                LEFT JOIN archive a ON m.id = a.memory_id
                WHERE {where_clause}
                ORDER BY m.temperature DESC
                LIMIT %s
            """, params)
            rows = cur.fetchall()

        # Score overlap in Python
        results = []
        for r in rows:
            overlap = 0
            for col, val in fields.items():
                if r.get(col) == val:
                    overlap += 1
            if overlap >= min_overlap:
                mem = _row_to_mem(r)
                d = mem.to_dict()
                d["context_overlap"] = overlap
                if r.get("archived_raw"):
                    d["archived_raw"] = r["archived_raw"]
                results.append(d)

        results.sort(key=lambda x: (-x["context_overlap"], -x["temperature"]))
        return results[:top_k]

    # ── Archive operations ───────────────────────────────────────

    def archive_raw(self, memory_id: str, raw_content: str, encoding_context: dict = None):
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO archive (memory_id, raw_content, encoding_context)
                VALUES (%s, %s, %s)
                ON CONFLICT (memory_id) DO NOTHING
            """, (memory_id, raw_content, json.dumps(encoding_context or {})))
        self.conn.commit()

    # ── Retrieval logging (for EWC) ──────────────────────────────

    def log_retrieval(self, query_text: str, query_embedding: List[float],
                      result_ids: List[str], top_k_ids: List[str]):
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO retrieval_log (query_text, query_embedding, results, top_k_ids)
                VALUES (%s, %s, %s::uuid[], %s::uuid[])
            """, (query_text, query_embedding, result_ids, top_k_ids))
        self.conn.commit()

    def get_total_query_count(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM retrieval_log")
            return cur.fetchone()[0]

    # ── Recent embeddings (for novelty scoring) ──────────────────

    def get_recent_embeddings(self, limit: int = 50) -> List[List[float]]:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT embedding FROM memories
                WHERE embedding IS NOT NULL
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))
            rows = cur.fetchall()
        results = []
        for row in rows:
            emb = row[0]
            if emb is not None:
                results.append(list(emb))
        return results

    # ── Batch update ─────────────────────────────────────────────

    def update_fields(self, memory_id: str, fields: dict):
        if not fields:
            return
        set_parts = []
        params = []
        for k, v in fields.items():
            set_parts.append(f"{k} = %s")
            params.append(v)
        params.append(memory_id)
        sql = f"UPDATE memories SET {', '.join(set_parts)} WHERE id = %s"
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
        self.conn.commit()

    # ── Stats ────────────────────────────────────────────────────

    def stats(self) -> dict:
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    COALESCE(AVG(temperature), 0) as avg_temp,
                    SUM(CASE WHEN temperature >= 0.85 THEN 1 ELSE 0 END) as scorching,
                    SUM(CASE WHEN temperature >= 0.70 AND temperature < 0.85 THEN 1 ELSE 0 END) as hot,
                    SUM(CASE WHEN temperature >= 0.55 AND temperature < 0.70 THEN 1 ELSE 0 END) as warm,
                    SUM(CASE WHEN temperature >= 0.40 AND temperature < 0.55 THEN 1 ELSE 0 END) as room,
                    SUM(CASE WHEN temperature >= 0.25 AND temperature < 0.40 THEN 1 ELSE 0 END) as cool,
                    SUM(CASE WHEN temperature >= 0.10 AND temperature < 0.25 THEN 1 ELSE 0 END) as cold,
                    SUM(CASE WHEN temperature < 0.10 THEN 1 ELSE 0 END) as trace,
                    SUM(CASE WHEN region = 'hippocampus' THEN 1 ELSE 0 END) as hippocampus,
                    SUM(CASE WHEN region = 'neocortex' THEN 1 ELSE 0 END) as neocortex,
                    SUM(CASE WHEN region = 'archive' THEN 1 ELSE 0 END) as archive_count
                FROM memories
            """)
            row = cur.fetchone()

        return {
            "total": row["total"] or 0,
            "avg_temp": round(float(row["avg_temp"] or 0), 4),
            "bands": {
                "scorching": row["scorching"] or 0,
                "hot": row["hot"] or 0,
                "warm": row["warm"] or 0,
                "room": row["room"] or 0,
                "cool": row["cool"] or 0,
                "cold": row["cold"] or 0,
                "trace": row["trace"] or 0,
            },
            "regions": {
                "hippocampus": row["hippocampus"] or 0,
                "neocortex": row["neocortex"] or 0,
                "archive": row["archive_count"] or 0,
            },
        }

    # ── Bulk prune ───────────────────────────────────────────────

    def prune_below_temp(self, threshold: float) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM memories WHERE temperature < %s RETURNING id",
                (threshold,)
            )
            pruned = cur.rowcount
        self.conn.commit()
        return pruned

    def clear_all(self):
        """Delete all memories and related data. Used for benchmark isolation."""
        for table in ["memory_entities", "entity_edges", "entities",
                      "retrieval_log", "archive", "procedures", "memories"]:
            try:
                with self.conn.cursor() as cur:
                    cur.execute(f"DELETE FROM {table}")
                self.conn.commit()
            except Exception:
                self.conn.rollback()
        logger.info("Cleared all memories from store")

    def close(self):
        self.conn.close()


# ── Helpers ──────────────────────────────────────────────────────

def _mem_to_params(mem: Memory) -> dict:
    import numpy as np
    embedding = None
    if mem.embedding is not None:
        embedding = np.array(mem.embedding, dtype=np.float32)

    # Signal computations sometimes return numpy.float32 (cosine_similarity
    # over numpy arrays). psycopg2 doesn't know how to adapt those, so coerce
    # to native Python floats at the storage boundary.
    def _f(x):
        return float(x) if x is not None else None

    def _flist(xs):
        return [float(x) for x in xs] if xs else xs

    return {
        "id": mem.id,
        "raw_content": mem.raw_content,
        "structured_summary": mem.structured_summary,
        "knowledge_nugget": mem.knowledge_nugget,
        "entity_edges": mem.entity_edges,
        "temperature": _f(mem.temperature),
        "region": mem.region,
        "da_relevance": _f(mem.da_relevance),
        "ne_novelty": _f(mem.ne_novelty),
        "usage_score": _f(mem.usage_score),
        "da_history": _flist(mem.da_history),
        "ne_history": _flist(mem.ne_history),
        "signal_variance": _f(mem.signal_variance),
        "retrieval_hits": mem.retrieval_hits,
        "retrieval_importance": _f(mem.retrieval_importance),
        "embedding": embedding,
        "created_at": mem.created_at,
        "last_accessed": mem.last_accessed,
        "last_consolidated": mem.last_consolidated,
        "access_times": mem.access_times,
        "access_count": mem.access_count,
        "source": mem.source,
        "context": mem.context,
        "location": mem.location,
        "time_of_day": mem.time_of_day,
        "day_of_week": mem.day_of_week,
        "co_entities": mem.co_entities,
        "preceding_memory_id": mem.preceding_memory_id,
        "compression_level": mem.compression_level,
        "tokens_original": mem.tokens_original,
        "tokens_current": mem.tokens_current,
        "token_weights": json.dumps(mem.token_weights) if mem.token_weights else None,
        "knowledge_nuggets": json.dumps(mem.knowledge_nuggets) if mem.knowledge_nuggets else None,
        "last_compressed_at": mem.last_compressed_at,
        "compression_temperature": _f(mem.compression_temperature),
        "metadata": json.dumps(mem.metadata),
    }


import re as _re

# Stop words to strip before building FTS queries
_FTS_STOP = {
    'i', 'me', 'my', 'we', 'our', 'you', 'your', 'the', 'a', 'an',
    'is', 'am', 'are', 'was', 'were', 'been', 'be',
    'do', 'does', 'did', 'have', 'has', 'had',
    'will', 'would', 'could', 'should', 'can', 'may', 'might',
    'to', 'of', 'in', 'on', 'at', 'for', 'with', 'from', 'by',
    'and', 'or', 'but', 'not', 'no', 'so', 'if', 'than',
    'that', 'this', 'these', 'those', 'it', 'its',
    'how', 'what', 'which', 'where', 'when', 'who', 'whom', 'why',
}


def _extract_content_words(query: str) -> list:
    """Extract meaningful content words from a query, stripping stop words."""
    words = _re.findall(r'[a-zA-Z]+', query.lower())
    return [w for w in words if w not in _FTS_STOP and len(w) > 2]


def _build_or_tsquery(query: str) -> str:
    """Build an OR-joined tsquery string from a natural language query.

    'How long is my daily commute to work?' -> 'long | daily | commute | work'
    Returns empty string if no content words remain.
    """
    words = _extract_content_words(query)
    if not words:
        return ''
    return ' | '.join(words)


def _row_to_mem(row: dict) -> Memory:
    embedding = None
    if row.get("embedding") is not None:
        emb = row["embedding"]
        if hasattr(emb, "tolist"):
            embedding = emb.tolist()
        elif isinstance(emb, list):
            embedding = emb

    access_times = row.get("access_times") or []
    if isinstance(access_times, list):
        access_times = [
            t if isinstance(t, datetime) else t
            for t in access_times
        ]

    return Memory(
        id=str(row["id"]),
        raw_content=row.get("raw_content"),
        structured_summary=row.get("structured_summary"),
        knowledge_nugget=row.get("knowledge_nugget"),
        entity_edges=row.get("entity_edges"),
        temperature=float(row["temperature"]),
        region=row["region"],
        da_relevance=float(row.get("da_relevance", 0.3)),
        ne_novelty=float(row.get("ne_novelty", 0.5)),
        usage_score=float(row.get("usage_score", 0.0)),
        da_history=list(row.get("da_history") or []),
        ne_history=list(row.get("ne_history") or []),
        signal_variance=float(row.get("signal_variance", 0.5)),
        retrieval_hits=int(row.get("retrieval_hits", 0)),
        retrieval_importance=float(row.get("retrieval_importance", 0.0)),
        embedding=embedding,
        created_at=row.get("created_at") or datetime.now(timezone.utc),
        last_accessed=row.get("last_accessed") or datetime.now(timezone.utc),
        last_consolidated=row.get("last_consolidated"),
        access_times=access_times,
        access_count=int(row.get("access_count", 0)),
        source=row.get("source", "direct"),
        context=row.get("context", ""),
        location=row.get("location", ""),
        time_of_day=row.get("time_of_day", ""),
        day_of_week=row.get("day_of_week", ""),
        co_entities=list(row.get("co_entities") or []),
        preceding_memory_id=str(row["preceding_memory_id"]) if row.get("preceding_memory_id") else None,
        compression_level=row.get("compression_level", "raw"),
        tokens_original=row.get("tokens_original"),
        tokens_current=row.get("tokens_current"),
        token_weights=row.get("token_weights"),
        knowledge_nuggets=row.get("knowledge_nuggets"),
        last_compressed_at=row.get("last_compressed_at"),
        compression_temperature=float(row["compression_temperature"]) if row.get("compression_temperature") is not None else None,
        metadata=row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
    )
