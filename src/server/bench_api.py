"""
Bench API — HTTP server exposing Dendric for the memory-lifecycle-bench harness.

Implements the TargetSystemAdapter contract from
/Users/rileycoleman/Benchmark/memory-lifecycle-bench/harness/adapter.py:

    POST /api/reset         → clear all memory
    POST /api/ingest        → {text, session_id, timestamp, metadata}
    POST /api/consolidate   → {cycles}
    POST /api/query         → {question, top_k}  (full retrieval pipeline + Claude answer)
    GET  /api/stats         → memory counts & distribution

Shares the full retrieval pipeline from run_longmemeval.py (multi-query, temporal,
aggregation boost) so bench results reflect the "full system".

Run:
    python -m src.server.bench_api
Listens on http://localhost:5050 (matches bench config.yaml).
"""

from __future__ import annotations

import os
import sys
import time
import logging
import threading
from typing import Optional

from flask import Flask, request, jsonify

# Make the LME runner importable so we can reuse answer_question().
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.engine.core.engine import MemoryEngine
from src.engine.config import EngineConfig
from src.scripts.run_longmemeval import answer_question

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bench_api")

app = Flask(__name__)

# ── Engine (shared, stateful across requests) ────────────────────────────
_engine_lock = threading.Lock()
_engine: Optional[MemoryEngine] = None


def _envbool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes")


# Map env var name → EngineConfig flag name. Exposed so the ablation harness
# can flip components on/off via the environment without code changes.
_ABLATION_ENV_FLAGS = {
    "DENDRIC_ENABLE_VECTOR_PATH": "enable_vector_path",
    "DENDRIC_ENABLE_KEYWORD_PATH": "enable_keyword_path",
    "DENDRIC_ENABLE_ASSOCIATIVE_PATH": "enable_associative_path",
    "DENDRIC_ENABLE_GRAPH_PATH": "enable_graph_path",
    "DENDRIC_ENABLE_TEMPORAL_DECOMPOSITION": "enable_temporal_decomposition",
    "DENDRIC_ENABLE_TEMPORAL_RERANK": "enable_temporal_rerank",
    "DENDRIC_ACTIVATION_USE_GANE": "activation_use_gane",
    "DENDRIC_ACTIVATION_USE_NOISE": "activation_use_noise",
    "DENDRIC_ENABLE_LIFECYCLE_MODULATION": "enable_lifecycle_modulation",
}


def _get_engine() -> MemoryEngine:
    global _engine
    if _engine is None:
        # recall_mutates_state defaults to False here so benchmark runs are
        # deterministic across re-runs and query orderings. Set
        # DENDRIC_RECALL_MUTATES_STATE=1 to opt back in for production-style runs.
        kwargs = dict(
            db_url=os.environ.get("DATABASE_URL", "postgresql://localhost:5432/dendric_bench"),
            novelty_gate=float(os.environ.get("NOVELTY_GATE", "0.05")),
            overfetch_multiplier=int(os.environ.get("OVERFETCH_MULTIPLIER", "5")),
            recall_mutates_state=_envbool("DENDRIC_RECALL_MUTATES_STATE", False),
            # Persona: the implicit owner of this memory stream. First-person
            # memories ("I did X") don't mention the persona by name, so for
            # queries like "Has Jamie had X?" to reach those memories we need
            # an explicit persona entity that every memory links to. Default
            # 'jamie' matches the bench corpus; override via env for other runs.
            persona=os.environ.get("DENDRIC_PERSONA", "jamie"),
        )
        # Pull ablation flags. Default of every flag is True (full system).
        for env_name, field_name in _ABLATION_ENV_FLAGS.items():
            kwargs[field_name] = _envbool(env_name, True)

        config = EngineConfig(**kwargs)
        _engine = MemoryEngine(config=config)

        ablated = [n for n, f in _ABLATION_ENV_FLAGS.items() if not getattr(config, f)]
        ablated_msg = f" ablated={ablated}" if ablated else ""
        logger.info(
            f"engine initialized on {config.db_url} "
            f"(recall_mutates_state={config.recall_mutates_state}){ablated_msg}"
        )
    return _engine


# ── /api/reset ───────────────────────────────────────────────────────────
@app.route("/api/reset", methods=["POST"])
def reset():
    with _engine_lock:
        eng = _get_engine()
        eng.store.clear_all()
        eng._cycle_count = 0
        logger.info("memory cleared")
    return jsonify({"success": True})


# ── /api/ingest ──────────────────────────────────────────────────────────
@app.route("/api/ingest", methods=["POST"])
def ingest():
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    session_id = data.get("session_id", 0)
    timestamp = data.get("timestamp") or ""
    metadata = data.get("metadata") or {}

    if not text:
        return jsonify({"success": False, "memory_id": None, "gated": False,
                        "gate_reason": "empty text"}), 400

    # Build a context string matching LME convention: "TIMESTAMP | session_N"
    context = f"{timestamp} | session_{session_id}" if timestamp else f"session_{session_id}"

    with _engine_lock:
        eng = _get_engine()
        mem = eng.remember(content=text, source=metadata.get("source", "bench"), context=context)

    # Dendric's novelty gate sets temperature=0.1 and region=neocortex on gate.
    gated = bool(mem.get("temperature", 1.0) <= 0.1 and mem.get("region") == "neocortex")
    gate_reason = f"novelty below threshold (NE={mem.get('ne_novelty', 0):.3f})" if gated else None

    return jsonify({
        "success": True,
        "memory_id": mem.get("id"),
        "gated": gated,
        "gate_reason": gate_reason,
    })


# ── /api/consolidate ─────────────────────────────────────────────────────
@app.route("/api/consolidate", methods=["POST"])
def consolidate():
    data = request.get_json(silent=True) or {}
    cycles = int(data.get("cycles", 1))

    total_aggregates = 0
    with _engine_lock:
        eng = _get_engine()
        for i in range(cycles):
            try:
                result = eng.consolidate()
                if isinstance(result, dict):
                    total_aggregates += result.get("aggregates_created", 0)
            except Exception as e:
                logger.warning(f"consolidation cycle {i+1} failed: {e}")

    return jsonify({
        "memories_compressed": total_aggregates,
        "cycles": cycles,
    })


# ── /api/query ───────────────────────────────────────────────────────────
@app.route("/api/query", methods=["POST"])
def query():
    data = request.get_json(force=True) or {}
    question = (data.get("question") or "").strip()
    top_k = int(data.get("top_k", 10))
    question_timestamp = data.get("question_timestamp")  # optional, for temporal reasoning

    if not question:
        return jsonify({"error": "empty question"}), 400

    t0 = time.time()
    with _engine_lock:
        eng = _get_engine()
        try:
            answer, retrieved = answer_question(
                question=question,
                engine=eng,
                top_k=top_k,
                use_cot=True,
                question_timestamp=question_timestamp,
                return_retrieved=True,
            )
        except Exception as e:
            logger.exception(f"query failed: {e}")
            return jsonify({"error": str(e)}), 500
    total_ms = (time.time() - t0) * 1000.0

    # Rough split: we don't time retrieval and generation separately inside
    # answer_question, so report total as generation and retrieval=0. The
    # harness uses these for reporting, not scoring.
    memories_out = []
    for i, r in enumerate(retrieved):
        mem_id = r.get("id")
        mem = eng.store.get(mem_id) if mem_id else None
        content = mem.raw_content if mem else r.get("content", r.get("raw_content", ""))
        memories_out.append({
            "id": mem_id,
            "text": content,
            "session_id": _extract_session(r.get("context", "")),
            "timestamp": _extract_timestamp(r.get("context", "")),
            "score": float(r.get("fusion_score", r.get("similarity", r.get("score", 0.0)))),
        })

    return jsonify({
        "answer": answer,
        "retrieval": {
            "memories": memories_out,
            "query_time_ms": 0.0,
            "paths_used": list(set(
                p for r in retrieved for p in r.get("retrieval_paths", [])
            )) or ["vector"],
        },
        "generation_time_ms": total_ms,
    })


# ── /api/stats ───────────────────────────────────────────────────────────
@app.route("/api/stats", methods=["GET"])
def stats():
    with _engine_lock:
        eng = _get_engine()
        s = eng.stats()

    bands = s.get("bands", {})
    # Map Dendric bands → adapter's hot/warm/cold buckets.
    hot = bands.get("scorching", 0) + bands.get("hot", 0)
    warm = bands.get("warm", 0) + bands.get("room", 0)
    cold = bands.get("cool", 0) + bands.get("cold", 0) + bands.get("trace", 0)

    # Entity count — cheap query against entities table.
    entity_count = 0
    try:
        with eng.store.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM entities")
            row = cur.fetchone()
            entity_count = int(row[0] or 0) if row else 0
    except Exception:
        pass

    return jsonify({
        "total_memories": s.get("total", 0),
        "temperature_distribution": {"hot": hot, "warm": warm, "cold": cold},
        "entity_count": entity_count,
        "gated_count": 0,  # Dendric stores gated memories at temp=0.1; not tracked separately.
        "raw": s,
    })


# ── helpers ──────────────────────────────────────────────────────────────
def _extract_session(context: str):
    if not context:
        return None
    tail = context.split("|")[-1].strip() if "|" in context else context
    import re
    m = re.search(r"session[_ ]?(\d+)", tail)
    return int(m.group(1)) if m else None


def _extract_timestamp(context: str) -> str:
    if not context or "|" not in context:
        return ""
    return context.split("|")[0].strip()


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


if __name__ == "__main__":
    port = int(os.environ.get("BENCH_API_PORT", "5050"))
    print(f"\n  Dendric Bench API — http://localhost:{port}")
    print(f"  Endpoints: /api/reset /api/ingest /api/consolidate /api/query /api/stats\n")
    # threaded=False because the engine + psycopg2 connection aren't thread-safe
    # and we already serialize with _engine_lock. Single worker is fine.
    app.run(host="0.0.0.0", port=port, debug=False, threaded=False)
