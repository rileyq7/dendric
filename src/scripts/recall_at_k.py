"""Memory-level recall@k — substrate-only retrieval metric.

Measures: for a query whose answer lives in known ground-truth memories, did
any / all of them appear in the top-k retrieved set? This isolates substrate
quality from answer-model synthesis — a number that's valid regardless of
whether Claude/GPT synthesizes the answer correctly.

Three variants reported (each query gets all three, we average separately):
  recall_any @k  — 1 if ANY gold memory is in top-k, else 0.  Permissive.
  recall_all @k  — 1 if EVERY gold memory is in top-k, else 0.  Strict.
  recall_frac@k  — (# gold memories found) / (# gold memories).  Fractional.

For aggregation queries where the gold is scattered across many turns,
recall_all is punitive and recall_frac is the honest one. For single-
mention questions all three tend to agree.

Two corpora:
  longmemeval: uses has_answer=true turn markers as gold. Per-question
      ingest into a fresh DB, no consolidation — tests COLD retrieval.
  meridian_deep: uses a hand-annotated probe file (gold memory ids
      resolved manually). Tests AGED retrieval on real personal data.

Usage:
    python -m src.scripts.recall_at_k --corpus longmemeval
    python -m src.scripts.recall_at_k --corpus meridian_deep \
        --annotations probes/meridian_recall_gold.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.engine.config import EngineConfig, PATH_NAMES, leave_one_out_config, plain_rag_config  # noqa: E402
from src.engine.core.engine import MemoryEngine  # noqa: E402


def _build_engine(
    db_url: str, persona: str, config_name: str,
    overrides: dict | None = None,
) -> MemoryEngine:
    """Construct a MemoryEngine with the named preset.
      full        — production config (lifecycle, 4 paths, déjà-vu, all knobs on)
      plain_rag   — baseline: vector-only, no lifecycle, no fancy paths
      no_<path>   — full config with one path disabled (leave-one-out).
                    path ∈ {vector, keyword, associative, graph}
    New configs can be added here without touching the rest of the harness.

    overrides: optional dict of EngineConfig field → value. Applied AFTER
    the preset constructs the config, so e.g. a parameter sweep can run
    'full' as a base and tweak archive_rrf_boost without writing a new
    preset for every grid point.
    """
    if config_name == "plain_rag":
        cfg = plain_rag_config(db_url=db_url, persona=persona)
    elif config_name == "full":
        cfg = EngineConfig(db_url=db_url, persona=persona, recall_mutates_state=False)
    elif config_name.startswith("no_"):
        path = config_name[len("no_"):]
        if path not in PATH_NAMES:
            raise ValueError(
                f"Unknown LOO config {config_name!r}. "
                f"Expected no_<path> with path in {PATH_NAMES}."
            )
        cfg = leave_one_out_config(disabled_path=path, db_url=db_url, persona=persona)
    else:
        raise ValueError(f"Unknown config: {config_name!r}")

    if overrides:
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise ValueError(
                    f"Unknown EngineConfig field {k!r}. "
                    f"Cannot apply override {k}={v!r}."
                )
            setattr(cfg, k, v)
    return MemoryEngine(config=cfg)
from src.scripts.run_longmemeval import (  # noqa: E402
    load_longmemeval_data,
    ingest_longmemeval_item,
)


logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
logger = logging.getLogger("recall_at_k")


# ── Context-string parsing (Dendric-specific) ────────────────────────────

_SESSION_TURN_RE = re.compile(r"session_(\d+)_turn_(\d+)")
_SESSION_CHUNK_RE = re.compile(r"session_(\d+)_chunk_(\d+)")
_SESSION_SUMMARY_RE = re.compile(r"session_(\d+)_summary")


def parse_context(ctx: str | None) -> dict:
    """Parse Dendric's context string into structured fields.
    Expected formats (from longmemeval ingest):
      "2023/05/20 (Sat) 14:38 | session_1_turn_0"
      "2023/05/20 (Sat) 14:38 | session_1_chunk_0"
      "2023/05/20 (Sat) 14:38 | session_1_summary"
    """
    if not ctx:
        return {}
    out = {}
    m = _SESSION_TURN_RE.search(ctx)
    if m:
        out["session_idx"] = int(m.group(1))
        out["turn_idx"] = int(m.group(2))
        out["granularity"] = "turn"
        return out
    m = _SESSION_CHUNK_RE.search(ctx)
    if m:
        out["session_idx"] = int(m.group(1))
        out["chunk_idx"] = int(m.group(2))
        out["granularity"] = "chunk"
        return out
    m = _SESSION_SUMMARY_RE.search(ctx)
    if m:
        out["session_idx"] = int(m.group(1))
        out["granularity"] = "summary"
        return out
    return {}


# ── Matching: which retrieved memories are "gold" hits? ──────────────────

@dataclass
class GoldSpec:
    """Describes what constitutes a ground-truth hit for one query.

    gold_turns: set of (session_idx, turn_idx) tuples. Any retrieved memory
        whose context matches one of these counts as a hit.
    gold_sessions: set of session_idx. Any retrieved chunk/summary from
        one of these sessions also counts — the user agreed that finding
        any chunk of the answer-bearing session satisfies "retrieval got
        us to the right place."
    n_expected: how many distinct gold_turns we're looking for (for the
        _frac metric). If 0, query is unanswerable and we score differently.
    """
    gold_turns: set[tuple[int, int]] = field(default_factory=set)
    gold_sessions: set[int] = field(default_factory=set)

    @property
    def n_expected(self) -> int:
        # Each gold turn is an independent piece of evidence.
        return len(self.gold_turns)


def is_hit(memory_ctx: str, spec: GoldSpec) -> bool:
    """Does this retrieved memory cover any gold-bearing turn?"""
    parsed = parse_context(memory_ctx)
    if not parsed:
        return False
    sidx = parsed.get("session_idx")
    if sidx is None:
        return False
    # Turn-level hit: exact (session, turn) match
    if parsed["granularity"] == "turn":
        return (sidx, parsed["turn_idx"]) in spec.gold_turns
    # Chunk/summary hit: any memory from the gold-bearing session counts,
    # per design decision — a chunk spans multiple turns and the retrieval
    # "got us to the right place" even if it didn't pinpoint the specific turn.
    return sidx in spec.gold_sessions


def which_turns_covered(retrieved: list[dict], spec: GoldSpec) -> set[tuple[int, int]]:
    """Return the set of gold (session, turn) tuples that any retrieved
    memory covers. Used for recall_all and recall_frac computation.

    A chunk/summary from a gold-bearing session covers ALL gold turns in
    that session (we can't tell which specific turn without inspecting the
    chunk text, and the design decision says any chunk of the right
    session counts as a hit).
    """
    covered: set[tuple[int, int]] = set()
    for r in retrieved:
        parsed = parse_context(r.get("context"))
        if not parsed:
            continue
        sidx = parsed.get("session_idx")
        if sidx is None:
            continue
        if parsed["granularity"] == "turn":
            key = (sidx, parsed.get("turn_idx", -1))
            if key in spec.gold_turns:
                covered.add(key)
        else:
            # Chunk/summary: covers every gold turn in that session
            for (s, t) in spec.gold_turns:
                if s == sidx:
                    covered.add((s, t))
    return covered


# ── Metrics ──────────────────────────────────────────────────────────────

@dataclass
class RecallRow:
    query: str
    k: int
    n_retrieved: int
    n_expected: int
    n_covered: int
    recall_any: float   # 0 or 1
    recall_all: float   # 0 or 1
    recall_frac: float  # [0, 1]


def score_one(query: str, retrieved: list[dict], spec: GoldSpec, k: int) -> RecallRow:
    topk = retrieved[:k]
    covered = which_turns_covered(topk, spec)
    n_exp = spec.n_expected
    if n_exp == 0:
        # Unanswerable question — if no gold exists, we conventionally
        # treat recall as "not applicable" but return zeros so it still
        # folds into aggregates with a clear signal. Caller should filter
        # these out when reporting.
        return RecallRow(query, k, len(topk), 0, 0, 0.0, 0.0, 0.0)
    return RecallRow(
        query=query, k=k,
        n_retrieved=len(topk),
        n_expected=n_exp,
        n_covered=len(covered),
        recall_any=1.0 if covered else 0.0,
        recall_all=1.0 if len(covered) == n_exp else 0.0,
        recall_frac=len(covered) / n_exp,
    )


# ── LongMemEval driver ───────────────────────────────────────────────────

def build_longmemeval_spec(item: dict) -> GoldSpec:
    spec = GoldSpec()
    sessions = item.get("haystack_sessions", []) or []
    for sidx, turns in enumerate(sessions):
        for tidx, turn in enumerate(turns):
            if turn.get("has_answer"):
                spec.gold_turns.add((sidx, tidx))
                spec.gold_sessions.add(sidx)
    return spec


def eval_longmemeval(
    data_path: str | None,
    db_url: str,
    k_values: Iterable[int],
    max_questions: int | None,
    question_types: list[str] | None,
    consolidation_cycles: int,
    per_type: int | None,
    config_name: str = "full",
    overrides: dict | None = None,
    skip_ingest: bool = False,
) -> list[RecallRow]:
    data = load_longmemeval_data(data_path)

    if question_types:
        data = [d for d in data if d.get("question_type") in question_types]

    if per_type:
        from collections import defaultdict
        by_type: dict[str, list[dict]] = defaultdict(list)
        for item in data:
            by_type[item.get("question_type", "unknown")].append(item)
        sampled = []
        for t in sorted(by_type.keys()):
            sampled.extend(by_type[t][:per_type])
        data = sampled

    if max_questions:
        data = data[:max_questions]

    rows: list[RecallRow] = []
    k_list = sorted(set(k_values))
    max_k = max(k_list)

    for i, item in enumerate(data):
        q_text = item.get("question", "")
        q_id = item.get("question_id", f"q{i}")
        q_type = item.get("question_type", "unknown")

        spec = build_longmemeval_spec(item)
        if spec.n_expected == 0:
            # Skip abstention/unanswerable — they'd inflate zeros
            continue

        # Fresh DB per question — LongMemEval uses per-question haystacks.
        # Use a prefixed DB name to avoid clobbering anything.
        per_q_db = f"{db_url}_{q_id[:12].replace('-', '')}"
        ingest_engine = None
        engine = None
        try:
            _ensure_db(per_q_db)
            # Ingest ALWAYS uses the full config so the corpus in the DB
            # is the same across comparisons — only retrieval behavior
            # differs. Otherwise an ablation that skips entity extraction
            # at ingest would be comparing different DBs, not different
            # retrieval strategies.
            #
            # skip_ingest=True re-uses the per-question DB as-is. Used by
            # parameter sweeps where ingest is identical across sweep points
            # (the swept knobs only affect retrieval). Saves 5+ min/run.
            if not skip_ingest:
                ingest_engine = MemoryEngine(config=EngineConfig(
                    db_url=per_q_db, persona="", recall_mutates_state=False,
                ))
                ingest_engine.store.clear_all()
                ingest_longmemeval_item(item, ingest_engine, consolidation_cycles=consolidation_cycles)

            # Retrieval engine: uses the requested config preset
            engine = _build_engine(per_q_db, "", config_name, overrides=overrides)

            retrieved = engine.recall(query=q_text, top_k=max_k, reheat=False)
            for k in k_list:
                row = score_one(q_text, retrieved, spec, k)
                # Attach question_type to the row via a sidecar dict — we
                # carry it on the object via an ad-hoc attribute for grouping
                row_with_type = row
                row_with_type.__dict__["question_type"] = q_type
                rows.append(row_with_type)
        except Exception as e:
            logger.exception(f"Question {q_id} failed: {e}")
        finally:
            # Close per-question engines so connections don't accumulate
            # across hundreds of questions or across successive configs in
            # the ablation driver. Without this, Postgres max_connections
            # (default 100) deadlocks the run around the second config.
            for eng in (ingest_engine, engine):
                if eng is not None:
                    try:
                        eng.close()
                    except Exception:
                        pass

        if (i + 1) % 10 == 0:
            logger.warning(f"Processed {i + 1}/{len(data)} questions")

    return rows


def _ensure_db(db_url: str) -> None:
    """Create DB + vector extension if it doesn't exist. Uses the DB name
    from the URL; same pattern as run_longmemeval's per-worker DB logic."""
    import psycopg2
    from urllib.parse import urlparse
    parsed = urlparse(db_url)
    db_name = parsed.path.lstrip("/")
    admin_url = db_url.rsplit("/", 1)[0] + "/postgres"
    conn = psycopg2.connect(admin_url)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
            if not cur.fetchone():
                cur.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        conn.close()
    # Ensure extension in the target DB
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    finally:
        conn.close()


# ── meridian_deep driver ─────────────────────────────────────────────────

def eval_meridian_deep(
    db_url: str,
    annotations_path: str,
    k_values: Iterable[int],
    config_name: str = "full",
    overrides: dict | None = None,
) -> list[RecallRow]:
    """Evaluate against a hand-annotated probe file. Format:
        [
          {
            "query": "...",
            "gold_memory_ids": ["uuid", ...]   # OR
            "gold_context_matches": ["substring1", "substring2"]
          },
          ...
        ]
    The context-matches form lets you annotate by substring rather than
    needing to look up every gold memory's id ahead of time.
    """
    with open(annotations_path) as f:
        annotations = json.load(f)

    engine = _build_engine(
        db_url=db_url,
        persona=os.environ.get("PERSONA", "riley"),
        config_name=config_name,
        overrides=overrides,
    )

    rows: list[RecallRow] = []
    k_list = sorted(set(k_values))
    max_k = max(k_list)

    try:
        for ann in annotations:
            query = ann["query"]
            gold_ids = set(ann.get("gold_memory_ids", []) or [])
            gold_substrings = ann.get("gold_context_matches", []) or []

            retrieved = engine.recall(query=query, top_k=max_k, reheat=False)

            # Build a "covered" set differently — here we just count how many
            # retrieved memories match gold. Each gold id / substring is one
            # expected unit.
            n_expected = len(gold_ids) + len(gold_substrings)
            if n_expected == 0:
                continue

            for k in k_list:
                topk = retrieved[:k]
                covered_ids: set[str] = set()
                covered_substrings: set[str] = set()
                for r in topk:
                    rid = str(r.get("id", ""))
                    if rid in gold_ids:
                        covered_ids.add(rid)
                    text = (r.get("raw_content") or "").lower()
                    ctx = (r.get("context") or "").lower()
                    for s in gold_substrings:
                        sl = s.lower()
                        if sl in text or sl in ctx:
                            covered_substrings.add(s)
                n_covered = len(covered_ids) + len(covered_substrings)
                rows.append(RecallRow(
                    query=query, k=k,
                    n_retrieved=len(topk),
                    n_expected=n_expected,
                    n_covered=n_covered,
                    recall_any=1.0 if n_covered > 0 else 0.0,
                    recall_all=1.0 if n_covered == n_expected else 0.0,
                    recall_frac=n_covered / n_expected,
                ))
    finally:
        # Close so successive calls (e.g. the ablation driver looping
        # through 6 configs) don't accumulate PG connections.
        try:
            engine.close()
        except Exception:
            pass

    return rows


# ── Reporting ────────────────────────────────────────────────────────────

def summarize_repeats(all_runs: list[list[RecallRow]]) -> dict:
    """Variance/stability summary across repeated runs of the same eval.

    For each k, compute mean ± SD of (any, all, frac) across the N runs.
    If SD is tight (<2pp), the substrate is deterministic under this
    config. If SD is wide, there's non-determinism worth tracking down
    before making headline claims.
    """
    from collections import defaultdict
    import math

    # Aggregate per run → per k
    per_run_per_k: list[dict[int, dict]] = []
    for run_rows in all_runs:
        per_k: dict[int, dict] = defaultdict(lambda: {"n": 0, "any": 0.0, "all": 0.0, "frac": 0.0})
        for r in run_rows:
            per_k[r.k]["n"] += 1
            per_k[r.k]["any"] += r.recall_any
            per_k[r.k]["all"] += r.recall_all
            per_k[r.k]["frac"] += r.recall_frac
        per_run_per_k.append({
            k: {"any": v["any"]/v["n"], "all": v["all"]/v["n"], "frac": v["frac"]/v["n"], "n": v["n"]}
            for k, v in per_k.items()
        })

    # Collect across runs
    ks = sorted({k for r in per_run_per_k for k in r.keys()})
    out: dict[int, dict] = {}
    for k in ks:
        vals_any = [r[k]["any"] for r in per_run_per_k if k in r]
        vals_all = [r[k]["all"] for r in per_run_per_k if k in r]
        vals_frac = [r[k]["frac"] for r in per_run_per_k if k in r]
        n = per_run_per_k[0][k]["n"] if k in per_run_per_k[0] else 0

        def _mean_sd(xs):
            m = sum(xs) / len(xs)
            sd = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)) if len(xs) > 1 else 0.0
            return m, sd

        m_any, sd_any = _mean_sd(vals_any)
        m_all, sd_all = _mean_sd(vals_all)
        m_frac, sd_frac = _mean_sd(vals_frac)
        out[k] = {
            "n_queries": n,
            "n_runs": len(vals_any),
            "any_mean": m_any, "any_sd": sd_any,
            "all_mean": m_all, "all_sd": sd_all,
            "frac_mean": m_frac, "frac_sd": sd_frac,
            "any_per_run": vals_any,
            "all_per_run": vals_all,
            "frac_per_run": vals_frac,
        }
    return out


def print_variance_report(summary: dict, title: str) -> None:
    print(f"\n{'='*78}")
    print(f" {title}")
    print(f"{'='*78}")
    print(f"{'k':>4}  {'n_q':>4}  {'runs':>4}  "
          f"{'any mean±sd':>16}  {'all mean±sd':>16}  {'frac mean±sd':>16}")
    print("-" * 78)
    for k, v in sorted(summary.items()):
        def fmt(m, sd):
            return f"{m*100:5.1f}% ± {sd*100:4.1f}pp"
        def fmt_frac(m, sd):
            return f"{m:.3f} ± {sd:.3f}"
        print(
            f"{k:>4}  {v['n_queries']:>4}  {v['n_runs']:>4}  "
            f"{fmt(v['any_mean'], v['any_sd']):>16}  "
            f"{fmt(v['all_mean'], v['all_sd']):>16}  "
            f"{fmt_frac(v['frac_mean'], v['frac_sd']):>16}"
        )
    # Per-run raw values so any outlier is obvious
    print("\n  per-run values (to spot outliers):")
    for k in sorted(summary.keys()):
        v = summary[k]
        anys = ", ".join(f"{x*100:.1f}%" for x in v["any_per_run"])
        alls = ", ".join(f"{x*100:.1f}%" for x in v["all_per_run"])
        print(f"    k={k}  any: [{anys}]   all: [{alls}]")


def summarize(rows: list[RecallRow], group_by: str | None = None) -> dict:
    """Aggregate rows by k and optional grouping key."""
    from collections import defaultdict
    by_k: dict[int, dict] = defaultdict(lambda: {
        "n": 0, "any": 0.0, "all": 0.0, "frac": 0.0
    })
    # Optional grouping: group_by is an attribute name stored on the row
    # (e.g., 'question_type'). We only use it for longmemeval rows.
    by_group_k: dict[tuple[str, int], dict] = defaultdict(lambda: {
        "n": 0, "any": 0.0, "all": 0.0, "frac": 0.0
    })

    for r in rows:
        by_k[r.k]["n"] += 1
        by_k[r.k]["any"] += r.recall_any
        by_k[r.k]["all"] += r.recall_all
        by_k[r.k]["frac"] += r.recall_frac
        if group_by:
            g = r.__dict__.get(group_by, "unknown")
            key = (g, r.k)
            by_group_k[key]["n"] += 1
            by_group_k[key]["any"] += r.recall_any
            by_group_k[key]["all"] += r.recall_all
            by_group_k[key]["frac"] += r.recall_frac

    def _norm(d):
        n = d["n"]
        if n == 0:
            return {"n": 0, "any": 0.0, "all": 0.0, "frac": 0.0}
        return {
            "n": n,
            "any": d["any"] / n,
            "all": d["all"] / n,
            "frac": d["frac"] / n,
        }

    return {
        "overall": {k: _norm(v) for k, v in sorted(by_k.items())},
        "by_group": {f"{g}_k{k}": _norm(v) for (g, k), v in sorted(by_group_k.items())},
    }


def print_report(summary: dict, title: str) -> None:
    print(f"\n{'='*78}")
    print(f" {title}")
    print(f"{'='*78}")
    print(f"{'k':>4}  {'n':>5}  {'any':>7}  {'all':>7}  {'frac':>7}")
    print("-" * 44)
    for k, v in summary["overall"].items():
        print(f"{k:>4}  {v['n']:>5}  {v['any']:>6.1%}  {v['all']:>6.1%}  {v['frac']:>6.3f}")

    by_group = summary.get("by_group", {})
    if by_group:
        print("\n  by group:")
        for gk, v in sorted(by_group.items()):
            # gk is "group_k{k}" format
            print(f"    {gk:<45}  n={v['n']:>3}  any={v['any']:>5.1%}  all={v['all']:>5.1%}  frac={v['frac']:>5.3f}")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["longmemeval", "meridian_deep"], required=True)
    parser.add_argument(
        "--config",
        choices=["full", "plain_rag"] + [f"no_{p}" for p in PATH_NAMES],
        default="full",
        help="Retrieval config preset. full = production. "
             "plain_rag = vector-only baseline. "
             "no_<path> = leave-one-out ablation (path ∈ {vector,keyword,associative,graph}).",
    )
    parser.add_argument("--k", default="5,10,25",
                        help="Comma-separated k values to evaluate at")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Run the eval N times and report mean±SD "
                             "(meridian_deep only — longmemeval per-run "
                             "ingest is too expensive)")
    parser.add_argument("--output", default=None,
                        help="Path to write raw rows as JSON")
    # longmemeval-specific
    parser.add_argument("--data", default=None)
    parser.add_argument("--db-prefix", default="postgresql://localhost:5432/recall_at_k",
                        help="Per-question DB url prefix (longmemeval only)")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--types", nargs="+", default=None)
    parser.add_argument("--per-type", type=int, default=None)
    parser.add_argument("--cycles", type=int, default=1,
                        help="Consolidation cycles post-ingest (longmemeval)")
    # meridian_deep-specific
    parser.add_argument("--db", default="postgresql://localhost:5432/meridian_deep")
    parser.add_argument("--annotations", default=None,
                        help="Path to meridian_deep gold annotations JSON")
    # parameter sweep support
    parser.add_argument(
        "--override", action="append", default=[],
        help="Patch an EngineConfig field after preset construction. "
             "Repeatable. Format: key=value, where value is parsed as JSON "
             "(so 1.8 → float, true → bool, etc.). "
             "Example: --override archive_rrf_boost=1.8 --override mod_temp_lift=0.4",
    )
    parser.add_argument(
        "--skip-ingest", action="store_true",
        help="Re-use existing per-question DBs without clearing or re-ingesting "
             "(longmemeval only). Use this when sweeping retrieval-only knobs "
             "after one initial ingest pass.",
    )
    args = parser.parse_args()

    overrides: dict = {}
    for pair in args.override:
        if "=" not in pair:
            print(f"ERROR: --override expects key=value, got {pair!r}", file=sys.stderr)
            sys.exit(1)
        key, raw_val = pair.split("=", 1)
        try:
            value = json.loads(raw_val)
        except json.JSONDecodeError:
            # Fall back to string if not valid JSON
            value = raw_val
        overrides[key.strip()] = value

    k_values = [int(x.strip()) for x in args.k.split(",") if x.strip()]

    if args.corpus == "longmemeval":
        rows = eval_longmemeval(
            data_path=args.data,
            db_url=args.db_prefix,
            k_values=k_values,
            max_questions=args.max_questions,
            question_types=args.types,
            consolidation_cycles=args.cycles,
            per_type=args.per_type,
            config_name=args.config,
            overrides=overrides or None,
            skip_ingest=args.skip_ingest,
        )
        summary = summarize(rows, group_by="question_type")
        print_report(summary, f"LongMemEval recall@k [{args.config}] — {len(rows)} question-k pairs")
    else:
        if not args.annotations:
            print("--annotations is required for meridian_deep", file=sys.stderr)
            sys.exit(1)
        if args.repeats > 1:
            all_runs: list[list[RecallRow]] = []
            for i in range(args.repeats):
                logger.warning(f"Repeat {i+1}/{args.repeats}...")
                rows_i = eval_meridian_deep(
                    db_url=args.db,
                    annotations_path=args.annotations,
                    k_values=k_values,
                    config_name=args.config,
                    overrides=overrides or None,
                )
                all_runs.append(rows_i)
            variance_summary = summarize_repeats(all_runs)
            print_variance_report(
                variance_summary,
                f"meridian_deep recall@k [{args.config}] — {args.repeats} repeats",
            )
            if args.output:
                out = {
                    "variance_summary": {str(k): v for k, v in variance_summary.items()},
                    "all_runs": [[r.__dict__ for r in run] for run in all_runs],
                }
                with open(args.output, "w") as f:
                    json.dump(out, f, indent=2, default=str)
                print(f"\nRaw data written to {args.output}")
            return
        rows = eval_meridian_deep(
            db_url=args.db,
            annotations_path=args.annotations,
            k_values=k_values,
            config_name=args.config,
            overrides=overrides or None,
        )
        summary = summarize(rows)
        print_report(summary, f"meridian_deep recall@k [{args.config}] — {len(rows)} query-k pairs")

    if args.output:
        out = {
            "summary": summary,
            "rows": [r.__dict__ for r in rows],
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nRaw rows written to {args.output}")


if __name__ == "__main__":
    main()
