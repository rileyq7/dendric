#!/usr/bin/env python3
"""
Stability Check — verify recall is deterministic across repeated runs.

Goal: gate the system as "measurement-ready" before any ablation work.

What it does:
  1. Spin up an isolated engine with recall_mutates_state=False
  2. Ingest a small fixed corpus
  3. Run a fixed query set TWICE
  4. Diff the per-query top-k memory IDs between run 1 and run 2

Pass: every query returns identical memory IDs in identical order.
Fail: prints the first divergent query and the diff, then exits 1.

A failure means there is still a hidden mutation source somewhere in the
recall path. Likely candidates: (a) consolidation triggered implicitly,
(b) noise in the activation equation leaking into recall (recall reads
temperature, but consolidation is what calls compute_temperature with
noise=True — they should be decoupled), (c) cache state inside the
retrieval paths, (d) ORDER BY ties broken nondeterministically by Postgres.

Usage:
    python -m src.scripts.stability_check
    python -m src.scripts.stability_check --queries 20 --corpus 50
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.engine.core.engine import MemoryEngine
from src.engine.config import EngineConfig

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("stability_check")


# Small fixed corpus — enough variety that the 4 retrieval paths each have
# something to bite on, but small enough to ingest in seconds.
CORPUS = [
    ("2025/01/05 (Sun) 09:00 | session_1",  "I started learning Spanish on Duolingo today. Aiming for 15 minutes a day."),
    ("2025/01/05 (Sun) 14:30 | session_1",  "Picked up a copy of Project Hail Mary by Andy Weir at the bookstore."),
    ("2025/01/06 (Mon) 08:15 | session_2",  "Morning run was 3.2 miles, average pace 9:42. Knee felt fine."),
    ("2025/01/06 (Mon) 19:00 | session_2",  "Cooked a Thai green curry for dinner. Used too much fish sauce."),
    ("2025/01/07 (Tue) 10:00 | session_3",  "Standup meeting: Jamie is blocked on the Prism integration, needs auth tokens."),
    ("2025/01/07 (Tue) 21:30 | session_3",  "Finished chapter 4 of Project Hail Mary. The Rocky reveal is great."),
    ("2025/01/08 (Wed) 09:45 | session_4",  "Mango (the cat) knocked over the water glass on the desk this morning."),
    ("2025/01/08 (Wed) 16:00 | session_4",  "Got the Prism auth tokens working. Pushed the PR for review."),
    ("2025/01/09 (Thu) 08:00 | session_5",  "Mango got hurt — limping on the back left leg. Vet appointment scheduled for tomorrow."),
    ("2025/01/09 (Thu) 12:30 | session_5",  "Tried a new ramen place on 4th street. Tonkotsu was excellent."),
    ("2025/01/10 (Fri) 11:00 | session_6",  "Vet visit: Mango has a sprain, needs to rest for a week. No jumping."),
    ("2025/01/10 (Fri) 18:00 | session_6",  "Spanish streak hit 5 days. Learning numbers and colors this week."),
    ("2025/01/11 (Sat) 10:00 | session_7",  "Painted the Tamiya model kit — Vallejo paints work much better than Citadel for thin layers."),
    ("2025/01/12 (Sun) 09:00 | session_8",  "Mango is doing better, only mild limp now. Still keeping him off the cat tree."),
    ("2025/01/12 (Sun) 14:00 | session_8",  "Started chapter 7 of Project Hail Mary. About a third of the way through."),
]

QUERIES = [
    "What book am I reading?",
    "How is Mango doing?",
    "What was happening around the time Mango got hurt?",
    "How many days have I been doing Spanish?",
    "Tell me about the Prism work",
    "What paints do I use for models?",
    "Have I been running recently?",
    "What restaurants have I tried?",
]


def run_query_set(engine: MemoryEngine, queries: List[str], top_k: int) -> List[List[str]]:
    """Return per-query list of retrieved memory IDs in rank order."""
    results = []
    for q in queries:
        recalled = engine.recall(q, top_k=top_k)
        ids = [r["id"] for r in recalled]
        results.append(ids)
    return results


def diff_runs(run1: List[List[str]], run2: List[List[str]], queries: List[str]) -> List[dict]:
    """Compare two runs query-by-query. Returns list of divergence records."""
    divergences = []
    for i, (q, ids1, ids2) in enumerate(zip(queries, run1, run2)):
        if ids1 != ids2:
            divergences.append({
                "query_index": i,
                "query": q,
                "run1_ids": ids1,
                "run2_ids": ids2,
                "shared": [x for x in ids1 if x in ids2],
                "only_in_run1": [x for x in ids1 if x not in ids2],
                "only_in_run2": [x for x in ids2 if x not in ids1],
            })
    return divergences


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db-url", default=os.environ.get(
        "DATABASE_URL", "postgresql://localhost:5432/dendric_stability"
    ))
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--corpus", type=int, default=len(CORPUS),
                   help="How many corpus items to ingest (default: all)")
    p.add_argument("--queries", type=int, default=len(QUERIES),
                   help="How many queries to run (default: all)")
    args = p.parse_args()

    corpus = CORPUS[:args.corpus]
    queries = QUERIES[:args.queries]

    print(f"Stability check: {len(corpus)} corpus items, {len(queries)} queries, top_k={args.top_k}")
    print(f"DB: {args.db_url}")

    config = EngineConfig(
        db_url=args.db_url,
        recall_mutates_state=False,  # The whole point — must be off
    )
    engine = MemoryEngine(config=config)

    # Clean slate
    print("Clearing store...")
    engine.store.clear_all()

    # Ingest
    print(f"Ingesting {len(corpus)} memories...")
    for ctx, text in corpus:
        engine.remember(content=text, source="stability_check", context=ctx)

    # Run twice
    print("Running query set, pass 1...")
    run1 = run_query_set(engine, queries, args.top_k)

    print("Running query set, pass 2...")
    run2 = run_query_set(engine, queries, args.top_k)

    # Diff
    divergences = diff_runs(run1, run2, queries)

    if not divergences:
        print(f"\n✓ STABLE: all {len(queries)} queries returned identical results across both runs.")
        print("  System is measurement-ready. Proceed to ablation.")
        return 0

    print(f"\n✗ UNSTABLE: {len(divergences)}/{len(queries)} queries diverged between runs.\n")
    for d in divergences:
        print(f"  Query #{d['query_index']}: {d['query']!r}")
        print(f"    Run 1 IDs: {d['run1_ids']}")
        print(f"    Run 2 IDs: {d['run2_ids']}")
        print(f"    Only in run 1: {d['only_in_run1']}")
        print(f"    Only in run 2: {d['only_in_run2']}")
        print()

    print("Hidden mutation source remains. Likely suspects:")
    print("  - Activation equation noise leaking into recall (check noise=True call sites)")
    print("  - Implicit consolidation between runs (check engine.recall for stray consolidate calls)")
    print("  - Cache state in retrieval paths (vector/keyword/associative/graph)")
    print("  - Postgres ORDER BY tie-breaking (add deterministic secondary sort)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
