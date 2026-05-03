"""Diagnostic: why is the associative path net-harmful on meridian_deep?

The path ablation showed `no_associative` beats `full` by +6.7pp recall_any@5
on meridian_deep at both sa_decay=0.5 and sa_decay=0.7. This script
inspects what's actually different in top-k between the two configs to
test two hypotheses:

  (a) Archive memories surfaced by associative_archive trigger push good
      non-archive candidates out of top-k.
  (b) Spreading-activation hits on persona-adjacent / high-fan-out
      entities introduce off-topic candidates that out-rank gold.

For each probe, we report:
  - full top-10: rank, region, retrieval_paths, gold-hit?, content snippet
  - no_associative top-10: same shape
  - Diff: which gold dropped out / which dropped in / which was pushed
    from top-5 to top-6+

Usage:
  DATABASE_URL=postgresql://localhost:5432/meridian_deep PERSONA=riley \
      python -m src.scripts.associative_diagnostic
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.engine.config import EngineConfig, leave_one_out_config  # noqa: E402
from src.engine.core.engine import MemoryEngine  # noqa: E402


def _gold_substrings(ann: dict) -> list[str]:
    return [s.lower() for s in ann.get("gold_context_matches", []) or []]


def _is_gold_hit(memory: dict, gold_substrings: list[str]) -> list[str]:
    """Return the substrings this memory matches (may be empty)."""
    text = (memory.get("raw_content") or "").lower()
    ctx = (memory.get("context") or "").lower()
    return [s for s in gold_substrings if s in text or s in ctx]


def _summarize_topk(
    label: str, results: list[dict], gold_substrings: list[str], k: int = 10,
) -> tuple[set[int], int]:
    """Print top-k details and return (gold-hit ranks, total archive count)."""
    gold_ranks: set[int] = set()
    archive_count = 0
    for i, r in enumerate(results[:k]):
        rank = i + 1
        region = r.get("region", "?")
        paths = r.get("retrieval_paths", [])
        text = (r.get("raw_content") or "")[:60].replace("\n", " ")
        hits = _is_gold_hit(r, gold_substrings)
        gold_marker = "★" if hits else " "
        if hits:
            gold_ranks.add(rank)
        if region == "archive":
            archive_count += 1
        paths_str = ",".join(p[:8] for p in paths)
        print(f"  {gold_marker} {rank:>2}. [{region[:8]:<8}] {paths_str[:25]:<25} {text}")
    return gold_ranks, archive_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probe-indices", default=None,
        help="Comma-separated 1-based probe indices to inspect (default: all)",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--decay", type=float, default=None,
        help="Override sa_decay (default: production value 0.5).",
    )
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL", "postgresql://localhost:5432/meridian_deep")
    annotations_path = "/Users/rileycoleman/meridian/probes/meridian_recall_gold.json"
    persona = os.environ.get("PERSONA", "riley")

    with open(annotations_path) as f:
        annotations = json.load(f)

    if args.probe_indices:
        indices = [int(x.strip()) - 1 for x in args.probe_indices.split(",") if x.strip()]
        annotations = [annotations[i] for i in indices if 0 <= i < len(annotations)]

    # Build both engines once
    overrides = {}
    if args.decay is not None:
        overrides["sa_decay"] = args.decay

    full_cfg = EngineConfig(db_url=db_url, persona=persona, recall_mutates_state=False)
    for k, v in overrides.items():
        setattr(full_cfg, k, v)
    full_engine = MemoryEngine(config=full_cfg)

    noassoc_cfg = leave_one_out_config(
        disabled_path="associative", db_url=db_url, persona=persona,
    )
    for k, v in overrides.items():
        setattr(noassoc_cfg, k, v)
    noassoc_engine = MemoryEngine(config=noassoc_cfg)

    print(f"DB:          {db_url}")
    print(f"Annotations: {annotations_path}")
    print(f"sa_decay:    {full_cfg.sa_decay}")
    print()

    # Aggregate stats
    n_probes = len(annotations)
    full_recall_any5 = 0
    noassoc_recall_any5 = 0
    full_archive_in_top10 = 0
    noassoc_archive_in_top10 = 0

    for idx, ann in enumerate(annotations, 1):
        query = ann["query"]
        gold = _gold_substrings(ann)
        if not gold:
            continue

        print(f"\n{'='*100}")
        print(f"Probe {idx}: {query}")
        print(f"  gold markers: {gold}")
        print(f"{'='*100}")

        full_results = full_engine.recall(query=query, top_k=args.k, reheat=False)
        noassoc_results = noassoc_engine.recall(query=query, top_k=args.k, reheat=False)

        print(f"\n  FULL (with associative):")
        full_ranks, full_arch = _summarize_topk(
            "full", full_results, gold, args.k)
        print(f"\n  NO_ASSOCIATIVE:")
        noassoc_ranks, noassoc_arch = _summarize_topk(
            "no_associative", noassoc_results, gold, args.k)

        # Quick diff
        full_top5 = set([r for r in full_ranks if r <= 5])
        noassoc_top5 = set([r for r in noassoc_ranks if r <= 5])

        if full_recall_any5 := (1 if full_top5 else 0):
            full_recall_any5 = full_recall_any5  # noqa
        full_archive_in_top10 += full_arch
        noassoc_archive_in_top10 += noassoc_arch

        full_any5 = 1 if any(r <= 5 for r in full_ranks) else 0
        noassoc_any5 = 1 if any(r <= 5 for r in noassoc_ranks) else 0
        if full_any5 != noassoc_any5:
            print(f"\n  >>> DIVERGENCE @5: full={full_any5}, no_associative={noassoc_any5} <<<")
            if noassoc_any5 and not full_any5:
                print("      (associative path PUSHED gold OUT of top-5 — hypothesis (a) candidate)")
            elif full_any5 and not noassoc_any5:
                print("      (associative path FOUND gold that other paths missed)")

    print(f"\n{'='*100}")
    print("Summary across probes:")
    print(f"  Total archive memories in top-{args.k}, full config:           {full_archive_in_top10}")
    print(f"  Total archive memories in top-{args.k}, no_associative config: {noassoc_archive_in_top10}")
    print(f"  Difference (archives 'displaced' by removing associative):    {full_archive_in_top10 - noassoc_archive_in_top10}")


if __name__ == "__main__":
    main()
