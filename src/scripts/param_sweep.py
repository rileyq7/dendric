"""Parameter sensitivity sweep — substrate-only, recall@k-based.

For each (knob, value) point in a 1D grid, runs the same recall@k eval
with `--config full` plus `--override knob=value`. Builds a per-knob
delta table comparing every value to the default and to the
default-config (current production) baseline.

Knobs swept (1D, independently):
  archive_rrf_boost          [1.0, 1.4, 1.8, 2.2, 2.6]   default 1.8
  archive_trigger_threshold  [0.0, 0.3, 0.5, 0.7, 0.9]   default 0.7
  mod_temp_lift              [0.0, 0.2, 0.4, 0.6, 0.8]   default 0.4

Why 1D first: each knob is conceptually independent. boost affects RRF
rank, threshold gates déjà-vu firing, temp_lift adjusts modulation
strength. If a 1D sweep reveals interaction (e.g. boost effect varies
strongly with threshold), follow up with a targeted 2D grid.

Usage:
  # meridian_deep (fast — single DB, retrieval-only sweep)
  python -m src.scripts.param_sweep --corpus meridian_deep \
      --db postgresql://localhost:5432/meridian_deep \
      --annotations /Users/rileycoleman/meridian/probes/meridian_recall_gold.json \
      --output-dir param_sweep_results/meridian_deep

  # longmemeval (uses --skip-ingest after the first point per knob to
  # re-use per-question DBs across sweep values)
  python -m src.scripts.param_sweep --corpus longmemeval --per-type 5 \
      --db-prefix postgresql://postgres:postgres@db:5432/recall_per5 \
      --output-dir param_sweep_results/longmemeval
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# Knob → (default value, list of grid points). The default value is
# included in the grid so every sweep produces a row for "current
# production setting" as a sanity-check anchor.
KNOBS: dict[str, tuple[float, list[float]]] = {
    "archive_rrf_boost": (1.8, [1.0, 1.4, 1.8, 2.2, 2.6]),
    "archive_trigger_threshold": (0.7, [0.0, 0.3, 0.5, 0.7, 0.9]),
    "mod_temp_lift": (0.4, [0.0, 0.2, 0.4, 0.6, 0.8]),
}


def _run_point(
    knob: str, value: float, corpus: str, args, k_values: list[int],
    out_dir: Path, skip_ingest: bool,
) -> dict:
    """Run one (knob, value) point as a recall_at_k subprocess."""
    k_str = ",".join(str(k) for k in k_values)
    raw_out = out_dir / f"sweep_{corpus}_{knob}_{value}_raw.json"

    cmd = [
        sys.executable, "-m", "src.scripts.recall_at_k",
        "--corpus", corpus,
        "--config", "full",
        "--k", k_str,
        "--output", str(raw_out),
        "--override", f"{knob}={value}",
    ]
    if corpus == "meridian_deep":
        cmd += ["--db", args.db, "--annotations", args.annotations]
    else:
        cmd += ["--db-prefix", args.db_prefix]
        if args.data:
            cmd += ["--data", args.data]
        if args.per_type:
            cmd += ["--per-type", str(args.per_type)]
        if args.cycles is not None:
            cmd += ["--cycles", str(args.cycles)]
        if skip_ingest:
            cmd += ["--skip-ingest"]

    print(f"  → subprocess: --override {knob}={value}"
          + (" (skip-ingest)" if skip_ingest else ""), flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"recall_at_k subprocess failed for {knob}={value} "
            f"(exit {proc.returncode}); see log above"
        )
    if not raw_out.exists():
        raise RuntimeError(f"no output for {knob}={value}")
    raw = json.loads(raw_out.read_text())
    summary = raw["summary"]
    summary["overall"] = {int(k): v for k, v in summary.get("overall", {}).items()}
    return summary


def _knob_table(
    knob: str, default: float, results: dict[float, dict], k_values: list[int],
) -> str:
    """Build a per-knob markdown table — one row per swept value, all k columns."""
    lines = [f"## {knob}\n"]
    lines.append(f"Default: **{default}**. Δ rows compare against the default-value row.\n")
    header = ["value"]
    for k in k_values:
        header += [f"any@{k}", f"all@{k}", f"frac@{k}"]
    header.append("Δ any@5 vs default")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")

    default_overall = results[default]["overall"]
    default_any5 = default_overall.get(min(k_values), {}).get("any", 0.0)

    for val in sorted(results.keys()):
        overall = results[val]["overall"]
        row = [f"{val}{'  (def)' if val == default else ''}"]
        for k in k_values:
            v = overall.get(k, {})
            row += [
                f"{v.get('any', 0):.1%}",
                f"{v.get('all', 0):.1%}",
                f"{v.get('frac', 0):.3f}",
            ]
        d_any = overall.get(min(k_values), {}).get("any", 0.0) - default_any5
        row.append(f"{d_any*100:+.1f}pp" if val != default else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", choices=["meridian_deep", "longmemeval"], required=True)
    p.add_argument("--knobs", nargs="+", default=list(KNOBS.keys()),
                   help="Subset of knobs to sweep (default: all)")
    p.add_argument("--k", default="5,10,25", help="Comma-separated k values")
    p.add_argument("--output-dir", default="param_sweep_results",
                   help="Where to drop per-point JSONs and the summary markdown")
    # meridian_deep
    p.add_argument("--db", default=None)
    p.add_argument("--annotations", default=None)
    # longmemeval
    p.add_argument("--data", default=None)
    p.add_argument("--db-prefix", default="postgresql://localhost:5432/recall_at_k")
    p.add_argument("--per-type", type=int, default=None)
    p.add_argument("--cycles", type=int, default=1)
    p.add_argument(
        "--reuse-ingest", action="store_true",
        help="Skip ingest from the first point onward (longmemeval only). "
             "Use when per-question DBs at --db-prefix are already populated "
             "from a prior run with the same questions.",
    )
    args = p.parse_args()

    if args.corpus == "meridian_deep":
        if not args.db or not args.annotations:
            print("--db and --annotations required for meridian_deep", file=sys.stderr)
            sys.exit(1)
    else:
        if not args.per_type:
            print("--per-type required for longmemeval", file=sys.stderr)
            sys.exit(1)

    unknown = [k for k in args.knobs if k not in KNOBS]
    if unknown:
        print(f"Unknown knob(s): {unknown}. Choices: {list(KNOBS)}", file=sys.stderr)
        sys.exit(1)

    k_values = [int(x.strip()) for x in args.k.split(",") if x.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_sections: list[str] = []
    md_sections.append(
        f"# Parameter sensitivity sweep — {args.corpus} — "
        f"{datetime.now().isoformat(timespec='seconds')}\n"
    )

    # For longmemeval, --reuse-ingest skips ingest on every point (DBs
    # populated by an earlier run). Otherwise the first point of the
    # first knob does full ingest, and subsequent points re-use those
    # DBs. Either way: ingest happens at most once across the whole
    # sweep, since retrieval-only knobs can't change what's stored.
    longmemeval_first_pass = (args.corpus == "longmemeval") and not args.reuse_ingest

    for knob in args.knobs:
        default, grid = KNOBS[knob]
        print(f"\n=== Sweeping {knob} (default={default}) over {grid} ===", flush=True)
        per_knob: dict[float, dict] = {}
        for i, val in enumerate(grid):
            skip_ingest = args.corpus == "longmemeval" and not longmemeval_first_pass
            try:
                summary = _run_point(knob, val, args.corpus, args, k_values, out_dir, skip_ingest)
            except Exception as e:
                print(f"  point {knob}={val} failed: {e}", flush=True)
                continue
            per_knob[val] = summary
            longmemeval_first_pass = False

            overall = summary["overall"]
            v5 = overall.get(min(k_values), {})
            print(f"    any@{min(k_values)}={v5.get('any', 0):.1%}  "
                  f"all={v5.get('all', 0):.1%}  frac={v5.get('frac', 0):.3f}",
                  flush=True)

        if default not in per_knob:
            print(f"  WARNING: default value {default} not in completed runs for {knob}; "
                  f"deltas will use the lowest completed value as anchor", flush=True)
            anchor = min(per_knob.keys()) if per_knob else default
        else:
            anchor = default

        md_sections.append(_knob_table(knob, anchor, per_knob, k_values))

    summary_path = out_dir / f"param_sweep_{args.corpus}_{datetime.now():%Y%m%d_%H%M%S}.md"
    summary_path.write_text("\n".join(md_sections))
    print()
    print("\n".join(md_sections))
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
