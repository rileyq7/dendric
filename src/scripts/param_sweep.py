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
    # sa_decay: 0.3 keeps the binary-seed-gate behavior; 0.5 = current
    # production default; 0.7 makes hop-1 neighbors reach the 0.7 trigger
    # threshold (graduated regime); 0.9 is past the runaway point per the
    # decay diagnostic (one query pulls 80k archive memories).
    "sa_decay": (0.5, [0.3, 0.5, 0.7]),
}


# Targeted 2D grids — for revealing knob interactions that 1D sweeps miss.
# Each entry: ((knob_a, grid_a), (knob_b, grid_b)).
GRIDS_2D: dict[str, tuple] = {
    # Tests the hypothesis that archive_trigger_threshold only becomes
    # graduated (rather than binary) at sa_decay=0.7, so the threshold's
    # apparent inertness in 1D is conditional on decay.
    "threshold_x_decay": (
        ("archive_trigger_threshold", [0.0, 0.3, 0.5, 0.7, 0.9]),
        ("sa_decay", [0.3, 0.5, 0.7]),
    ),
}


def _run_point(
    overrides: dict[str, float], corpus: str, args, k_values: list[int],
    out_dir: Path, skip_ingest: bool, label: str | None = None,
) -> dict:
    """Run one sweep point as a recall_at_k subprocess. `overrides` carries
    one or more knob=value pairs (one for 1D, two for 2D, etc)."""
    k_str = ",".join(str(k) for k in k_values)
    label = label or "_".join(f"{k}_{v}" for k, v in overrides.items())
    raw_out = out_dir / f"sweep_{corpus}_{label}_raw.json"

    cmd = [
        sys.executable, "-m", "src.scripts.recall_at_k",
        "--corpus", corpus,
        "--config", "full",
        "--k", k_str,
        "--output", str(raw_out),
    ]
    for knob, value in overrides.items():
        cmd += ["--override", f"{knob}={value}"]
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

    overrides_str = " ".join(f"{k}={v}" for k, v in overrides.items())
    print(f"  → subprocess: {overrides_str}"
          + (" (skip-ingest)" if skip_ingest else ""), flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"recall_at_k subprocess failed for {overrides_str} "
            f"(exit {proc.returncode}); see log above"
        )
    if not raw_out.exists():
        raise RuntimeError(f"no output for {overrides_str}")
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


def _grid2d_table(
    knob_a: str, grid_a: list[float], knob_b: str, grid_b: list[float],
    results: dict[tuple[float, float], dict], k: int,
) -> str:
    """Build a 2D heatmap-style markdown table at one specific k.
    Rows are knob_a values, columns are knob_b values, cells are recall_any@k.
    Useful for spotting interaction patterns (a row that varies in
    one column block but not another)."""
    lines = [f"## 2D sweep: {knob_a} × {knob_b}  (recall_any@{k})\n"]
    lines.append(f"Rows: `{knob_a}`. Columns: `{knob_b}`.\n")
    header = [f"`{knob_a}` ↓ / `{knob_b}` →"] + [str(b) for b in grid_b]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for a in grid_a:
        row = [str(a)]
        for b in grid_b:
            cell = results.get((a, b))
            if cell is None:
                row.append("—")
                continue
            v = cell["overall"].get(k, {})
            row.append(f"{v.get('any', 0):.1%}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Same table for recall_frac@k — often more informative on small n
    lines.append(f"\n### recall_frac@{k}\n")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for a in grid_a:
        row = [str(a)]
        for b in grid_b:
            cell = results.get((a, b))
            if cell is None:
                row.append("—")
                continue
            v = cell["overall"].get(k, {})
            row.append(f"{v.get('frac', 0):.3f}")
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
    p.add_argument(
        "--grid2d", default=None, choices=list(GRIDS_2D),
        help="Run a targeted 2D grid (interaction sweep) instead of the "
             "default per-knob 1D sweeps. Use when 1D results suggest "
             "two knobs interact.",
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

    if args.grid2d:
        if args.grid2d not in GRIDS_2D:
            print(f"Unknown 2D grid {args.grid2d!r}. Choices: {list(GRIDS_2D)}",
                  file=sys.stderr)
            sys.exit(1)
        (knob_a, grid_a), (knob_b, grid_b) = GRIDS_2D[args.grid2d]
        print(f"\n=== 2D sweep: {knob_a} × {knob_b} ({len(grid_a)} × {len(grid_b)} = "
              f"{len(grid_a) * len(grid_b)} points) ===", flush=True)
        results_2d: dict[tuple[float, float], dict] = {}
        for va in grid_a:
            for vb in grid_b:
                skip_ingest = args.corpus == "longmemeval" and not longmemeval_first_pass
                label = f"{knob_a}_{va}__{knob_b}_{vb}"
                try:
                    summary = _run_point(
                        {knob_a: va, knob_b: vb}, args.corpus, args, k_values,
                        out_dir, skip_ingest, label=label,
                    )
                except Exception as e:
                    print(f"  point {label} failed: {e}", flush=True)
                    continue
                results_2d[(va, vb)] = summary
                longmemeval_first_pass = False
                overall = summary["overall"]
                v5 = overall.get(min(k_values), {})
                print(f"    any@{min(k_values)}={v5.get('any', 0):.1%}  "
                      f"all={v5.get('all', 0):.1%}  frac={v5.get('frac', 0):.3f}",
                      flush=True)
        md_sections.append(_grid2d_table(
            knob_a, grid_a, knob_b, grid_b, results_2d, min(k_values)))
    else:
        for knob in args.knobs:
            default, grid = KNOBS[knob]
            print(f"\n=== Sweeping {knob} (default={default}) over {grid} ===", flush=True)
            per_knob: dict[float, dict] = {}
            for i, val in enumerate(grid):
                skip_ingest = args.corpus == "longmemeval" and not longmemeval_first_pass
                try:
                    summary = _run_point(
                        {knob: val}, args.corpus, args, k_values, out_dir, skip_ingest,
                    )
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
