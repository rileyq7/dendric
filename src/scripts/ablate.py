#!/usr/bin/env python3
"""
Ablation Harness — measure the marginal contribution of each Dendric component.

For each ablation:
  1. Spin up bench_api with the right env vars (toggling one component off)
  2. Wait for it to come up
  3. Reset the bench DB
  4. Run the memory-lifecycle-bench full pipeline against it
  5. Capture the report JSON
  6. Stop the server
  7. Compare aggregate + per-category scores against baseline

Output: ablation_results.md with a delta table.

Assumptions:
  - bench_api is at src/server/bench_api.py
  - memory-lifecycle-bench harness is at /Users/rileycoleman/Benchmark/memory-lifecycle-bench
  - OPENAI_API_KEY and ANTHROPIC_API_KEY are set in the env
  - Postgres is running with the dendric_bench database

Usage:
  # First: capture baseline (all features on)
  python -m src.scripts.ablate --baseline

  # Then run all ablations (compares each against the baseline)
  python -m src.scripts.ablate --all

  # Or run a specific ablation
  python -m src.scripts.ablate --ablate enable_contiguity
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests

ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = Path("/Users/rileycoleman/Benchmark/memory-lifecycle-bench")
RESULTS_DIR = ROOT / "ablation_results"

# Component env vars (must match _ABLATION_ENV_FLAGS in src/server/bench_api.py)
ABLATIONS = {
    "enable_temporal_decomposition": "DENDRIC_ENABLE_TEMPORAL_DECOMPOSITION",
    "enable_temporal_rerank": "DENDRIC_ENABLE_TEMPORAL_RERANK",
    "enable_vector_path": "DENDRIC_ENABLE_VECTOR_PATH",
    "enable_keyword_path": "DENDRIC_ENABLE_KEYWORD_PATH",
    "enable_associative_path": "DENDRIC_ENABLE_ASSOCIATIVE_PATH",
    "enable_graph_path": "DENDRIC_ENABLE_GRAPH_PATH",
    "enable_lifecycle_modulation": "DENDRIC_ENABLE_LIFECYCLE_MODULATION",
    "activation_use_gane": "DENDRIC_ACTIVATION_USE_GANE",
    "activation_use_noise": "DENDRIC_ACTIVATION_USE_NOISE",
}


def start_bench_api(disabled_flag: Optional[str] = None) -> subprocess.Popen:
    """Spin up bench_api with one ablation flag turned off (or none for baseline)."""
    env = os.environ.copy()
    env["BENCH_API_PORT"] = "5050"
    env["DATABASE_URL"] = env.get("DATABASE_URL", "postgresql://localhost:5432/dendric_bench")
    if disabled_flag:
        env[disabled_flag] = "0"
    log_path = RESULTS_DIR / f"server_{disabled_flag or 'baseline'}.log"
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.server.bench_api"],
        cwd=str(ROOT),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    return proc


def wait_for_server(timeout_s: int = 30) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get("http://localhost:5050/api/stats", timeout=2)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.5)
    return False


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def reset_bench_db() -> None:
    requests.post("http://localhost:5050/api/reset", timeout=30).raise_for_status()


def run_bench_pipeline(top_k: Optional[int] = None) -> Dict:
    """Run the lifecycle-bench full pipeline. Returns the parsed results JSON."""
    # Snapshot pre-run report files so we can identify the new one.
    reports_dir = BENCH_ROOT / "reports/generated"
    before = set(reports_dir.glob("results_*.json"))

    cmd = [sys.executable, "scripts/run_full.py", "--target"]
    if top_k is not None:
        cmd.extend(["--top-k", str(top_k)])
    proc = subprocess.run(cmd, cwd=str(BENCH_ROOT), capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(
            f"bench run failed: returncode={proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout[-2000:]}\n"
            f"--- stderr ---\n{proc.stderr[-2000:]}"
        )

    after = set(reports_dir.glob("results_*.json"))
    new = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
    target = new[0] if new else (reports_dir / "results.json")
    if not target.exists():
        raise RuntimeError(f"no results JSON produced; checked {target}")
    return json.loads(target.read_text())


def run_one(label: str, disabled_flag: Optional[str], top_k: Optional[int] = None) -> Dict:
    print(f"\n=== Running: {label} ===", flush=True)
    proc = start_bench_api(disabled_flag)
    try:
        if not wait_for_server():
            raise RuntimeError(f"bench_api did not come up for {label}")
        reset_bench_db()
        results = run_bench_pipeline(top_k=top_k)
    finally:
        stop_server(proc)

    out_path = RESULTS_DIR / f"results_{label}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"  → score={results['aggregate']['mean_score']:.4f} "
          f"correct={results['aggregate']['correct_pct']:.1f}% "
          f"saved={out_path.name}", flush=True)
    return results


def write_summary(baseline: Dict, ablations: Dict[str, Dict]) -> Path:
    base_score = baseline["aggregate"]["mean_score"]
    base_cats = {k: v["score"] for k, v in baseline["categories"].items()}

    lines = []
    lines.append(f"# Ablation Results — {datetime.now().isoformat(timespec='seconds')}\n")
    lines.append(f"Baseline mean_score: **{base_score:.4f}** "
                 f"({baseline['aggregate']['correct_pct']:.1f}% correct, "
                 f"{baseline['aggregate']['total_questions']} questions)\n")

    # Aggregate delta table
    lines.append("## Aggregate score delta\n")
    lines.append("| Component disabled | mean_score | Δ vs baseline | correct% | verdict |")
    lines.append("|---|---|---|---|---|")
    rows = []
    for label, results in ablations.items():
        score = results["aggregate"]["mean_score"]
        delta = score - base_score
        verdict = "**helps**" if delta < -0.01 else ("hurts" if delta > 0.01 else "neutral")
        rows.append((delta, label, score, results["aggregate"]["correct_pct"], verdict))
    # Sort by delta ascending (biggest score loss when disabled = most valuable component)
    for delta, label, score, pct, verdict in sorted(rows):
        sign = "+" if delta >= 0 else ""
        lines.append(f"| `{label}` off | {score:.4f} | {sign}{delta:+.4f} | {pct:.1f}% | {verdict} |")

    # Per-category delta table
    lines.append("\n## Per-category delta (positive = ablation helped, negative = ablation hurt)\n")
    cats = list(base_cats.keys())
    header = "| Component disabled | " + " | ".join(cats) + " |"
    sep = "|---|" + "---|" * len(cats)
    lines.append(header)
    lines.append(sep)
    for label, results in ablations.items():
        cat_scores = {k: v["score"] for k, v in results["categories"].items()}
        deltas = []
        for cat in cats:
            d = cat_scores.get(cat, 0.0) - base_cats[cat]
            deltas.append(f"{d:+.3f}" if abs(d) > 0.001 else "—")
        lines.append(f"| `{label}` off | " + " | ".join(deltas) + " |")

    lines.append("\n## Interpretation\n")
    lines.append("- **Negative Δ** when a component is disabled means that component was *helping* — keep it.")
    lines.append("- **Positive Δ** when a component is disabled means that component was *hurting* — strip or fix it.")
    lines.append("- **Near-zero Δ** means the component was net-neutral — strip it for simplicity unless it has another purpose.")

    out_path = RESULTS_DIR / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    out_path.write_text("\n".join(lines))
    return out_path


def main():
    global RESULTS_DIR
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", action="store_true", help="Run baseline only and save it")
    p.add_argument("--all", action="store_true", help="Run baseline + all ablations + summary")
    p.add_argument("--ablate", help="Run a single ablation by config flag name")
    p.add_argument("--top-k", type=int, default=None, help="Override bench top_k (default: bench config)")
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = p.parse_args()

    RESULTS_DIR = Path(args.results_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not (args.baseline or args.all or args.ablate):
        p.print_help()
        return 1

    if args.baseline or args.all:
        baseline = run_one("baseline", None, top_k=args.top_k)
    else:
        # Reuse most recent baseline if present
        baseline_path = RESULTS_DIR / "results_baseline.json"
        if not baseline_path.exists():
            print(f"ERROR: no baseline at {baseline_path}. Run --baseline first.", file=sys.stderr)
            return 1
        baseline = json.loads(baseline_path.read_text())

    ablations = {}
    if args.all:
        for label, env_var in ABLATIONS.items():
            ablations[label] = run_one(label, env_var, top_k=args.top_k)
    elif args.ablate:
        if args.ablate not in ABLATIONS:
            print(f"ERROR: unknown flag {args.ablate}. Choices: {list(ABLATIONS)}", file=sys.stderr)
            return 1
        ablations[args.ablate] = run_one(args.ablate, ABLATIONS[args.ablate], top_k=args.top_k)

    if ablations:
        summary_path = write_summary(baseline, ablations)
        print(f"\nSummary written: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
