"""Path ablation harness — substrate-only, recall@k-based.

Runs the same recall@k eval across 6 configs so each retrieval path's
marginal contribution can be attributed:

  plain_rag       — vector only, no lifecycle, no fancy paths (floor)
  no_vector       — leave-one-out: full minus vector path
  no_keyword      — leave-one-out: full minus keyword path
  no_associative  — leave-one-out: full minus associative path
  no_graph        — leave-one-out: full minus graph path
  full            — production (ceiling)

For each config, we report recall_any@k, recall_all@k, recall_frac@k
and a delta vs. the `full` ceiling. Negative delta when a path is
disabled means the path was helping. Positive delta means it was hurting.

Corpora:
  synthetic  — smoke test. All configs should hit ~100% since markers are
               distinctive. If they don't, the harness is broken.
  meridian_deep — real personal corpus + hand-annotated probes.
  longmemeval — external benchmark. Expensive: fresh per-question DB.

Usage:
  # Smoke test against the synthetic corpus (requires seed_synthetic_corpus first)
  python -m src.scripts.path_ablate --corpus synthetic \
      --annotations src/scripts/synthetic_gold.json \
      --db postgresql://postgres:postgres@db:5432/dendric

  # Real run against LongMemEval, 5 questions per type
  python -m src.scripts.path_ablate --corpus longmemeval --per-type 5

  # Against meridian_deep
  python -m src.scripts.path_ablate --corpus meridian_deep \
      --annotations probes/meridian_recall_gold.json \
      --db postgresql://localhost:5432/meridian_deep
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

from src.engine.config import PATH_NAMES  # noqa: E402


# Order matters for report readability — baseline floor first, ablations
# in the same order as PATH_NAMES, ceiling last.
CONFIGS = [
    "plain_rag",
    *[f"no_{p}" for p in PATH_NAMES],
    "full",
]


def _drop_per_q_dbs(db_prefix: str) -> None:
    """Drop all DBs whose name starts with the prefix's DB name.

    Called between configs in the longmemeval path. Keeps disk bounded:
    otherwise each config's clear+re-ingest leaves dead tuples that
    autovacuum can't keep up with, and at 30 questions × 6 configs the
    VM disk fills up. Dropping is cheap vs re-ingesting; subsequent
    configs will recreate the per-question DBs on demand.
    """
    import psycopg2
    from urllib.parse import urlparse
    parsed = urlparse(db_prefix)
    prefix_name = parsed.path.lstrip("/")
    admin_url = db_prefix.rsplit("/", 1)[0] + "/postgres"
    try:
        conn = psycopg2.connect(admin_url)
        conn.autocommit = True
    except Exception as e:
        # If we can't even reach the admin DB, let the real run surface
        # the error. Cleanup is best-effort.
        print(f"  (cleanup skipped: {e})", flush=True)
        return
    dropped = 0
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT datname FROM pg_database WHERE datname LIKE %s",
                (f"{prefix_name}_%",),
            )
            names = [r[0] for r in cur.fetchall()]
            for name in names:
                try:
                    cur.execute(f'DROP DATABASE IF EXISTS "{name}"')
                    dropped += 1
                except Exception as e:
                    # Most likely: another connection is still attached.
                    # Not fatal — just means one DB sticks around this round.
                    print(f"  (could not drop {name}: {e})", flush=True)
    finally:
        conn.close()
    if dropped:
        print(f"  cleanup: dropped {dropped} DBs with prefix '{prefix_name}_'",
              flush=True)


def _run_config(config_name: str, corpus: str, args, k_values: list[int],
                out_dir: Path) -> dict:
    """Run one config as a subprocess, return the summarize() dict.

    Subprocess isolation is important here: the longmemeval data file is
    265 MB and parsing it inflates to ~2 GB in Python. Running 6 configs
    in one process blows through the Docker 3.8 GB memory budget around
    config 2. A fresh process per config frees everything at teardown.
    """
    k_str = ",".join(str(k) for k in k_values)
    raw_out = out_dir / f"path_ablate_{corpus}_{config_name}_raw.json"

    cmd = [
        sys.executable, "-m", "src.scripts.recall_at_k",
        "--corpus", "meridian_deep" if corpus == "synthetic" else corpus,
        "--config", config_name,
        "--k", k_str,
        "--output", str(raw_out),
    ]

    if corpus in ("meridian_deep", "synthetic"):
        cmd += ["--db", args.db, "--annotations", args.annotations]
    else:
        cmd += ["--db-prefix", args.db_prefix]
        if args.data:
            cmd += ["--data", args.data]
        if args.max_questions:
            cmd += ["--max-questions", str(args.max_questions)]
        if args.types:
            cmd += ["--types", *args.types]
        if args.per_type:
            cmd += ["--per-type", str(args.per_type)]
        if args.cycles is not None:
            cmd += ["--cycles", str(args.cycles)]

    print(f"  → subprocess: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"recall_at_k subprocess failed for config={config_name} "
            f"(exit {proc.returncode}); see log above"
        )

    if not raw_out.exists():
        raise RuntimeError(
            f"recall_at_k produced no output file for config={config_name}"
        )
    raw = json.loads(raw_out.read_text())
    summary = raw["summary"]
    # recall_at_k serializes int k as string in JSON; normalize back so
    # the delta table's int-keyed lookups work.
    summary["overall"] = {int(k): v for k, v in summary.get("overall", {}).items()}
    return summary


def _delta_table(results: dict[str, dict], k_values: list[int]) -> str:
    """Build a markdown delta table comparing each config to `full`."""
    full = results["full"]["overall"]

    lines = []
    lines.append(f"# Path ablation — {datetime.now().isoformat(timespec='seconds')}\n")

    # Aggregate scores for each k
    for k in k_values:
        lines.append(f"## recall@{k}\n")
        lines.append("| config | any | all | frac | Δ any vs full | Δ frac vs full | verdict |")
        lines.append("|---|---|---|---|---|---|---|")
        full_any = full.get(k, {}).get("any", 0.0)
        full_frac = full.get(k, {}).get("frac", 0.0)
        for cfg in CONFIGS:
            v = results[cfg]["overall"].get(k)
            if v is None:
                continue
            d_any = v["any"] - full_any
            d_frac = v["frac"] - full_frac
            if cfg == "full":
                verdict = "— (ceiling)"
            elif cfg == "plain_rag":
                verdict = "— (floor)"
            elif d_any < -0.01:
                verdict = f"**path helps** (+{-d_any*100:.1f}pp any)"
            elif d_any > 0.01:
                verdict = f"path hurts ({d_any*100:+.1f}pp any)"
            else:
                verdict = "neutral"
            lines.append(
                f"| `{cfg}` | {v['any']:.1%} | {v['all']:.1%} | {v['frac']:.3f} | "
                f"{d_any*100:+.1f}pp | {d_frac:+.3f} | {verdict} |"
            )
        lines.append("")

    lines.append("## Reading the table\n")
    lines.append(
        "- **`no_X`** rows show recall when path X is disabled. A large "
        "**negative Δ** means that path was contributing substantially — "
        "disabling it hurt. A positive Δ means the path was net-harmful "
        "in this config (investigate, don't silently keep it)."
    )
    lines.append(
        "- **`plain_rag`** is the vector-only floor. The gap between "
        "`plain_rag` and `full` is the *total* lift Dendric's architecture "
        "provides over standard RAG."
    )
    lines.append(
        "- **`full`** is the ceiling. All deltas are reported against it."
    )
    return "\n".join(lines)


def _print_delta_table(md: str) -> None:
    # Monospace is fine; the markdown renders as readable plaintext too.
    for line in md.splitlines():
        print(line)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", choices=["synthetic", "meridian_deep", "longmemeval"], required=True)
    p.add_argument("--k", default="5,10,25", help="Comma-separated k values")
    p.add_argument("--output-dir", default="ablation_results",
                   help="Directory to drop the summary markdown + raw per-config JSON")
    # meridian_deep / synthetic
    p.add_argument("--db", default=None,
                   help="Postgres URL for meridian_deep/synthetic (e.g. "
                        "postgresql://postgres:postgres@db:5432/dendric)")
    p.add_argument("--annotations", default=None,
                   help="Gold annotations JSON for meridian_deep/synthetic")
    # longmemeval
    p.add_argument("--data", default=None)
    p.add_argument("--db-prefix", default="postgresql://localhost:5432/recall_at_k")
    p.add_argument("--max-questions", type=int, default=None)
    p.add_argument("--types", nargs="+", default=None)
    p.add_argument("--per-type", type=int, default=None)
    p.add_argument("--cycles", type=int, default=1)
    args = p.parse_args()

    if args.corpus in ("meridian_deep", "synthetic"):
        if not args.annotations or not args.db:
            print("--annotations and --db are required for meridian_deep/synthetic",
                  file=sys.stderr)
            sys.exit(1)

    k_values = [int(x.strip()) for x in args.k.split(",") if x.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for cfg in CONFIGS:
        # For longmemeval, each config creates 30 per-question DBs and then
        # clear_all()s between configs. Postgres doesn't physically reclaim
        # deleted rows until VACUUM, so repeated clear+ingest bloats each DB.
        # Dropping the per-prefix DBs between configs keeps disk bounded.
        if args.corpus == "longmemeval":
            _drop_per_q_dbs(args.db_prefix)
        print(f"\n=== Running config: {cfg} ===", flush=True)
        summary = _run_config(cfg, args.corpus, args, k_values, out_dir)
        results[cfg] = summary

        # Persist per-config summary so partial runs aren't wasted
        per_cfg_path = out_dir / f"path_ablate_{args.corpus}_{cfg}.json"
        per_cfg_path.write_text(json.dumps(summary, indent=2, default=str))

        # Quick live feedback
        overall = summary["overall"]
        if k_values[0] in overall:
            v = overall[k_values[0]]
            print(f"  → any@{k_values[0]}={v['any']:.1%} "
                  f"all@{k_values[0]}={v['all']:.1%} "
                  f"frac@{k_values[0]}={v['frac']:.3f}")

    md = _delta_table(results, k_values)
    summary_path = out_dir / f"path_ablate_{args.corpus}_{datetime.now():%Y%m%d_%H%M%S}.md"
    summary_path.write_text(md)

    print()
    _print_delta_table(md)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
