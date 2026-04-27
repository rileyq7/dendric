"""Seed a deterministic synthetic corpus for reproducible recall@k runs.

Motivation: a reviewer who wants to validate our numbers shouldn't need
LongMemEval (gated download), personal Meridian data (private), or
OpenAI-style persona logs. This script generates a small, fully
self-contained corpus + annotations file that the recall_at_k harness
can consume in its meridian_deep mode.

Design:
  - Fixed random seed → same corpus every run.
  - Memories are short, natural-language, and each carries a distinctive
    unique marker substring that the annotations can key off via
    gold_context_matches.
  - A mix of "gold" memories (which some query should retrieve) and
    "distractor" memories (plausible but irrelevant), so the metric
    actually discriminates.

Output:
  - Memories loaded into the DB pointed to by DATABASE_URL.
  - data/synthetic_gold.json — annotations consumable by recall_at_k
    --corpus meridian_deep.

Usage:
    python -m src.scripts.seed_synthetic_corpus
    python -m src.scripts.recall_at_k --corpus meridian_deep \
        --annotations data/synthetic_gold.json --k 5,10,25
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.engine.config import EngineConfig  # noqa: E402
from src.engine.core.engine import MemoryEngine  # noqa: E402


SEED = 20260424
N_DISTRACTORS = 80

# Gold set: each entry pairs a query with memories that should be retrieved.
# The "marker" string is what recall_at_k matches on via gold_context_matches.
GOLD = [
    {
        "query": "What city did I visit for the neuroscience conference?",
        "memories": [
            "Flew to Lisbon last spring for the FENS neuroscience meeting. Stayed near Alfama.",
        ],
        "markers": ["Lisbon"],
    },
    {
        "query": "What's the name of my daughter's piano teacher?",
        "memories": [
            "Amelia's piano teacher Ms. Harrington moved her lesson slot to Wednesdays.",
        ],
        "markers": ["Harrington"],
    },
    {
        "query": "Which medication am I taking for migraines?",
        "memories": [
            "Neurologist started me on topiramate 25mg for migraine prophylaxis in February.",
        ],
        "markers": ["topiramate"],
    },
    {
        "query": "What was the bug that broke the Friday deploy?",
        "memories": [
            "Friday deploy rollback was triggered by a missing index on users.email_lower.",
        ],
        "markers": ["users.email_lower"],
    },
    {
        "query": "When is my parents' anniversary?",
        "memories": [
            "Mom and Dad's wedding anniversary is September 14th — 40 years this fall.",
        ],
        "markers": ["September 14"],
    },
    {
        "query": "What car did I test drive last weekend?",
        "memories": [
            "Drove a Rivian R1S last Saturday at the Berkeley dealership. Liked the ride height.",
        ],
        "markers": ["Rivian R1S"],
    },
    {
        "query": "Who is my primary care doctor now?",
        "memories": [
            "Switched PCP to Dr. Ostrowski at One Medical Market Street branch.",
        ],
        "markers": ["Ostrowski"],
    },
    {
        "query": "What's the wifi password at the Tahoe cabin?",
        "memories": [
            "Tahoe cabin wifi password is ManzanitaRidge42 — written on the fridge.",
        ],
        "markers": ["ManzanitaRidge42"],
    },
    {
        "query": "Which book did Priya recommend at book club?",
        "memories": [
            "Priya pushed hard for The Overstory by Richard Powers at last month's book club.",
        ],
        "markers": ["Overstory"],
    },
    {
        "query": "What's my climbing gym membership expiration?",
        "memories": [
            "Dogpatch Boulders membership auto-renews on July 3rd every year.",
        ],
        "markers": ["Dogpatch Boulders"],
    },
]

DISTRACTOR_TEMPLATES = [
    "Picked up groceries at {store} on {day}: milk, eggs, sourdough.",
    "Meeting with {name} rescheduled to {day} afternoon.",
    "Watched {show} season finale last night; plot was wobbly.",
    "Ran {distance} miles along the {trail} trail this morning.",
    "Ordered {item} from {vendor}, arrives next week.",
    "Finished chapter {n} of the {book} — taking notes.",
    "Tried a new recipe for {dish}; used too much salt.",
    "Paid {bill} bill online, autopay was broken again.",
    "Called {relative} to check in, quick chat.",
    "Morning journal: feeling {mood} about the {project} deadline.",
]

DISTRACTOR_FILL = {
    "store": ["Rainbow", "Bi-Rite", "Trader Joe's", "Safeway"],
    "day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    "name": ["Marcus", "Jenna", "Oliver", "Saachi", "Devon"],
    "show": ["Severance", "The Bear", "Shogun", "Slow Horses"],
    "distance": ["3", "5", "7", "10"],
    "trail": ["Presidio", "Lands End", "Mount Sutro", "Bay"],
    "item": ["a new desk lamp", "running shoes", "a backpack", "noise-canceling headphones"],
    "vendor": ["REI", "Uniqlo", "Backcountry", "Muji"],
    "n": ["3", "5", "7", "11"],
    "book": ["biography of Lincoln", "Dune", "Project Hail Mary", "Sapiens"],
    "dish": ["shakshuka", "ramen", "thai curry", "risotto"],
    "bill": ["electricity", "internet", "gas", "water"],
    "relative": ["mom", "dad", "my sister", "Uncle Ray"],
    "mood": ["anxious", "optimistic", "flat", "restless"],
    "project": ["Q2 launch", "migration", "hiring", "refactor"],
}


def generate_distractors(rng: random.Random, n: int) -> list[str]:
    out = []
    for _ in range(n):
        tpl = rng.choice(DISTRACTOR_TEMPLATES)
        filled = tpl.format(**{k: rng.choice(v) for k, v in DISTRACTOR_FILL.items()})
        out.append(filled)
    return out


def main():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set.", file=sys.stderr)
        print("  In docker compose: DATABASE_URL is injected automatically.", file=sys.stderr)
        print("  Outside docker:   export DATABASE_URL=postgresql://...", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(SEED)

    print(f"Seeding synthetic corpus into {db_url}")

    # pgvector image ships the extension binaries but doesn't auto-create
    # the extension in each DB. Ensure it exists before the engine tries
    # to register_vector() on connect.
    import psycopg2
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    finally:
        conn.close()

    engine = MemoryEngine(config=EngineConfig(
        db_url=db_url, persona="synthetic", recall_mutates_state=False,
    ))
    engine.store.clear_all()

    total = 0
    for entry in GOLD:
        for mem_text in entry["memories"]:
            engine.remember(content=mem_text, source="synthetic_gold", context="synthetic")
            total += 1

    for text in generate_distractors(rng, N_DISTRACTORS):
        engine.remember(content=text, source="synthetic_distractor", context="synthetic")
        total += 1

    annotations = [
        {"query": entry["query"], "gold_context_matches": entry["markers"]}
        for entry in GOLD
    ]

    # Write alongside the seeder rather than into data/ — in the docker
    # compose stack data/ is mounted read-only (intent: corpus is immutable
    # during an eval run). The annotations are a generated artifact, not
    # raw data, so they belong with the script that produced them.
    out_path = Path(__file__).resolve().parent / "synthetic_gold.json"
    with open(out_path, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"Loaded {total} memories ({len(GOLD)} gold + {N_DISTRACTORS} distractors).")
    print(f"Annotations written to {out_path}")
    print()
    print("Next:")
    print(f"  python -m src.scripts.recall_at_k --corpus meridian_deep \\")
    print(f"      --annotations src/scripts/synthetic_gold.json \\")
    print(f"      --db {db_url} --k 5,10,25")


if __name__ == "__main__":
    main()
