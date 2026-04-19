# Dendric — Episodic Memory for AI

> Biologically-grounded memory consolidation. Lifecycle dynamics — temperature, signals, decay, reheat, archive — are load-bearing at every retrieval, not just stored as metadata.

## What it is

A memory system for AI agents (or persona-driven chat) where every memory is a living object. Memories are ingested with three signal values (DA/NE/GABA), live in the hippocampal hot buffer at full fidelity, and consolidate into the cortical store through a sleep cycle that decays temperature, erodes tokens differentially, and migrates cold memories into a dormant archive that's reachable only via associative trigger.

Retrieval fuses four parallel paths via RRF, then modulates the fused score by lifecycle state — a hot, goal-relevant memory outranks a semantically-closer cold one, but the bias is bounded so an overwhelming match still surfaces. The four paths are real and complementary:

- **Vector** — dense semantic similarity (`text-embedding-3-small`, 1536-dim, pgvector)
- **Keyword** — Postgres FTS + ILIKE fallback for exact-token recall
- **Graph** — entity-fan match on co-occurrence edges
- **Associative (spreading activation)** — seeds at query entities, propagates through co-occurrence edges with decay-over-hops, returns memories linked to the most-activated nodes. Falls back to seeding at the persona node when no query entity resolves.

A déjà-vu trigger extends the associative path into the archive: when entity activation crosses a threshold, archived memories linked to that entity surface. Otherwise archive stays dormant.

## Architecture

```
INGEST                                 CONSOLIDATE (sleep cycle)
─────────                              ─────────────────────────
text → embed                           ACT-R power-law decay
     → DA/NE/GABA                      DA-boosted memories resist decay
     → novelty gate                    GABA-suppressed accelerate down
     → entities + co-occurrence        Token erosion (stopwords shed first)
     → persona link                    Region migration: hot → warm → cold
     → store (hippocampus, T=1.0)      Archived memories preserved but dormant
                                       Probabilistic pruning of dead memories

RECALL
──────
query → 4 paths in parallel:
        ├─ vector (cosine, region != archive)
        ├─ keyword (FTS, region != archive)
        ├─ graph (entity-fan walk, region != archive)
        └─ associative (spreading activation, can pull archive on déjà-vu)
     → RRF fusion
     → lifecycle modulation: final_score = RRF × bounded f(temp, DA, NE)
     → top_k
```

Persona injection: every ingested memory is linked to the persona entity (the implicit owner of the memory stream). First-person episodes ("I did X") are linked to the persona node so queries by name ("Has X done Y?") can reach them. Persona acts as a fallback seed for spreading activation, with softer initial activation and fan-out normalization so it doesn't swamp specific-entity matches.

Compression is purely deterministic — token-level erosion + entity-edge extraction. No LLM in the consolidation path.

## Quick Start

```bash
# Install
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...      # required (real embeddings)
export ANTHROPIC_API_KEY=sk-...   # required if using bench harness

# Initialize Postgres
createdb dendric
psql dendric -c "CREATE EXTENSION IF NOT EXISTS vector;"
python -c "import psycopg2; from src.engine.storage.migrations import run_migrations; run_migrations(psycopg2.connect('postgresql://localhost:5432/dendric'))"

# Use as a library
python -c "
from src.engine.core.engine import MemoryEngine
from src.engine.config import EngineConfig
eng = MemoryEngine(config=EngineConfig(db_url='postgresql://localhost:5432/dendric', persona='alex'))
eng.remember('Took Mango for a walk this morning.', context='session_1')
results = eng.recall('What did I do with Mango?', top_k=5)
"

# Or run as HTTP server (memory-lifecycle-bench compatible)
python -m src.server.bench_api  # listens on :5050
```

## Validation

Three design-invariant tests covering the load-bearing mechanisms — run independently of any benchmark:

```bash
DATABASE_URL=postgresql://localhost:5432/dendric_stability \
  python -m src.scripts.per_signal_recall_check
```

Checks: lifecycle modulation tilts ranking by signal state, spreading activation reaches co-occurring entities, déjà-vu trigger surfaces archived memories only when relevant.

Stability check (deterministic recall under benchmark mode):

```bash
DATABASE_URL=postgresql://localhost:5432/dendric_stability \
  python -m src.scripts.stability_check
```

Benchmark against memory-lifecycle-bench harness:

```bash
python -m src.scripts.ablate --baseline --top-k 5
```

## Configuration

All tunable parameters in `src/engine/config.py`. Key knobs:

- `persona` — implicit owner of the memory stream
- `archive_trigger_threshold` (0.7) — entity activation level required for déjà-vu
- `mod_temp_lift`, `mod_da_lift`, `mod_ne_penalty` — lifecycle modulation strength (bounded to `[mod_min, mod_max]`)
- `recall_mutates_state` — when False, recall is read-only (required for benchmark determinism)
- `enable_*` flags — ablate individual paths and equation stages

## Project Structure

```
src/
├── engine/
│   ├── core/            # Engine, signals, activation, compression, erosion
│   ├── retrieval/       # vector, keyword, associative (spreading), graph, fusion, temporal
│   ├── storage/         # Postgres + entity graph
│   └── embeddings/      # OpenAI batch API wrapper
├── scripts/             # bench harnesses, validation, lifecycle runner
└── server/              # bench_api (HTTP adapter)
```

## License

MIT
