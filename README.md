# Dendric

**A biologically-grounded episodic memory substrate for AI agents.**

Memories are living objects with a temperature, three neuromodulator signals
(DA / NE / GABA), and a lifecycle — hot in the hippocampus, warm in the
neocortex, dormant in archive. Retrieval fuses four parallel paths and tilts
the ranking by lifecycle state. Compression is deterministic — no LLM in the
sleep cycle. Forgetting is differential — important tokens survive longer
than filler.

```
                    ╭────────────────────────────────────╮
                    │ hippocampus → neocortex → archive  │
                    ╰────────────────────────────────────╯
                          T=1.0          T=0.5         T<0.10
                          full raw       summary       dormant
                          fidelity       + nuggets     (déjà-vu only)
```

For the technically rigorous deep-dive, see
[`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md).
For empirical findings (ablations, sweeps, baselines), see
[`docs/RIGOR_FINDINGS.md`](docs/RIGOR_FINDINGS.md).

---

## Why this exists

Most "memory for LLMs" projects are a vector store, a chunker, and a prompt.
They store every chunk forever and rank by cosine. That collapses three
distinct functions — encoding, consolidation, recall — into a single read/
write path, which is why they degrade on aged corpora and choke on multi-
session questions.

Dendric makes the bet that **the lifecycle of a memory is the substrate**.
The same memory, queried today and a year from now, ranks differently
because it has *aged*: temperature decayed, novelty faded, GABA accumulated
through disuse, EWC importance grew if it kept getting recalled, raw text
got eroded down to its load-bearing tokens, and at some point the cold
remainder migrated into a dormant archive that's only reachable when an
associative trigger drags it back into awareness.

This README is the concise version. The
[Technical Report](docs/TECHNICAL_REPORT.md) explains every formula with code.

---

## How it works (sketch)

```
INGEST                                      CONSOLIDATE (sleep cycle)
─────────                                   ─────────────────────────
text → embed (1536-dim, OpenAI)             ACT-R power-law decay
     → DA (relevance)                       7-stage GANE activation
       NE (novelty / inverted-U)            DA-protected, NE-modulated,
       GABA (redundancy + staleness)         GABA-suppressed
     → novelty gate                         Token erosion (stopwords first)
     → entity extraction (regex, no LLM)    Region migration: hipp → neo → archive
     → persona link                         MESU probabilistic pruning
     → store @ T=1.0                        Compression cascade fires by temp:
                                              0.70 → structured summary
                                              0.40 → knowledge nuggets
                                              0.20 → entity edges
                                              0.10 → archive raw

RECALL
──────
query → 4 parallel paths:
        ├─ vector       (pgvector cosine, region != archive)
        ├─ keyword      (FTS + ILIKE fallback, region != archive)
        ├─ graph        (entity-fan walk, region != archive)
        └─ associative  (BFS spreading activation, can pull archive on déjà-vu)
     → temporal decomposition (extra retrievals for time-anchored Qs)
     → RRF fusion + session-diversity penalty + recency tiebreaker
     → lifecycle modulation: fused × bounded f(temp, DA, NE)
     → top_k
```

**The two pieces of novel mechanism worth the attention:**

1. **The Dendric activation equation (GANE).** A 7-stage formula composing
   ACT-R power-law decay, additive DA importance, an asymmetric inverted-U
   over NE, GANE feedback for winner-take-more / loser-take-less competition,
   hybrid divisive+subtractive GABA, and arousal-modulated noise. Replaces
   the linear "decay − boost" formula that comparable systems use.

2. **Token erosion.** Per-token importance scoring at ingest, differential
   per-cycle decay (high-importance tokens shed slowly, stopwords shed fast),
   and a `read_at_temperature` reconstructor that surfaces a graceful low-
   fidelity version of the memory based on its current temperature. *No LLM
   in the consolidation path.*

Both are explained in detail with full code in [the Technical
Report](docs/TECHNICAL_REPORT.md#6-the-7-stage-dendric-activation-equation-gane).

---

## The activation equation (the thing GANE points at)

```
Stage 1   ACT-R base:          B = ln( Σ t_i^(-d) )                 d=0.3
Stage 2   DA additive boost:   A₁ = B − τ + β·DA + s                τ=2.0, β=2.2
Stage 3   NE inverted-U:       I_U = w·exp(−(NE−μ)² / 2σ²)          μ=0.65, σ=0.18
Stage 4   Pre-GANE salience:   a_sal = A₁ + I_U
Stage 5   GANE feedback:       (1 + α·DA) · η·NE · tanh(λ·a_sal)    α=0.2, η=0.05
Stage 6   GABA hybrid:         a = (a_sal + GANE) / (1 + δ·GABA) − γ·GABA
Stage 7   Logistic noise:      a += s₀(1 + ρ·NE) · ln(u/(1−u))
Final     Sigmoid → temp ∈ [0, 1]
```

Implementation (`src/engine/core/activation.py`):

```python
def compute_temperature(accesses_days_ago, da_relevance, ne_novelty,
                        gaba_inhibition, spreading_activation=0.0,
                        noise=True, use_gane=True, ...):
    # Stage 1 — ACT-R power-law decay
    times = [max(1/1440, t) for t in accesses_days_ago] or [1.0]
    B = math.log(sum(t ** (-d) for t in times))

    # Stage 2 — DA enters ADDITIVELY (multiplicative on negative B amplifies decay)
    A1 = B - tau + beta * da_relevance + spreading_activation

    # Stage 3 — Asymmetric inverted-U with peak at μ_NE = 0.65
    iu = w_ne * math.exp(-((ne_novelty - mu_ne) ** 2) / (2 * sigma_ne ** 2))

    # Stage 4 — Pre-GANE salience
    a_sal = A1 + iu

    # Stage 5 — GANE: winner-take-more / loser-take-less; tanh shapes competition.
    # DA gates how much GANE matters at all.
    if use_gane:
        g = 1 + alpha * da_relevance
        gane = g * eta * ne_novelty * math.tanh(lam * a_sal)
    else:
        gane = 0.0

    # Stage 6 — GABA hybrid (divisive + subtractive)
    a_pre = a_sal + gane
    a = a_pre / (1 + delta_gaba * gaba_inhibition) - gamma_gaba * gaba_inhibition

    # Stage 7 — Arousal-modulated logistic noise
    if noise:
        s = s0 * (1 + rho * ne_novelty)
        u = max(1e-10, min(1 - 1e-10, random.random()))
        a += s * math.log(u / (1 - u))

    a_clipped = max(-20, min(20, a))
    return 1.0 / (1.0 + math.exp(-a_clipped))
```

The validation suite asserts the critical NE inverted-U property:
`temp(NE=1.0) < temp(NE=0.65)`. Without it the curve has been silently
inverted and the substrate is broken.

---

## Token erosion (the other novel piece)

Each memory carries per-token importance scores from ingest:

```python
# src/engine/core/erosion.py
def score_token_importance(tokens, known_entities=None):
    """Signals (max positive + sum negatives, clamped to [0.05, 0.95]):
       - Entity:    capitalized mid-sentence OR in entity list  → 0.95
       - Numeric:   contains digits OR is a unit (kg, mg, %)     → 0.90 / 0.85
       - Rarity:    not stopword, length > 3                     → 0.5–0.8
       - Proximity: within ±2 of an entity or number             → +0.4
       - Stopword:  in stopword set                              → -0.3
    """
```

Each consolidation cycle, weights decay differentially by importance:

```python
def erode_tokens(token_weights, base_decay=0.15):
    """High-importance tokens decay slowly; low-importance tokens decay fast."""
    for tw in token_weights:
        importance = tw.get('importance', 0.5)
        decay_rate = base_decay * (1.0 - importance)
        tw['weight'] = tw.get('weight', 1.0) * (1.0 - decay_rate)
    return token_weights
```

A token at importance=0.95 decays at 0.75% per cycle; a stopword at
importance=0.05 decays at 14.25%. After 30 cycles the entity is at 0.79;
the stopword is at 0.0094.

Reconstruction at any temperature:

```python
def read_at_temperature(token_weights, temperature):
    """temp 1.0 → all tokens; temp 0.5 → only weight > 0.5; temp 0.1 → only > 0.9."""
    if temperature >= 0.95:
        return ' '.join(tw['token'] for tw in token_weights)
    threshold = 1.0 - temperature
    visible = [tw['token'] for tw in token_weights
               if tw.get('weight', 1.0) > threshold]
    return ' '.join(visible) if visible else "(eroded to empty)"
```

A memory that started as

> *"Walked Mango along the canal at 7 in the morning while it was raining."*

might after months of disuse read as

> *"Mango canal 7 morning"*

without a single LLM call. **Token erosion *is* the summary.**

---

## Déjà-vu — the archive trigger

Archived memories (`region = 'archive'`, T ≤ 0.10) are filtered out of
vector / keyword / graph paths via `WHERE region != 'archive'`. The *only*
way they reach the candidate set is through the spreading-activation path's
archive trigger:

```python
# src/engine/retrieval/associative.py
if archive_trigger_threshold is not None:
    archive_triggered_entities = [
        eid for eid, act in activated if act >= archive_trigger_threshold
    ]
    for eid in archive_triggered_entities:
        for mid in eg.get_memories_for_entity(eid):
            archive_triggered_ids.append(str(mid))
```

Two compensations let archive hits actually surface in top_k:

```python
# fusion.py — archive memories are single-path by construction; without the
# boost they always lose RRF to multi-path active-region results.
if "associative_archive" in path_tags and archive_rrf_boost != 1.0:
    rrf_score *= archive_rrf_boost            # default 1.8

# fusion.py — déjà-vu firing is itself a strong signal; don't double-penalize
# the archive's low temperature in lifecycle modulation.
if is_archive and archive_modulation_override != 1.0:
    mod = archive_modulation_override         # default 1.3
```

The design-invariant test (`src/scripts/per_signal_recall_check.py`):
querying *"What did I do with Mango?"* must surface an archived Mango
memory; querying *"Tell me about pancakes."* must leave it dormant.

---

## Empirical results

(Full log: [`docs/RIGOR_FINDINGS.md`](docs/RIGOR_FINDINGS.md).)

### Aged personal corpus (`meridian_deep`, n=15, after fan-out fix)

| k | metric | full | plain_rag (vector-only) | Δ |
|---|---|---|---|---|
| 5  | recall_any  | **86.7%** | 80.0% | **+6.7pp** |
| 5  | recall_frac | **0.640** | 0.560 | **+0.080** |
| 10 | recall_any  | **93.3%** | 86.7% | **+6.7pp** |
| 10 | recall_all  | **46.7%** | 40.0% | **+6.7pp** |
| 10 | recall_frac | **0.720** | 0.627 | **+0.093** |
| 25 | recall_any  | 93.3% | 93.3% | 0.0pp |

Convergence at k=25 means the architecture is doing **ranking** (not
candidate discovery). Lifecycle modulation lifts good candidates higher
in the list.

### LongMemEval (n=30 stratified, fresh per-question DB)

| k | recall_any | recall_all | recall_frac |
|---|---|---|---|
| 5  | 83.3% | 53.3% | 0.663 |
| 10 | 90.0% | 63.3% | 0.742 |
| 25 | 93.3% | 70.0% | 0.814 |

100% recall_all on 3/6 question types (single-session-user, -assistant,
-preference). Multi-session and temporal-reasoning are the honest
weakness — the substrate finds the right session (60-80% any) but
"every scattered mention" requires aggregation that lives at the answer
layer.

### Determinism

Under `recall_mutates_state=False`, retrieval is bit-identical across
repeated runs (SD = 0pp at k ∈ {5, 10, 25}, 5 repeats × 15 probes).
This is what makes every other number above a stable point estimate
rather than a noisy draw.

---

## Quick start

```bash
# Install
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...           # required (real embeddings)

# Initialize Postgres + pgvector
createdb dendric
psql dendric -c "CREATE EXTENSION IF NOT EXISTS vector;"
python -c "import psycopg2; from src.engine.storage.migrations import run_migrations; \
           run_migrations(psycopg2.connect('postgresql://localhost:5432/dendric'))"

# Use as a library
python <<'PY'
from src.engine.core.engine import MemoryEngine
from src.engine.config import EngineConfig

eng = MemoryEngine(config=EngineConfig(
    db_url="postgresql://localhost:5432/dendric",
    persona="alex",
))

eng.remember("Took Mango for a walk this morning.", context="session_1")
eng.remember("Mango spotted a fox in the garden.",  context="session_2")
results = eng.recall("What did I do with Mango?", top_k=5)
for r in results:
    print(f"  T={r['temperature']:.2f}  {r['raw_content']}")
PY

# Run as HTTP server (memory-lifecycle-bench compatible)
python -m src.server.bench_api          # listens on :5050
```

### Reproducible benchmark

```bash
docker compose build
docker compose run --rm dendric python -m src.scripts.seed_synthetic_corpus
docker compose run --rm dendric python -m src.scripts.recall_at_k \
    --corpus meridian_deep \
    --annotations src/scripts/synthetic_gold.json \
    --db postgresql://postgres:postgres@db:5432/dendric \
    --k 5,10,25
```

Synthetic should hit `recall_any@5 ≈ 1.0` (markers are distinctive — this
is *harness validation*, not benchmarking). For real corpora see
[`REPRODUCE.md`](REPRODUCE.md).

### Validation invariants

```bash
DATABASE_URL=postgresql://localhost:5432/dendric_stability \
    python -m src.scripts.per_signal_recall_check
```

Three design-level invariants — independent of any benchmark:

1. **Lifecycle modulation tilts ranking by signal state.**
2. **Spreading activation reaches co-occurring entities.**
3. **Déjà-vu trigger reaches archived memories — and only when relevant.**

If any fails, the substrate isn't doing what it claims regardless of
benchmark performance.

---

## Configuration

Every tunable lives in `src/engine/config.py`. The knobs that matter most:

| Knob | Default | What it does |
|---|---|---|
| `persona` | `""` | Implicit owner of the memory stream; auto-linked to every memory with reduced salience (0.2) |
| `archive_trigger_threshold` | `0.7` | Entity activation needed for déjà-vu to pull from archive |
| `archive_rrf_boost` | `1.8` | Multiplier on archive RRF score (compensates for single-path disadvantage) |
| `archive_modulation_override` | `1.3` | Bypasses cold-temperature penalty on déjà-vu firings |
| `mod_temp_lift` / `mod_da_lift` / `mod_ne_penalty` | `0.4 / 0.3 / 0.3` | Lifecycle-modulation strength |
| `mod_min` / `mod_max` | `0.7 / 1.6` | Bounded modulation — strong RRF can still surface cold |
| `sa_decay` / `sa_max_hops` | `0.5 / 2` | Spreading-activation per-hop decay and BFS depth |
| `sa_fanout_norm_exponent` | `0.5` | Hub-entity normalization (0.5=sqrt, 1.0=strict 1/N) |
| `recall_mutates_state` | env / `True` | When `False`, recall is read-only — required for benchmark determinism |
| `enable_<path>` flags | `True` | Per-path ablation switches |
| `enable_lifecycle_modulation` | `True` | The bias that turns lifecycle from stored model into retrieval signal |
| `activation_use_gane` / `activation_use_noise` | `True / True` | GANE feedback and arousal noise stages of the activation equation |

Two presets ship for ablation:

```python
plain_rag_config(db_url, persona)               # vector-only baseline
leave_one_out_config(disabled_path, db_url, ...) # full minus one path
```

---

## Project structure

```
src/engine/
├── core/
│   ├── engine.py              # Public API: remember, recall, consolidate
│   ├── activation.py          # 7-stage Dendric activation equation (GANE)
│   ├── signals_enhanced.py    # DA / NE / GABA computation
│   ├── erosion.py             # Token-level importance + differential decay
│   ├── compression.py         # Deterministic cascade: summary, nuggets, edges
│   ├── pruning.py             # MESU uncertainty-weighted pruning
│   ├── protection.py          # EWC retrieval-importance accumulation
│   ├── entity_extraction.py   # Regex/heuristic entity extraction
│   └── memory.py              # Memory dataclass + temperature/band helpers
├── retrieval/
│   ├── vector.py              # Path 1: pgvector cosine
│   ├── keyword.py             # Path 2: tsvector FTS + ILIKE
│   ├── associative.py         # Path 3: BFS spreading activation + déjà-vu
│   ├── graph.py               # Path 4: entity-fan walk
│   ├── fusion.py              # RRF + lifecycle modulation
│   ├── temporal.py            # Temporal-query detection + re-rank
│   └── temporal_decomposer.py # Decomposes time-anchored queries
├── storage/
│   ├── postgres.py            # Postgres + pgvector backend
│   ├── entity_graph.py        # Entities, links, weighted edges
│   └── migrations.py          # Idempotent schema migrations
├── embeddings/embed.py        # OpenAI batch-API wrapper
└── config.py                  # All tunables; presets for ablation

src/scripts/                   # Bench harnesses, validation, sweeps
src/server/bench_api.py        # HTTP adapter for memory-lifecycle-bench

docs/
├── TECHNICAL_REPORT.md        # The deep dive — every formula with code
├── RIGOR_FINDINGS.md          # Empirical log: ablations, sweeps, baselines
├── ARCHITECTURE.md            # (legacy session summary)
├── DEPLOYMENT.md              # Production notes
├── PERFORMANCE.md             # Scale / latency characterization
└── ENTITY_EXTRACTION.md       # Entity extractor design notes

REPRODUCE.md                   # Clone-to-results recipe
docker-compose.yml             # pgvector/pgvector:pg16 + dendric services
```

---

## Status

- ✅ **Substrate**: stable. Determinism (SD=0) verified on 15-probe ×
  5-repeat sweep.
- ✅ **Reproducibility**: Docker stack + synthetic corpus runs end-to-end
  in under 10 minutes.
- ✅ **Architecture earns its keep on aged corpora** at small k
  (+6.7pp recall_any@10 over plain RAG on `meridian_deep`).
- ⚠️ **Architecture is benchmark-dependent.** On per-question fresh-haystack
  benchmarks (LongMemEval), full ≈ plain RAG; the lifecycle has nothing to
  act on. This is in-scope for the design.
- ⚠️ **Multi-session aggregation is weak** (60% any@10 on LongMemEval
  multi-session, n=5). Substrate finds the session; surfacing every turn
  needs work above the substrate.
- 🔬 **Open**: long-horizon (100+ cycle) consolidation stability,
  fan-out exponent sweep across [0.5, 0.7, 0.85, 1.0], LongMemEval at
  full scale (n=500), second-person blind annotation pass.

---

## Citation

If you use Dendric in research, please cite the relevant published references
for ACT-R, GANE, EWC, and MESU listed in
[`docs/TECHNICAL_REPORT.md#18-references`](docs/TECHNICAL_REPORT.md#18-references).
A Dendric-specific writeup is in preparation.

## License

MIT.
