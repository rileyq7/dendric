# Dendric — A Technical Report

**A biologically-grounded episodic memory substrate for AI agents.**

> Lifecycle dynamics — temperature, neuromodulator signals, decay, reheat, archive — are
> *load-bearing at every retrieval*, not stored as decorative metadata. Memories
> consolidate via a deterministic sleep cycle that mirrors hippocampal-to-cortical
> transfer, erodes tokens differentially, and migrates cold items into a dormant
> archive that is reachable only through associative déjà-vu. Retrieval fuses four
> independent paths and then tilts the ranking by lifecycle state.

This document is the technically-rigorous companion to `README.md`. Every claim is
backed by a code reference and (where applicable) an empirical result from
`docs/RIGOR_FINDINGS.md`.

---

## Table of contents

1. [Design intent](#1-design-intent)
2. [Architecture overview](#2-architecture-overview)
3. [Storage substrate](#3-storage-substrate)
4. [Ingest pipeline](#4-ingest-pipeline)
5. [Neuromodulator signals — DA, NE, GABA](#5-neuromodulator-signals--da-ne-gaba)
6. [The 7-stage Dendric activation equation (GANE)](#6-the-7-stage-dendric-activation-equation-gane)
7. [Token erosion — differential, importance-weighted forgetting](#7-token-erosion--differential-importance-weighted-forgetting)
8. [Deterministic compression cascade](#8-deterministic-compression-cascade)
9. [Consolidation: the sleep cycle](#9-consolidation-the-sleep-cycle)
10. [Retrieval — four parallel paths](#10-retrieval--four-parallel-paths)
11. [RRF fusion + lifecycle modulation at rank](#11-rrf-fusion--lifecycle-modulation-at-rank)
12. [Déjà-vu — the archive trigger](#12-déjà-vu--the-archive-trigger)
13. [Persona, fan-out normalization, and entity hubs](#13-persona-fan-out-normalization-and-entity-hubs)
14. [MESU pruning + EWC protection](#14-mesu-pruning--ewc-protection)
15. [Empirical findings](#15-empirical-findings)
16. [Configuration surface and ablation harness](#16-configuration-surface-and-ablation-harness)
17. [Honest limitations](#17-honest-limitations)
18. [References](#18-references)

---

## 1. Design intent

Most "memory for LLMs" systems are a vector store, a chunker, and an answer
prompt. They store every chunk forever, retrieve by cosine, and call that
"long-term memory." This collapses three distinct biological functions —
encoding, consolidation, and recall — into a single read/write path.

Dendric is built around a different bet: **the *lifecycle* of a memory is the
substrate**. Whether a memory is hot or cold, whether the agent's current goals
make it dopaminergically relevant, whether something else is currently shouting
at the same entity — these are the things that should drive what surfaces, not
just embedding cosine.

Concretely, Dendric promises and tests three design invariants
(`src/scripts/per_signal_recall_check.py`):

1. **Lifecycle modulation tilts ranking by state.** A hot, goal-relevant memory
   outranks a cold one when both are otherwise comparable — but the bias is
   bounded, so an overwhelmingly strong semantic match on a cold memory still
   surfaces.
2. **Spreading activation reaches co-occurring entities.** Querying for `Mango`
   surfaces a memory about Tamiya thinner via the entity graph, not just direct
   vector hits.
3. **The déjà-vu trigger reaches archived memories — and only when relevant.**
   Archived memories are dormant by default (filtered out of vector / keyword /
   graph paths) but become reachable when entity activation crosses a threshold.

These invariants are checked on a real Postgres on every CI-eligible run; the
benchmark numbers are downstream of the substrate behaving correctly.

---

## 2. Architecture overview

```
INGEST                                 CONSOLIDATE (sleep cycle)
─────────                              ─────────────────────────
text → embed (1536-dim)                ACT-R power-law decay (Stage 1)
     → DA / NE / GABA                  DA-boosted memories resist decay
     → novelty gate (NE < 0.15 → cold) GABA-suppressed accelerate down
     → entities + co-occurrence        7-stage GANE activation per memory
     → persona link                    Token erosion (stopwords shed first)
     → store @ T=1.0 (hippocampus)     Region migration: hipp → neo → archive
                                       MESU probabilistic pruning
                                       Entity-graph fan-effect decay + prune

RECALL
──────
query → 4 paths in parallel:
        ├─ vector      (cosine, region != archive)
        ├─ keyword     (FTS + ILIKE, region != archive)
        ├─ graph       (entity-fan walk, region != archive)
        └─ associative (spreading activation, can pull archive on déjà-vu)
     → temporal decomposition (extra vec/kw queries for time-anchored Q's)
     → RRF fusion w/ session-diversity penalty + recency tiebreaker
     → lifecycle modulation: fused × bounded f(temp, DA, NE)
     → temporal re-rank (only if temporal query)
     → top_k
```

Source map (`src/engine/`):

| Module | Role |
|---|---|
| `core/engine.py` | Public API: `remember`, `recall`, `consolidate`, `forget`, `stats` |
| `core/activation.py` | 7-stage Dendric activation equation (GANE) |
| `core/signals_enhanced.py` | DA / NE / GABA computation |
| `core/erosion.py` | Token-level importance scoring + differential decay |
| `core/compression.py` | Deterministic 3-tier compression (summary, nuggets, edges) |
| `core/pruning.py` | MESU uncertainty-weighted probabilistic pruning |
| `core/protection.py` | EWC retrieval-importance accumulation |
| `core/entity_extraction.py` | Regex/heuristic entity extraction (no LLM) |
| `retrieval/vector.py` | Path 1: pgvector cosine |
| `retrieval/keyword.py` | Path 2: tsvector FTS + ILIKE |
| `retrieval/associative.py` | Path 3: BFS spreading activation + déjà-vu |
| `retrieval/graph.py` | Path 4: entity-fan walk |
| `retrieval/fusion.py` | Reciprocal Rank Fusion + lifecycle modulation |
| `retrieval/temporal.py` | Temporal-query detection + re-rank |
| `retrieval/temporal_decomposer.py` | Decomposes "X 3 days ago" into event + time |
| `storage/postgres.py` | Postgres + pgvector backend |
| `storage/entity_graph.py` | Entities, memory↔entity links, entity edges |
| `storage/migrations.py` | Schema + idempotent migrations |
| `embeddings/embed.py` | OpenAI batch-API wrapper |
| `config.py` | All tunable parameters; presets for ablation |

---

## 3. Storage substrate

Postgres with the `pgvector` extension. The schema lives in
`src/engine/storage/migrations.py` and is applied idempotently every connect.
Five logical tables:

```sql
-- Primary memory record (one row per stored episode)
CREATE TABLE memories (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Content at every compression level (only the appropriate ones are populated)
    raw_content     TEXT,
    structured_summary TEXT,
    knowledge_nugget TEXT,
    entity_edges    TEXT,

    -- Lifecycle state
    temperature     FLOAT NOT NULL DEFAULT 1.0,
    region          TEXT NOT NULL DEFAULT 'hippocampus',  -- | neocortex | archive

    -- Neuromodulator signals
    da_relevance    FLOAT NOT NULL DEFAULT 0.3,
    ne_novelty      FLOAT NOT NULL DEFAULT 0.5,
    usage_score     FLOAT NOT NULL DEFAULT 0.0,

    -- MESU uncertainty tracking
    da_history      FLOAT[] DEFAULT '{}',
    ne_history      FLOAT[] DEFAULT '{}',
    signal_variance FLOAT DEFAULT 0.5,

    -- EWC retrieval importance (resists compression/pruning when high)
    retrieval_hits  INTEGER DEFAULT 0,
    retrieval_importance FLOAT DEFAULT 0.0,

    -- Retrieval indexes
    embedding       vector(1536),       -- text-embedding-3-small
    content_tsv     tsvector,           -- weighted A/B/C/D over the 4 levels

    -- Token-level erosion (differential decay weights per token)
    token_weights   JSONB,
    ...
);

-- Cold storage. Raw content of archived memories is moved here so the
-- active table stays lean; the memories row keeps lifecycle state and
-- can still be reanimated via déjà-vu.
CREATE TABLE archive (
    memory_id        UUID PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    raw_content      TEXT NOT NULL,
    original_embedding vector(1536),
    archived_at      TIMESTAMPTZ DEFAULT now(),
    encoding_context JSONB DEFAULT '{}'
);

-- Entity graph: entities, links, weighted co-occurrence edges
CREATE TABLE entities (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name           TEXT NOT NULL,
    entity_type    TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    first_seen     TIMESTAMPTZ DEFAULT now(),
    last_seen      TIMESTAMPTZ,
    mention_count  INTEGER DEFAULT 1,
    session_ids    TEXT[] DEFAULT '{}',
    UNIQUE(canonical_name, entity_type)
);

CREATE TABLE memory_entities (
    memory_id  UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    entity_id  UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    salience   FLOAT DEFAULT 0.5,            -- 0.5 for normal, 0.2 for persona
    position   INTEGER,
    PRIMARY KEY (memory_id, entity_id)
);

CREATE TABLE entity_edges (
    entity_a            UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    entity_b            UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    weight              FLOAT DEFAULT 1.0,
    co_occurrence_count INTEGER DEFAULT 1,
    last_reinforced     TIMESTAMPTZ DEFAULT now(),
    predicate           TEXT DEFAULT 'co_occurs',
    PRIMARY KEY (entity_a, entity_b),
    CHECK (entity_a < entity_b)             -- canonical ordering, no dupes
);

-- Retrieval log (drives EWC importance updates)
CREATE TABLE retrieval_log (
    id              SERIAL PRIMARY KEY,
    query_text      TEXT NOT NULL,
    query_embedding vector(1536),
    results         UUID[] NOT NULL,
    top_k_ids       UUID[] NOT NULL,
    timestamp       TIMESTAMPTZ DEFAULT now()
);
```

The `content_tsv` column is auto-maintained by a trigger that gives raw_content
the highest weight and entity_edges the lowest, so as a memory compresses, FTS
relevance gracefully degrades:

```sql
CREATE OR REPLACE FUNCTION memories_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.content_tsv :=
        setweight(to_tsvector('english', coalesce(NEW.raw_content, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(NEW.structured_summary, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(NEW.knowledge_nugget, '')), 'C') ||
        setweight(to_tsvector('english', coalesce(NEW.entity_edges, '')), 'D');
    RETURN NEW;
END
$$ LANGUAGE plpgsql;
```

The migration also contains a pgvector dimension-mismatch self-healer that
TRUNCATEs+rebuilds vector columns only when `atttypmod != 1536` — a previous
buggy comparator (`!= 1540`) silently destroyed every long-lived deployment on
boot. The fix is documented inline:

```python
# IMPORTANT: pgvector stores the declared dimension N directly in
# atttypmod (so vector(1536) → atttypmod=1536). Old code here checked
# against 1540, which is wrong, and silently TRUNCATEd every boot.
# That's a destructive bug for any long-lived deployment.
```

---

## 4. Ingest pipeline

`MemoryEngine.remember(content, source, context)` does the following, in order
(`src/engine/core/engine.py:93`):

1. **Embed** the content (OpenAI `text-embedding-3-small`, 1536-dim).
2. **Compute DA** at ingest from access history (none yet) → baseline.
3. **Compute NE / GABA** by cosine-comparing the new embedding to the 50 most
   recent stored embeddings.
4. **Extract entities** with `extract_entities()` — regex/heuristic, no LLM.
   Filter to `named` and `numeric` types of length ≥ 3, strip trailing
   punctuation, exclude `ENTITY_STOPWORDS`.
5. **Novelty gate.** If `NE < 0.15` (config: `novelty_gate`), the memory is
   stored cold (`temp=0.1`, `region=neocortex`) — entity stats still accumulate
   so future related memories find the graph, but the memory itself is born
   suppressed:
   ```python
   if ne < self.config.novelty_gate:
       logger.info(f"Gated out (NE={ne:.3f}): {content[:50]}...")
       mem = Memory(temperature=0.1, region="neocortex", ...)
       self.store.store(mem)
       self._link_entities(entity_graph, mem.id, entities, session_id)
       return mem.to_dict()
   ```
6. **Build the Memory record** with `temp=1.0`, `region=hippocampus`, time-of-
   day derived from the ingest timestamp.
7. **Initialize token-level erosion weights** by scoring each token's importance
   (entity / numeric / rarity / proximity / stopword penalty — see §7).
8. **Persist + link entities to the graph.** Persona is linked separately with
   reduced salience (0.2 vs 0.5), and persona is *excluded from co-occurrence
   edges* (otherwise every entity would end up edge-connected to the persona,
   destroying spreading-activation specificity).

Batch ingest (`remember_batch`) is a single OpenAI API call for all embeddings,
a sliding-window novelty deque sized to `novelty_window` (200), and a single
DB transaction for all memory inserts. This is what makes 50k-item Claude-export
imports tractable; the naïve O(n²) cosine recompute would take days.

---

## 5. Neuromodulator signals — DA, NE, GABA

`src/engine/core/signals_enhanced.py`. Three signals, each computable without
any LLM call, each with a literature-grounded interpretation.

### Dopamine (DA): relevance / importance

```python
def compute_da(access_count, avg_outcome, user_rating, goal_alignment=0.0,
               active_goal_weight=0.4) -> float:
    freq = access_count / (access_count + 5)             # saturating frequency
    outcome_norm = (avg_outcome + 1) / 2                  # [-1,1] → [0,1]
    rating_norm  = (user_rating + 1) / 2

    da = (
        0.25 * freq
        + active_goal_weight * goal_alignment
        + 0.20 * outcome_norm
        + 0.15 * rating_norm
    )
    return min(1.0, da)
```

Ingest passes zeros for the outcome / rating / goal-alignment terms (no
retrieval history yet); DA is recomputed during consolidation as access counts
accumulate.

### Norepinephrine (NE): novelty / surprise

NE is dissimilarity-to-nearest-neighbor, decayed by age:

```python
def compute_ne(embedding, store_embeddings, age_days=1.0,
               recency_decay_rate=0.05) -> float:
    if not store_embeddings:
        return 1.0
    similarities = [cosine_similarity(embedding, other) for other in store_embeddings]
    nearest_sim = max(similarities) if similarities else 0.0
    novelty = 1.0 - nearest_sim
    recency_decay = math.exp(-recency_decay_rate * age_days)
    return max(0.0, min(1.0, novelty * recency_decay))
```

The downstream activation equation does *not* treat NE linearly. It uses an
**asymmetric Gaussian inverted-U** with peak at `μ_NE = 0.65`:

```python
iu = w_ne * math.exp(-((ne_novelty - mu_ne) ** 2) / (2 * sigma_ne ** 2))
```

This phenomenologically captures dual-receptor pharmacology (Avery et al.
2013): low NE underdrives encoding, optimal NE maximally enhances it, NE
overload (→ 1.0) suppresses it. The validation suite specifically asserts
`temp(NE=1.0) < temp(NE=0.65)` — without this property the curve is broken.

### GABA: inhibition / staleness

GABA combines redundancy (cosine to the most-similar stored memory) with
staleness (logistic curve in age-since-access):

```python
def compute_gaba(embedding, store_embeddings, age_days=1.0,
                 stale_threshold=30.0, recency_window=10.0) -> float:
    if store_embeddings:
        similarities = [cosine_similarity(embedding, other) for other in store_embeddings]
        max_sim = max(similarities) if similarities else 0.0
        redundancy = max(0.0, (max_sim - 0.8) * 5)         # 0.8-1.0 sim → 0-1
    else:
        redundancy = 0.0
    staleness = 1.0 / (1.0 + math.exp(-(age_days - stale_threshold) / recency_window))
    return min(1.0, 0.5 * redundancy + 0.5 * staleness)
```

GABA is what makes a memory passively decay even when nothing actively pushes
it down; it is also recomputed each cycle as `1 − e^(−0.3·cycles_since_access)`
so disuse alone drives it toward saturation.

---

## 6. The 7-stage Dendric activation equation (GANE)

This is the core novelty of the substrate. The full equation lives in
`src/engine/core/activation.py:25` and is composed of seven phenomenologically-
motivated stages. Defaults shown here were tuned against eight research-derived
test scenarios in the same file.

The formula, in mathematical form:

```
Stage 1 (ACT-R base activation):
    B = ln( Σ_i  t_i^(-d) )            with d = 0.3, t_i in days

Stage 2 (DA importance + spreading activation):
    A₁ = B - τ + β · DA + s            with τ = 2.0, β = 2.2

Stage 3 (NE inverted-U, asymmetric Gaussian):
    I_U = w_NE · exp( -(NE - μ_NE)² / (2σ_NE²) )
                                       with w_NE=0.4, μ_NE=0.65, σ_NE=0.18

Stage 4 (Pre-GANE salience):
    a_sal = A₁ + I_U

Stage 5 (GANE feedback — winner-take-more / loser-take-less):
    g    = 1 + α · DA                  with α = 0.2
    GANE = g · η · NE · tanh(λ · a_sal) with η=0.05, λ=0.6

Stage 6 (GABA hybrid inhibition — divisive + subtractive):
    a_pre = a_sal + GANE
    a     = a_pre / (1 + δ · GABA) - γ · GABA   with δ=1.2, γ=0.8

Stage 7 (arousal-modulated logistic noise):
    s = s₀ · (1 + ρ · NE)              with s₀=0.15, ρ=0.5
    a += s · ln( u / (1 - u) ),  u ∼ U(0,1)

Final:
    temperature = σ(a)                 (sigmoid → [0, 1])
```

The full implementation:

```python
def compute_temperature(
    accesses_days_ago, da_relevance, ne_novelty, gaba_inhibition,
    spreading_activation=0.0, noise=True, use_gane=True,
    d=0.3, tau=2.0, beta=2.2, alpha=0.2,
    w_ne=0.4, mu_ne=0.65, sigma_ne=0.18,
    eta=0.05, lam=0.6,
    delta_gaba=1.2, gamma_gaba=0.8,
    s0=0.15, rho=0.5,
) -> float:
    # ── Stage 1: Base-level activation (ACT-R power-law decay) ──
    times = [max(1 / 1440, t) for t in accesses_days_ago] or [1.0]
    B = math.log(sum(t ** (-d) for t in times))

    # ── Stage 2: DA importance boost + spreading activation ──
    # DA is ADDITIVE (importance floor), not multiplicative.
    # Multiplicative gain on negative B amplifies decay — wrong behavior.
    A1 = B - tau + beta * da_relevance + spreading_activation

    # ── Stage 3: NE inverted-U (Gaussian approximation) ──
    iu = w_ne * math.exp(-((ne_novelty - mu_ne) ** 2) / (2 * sigma_ne ** 2))

    # ── Stage 4: Pre-GANE salience ──
    a_sal = A1 + iu

    # ── Stage 5: GANE feedback (winner-take-more / loser-take-less) ──
    # tanh creates competition: above-mean memories amplified, below-mean suppressed
    # DA gates the gain: important memories are more responsive to NE dynamics
    if use_gane:
        g = 1 + alpha * da_relevance
        gane = g * eta * ne_novelty * math.tanh(lam * a_sal)
    else:
        gane = 0.0

    # ── Stage 6: GABA hybrid inhibition ──
    # Divisive: GABA_A shunting inhibition (compresses dynamic range)
    # Subtractive: GABA_B hyperpolarizing inhibition (shifts baseline down)
    a_pre = a_sal + gane
    a = a_pre / (1 + delta_gaba * gaba_inhibition) - gamma_gaba * gaba_inhibition

    # ── Stage 7: Arousal-modulated logistic noise ──
    if noise:
        s = s0 * (1 + rho * ne_novelty)
        u = max(1e-10, min(1 - 1e-10, random.random()))
        a += s * math.log(u / (1 - u))                 # logistic noise

    # ── Final sigmoid → temperature (0.0-1.0) ──
    a_clipped = max(-20, min(20, a))
    return 1.0 / (1.0 + math.exp(-a_clipped))
```

### What each stage does, and why

| Stage | Function | Motivation |
|---|---|---|
| 1 | ACT-R power-law decay over access history | Anderson & Lebiere (1998); base activation falls as t^-d, but multiple accesses contribute linearly inside the log |
| 2 | DA enters **additively**, not multiplicatively | A multiplicative DA gain on a negative B (which happens for old memories) would *amplify* decay, the opposite of "important things resist forgetting" |
| 3 | NE inverted-U around μ_NE=0.65 | Avery et al. (2013) dual-receptor pharmacology — moderate novelty maximally enhances encoding; overload suppresses |
| 4 | Compose pre-GANE salience | Just A₁ + I_U |
| 5 | **GANE: g · η · NE · tanh(λ · a_sal)** with g = 1 + α·DA | Mather et al. (2016) — winner-take-more / loser-take-less competition. The `tanh` shape means above-mean memories are amplified and below-mean memories are suppressed; DA gates how much GANE matters at all |
| 6 | GABA hybrid (divisive + subtractive) | Mitchell & Silver (2003) GABA_A shunting + Prescott & De Koninck (2003) GABA_B hyperpolarization; hot memories are protected by divisive normalization, cold memories take both barrels |
| 7 | Arousal-modulated logistic noise | Dancy et al. (2015); higher NE → higher stochasticity. Logistic noise (rather than Gaussian) so the sigmoid composition gives a closed-form interpretation |

The validated test scenarios are encoded in the file (`TEST_SCENARIOS`,
`validate_scenarios`); the critical property — `temp(NE=1.0) < temp(NE=0.65)` —
is asserted explicitly. Without it the inverted-U has been silently flattened
or inverted.

GANE is what makes the equation more than a fancy decay curve. Without it (set
`activation_use_gane=False`) the system reduces to weighted ACT-R + GABA. With
it, the system produces actual ranking competition between memories at the
*activation* level, before retrieval ever runs. The same `compute_temperature`
function is called at every consolidation cycle for every active memory; that
output is what gets written into the `temperature` column and what the
retrieval lifecycle modulator multiplies the fused score by.

---

## 7. Token erosion — differential, importance-weighted forgetting

`src/engine/core/erosion.py`. The other novel mechanism: instead of binary
compression ("keep raw text" vs "drop it"), each memory has a per-token
importance score and a per-token weight that decays differentially. High-
importance tokens (entities, numbers, units, rare nouns) survive longer than
stopwords and filler.

### Importance scoring (at ingest)

```python
def score_token_importance(tokens, known_entities=None) -> List[Dict]:
    entity_set = {e.lower() for e in (known_entities or [])}
    results = []
    for i, token in enumerate(tokens):
        signals = []

        # Entity signal: capitalized mid-sentence, or in entity list
        is_entity = (
            (i > 0 and token[0].isupper() and not token.isupper())
            or (token.lower() in entity_set)
        )
        if is_entity:
            signals.append(0.95)

        # Numeric signal
        if re.search(r'\d', token):
            signals.append(0.90)
        elif token.lower() in UNITS:                 # kg, mg, cm, %, ...
            signals.append(0.85)

        # Rarity signal
        if token.lower() not in STOPWORDS and len(token) > 3:
            rarity = 0.5 + min(0.3, len(token) * 0.03)
            signals.append(rarity)

        # Proximity signal (within ±2 tokens of an entity or number)
        window = tokens[max(0, i-2):min(len(tokens), i+3)]
        if any(w[0].isupper() or re.search(r'\d', w)
               for j, w in enumerate(window)
               if j != (i - max(0, i-2))):
            signals.append(0.4)

        # Stopword penalty
        if token.lower() in STOPWORDS:
            signals.append(-0.3)

        # Combine: max positive + sum negatives
        pos = [s for s in signals if s > 0]
        neg = [s for s in signals if s < 0]
        importance = (max(pos) if pos else 0.1) + sum(neg)
        importance = max(0.05, min(0.95, importance))

        results.append({'token': token,
                        'importance': round(importance, 3),
                        'weight': 1.0})
    return results
```

### Differential decay (each consolidation cycle)

```python
def erode_tokens(token_weights, base_decay=0.15) -> List[Dict]:
    """High-importance tokens decay slowly; low-importance tokens decay fast."""
    for tw in token_weights:
        importance = tw.get('importance', 0.5)
        decay_rate = base_decay * (1.0 - importance)
        tw['weight'] = tw.get('weight', 1.0) * (1.0 - decay_rate)
    return token_weights
```

The decay rate is gated by importance: a token with `importance=0.95` decays at
`0.15 × 0.05 = 0.75%` per cycle; a stopword with `importance=0.05` decays at
`0.15 × 0.95 = 14.25%` per cycle. After 30 cycles the entity is at
`0.79`; the stopword is at `0.0094`.

### Reading at temperature (graceful degradation)

```python
def read_at_temperature(token_weights, temperature) -> str:
    """
    Higher temperature = full fidelity (all tokens visible).
    Lower temperature = eroded (only high-importance tokens visible).

    threshold = 1.0 - temperature
      temp 1.0 → threshold 0.0 → all tokens
      temp 0.5 → threshold 0.5 → only weight > 0.5
      temp 0.1 → threshold 0.9 → only weight > 0.9
    """
    if temperature >= 0.95:
        return ' '.join(tw['token'] for tw in token_weights)
    threshold = 1.0 - temperature
    visible = [tw['token'] for tw in token_weights
               if tw.get('weight', 1.0) > threshold]
    return ' '.join(visible) if visible else "(eroded to empty)"
```

This is what `Memory.best_content` and `Memory.to_dict()` use to surface text
at any temperature. A memory that started as

> *"Walked Mango along the canal at 7 in the morning while it was raining."*

might after months of disuse and consolidation read as

> *"Mango canal 7 morning"*

without the system ever doing an LLM call to "summarize" it. Token erosion *is*
the summary.

---

## 8. Deterministic compression cascade

`src/engine/core/compression.py`. Three cascading outputs, each fired when the
memory's temperature crosses a threshold. **No LLM call.**

```python
COMPRESSION_THRESHOLDS = {
    "structured_summary": 0.70,    # extractive top-K sentences by importance
    "knowledge_nugget":   0.40,    # entity + (number|date|entity) pattern match
    "entity_edges":       0.20,    # verb-mediated entity relationships
    "archive":            0.10,    # raw_content moved to archive table
}
```

The compression engine operates on token-importance scores from ingest:

1. **`extract_compression_entities()`** — re-extracts entities with character
   spans and prioritizes temporal > numeric > named > concept. Earlier passes
   "claim" character ranges so later passes don't double-count.
2. **`split_sentences()`** — abbreviation-aware sentence splitter (lookbehind
   on Mr/Mrs/Dr/St/vs/etc/Jr/Sr).
3. **`score_sentences()`** — combines mean per-token importance with an
   entity-density boost (`1.0 + 0.3 · min(entity_count, 4)`).
4. **`extract_summary(scored, T)`** — keeps top `max(2, total · keep_ratio)`
   sentences in original order. `keep_ratio = clamp(T, 0.2, 0.8)` so the
   summary tightens as temperature falls.
5. **`extract_nuggets()`** — keeps sentences that are entity+entity,
   entity+number, or entity+date patterns. Emitted as middot-joined fact
   strings: `"Theo · Battersea Vets · 28kg"`.
6. **`extract_edges()`** — for every entity pair within a sentence, find a
   verb in between, map to a typed predicate (`went_to`, `lives_in`,
   `bought`, ...), weight by `(e1.importance + e2.importance) / 2 ·
   sentence.score`. Predicate defaults to `co_occurs` if no verb resolves.

Edge persistence at consolidation:

```python
if compressions.get("entity_edges"):
    edge_data = json.loads(compressions["entity_edges"])
    entity_graph = EntityGraphStore(self.store.conn)
    for edge in edge_data:
        entity_graph.upsert_compression_edge(
            source_name=edge['source'], source_type=edge['source_type'],
            target_name=edge['target'], target_type=edge['target_type'],
            predicate=edge['predicate'], weight=edge['weight'],
        )
```

This means a memory's compression doesn't just produce a smaller text — it
*feeds back* into the entity graph as typed edges, which in turn drive
spreading activation and graph retrieval. The substrate gets denser as
memories age.

---

## 9. Consolidation: the sleep cycle

`MemoryEngine.consolidate()` (`src/engine/core/engine.py:703`). One cycle
processes every active memory. Sketch:

```python
def consolidate(self) -> dict:
    self._cycle_count += 1
    now = datetime.now(timezone.utc)
    stats = {...}
    memories = self.store.get_all_active()

    # 0. Strengthen co-accessed entity edges; fan-effect decay on the graph
    pruned_edges = entity_graph.consolidate_entity_graph(last_consol)
    stats["edges_pruned"] = pruned_edges

    for mem in memories:
        # 1.  GABA grows with disuse:  GABA = 1 - exp(-0.3 · cycles_since_access)
        cycles_since_access = self._cycles_since_access(mem, now)
        mem.gaba_inhibition = min(0.95, 1.0 - np.exp(-0.3 * cycles_since_access))

        # 1a. EWC importance decays once per cycle (was per-query — too aggressive)
        mem.retrieval_importance *= (1.0 - self.config.importance_decay_rate)

        # 1b. Recompute DA from access history
        mem.da_relevance = compute_da(access_count=mem.access_count, ...)

        # 1c. NE decays toward 0 (novelty fades; old things stop being surprising)
        decayed_ne = mem.ne_novelty * math.exp(-0.05 * age_days)
        mem.ne_novelty = decayed_ne if decayed_ne > 1e-30 else 0.0

        # 2.  7-stage Dendric activation → new temperature
        sa = entity_graph.compute_spreading_activation(mem.id)
        new_temp = compute_temperature(
            accesses_days_ago=accesses_days_ago,
            da_relevance=mem.da_relevance,
            ne_novelty=mem.ne_novelty,
            gaba_inhibition=mem.gaba_inhibition,
            spreading_activation=sa * self.config.spreading_activation_weight,
            noise=self.config.activation_use_noise,
            use_gane=self.config.activation_use_gane,
        )

        # 3.  Update DA/NE histories for MESU uncertainty
        # 4.  Deterministic compression — only fire if temp crosses threshold
        #     AND EWC compression-resistance check rolls true
        # 5.  Token erosion — differential decay
        # 6.  Region migration:
        #       new_temp < 0.10  → archive (raw moved to archive table)
        #       new_temp < 0.65 AND in hippocampus → neocortex
        # 7.  MESU probabilistic pruning (skip if temp > 0.15 OR accessed in last 3 cycles)
        # 8.  Persist compression edges to entity graph (next cycle's SA can use them)
        # 9.  Single UPDATE with all new fields
```

A few subtle properties worth calling out:

- **Decay happens during consolidation, not during recall.** A bug in earlier
  versions decayed EWC importance every query, which made the importance
  landscape a function of query throughput rather than time. Now it decays
  once per cycle.
- **`spreading_activation_weight=0.3` modulates the SA contribution into
  Stage 2 of the activation equation.** This is how entity-graph centrality
  protects memories from decay.
- **Archive raw is *moved*, not copied.** The `memories` row keeps lifecycle
  state and embeddings; the raw text moves to the `archive` table. Déjà-vu
  reads back from `memories` (which is why archive memories still appear in
  the spreading-activation candidate set).

---

## 10. Retrieval — four parallel paths

`MemoryEngine.recall(query, top_k, ...)` (`src/engine/core/engine.py:339`).
Four independent retrieval paths run in parallel, each returning a ranked list
of memories with a normalized score. Each path is independently ablatable.

### Path 1 — Vector

`src/engine/retrieval/vector.py`. Cosine over pgvector. Excludes `region =
'archive'`.

```python
def vector_search(self, query_embedding, top_k=20, min_temp=0.0):
    cur.execute("""
        SELECT *, 1 - (embedding <=> %s::vector) as similarity
        FROM memories
        WHERE temperature >= %s
          AND embedding IS NOT NULL
          AND region != 'archive'
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (query_embedding, min_temp, query_embedding, top_k))
```

### Path 2 — Keyword (FTS + ILIKE fallback)

Postgres tsvector with weighted A/B/C/D over the four content levels. Three
tiers of escalation:

1. Strict AND `plainto_tsquery` — precise.
2. OR `to_tsquery` if AND returns < top_k — broad, but `ts_rank` puts best
   matches first.
3. ILIKE substring fallback on individual content words for the long tail
   (camelCase identifiers, embedded numerals, etc.).

### Path 3 — Associative (spreading activation)

`src/engine/retrieval/associative.py`. The path that earns the "biological"
framing of the system.

Algorithm (from the module docstring):

```
1. Resolve query entity names → entity_ids in the graph
2. activation[seed] = 1.0 for each seed (or 0.5 if persona-fallback)
3. For each hop in [1..max_hops]:
     For each entity with current activation:
       For each edge (entity ↔ neighbor):
         spread = current_act × edge_weight × decay^hop
         activation[neighbor] = max(activation[neighbor], spread)
   (max-aggregation over paths is more stable than sum at low hop counts)
4. For each entity with activation > threshold:
     Pull its linked memories, score = activation[entity] × salience
5. Aggregate per-memory scores (sum across all activated entities the
   memory is linked to), divided by num_linked**fanout_norm_exponent
6. Return top_k by score
```

The implementation, inlined for clarity:

```python
def spreading_activation_recall(
    query_entities, store, top_k=10,
    max_hops=2, decay=0.5, activation_threshold=0.05,
    min_temp=0.0, archive_trigger_threshold=None,
    persona="", persona_seed_activation=0.5, persona_fallback=True,
    fanout_norm_exponent=0.5,
):
    eg = EntityGraphStore(store.conn)

    # Resolve seeds; persona is held back as fallback
    non_persona_seeds = []
    persona_in_query = False
    for name in (query_entities or []):
        canonical = (name or "").strip().lower()
        if canonical == persona:
            persona_in_query = True
            continue
        eid = eg.get_entity_by_name(canonical)
        if eid is not None:
            non_persona_seeds.append((eid, 1.0))

    seed_ids_with_act = list(non_persona_seeds)
    needs_persona = persona and not non_persona_seeds and (
        persona_in_query or persona_fallback)
    if needs_persona:
        persona_eid = eg.get_entity_by_name(persona)
        if persona_eid is not None:
            seed_ids_with_act.append((persona_eid, persona_seed_activation))

    if not seed_ids_with_act:
        return []

    # BFS with max-aggregation
    activation = {eid: act for eid, act in seed_ids_with_act}
    frontier = [eid for eid, _ in seed_ids_with_act]
    for hop in range(1, max_hops + 1):
        next_frontier = []
        hop_factor = decay ** hop
        for eid in frontier:
            current_act = activation.get(eid, 0.0)
            if current_act < activation_threshold:
                continue
            for edge in eg.get_edges_for_entity(eid):
                neighbor = edge["entity_b"] if edge["entity_a"] == eid else edge["entity_a"]
                edge_weight = float(edge.get("weight", 1.0))
                spread = current_act * edge_weight * hop_factor
                if spread > activation.get(neighbor, 0.0):
                    activation[neighbor] = spread
                    next_frontier.append(neighbor)
        frontier = next_frontier
        if not frontier:
            break

    # Score memories: each entity's contribution / num_linked**exp
    activated = [(eid, act) for eid, act in activation.items()
                 if act >= activation_threshold]
    memory_scores, memory_paths = defaultdict(float), defaultdict(set)
    for eid, act in activated:
        linked = list(eg.get_memories_for_entity(eid))
        if not linked:
            continue
        norm = 1.0 / (len(linked) ** fanout_norm_exponent)
        per_memory_contrib = act * norm
        for mid in linked:
            memory_scores[str(mid)] += per_memory_contrib
            memory_paths[str(mid)].add(str(eid))

    # Optional: pull archive memories for entities that crossed déjà-vu threshold
    archive_triggered_ids = []
    if archive_trigger_threshold is not None:
        archive_triggered_entities = [
            eid for eid, act in activated if act >= archive_trigger_threshold
        ]
        for eid in archive_triggered_entities:
            for mid in eg.get_memories_for_entity(eid):
                archive_triggered_ids.append(str(mid))

    # Fetch + return
    rows = _fetch_memories(store.conn, list(memory_scores.keys()),
                           min_temp, archive_triggered_ids)
    ...
```

`max-aggregation` (rather than sum) over multiple paths to the same neighbor is
chosen because at small hop counts (default `max_hops=2`) summing duplicates
the same evidence; max gives the strongest single chain.

### Path 4 — Graph (entity-fan walk)

`src/engine/retrieval/graph.py`. Same starting seeds as Path 3, but a different
algorithm: extract query entities → find seed entity ids → walk 1 hop along
weighted edges → optionally walk 1 more hop (only for edges with weight >
0.3) → fetch all memories attached to reached entities → score by sum of
per-entity edge weights, optionally fan-out-normalized.

Where Path 3 produces a *score per memory by activation*, Path 4 produces a
*score per memory by edge weight to seeds*. They tend to agree on memories
that are tightly connected to a query entity but diverge on long-range
multi-hop reachability.

---

## 11. RRF fusion + lifecycle modulation at rank

`src/engine/retrieval/fusion.py`. Standard Reciprocal Rank Fusion with per-path
weights and three additions:

1. **Similarity bonus on the vector path** — `+ vector_weight · similarity ·
   0.02` so an actually-similar match beats a barely-similar one even at the
   same rank.
2. **Recency tiebreaker** — `+ recency_bonus / (1 + ln(1 + age_days))`. Bonus
   is small (~0.002) compared to RRF scores (~0.01–0.06), so it only breaks
   ties.
3. **Three-phase session-diversity selection** — for non-aggregation queries,
   ensure top_k spans up to 3 distinct sessions before falling back to score
   ranking with an exponential session penalty:

```python
def _session_penalized_select(scores, top_k, session_decay):
    # Phase 1: take #1 result regardless of session
    # Phase 2: fill from unseen sessions until min_sessions met
    # Phase 3: fill rest with session-penalized scores
    ...
```

After fusion, **lifecycle modulation** tilts the score:

```python
def apply_lifecycle_modulation(
    results, temp_lift=0.4, da_lift=0.3, ne_penalty=0.3,
    mod_min=0.7, mod_max=1.6, archive_modulation_override=1.0,
):
    """
    modulation = (1 + temp_lift · temp)
               × (1 + da_lift   · da)
               × (1 - ne_penalty · (1 - inverted_u(ne)))
    clamped to [mod_min, mod_max]
    """
    for r in results:
        is_archive = "associative_archive" in (r.get("retrieval_paths") or [])

        if is_archive and archive_modulation_override != 1.0:
            mod = archive_modulation_override   # déjà-vu firing is itself strong evidence
        else:
            temp = float(r.get("temperature", 0.5))
            signals = r.get("signals", {}) or {}
            da = float(signals.get("DA", r.get("da_relevance", 0.3)))
            ne = float(signals.get("NE", r.get("ne_novelty", 0.5)))
            mod = (
                (1.0 + temp_lift * temp)
                * (1.0 + da_lift * da)
                * (1.0 - ne_penalty * (1.0 - _inverted_u(ne)))
            )
            mod = max(mod_min, min(mod_max, mod))

        r["_modulation"] = round(mod, 4)
        base = float(r.get("fusion_score", 0.0))
        r["_unmodulated_fusion_score"] = base
        r["fusion_score"] = round(base * mod, 6)

    results.sort(key=lambda x: x.get("fusion_score", 0.0), reverse=True)
    return results
```

The `_inverted_u` function shares its parameters (μ_NE=0.65, σ_NE=0.18) with
the activation equation — there is exactly one novelty-value curve in the
system, used at both consolidation time and recall time. The modulation
multiplier is bounded to `[0.7, 1.6]` so the bias is *a tilt, not a filter*:
a strong RRF match on a cold memory still surfaces.

This is the change that makes the lifecycle "load-bearing at retrieval"
rather than "stored as metadata." The substrate isn't computing temperature
to look pretty in the dashboard — temperature shapes ranking on every query.

---

## 12. Déjà-vu — the archive trigger

Archived memories (region = 'archive', temperature ≤ 0.10) are filtered out of
the vector / keyword / graph paths via SQL `WHERE region != 'archive'`. The
*only* way they can reach the candidate set is through the spreading-activation
path's archive trigger:

```python
if archive_trigger_threshold is not None:
    archive_triggered_entities = [
        eid for eid, act in activated if act >= archive_trigger_threshold
    ]
    if archive_triggered_entities:
        for eid in archive_triggered_entities:
            for mid in eg.get_memories_for_entity(eid):
                archive_triggered_ids.append(str(mid))
```

Once in the candidate set, archive memories face a structural problem at RRF
fusion: they're single-path by construction (only the associative path can
return them), and a single-path hit has score `weight / (rrf_k + rank + 1) ≈
1.5/61 ≈ 0.025` — always less than a two-path active-region hit at any rank.
Without intervention, the threshold is meaningless because archive *never
surfaces*.

Two compensations live in the config:

```python
# fusion.py
if "associative_archive" in path_tags and archive_rrf_boost != 1.0:
    rrf_score *= archive_rrf_boost                  # default 1.8

# fusion.py / apply_lifecycle_modulation
if is_archive and archive_modulation_override != 1.0:
    mod = archive_modulation_override               # default 1.3, bypasses cold penalty
```

**`archive_rrf_boost`** lifts the single-path archive contribution before
fusion sum. **`archive_modulation_override`** bypasses the cold-temperature
penalty in lifecycle modulation, on the principle that déjà-vu firing is
*itself* a strong signal (entity activation crossed the threshold) so we
don't want to double-penalize the low temperature.

These are documented as "the highest safe values" rather than "the optimum" —
parameter sweeps showed the system is inert below default and harmful above
(`docs/RIGOR_FINDINGS.md` 2026-04-27). The mechanism does fire (per
`src/scripts/dejavu_diagnostic.py`), but the threshold acts as a binary "is
this a seed entity?" gate at default `decay=0.5` and `max_hops=2` — only
seeds reach 1.0; hop-1 neighbors max out at 0.5. Tightening decay (raising
sa_decay to 0.7+) would make hop-1 neighbors threshold-eligible.

The design-invariant test for this lives at
`src/scripts/per_signal_recall_check.py`:

```python
def check_dejavu_trigger() -> bool:
    eng = _make_engine()
    eng.remember("Walked Mango along the canal.",     context="session_1")
    eng.remember("Mango spotted a fox in the garden.", context="session_2")
    eng.remember("Vet check for Mango: weight stable.", context="session_3")
    # Archive the canal memory
    cur.execute("UPDATE memories SET region='archive', temperature=0.05 "
                "WHERE raw_content LIKE '%canal%'")

    mango_results = eng.recall("What did I do with Mango?", top_k=10)
    other_results = eng.recall("Tell me about pancakes.",    top_k=10)

    archive_in_mango = any(r.get("region") == "archive" for r in mango_results)
    archive_in_other = any(r.get("region") == "archive" for r in other_results)
    return archive_in_mango and not archive_in_other
```

Surfaces archive on entity-relevant query, dormant otherwise.

---

## 13. Persona, fan-out normalization, and entity hubs

### Persona injection

A first-person memory stream has an implicit owner: the agent (or persona).
Configured via `EngineConfig.persona`. If set, every ingested memory is linked
to the persona entity in the graph — but with two protections that prevent the
persona from washing out the system:

```python
def _link_entities(self, entity_graph, memory_id, entities, session_id):
    """Persona is special: it's the implicit owner, not a peer entity. It
    gets a memory→persona link (so spreading activation can fall back to
    it for persona-only queries) but NOT co-occurrence edges to other
    entities. Without this exception, every entity ends up edge-connected
    to the persona, and activation from any entity hops via persona to
    everything — destroying the specificity of spreading activation.
    """
    entities = self._with_persona(entities)
    persona_canonical = (self.config.persona or "").strip().lower()
    entity_ids = []
    non_persona_ids = []          # for co-occurrence edges (excludes persona)
    for name, etype, canonical in entities:
        eid = entity_graph.upsert_entity(canonical, etype, name)
        entity_graph.update_entity_session(eid, session_id)
        is_persona = canonical == persona_canonical
        salience = 0.2 if is_persona else 0.5
        entity_graph.insert_memory_entity(memory_id, eid, salience=salience)
        entity_ids.append(eid)
        if not is_persona:
            non_persona_ids.append(eid)
    # Co-occurrence edges between non-persona entities only.
    for i in range(len(non_persona_ids)):
        for j in range(i + 1, len(non_persona_ids)):
            entity_graph.upsert_entity_edge(non_persona_ids[i], non_persona_ids[j])
```

In spreading activation:

- Persona is held back as a *fallback* seed. If the query mentions any other
  entity that resolves in the graph, persona doesn't seed — its universal-
  owner membership would dilute the specific signal at small top_k.
- If no specific entity resolves, persona seeds with reduced activation
  (`persona_seed_activation=0.5`, vs 1.0 for other entities). This is what
  lets queries about the agent itself (e.g. "Has Jamie been to Portugal?")
  reach memories that say "I went to Portugal."

### Fan-out normalization

A subtler problem surfaced empirically (`docs/RIGOR_FINDINGS.md` 2026-04-27):
heavy-but-not-universal entities like `pitchwits` (541 linked memories) on the
test corpus inject off-topic candidates into RRF fusion. The persona problem
was fixed via the persona-exception above; the same mechanism for non-persona
hubs is `sa_fanout_norm_exponent`:

```python
# Each entity's per-memory contribution is divided by num_linked**exp.
# exp=0.5 (sqrt, default): gentle normalization
# exp=1.0 (strict): aggressively suppresses high-degree entities

norm = 1.0 / (len(linked) ** fanout_norm_exponent)
per_memory_contrib = act * norm
```

A 500-linked entity at exp=0.5 contributes `act/22` per memory; at exp=1.0 it
contributes `act/500`. The empirical result on `meridian_deep` showed strict
1/N normalization moves `recall_any@5` from 80.0% (full @ exp=0.5) to 86.7%
(full @ exp=1.0) and turns the associative path from net-harmful (+6.7pp when
ablated) to neutral (0.0pp when ablated). The same knob exists in parallel for
the graph path (`graph_fanout_norm_exponent`, default 0.0 for back-compat).

---

## 14. MESU pruning + EWC protection

### MESU (Metaplasticity from Synaptic Uncertainty)

`src/engine/core/pruning.py`. Inspired by the 2025 Nature Communications MESU
paper. Memories with high signal variance (DA / NE / usage swinging
unpredictably) are "uncertain about their value" and become more prunable.
Memories with low variance are confident and protected.

```python
def compute_signal_uncertainty(da_history, ne_history, usage_history,
                               min_samples=3) -> float:
    if len(da_history) < min_samples:
        return 0.5
    variances = []
    for history in [da_history, ne_history, usage_history]:
        if len(history) >= min_samples:
            window = history[-20:]
            mean = sum(window) / len(window)
            var = sum((x - mean) ** 2 for x in window) / len(window)
            variances.append(var)
    avg_variance = sum(variances) / len(variances)
    return min(1.0, avg_variance / 0.1)


def compute_prune_probability(temperature, uncertainty, retrieval_importance,
                              access_count, cycles_since_last_access,
                              base_prune_rate=0.05) -> float:
    vulnerability = (1.0 - temperature) * uncertainty
    recency_penalty = min(1.0, cycles_since_last_access / 50.0)
    protection = retrieval_importance * (1.0 - uncertainty)
    prob = base_prune_rate * vulnerability * recency_penalty * (1.0 - protection)

    # Hard floors
    if temperature > 0.15:                  # never prune above 0.15
        prob = 0.0
    if cycles_since_last_access < 3:        # never prune if accessed in last 3 cycles
        prob = 0.0
    return max(0.0, min(1.0, prob))
```

### EWC (Elastic Weight Consolidation)

`src/engine/core/protection.py`. Inspired by Kirkpatrick et al. (2017).
Memories that frequently appear in top_k retrieval results accumulate
"retrieval importance" that protects them from compression and pruning:

```python
def update_retrieval_importance(was_in_top_k, current_importance,
                                learning_rate=0.1, decay_rate=0.01) -> float:
    if was_in_top_k:
        new_importance = current_importance + learning_rate * (1.0 - current_importance)
    else:
        new_importance = current_importance * (1.0 - decay_rate)
    return max(0.0, min(1.0, new_importance))


def compute_compression_resistance(retrieval_importance, uncertainty,
                                   temperature) -> float:
    resistance = retrieval_importance * (1.0 - uncertainty)
    if temperature > 0.6:
        resistance = max(resistance, 0.3)        # warm memories get baseline
    return resistance
```

EWC was previously updated *per query, scanning up to 500 memories per recall*
— O(N) per recall with two failure modes: (a) latency and write contention
scale with corpus size, (b) the per-query decay made the importance landscape
a function of query throughput rather than time. The fix
(`ewc_update_max_decay = 0`) only updates retrieved memories per query; decay
runs once per consolidation cycle.

---

## 15. Empirical findings

Full log in `docs/RIGOR_FINDINGS.md`. Headline numbers:

### Determinism

Under `recall_mutates_state=False`, retrieval is bit-identical across runs
(SD = 0pp at k ∈ {5, 10, 25}, n=15 probes × 5 repeats on `meridian_deep`).
This is what makes the benchmark numbers stable point estimates rather than
noisy draws.

### Architecture vs plain RAG (vector-only baseline)

Same ingest, different retrieval stack. Persona-aware aged corpus
(`meridian_deep`, 17,822 memories, 85% archived after 12 consolidation
cycles, n=15 probes):

| k | metric | full | plain_rag | Δ |
|---|---|---|---|---|
| 5  | recall_any  | 80.0% | 80.0% | 0.0pp |
| 5  | recall_all  | 40.0% | 33.3% | **+6.7pp** |
| 5  | recall_frac | 0.607 | 0.560 | **+0.047** |
| 10 | recall_any  | 93.3% | 86.7% | **+6.7pp** |
| 10 | recall_all  | 46.7% | 40.0% | **+6.7pp** |
| 10 | recall_frac | 0.720 | 0.627 | **+0.093** |
| 25 | recall_any  | 93.3% | 93.3% | 0.0pp |

Convergence at k=25 means the architecture is doing **ranking** (lifting good
candidates higher in the list), not **candidate discovery** (finding things
plain RAG misses). Honest framing: *"Dendric matches plain RAG on short-
horizon benchmarks at no quality cost; on aged-corpus retrieval where archive
and lifecycle modulation are active, it gains +6.7pp recall_any@10 over a
vector-only baseline. Value is in re-ranking."*

After the `sa_fanout_norm_exponent=1.0` fix (2026-04-27), the @5 lead also
opens: full → 86.7% any, 0.640 frac, vs plain_rag's 80.0% / 0.560.

### Path ablation (leave-one-out, meridian_deep n=15, after fan-out fix)

Δ vs `full` at k=5; smaller magnitude = path is more redundant:

| config | Δ recall_any@5 | reading |
|---|---|---|
| `plain_rag`       |  0.0pp | vector alone hits the same `any` @ k=5 |
| `no_vector`       | -33.3pp at decay=0.7 | vector is the load-bearer |
| `no_keyword`      | -6.7pp to -20.0pp | keyword adds real signal |
| `no_associative`  |  0.0pp (post-fix) | path no longer hurts; was +6.7pp pre-fix |
| `no_graph`        | +6.7pp / 0.0pp | path is roughly neutral |

### LongMemEval stratified (n=30, per-question fresh DB, 1 cycle)

| k | recall_any | recall_all | recall_frac |
|---|---|---|---|
| 5  | 83.3% | 53.3% | 0.663 |
| 10 | 90.0% | 63.3% | 0.742 |
| 25 | 93.3% | 70.0% | 0.814 |

By question type at k=10:

| type | any | all |
|---|---|---|
| single-session-user        | 100% | 100% |
| single-session-assistant   | 100% | 100% |
| single-session-preference  | 100% | 100% |
| knowledge-update           | 100% |  80% |
| temporal-reasoning         |  80% |   0% |
| multi-session              |  60% |   0% |

100% on three categories is a strong substrate result; 0% `recall_all` on
multi-session and temporal-reasoning is the honest weakness — these require
*aggregation across turns*, which is a generation-time problem on top of the
retrieval substrate. The substrate "finds the right session" (60-80% any) but
"surfacing every scattered mention" (recall_all) is harder.

LongMemEval is per-question fresh-haystack with one consolidation cycle, which
is the regime where lifecycle modulation has nothing to act on — it's the
opposite extreme from `meridian_deep`. The honest reading:

> **Dendric's architecture is benchmark-dependent.** On per-question fresh-
> haystack benchmarks (LongMemEval), full ≈ plain RAG. On aged-corpus
> retrieval where archive and lifecycle modulation are active
> (meridian_deep), full earns +6.7pp at small k.

### Parameter sensitivity (sweeps, 2026-04-27)

Of the three nominally-tunable lifecycle knobs:

| knob | meridian_deep | longmemeval | reading |
|---|---|---|---|
| `archive_rrf_boost` | 1.0–1.8 inert; 2.2 → −20pp | inert across all | "highest safe value", not "optimum" |
| `archive_trigger_threshold` | inert across [0.0, 0.9] | inert across [0.0, 0.9] | binary at default sa_decay; not earning its keep as a continuous knob |
| `mod_temp_lift` | 0.0 → −13.3pp; 0.4–0.8 plateau | inert (no aged data) | the real tilt; load-bearing on aged corpora |

The big surprise was a *hidden 4th knob*: `sa_decay` (the spreading-activation
per-hop multiplier, previously a function default) is the actual driver. On
`meridian_deep` decay=0.5 → 0.7 gave +7.8pp recall_frac@5 (0.562 → 0.640);
on LongMemEval the same change gave +1.1pp (within noise).

### Region distribution after backdated 1-year ingest + 12 consolidation cycles

| region | count | % |
|---|---|---|
| hippocampus | 0 | 0% |
| neocortex | 2,692 | 15% |
| archive | 15,130 | 85% |

Pruning rate asymptotes to ~25 prunes/cycle by cycle 12 — the system isn't
draining; the lifecycle is doing real work.

### Reproducibility

End-to-end Docker stack: `docker compose build && seed_synthetic_corpus &&
recall_at_k` runs clean on 10-gold + 80-distractor synthetic corpus, hits
recall_any@5 = 100% as expected (markers are distinctive — this is *harness
validation*, not a benchmark). Full recipe in `REPRODUCE.md`.

---

## 16. Configuration surface and ablation harness

`src/engine/config.py` is the single source of truth for tunable parameters.
The presets pattern lets the ablation harness construct comparable configs:

```python
def plain_rag_config(db_url: str, persona: str = "") -> EngineConfig:
    """Baseline: vector-only retrieval, no lifecycle modulation,
    no associative/graph/keyword paths, no temporal decomposition,
    no archive trigger. Standard RAG."""
    return EngineConfig(
        db_url=db_url, persona=persona, recall_mutates_state=False,
        enable_keyword_path=False, enable_associative_path=False,
        enable_graph_path=False,
        enable_temporal_decomposition=False, enable_temporal_rerank=False,
        enable_lifecycle_modulation=False,
        keyword_weight=0.0, associative_weight=0.0, graph_weight=0.0,
        archive_rrf_boost=1.0, archive_trigger_threshold=0.0,
    )


def leave_one_out_config(disabled_path: str, db_url: str, persona: str = ""):
    """Full config with exactly one retrieval path disabled.
    Used by path_ablate.py to isolate each path's marginal contribution.
    disabled_path ∈ {'vector', 'keyword', 'associative', 'graph'}."""
    overrides = {
        "enable_vector_path": True,
        "enable_keyword_path": True,
        "enable_associative_path": True,
        "enable_graph_path": True,
    }
    overrides[f"enable_{disabled_path}_path"] = False
    return EngineConfig(db_url=db_url, persona=persona,
                        recall_mutates_state=False, **overrides)
```

Harnesses in `src/scripts/`:

| Script | What it does |
|---|---|
| `recall_at_k.py` | Substrate-only `recall_{any,all,frac}@k`. Two corpora (longmemeval, meridian_deep). Configurable via `--config full|plain_rag|no_<path>` |
| `path_ablate.py` | Runs all 4 leave-one-out configs + plain_rag + full as separate subprocesses, summarizes deltas |
| `param_sweep.py` | 1D and 2D parameter sweeps with `--reuse-ingest` to amortize per-question DBs |
| `per_signal_recall_check.py` | Three design-invariant tests (lifecycle modulation, spreading activation, déjà-vu) |
| `stability_check.py` | Repeated runs to verify SD = 0 under deterministic config |
| `dejavu_diagnostic.py` | Instruments per-entity activation values across probes |
| `associative_diagnostic.py` | Per-probe top-k inspection, full vs no_associative |
| `seed_synthetic_corpus.py` | Deterministic 10-gold + 80-distractor corpus for harness validation |

The ablation harness pattern is "one preset per config, identical ingest,
identical query path, only retrieval differs" — measured deltas attribute to
retrieval architecture, not "different systems."

---

## 17. Honest limitations

The findings document is explicit about what doesn't yet work:

- **`archive_trigger_threshold` is effectively binary at default sa_decay=0.5.**
  Only seed entities ever cross the 0.7 threshold; hop-1 neighbors max out at
  0.5, hop-2 at 0.25. Tuning the threshold across [0.0, 0.9] produces
  byte-identical recall@k. It either needs to be removed from the "tuned
  parameters" list or paired with sa_decay ≥ 0.7.
- **Multi-session and temporal-reasoning categories underperform on
  LongMemEval** (60% / 80% any@10, 0% all@10). Surfacing every scattered
  mention requires aggregation; the substrate finds the right *session* but
  not every relevant *turn*.
- **The associative path is fragile to dense personal corpora** without strict
  fan-out normalization. Default `sa_fanout_norm_exponent=0.5` (sqrt) is
  gentle enough that 500-linked hub entities still inject off-topic candidates;
  exp=1.0 fixes meridian_deep but increases dependency on the keyword path.
  The right operating point likely lives in [0.7, 0.85] and hasn't been swept.
- **Annotations on `meridian_deep` are author-authored**, not blind. Valid for
  development, weaker for publication claims.
- **n=15 (meridian_deep) and n=30 (longmemeval-stratified) are small**;
  one probe = 6.7pp / 3.3pp granularity. ±6.7pp deltas should be read as
  "worth investigating," not "statistically significant."
- **No long-horizon (100+ cycle) consolidation stability test yet.** The
  asymptote at ~25 prunes/cycle suggests stability, but hasn't been confirmed
  past cycle 12.

---

## 18. References

The mechanism design draws explicitly on:

- **ACT-R**: Anderson & Lebiere (1998). *The Atomic Components of Thought.*
  Power-law base activation, fan effect.
- **DA additive boost**: Zhang & Berridge (2009). *What is the role of
  dopamine in reward: hedonic impact, reward learning, or incentive salience?*
- **NE inverted-U**: Avery et al. (2013). *A unified theory of brainstem
  catecholaminergic dynamics.* Dual-receptor pharmacology.
- **GANE**: Mather, Clewett, Sakaki & Harley (2016). *Norepinephrine
  ignites local hotspots of neuronal excitation.* Winner-take-more / loser-
  take-less competitive amplification.
- **GABA hybrid**: Mitchell & Silver (2003) for GABA_A divisive shunting;
  Prescott & De Koninck (2003) for GABA_B subtractive hyperpolarization.
- **Noise-arousal coupling**: Dancy et al. (2015). *A computational model
  of arousal effects on attention.*
- **EWC**: Kirkpatrick et al. (2017). *Overcoming catastrophic forgetting
  in neural networks.* PNAS.
- **MESU**: *Metaplasticity from Synaptic Uncertainty.* Nature Communications
  (2025).
- **RRF**: Cormack, Clarke & Buettcher (2009). *Reciprocal Rank Fusion
  Outperforms Condorcet and Individual Rank Learning Methods.*

The implementation choices (additive vs multiplicative DA, max-aggregation vs
sum in spreading activation, fan-out exponent, archive RRF boost) are
empirically motivated and documented inline in the source.

---

*Generated 2026-05-03. For empirical updates, see
[`docs/RIGOR_FINDINGS.md`](RIGOR_FINDINGS.md). For the public-facing summary,
see the project [`README.md`](../README.md).*
