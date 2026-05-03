# Rigor & Validation Findings — Running Notes

Log of concrete findings during the phased rigor program for Dendric +
Meridian. Each entry is a result + its caveat + what it unlocks or blocks.
Goal: accumulate the material a workshop/conference paper would cite.

Format: newest on top. Each finding has:
- **What** (concrete result)
- **How measured** (repro command or script)
- **Caveat** (known limits, don't oversell)
- **Implication** (what this changes about subsequent work)

---

## 2026-04-27

### Parameter sensitivity sweeps — two of three knobs are inert, one is at boundary

**What.** 1D sweeps of three lifecycle knobs against both corpora.
For each knob, 5 grid points (default value included as anchor),
`config=full` for everything else. LongMemEval used `--reuse-ingest`
to amortize the per-question DBs from the path-ablation run.

**Knobs swept:**
- `archive_rrf_boost` (default 1.8): [1.0, 1.4, 1.8, 2.2, 2.6]
- `archive_trigger_threshold` (default 0.7): [0.0, 0.3, 0.5, 0.7, 0.9]
- `mod_temp_lift` (default 0.4): [0.0, 0.2, 0.4, 0.6, 0.8]

**Cross-corpus pattern (any@5):**

| knob | meridian_deep (n=15) | longmemeval (n=30) |
|---|---|---|
| `archive_rrf_boost` | 1.0–1.8 inert; 2.2 → −20pp; 2.6 → −20pp | 1.0–2.6 **completely inert** |
| `archive_trigger_threshold` | 0.0–0.9 **completely inert** | 0.0–0.9 **completely inert** |
| `mod_temp_lift` | 0.0 → −13.3pp; 0.2 → −6.7pp; 0.4–0.8 plateau | 0.0–0.8 **completely inert** (one-question blip at 0.2) |

**The blunt read.** Of the three "tunable" lifecycle parameters,
**two are non-functional in their advertised range** on the corpora
tested:

1. **`archive_trigger_threshold` does literally nothing on either corpus.**
   Every threshold value 0.0–0.9 produces byte-identical results. Strong
   suggestion that the spreading-activation path never crosses the
   threshold on any test query, so the gate never engages. The déjà-vu
   mechanism may not be firing at all on this probe set. (Diagnostic
   pending — see follow-up entry below.)

2. **`archive_rrf_boost` is inert below default and harmful above.** The
   1.0–1.8 range produces identical results on meridian_deep — meaning
   no archive memory crosses into top-5 at any of those values. Above
   1.8 the boost starts wrongly dominating (-20pp at 2.2). On
   LongMemEval the parameter is completely inert — consistent with
   per-question fresh-haystack benchmarks not exercising archive at all.
   So 1.8 is effectively *"the highest safe value"* not *"the optimum"*
   — there is no optimum because the boost only matters above the
   harmful threshold.

3. **`mod_temp_lift` is the one knob doing real work** — but only on
   meridian_deep, only below the default. 0.0 hurts (-13.3pp), 0.2 is
   intermediate (-6.7pp), 0.4–0.8 plateau. Validates that lifecycle
   modulation is load-bearing on aged corpora. Inert on LongMemEval
   because nothing has aged into the modulation regime (1 consolidation
   cycle, fresh DB).

**How measured.**
```bash
# meridian_deep
python -m src.scripts.param_sweep --corpus meridian_deep \
    --db postgresql://localhost:5432/meridian_deep \
    --annotations /Users/rileycoleman/meridian/probes/meridian_recall_gold.json \
    --k 5,10,25 --output-dir param_sweep_results/meridian_deep

# longmemeval (re-uses recall_per5_* DBs from path-ablation run)
docker compose run --rm dendric python -m src.scripts.param_sweep \
    --corpus longmemeval --per-type 5 --reuse-ingest \
    --db-prefix postgresql://postgres:postgres@db:5432/recall_per5 \
    --k 5,10,25 --output-dir param_sweep_results/longmemeval
```

Driver: `src/scripts/param_sweep.py`. Subprocess per grid point. The
`--reuse-ingest` flag re-uses already-populated per-question DBs across
sweep points (retrieval-only knobs don't change ingested data), turning
a 3-hour LongMemEval sweep into ~10 min.

**Caveat.**
- 1D sweeps assume independence. If knobs interact (e.g. boost only
  matters when threshold is below some value), 1D misses the interaction.
  Inert results in particular could be hiding 2D structure: threshold
  may be inert because boost is too low to surface archive hits even
  when threshold is crossed, or vice versa. Targeted 2D follow-up
  warranted on the boost × threshold pair specifically.
- meridian_deep n=15 means 1 probe = 6.7pp granularity. The 0.2
  mod_temp_lift result on LongMemEval is one question flipping.
- A noted methodology issue: cross-day OpenAI embedding drift caused
  one meridian_deep probe ("when is mary's brother visiting") to flip
  from frac=1.000 (yesterday) to frac=0.333 (today) at default config.
  Within-day variance is still SD=0; cross-day requires recording the
  embedding model version + run date. Doesn't invalidate the within-run
  comparisons that drive the sweep findings, but worth flagging.

**Implication.**
- **Most publication-blocking finding:** the architecture has
  parameters that may not be doing the work the architecture claims
  they do. Cannot honestly write "we tuned `archive_trigger_threshold`
  to 0.7" when 0.0 and 0.9 produce identical results.
- **Defensible salvage:** the architecture's value is in lifecycle
  modulation (`mod_temp_lift` at 0.4+ on aged corpora). That's a
  narrower but real claim.
- **Required follow-up before publication:** instrument
  spreading-activation values to verify whether the threshold is ever
  crossed on real queries. (Done — see next entry.)
- **Defensible parameter ranges for the paper:**
  - `archive_rrf_boost`: 1.0–1.8 indistinguishable; recommend 1.0
    (Occam) or document why 1.8 was chosen as defensive ceiling
  - `archive_trigger_threshold`: needs investigation; current default
    is unfalsifiable
  - `mod_temp_lift`: 0.4 is reasonable; 0.6 ties so 0.4 is fine

---

### Fan-out fix on associative path — `full` recovers from net-harmful to net-positive

**What.** Re-ran the path ablation on meridian_deep with
`sa_fanout_norm_exponent=1.0` (strict `1/num_linked`) instead of the
production default 0.5 (sqrt). The change is parallel to the original
diagnostic finding: the gentle sqrt normalization didn't tame
hub-but-not-universal entities like `pitchwits` (843 linked memories)
and `riley` (persona).

**recall@5 comparison (15 probes):**

| config | sa_fanout=0.5 (production) | sa_fanout=1.0 |
|---|---|---|
| `plain_rag` | any=80.0%, frac=0.560 | any=80.0%, frac=0.560 |
| `full` | any=80.0%, frac=0.607 | any=**86.7%**, frac=**0.640** |
| `no_associative` | +6.7pp (path hurts) | 0.0pp (neutral) |
| `no_graph` | +6.7pp (path hurts) | 0.0pp (neutral) |

**recall@10:**

| config | sa_fanout=0.5 | sa_fanout=1.0 |
|---|---|---|
| `plain_rag` | any=86.7%, frac=0.627 | any=86.7%, frac=0.627 |
| `full` | any=93.3%, frac=0.720 | any=93.3%, frac=0.720 |

**Two real wins.**

(a) `full` improves +6.7pp recall_any@5 (80.0 → 86.7) and
+0.033 recall_frac@5 (0.607 → 0.640) at the same `plain_rag` floor.
The architecture's lead over plain RAG widens from 0pp at k=5 to
+6.7pp at k=5, matching the existing +6.7pp at k=10.

(b) The associative path goes from net-harmful (+6.7pp when ablated)
to neutral (0.0pp when ablated). The architecture is no longer
*hurt* by its own complexity. `full` now matches `no_associative`,
which means the "best disable" is no longer needed.

**One caveat.** The fan-out fix shifts the dependency structure:
`no_keyword` deteriorated from -6.7pp at exp=0.5 to -20.0pp at
exp=1.0. With the associative path's contribution suppressed, the
system relies more on keyword. The total information across paths
is unchanged; what's changed is that the associative path stopped
*adding noise* into RRF fusion.

**How measured.**
```bash
python -m src.scripts.path_ablate --corpus meridian_deep \
    --db postgresql://localhost:5432/meridian_deep \
    --annotations /Users/rileycoleman/meridian/probes/meridian_recall_gold.json \
    --override sa_fanout_norm_exponent=1.0 --k 5,10,25 \
    --output-dir ablation_results/meridian_deep_sa_fanout_1
```

**Caveat (general).**
- Tested only on meridian_deep n=15. The fix may shift behavior on
  LongMemEval; need to re-run there to confirm no regression.
- Strict 1/num_linked is one operating point in a continuous space;
  intermediate values (0.7, 0.85) may give similar improvement with
  less collateral effect on the keyword-path dependency.
- The fix doesn't address the deeper issue surfaced by the
  associative-diagnostic deep-dive: some off-topic memories get
  entity-tagged at ingest with entities they don't really discuss.
  E.g., the "totally right — you can't price this from a whiteboard"
  memory got linked to both `pitchwits` and `role` despite being
  about pricing strategy, not roles. The fan-out fix dampens the
  *retrieval* impact of this; it doesn't fix the *tagging accuracy*
  upstream.
- The graph path's parallel `graph_fanout_norm_exponent` (added in
  the same commit) was tested but didn't move probe-level results
  on this same probe set: graph_recall scales scores uniformly
  within the path, so relative ranking inside the path is unchanged.
  Left in as future-work scaffolding.

**Implication.**
- **Defensible publication claim:** "Dendric's full architecture
  beats vector-only RAG by +6.7pp recall_any at k=5 *and* k=10 on
  aged personal corpora." Stronger than the previous "wins at k=10
  only" position.
- **The fan-out exponent should probably be promoted to a
  documented hyperparameter**, not left as a buried function
  default. Worth running a sensitivity sweep across exp ∈
  [0.5, 0.7, 0.85, 1.0] to find the right operating point — the
  jump from 0.5 to 1.0 is a big step and the optimum is probably
  in between.
- **Architecture story update:** the associative path's
  spreading-activation mechanism is correct in principle, the
  implementation just needed proper fan-out suppression on dense
  personal corpora. The fix is a one-line code change with a
  one-line config knob. Doesn't require re-architecting.

---

### Why is the associative path harmful — direct top-k diagnostic

**What.** Per-probe inspection of top-10 contents under `full` vs
`no_associative` configurations on meridian_deep at production decay.
Two hypotheses to discriminate:

  - (a) Archive memories pulled by the déjà-vu trigger displace good
    candidates out of top-k
  - (b) Off-topic memories pulled via spreading activation through
    high-fan-out entities (persona, frequently-mentioned topics)
    displace gold

**Result: hypothesis (b) is correct, (a) is wrong.**

- **Total archive memories across all 15 probes' top-10:** 2 in `full`,
  0 in `no_associative`. The déjà-vu trigger almost never surfaces
  archive memories into top-10.
- **The displacement IS happening, via off-topic entity-linked
  candidates.** Concrete example, probe 6 ("what is my role at
  pitchwits"):

  *full* top-5 ranks 4-5:
  - 4. `vector,keyword`  Riley's sharing their new Pitchwits employment agreement... ★ (gold)
  - 5. `associat,graph`  totally right — you can't price this from a whiteboard...

  *no_associative* top-4:
  - 4. `vector,keyword`  Riley's sharing their new Pitchwits employment agreement... ★ (gold)

  In `full`, an off-topic memory about pricing/strategy was inserted
  at rank 5 because it's linked to the `pitchwits` entity in the graph;
  this displaces the gold memory from rank 4 down to rank 6 (out of
  top-5).

**Why the fan-out normalization isn't enough.** The associative path
normalizes each entity's contribution by `1/sqrt(num_linked)`. The
`pitchwits` entity has 541 memories linked to it (per earlier
diagnostic). Per-memory contribution is `1.0 / sqrt(541) ≈ 0.043`.
That's small per memory, but applied across hundreds of pitchwits-
linked memories, the cumulative RRF rank contribution is enough to
push topically-irrelevant candidates above gold for entity-tight
queries. The normalization penalizes the persona-style hub-entity
problem (one entity linked to ~everything) but not the next layer
down: heavy-but-not-universal entities like `pitchwits`, `mary`,
`riley`'s topical clusters.

**How measured.**
```bash
DATABASE_URL=postgresql://localhost:5432/meridian_deep PERSONA=riley \
    python -m src.scripts.associative_diagnostic
```

Diagnostic: `src/scripts/associative_diagnostic.py`. Runs the full
recall pipeline twice per probe (full vs no_associative leave-one-out
preset), prints top-10 with retrieval_paths and gold-hit markers,
flags any probe where the @5 hit-status diverges, and aggregates
archive count.

**Caveat.**
- Inspected 15 meridian_deep probes; the n is small but the *mechanism*
  finding (off-topic entity hits, not archive displacement) is robust:
  archive count is 2 vs 0 across the whole sample, so hypothesis (a)
  is just empirically weak.
- This diagnostic uses `recall_at_k`'s gold-substring matching; if a
  gold marker happens to appear in an off-topic memory we'd miscount
  it. The probe-6 inspection was eyeballed to confirm the mechanism;
  scaling the diagnostic would require a more rigorous matching test.
- The fan-out math is the proximate cause; the deeper question is
  whether spreading activation through entity-linked memories is the
  right retrieval signal at all for first-person queries about
  agent-state ("what is my role"), where entity-graph connectivity
  is high but topical alignment is low.

**Implication.**
- The associative path's harm is fixable in principle without
  removing the path: tighten the fan-out normalization (e.g.,
  `1/num_linked` instead of `1/sqrt(num_linked)`, or a hub-entity
  whitelist that suppresses pitchwits/persona-cluster contribution
  on first-person queries).
- But: each fix risks breaking the cases where the path *does* help
  (e.g., probe 9 "when is the tuscany ceremony" surfaces gold at
  rank 1-3 in full via vector+associat, whereas no_associative also
  hits at rank 1-3 — the path doesn't lose info there but doesn't
  add either).
- **Honest publication position:** the associative path's
  spreading-activation mechanism is correct in principle but the
  implementation's fan-out tolerance is miscalibrated for high-
  density personal corpora. We can either (i) fix and re-evaluate,
  or (ii) report the finding and propose the fix as future work.
  Either is defensible; (i) is more useful but expands the scope.

---

### sa_decay sweep on LongMemEval — confirms regime-dependence

**What.** 1D sweep of `sa_decay` ∈ {0.3, 0.5, 0.7} on LongMemEval n=30
stratified, retrieval-only via `--reuse-ingest` against the existing
`recall_per5_*` per-question DBs.

| value | any@5 | all@5 | frac@5 |
|---|---|---|---|
| 0.3 | 83.3% | 56.7% | 0.686 |
| 0.5 (production) | 83.3% | 56.7% | 0.686 |
| 0.7 | 83.3% | 60.0% | 0.697 |

**Decay barely moves the needle on LongMemEval.** Compare to
meridian_deep where decay=0.5 → 0.7 gave +7.8pp recall_frac@5
(0.562 → 0.640). Here it gives +1.1pp (0.686 → 0.697) — one question
flipping at n=30, right at the noise floor.

**Implication.** The decay parameter's value-add is regime-dependent
exactly as the biological framing predicts: lifecycle parameters
earn their keep when there's a lifecycle. LongMemEval's per-question
fresh haystacks (1 ingest, 1 consolidation cycle, no aged archive)
don't engage spreading-activation's archive-trigger pathway, so the
decay value that gates archive accessibility is irrelevant.

**Caveat.**
- The `recall_per5_*` DBs were ingested at `sa_decay=0.5` baseline.
  Decay only affects retrieval, so reusing them is fair; but the
  one-cycle consolidation also can't show decay's effect on long-
  horizon retrieval, by construction.
- n=30 LongMemEval is still small. The 1-question blip at decay=0.7
  could be noise either way.

**How measured.**
```bash
docker compose run --rm dendric python -m src.scripts.param_sweep \
    --corpus longmemeval --per-type 5 --reuse-ingest \
    --db-prefix postgresql://postgres:postgres@db:5432/recall_per5 \
    --knobs sa_decay --k 5,10,25 \
    --output-dir param_sweep_results/longmemeval_decay
```

**Combined cross-corpus picture (recall_frac@5 at decay=0.5 vs 0.7):**

| corpus | decay=0.5 | decay=0.7 | Δ |
|---|---|---|---|
| meridian_deep | 0.562 | 0.640 | +0.078 |
| LongMemEval | 0.686 | 0.697 | +0.011 |

The 7× gap in decay-sensitivity between corpora is the regime-dependence
claim, sharpened.

---

### Path ablation at sa_decay=0.7 — associative path is net-harmful on meridian_deep

**What.** Re-ran the 4-path leave-one-out ablation on meridian_deep
with `sa_decay=0.7` (the better decay value from the 2D sweep) instead
of the production default 0.5, holding all other knobs at defaults.
Compared deltas to the production-decay run from 2026-04-25.

**recall@5 deltas vs `full` (smaller magnitude = path is redundant):**

| config | decay=0.5 (production) | decay=0.7 |
|---|---|---|
| `plain_rag` | 0.0pp any | 0.0pp any |
| `no_vector` | -20.0pp | **-33.3pp** |
| `no_keyword` | -6.7pp | -6.7pp |
| `no_associative` | **+6.7pp (hurts)** | **+6.7pp (hurts)** |
| `no_graph` | +6.7pp (hurts) | 0.0pp (neutral) |

**Two findings.**

**(a) Vector becomes more dominant at decay=0.7, not less.** The
`no_vector` ablation drops further (-33.3pp vs -20.0pp), and the
*other* non-vector paths shrink in their apparent contribution. So
moving to the better decay value doesn't rescue the redundant paths;
it makes vector even more singular as the load-bearing path.

**(b) The associative path is net-harmful at *both* decay values.**
`no_associative` beats `full` by +6.7pp recall_any@5 in both runs.
At decay=0.7, `no_associative` reaches recall_frac@5=0.640 — the
same number as the 2D-sweep optimum (threshold=0.3, decay=0.7). In
other words: the "best operating point" in the 2D sweep is the one
where the associative path's subtractive effect is minimized via
threshold=0.3, but disabling the path entirely matches that ceiling.

**How measured.**
```bash
python -m src.scripts.path_ablate --corpus meridian_deep \
    --db postgresql://localhost:5432/meridian_deep \
    --annotations /Users/rileycoleman/meridian/probes/meridian_recall_gold.json \
    --override sa_decay=0.7 --k 5,10,25 \
    --output-dir ablation_results/meridian_deep_decay07
```

`path_ablate.py` gained a `--override` pass-through flag in the same
session.

**Caveat.**
- Same n=15, same 1-probe-flips noise floor as the 2D sweep. The
  "associative is harmful" finding is consistent across decay values
  (so it's not noise on one decay), but the magnitude could shrink
  on a larger probe set.
- Only tested on meridian_deep. The path ablation on LongMemEval at
  decay=0.5 showed associative neutral (0.0pp) — could shift again
  at decay=0.7. Worth a re-run.
- We did not test what happens when the associative path is disabled
  AND threshold is moved to 0.3 simultaneously — possible those two
  effects don't compose linearly.

**Implication.**
- The architecture's claimed value via spreading-activation /
  associative retrieval is **structurally underperforming on aged
  corpora**. The associative path is the second-most-complex code
  path (entity graph BFS + max-aggregation + déjà-vu archive trigger
  + persona fallback) and it's actively subtractive at retrieval
  time.
- **Honest paper position:** vector is doing nearly all the work,
  keyword adds modest signal, graph is roughly neutral, and
  associative is harmful at the production decay default. The
  4-path-fusion claim is overstated in the implementation, even if
  the design intent (associative path enables aged-archive retrieval
  inaccessible to vector) is sound.
- **Architectural follow-up worth doing before publishing:** trace
  why associative hurts. Two hypotheses:
  - The path returns *too many* archive memories (per the diagnostic,
    most probes pull 50+ archive memories from one entity), and even
    after RRF fusion their cumulative weight pushes good non-archive
    candidates out of top-5
  - The fan-out normalization (1/sqrt(num_linked)) doesn't penalize
    high-degree entities enough; queries about persona-adjacent topics
    pull persona-linked archive noise

---

### Threshold × decay 2D sweep — real driver is decay, threshold remains inert

**What.** Targeted 2D grid: `archive_trigger_threshold ∈ [0.0, 0.3, 0.5,
0.7, 0.9]` × `sa_decay ∈ [0.3, 0.5, 0.7]`, on meridian_deep (n=15). To
make `sa_decay` and `sa_max_hops` configurable through the override
mechanism they had to be promoted from hardcoded function args into
`EngineConfig` fields and threaded through `engine.recall()`.

**recall_frac@5:**

| threshold ↓ / decay → | 0.3 | 0.5 (default) | 0.7 |
|---|---|---|---|
| 0.0 | 0.562 | 0.562 | 0.573 |
| 0.3 | 0.576 | 0.540 | **0.640** |
| 0.5 | 0.576 | 0.540 | 0.573 |
| 0.7 (default) | 0.576 | 0.540 | 0.573 |
| 0.9 | 0.576 | 0.540 | 0.573 |

**Two findings.**

**(a) The threshold is mostly inert *within* each decay column.** At
decay=0.5 (production default), every threshold ≥ 0.3 gives the same
result. At decay=0.3, ditto. Only at decay=0.7 does the threshold show
a single discriminating step (0.3 → 0.5 drops frac from 0.640 to 0.573);
above 0.5 it's inert again. This corroborates the diagnostic finding:
non-seed entities crossing the threshold either pull the same archive
memories the seed crossings already pull, or those memories don't make
it into top-5 anyway. The threshold is not a useful tuning knob at any
decay value.

**(b) Decay is the actual driver.** The current default `decay=0.5` is
the **worst** of the three decay values:
- decay=0.5: best frac@5 = 0.562
- decay=0.3: best frac@5 = 0.576 (+1.4pp)
- decay=0.7: best frac@5 = **0.640 (+7.8pp at threshold=0.3)**

So the production default sits at a local minimum on this corpus.
decay=0.7 with threshold=0.3 is +7.8pp recall_frac@5 over current
production — not a small effect.

**How measured.**
```bash
python -m src.scripts.param_sweep --corpus meridian_deep \
    --db postgresql://localhost:5432/meridian_deep \
    --annotations /Users/rileycoleman/meridian/probes/meridian_recall_gold.json \
    --grid2d threshold_x_decay --k 5,10,25 \
    --output-dir param_sweep_results/meridian_deep_2d
```

Driver: `src/scripts/param_sweep.py` with the new `--grid2d` flag and
the `threshold_x_decay` predefined grid in `GRIDS_2D`.

**Caveat.**
- n=15: 1 probe = 6.7pp on `recall_any`, 0.067 on `recall_frac`. The
  +7.8pp gain at decay=0.7 is just slightly above 1-probe noise. Need
  a larger probe set (or LongMemEval re-run with the new sweep range)
  before recommending a default change.
- decay=0.7 with threshold=0.3 may also have side effects we haven't
  measured: more non-seed entities cross the threshold, which means
  more archive memories pulled, which (per the diagnostic at decay=0.9)
  can blow up to runaway scale on highly-connected entities. The 0.640
  result is one operating point; at scale this regime might be more
  brittle.
- The diagnostic showed decay=0.9 has a probe pulling 80k+ archive
  memories on a single entity. We didn't include 0.9 in the recall@k
  sweep because the diagnostic already established it's past the
  useful range.

**Implication.**
- Honest paper claim updates:
  - **`archive_trigger_threshold` should be removed from the "tuned
    parameters" list.** Across 15 (threshold × decay) combinations,
    threshold only changes results in 1 cell. It's not earning its
    keep as a knob.
  - **`sa_decay` (currently buried as a function default) is a
    first-class architectural parameter** and warrants its own
    discussion. The production value 0.5 is suboptimal on the test
    corpus; 0.7 is meaningfully better.
- Publication-blocking → publication-shaping: the original "we tuned
  3 knobs" framing was wrong. The honest framing is "we surfaced a
  hidden 4th knob (decay) which is doing the real work, and one of
  the 3 explicit knobs is structurally redundant."
- **Recommended follow-ups before publishing:**
  - Re-run the path ablation with `sa_decay=0.7` to see if the
    associative path's marginal contribution changes (the n=15
    ablation showed it neutral; might shift to positive at the
    better decay value)
  - Re-run on LongMemEval to check whether decay matters there too
    or only on aged corpora — that's a regime-dependence claim
    worth confirming
  - Probe set expansion: add 5-10 more meridian_deep annotations
    so 1-probe noise floor is below the gains we're trying to claim

---

### Déjà-vu diagnostic — threshold is binary, not graduated

**What.** Direct instrumentation of the spreading-activation path
across all 15 meridian_deep probes, recording per-entity activation
values and which probes cross which thresholds.

| threshold | probes w/ ≥1 entity crossing | probes w/ archive memories available |
|---|---|---|
| 0.0 | 15 / 15 | 15 / 15 |
| 0.3 | 15 / 15 | 14 / 15 |
| 0.5 | 15 / 15 | 14 / 15 |
| 0.7 (default) | 13 / 15 | 12 / 15 |
| 0.9 | 13 / 15 | 12 / 15 |

**The corrected picture.** The earlier "threshold is inert" reading
was wrong about *mechanism*:

- The threshold **IS** being crossed: 13/15 probes have at least one
  entity ≥ 0.7 after spreading.
- Archive memories **ARE** being pulled: spot-checking probe 13
  ("when is mary's brother visiting") shows an archive hit at rank 4
  with `paths=['associative_archive']` — meaning only the déjà-vu
  trigger surfaced it.
- The threshold values 0.0 and 0.9 produce identical recall@k because
  every value in that range crosses on **the same dominant seed
  entities** — the threshold is not a discriminator inside [0.0, 0.9].

**Why.** With `decay=0.5` and `max_hops=2`, the activation values are:
- Seed entities: 1.0 (always)
- Hop-1 neighbors: ≤ 0.5
- Hop-2 neighbors: ≤ 0.25

Only seed entities ever reach the 0.7 default. So the threshold isn't
gating on "strong association" — it's gating on "is this a seed?"
That's binary, not graduated. The two probes that fall below 0.7 are
the persona-fallback cases ("who is my wife", "who is my CEO"),
where persona seeds at 0.5 and is the only seed.

**How measured.**
```bash
DATABASE_URL=postgresql://localhost:5432/meridian_deep PERSONA=riley \
    python -m src.scripts.dejavu_diagnostic
```

Diagnostic: `src/scripts/dejavu_diagnostic.py`. Re-implements the BFS
spread step from `associative.py` to get per-entity activation values
without doing the memory fetch, then queries the entity graph to count
archive memories per crossing entity.

**Caveat.**
- Only measures *whether* the trigger fires, not whether the resulting
  archive hits are *useful*. A trigger that fires but pulls irrelevant
  archive memories is also a problem.
- The "binary, not graduated" framing assumes default `decay=0.5`. With
  weaker decay (e.g. 0.7), hop-1 neighbors could reach 0.7+ and the
  threshold would become a discriminator. So the architectural claim
  could still be repaired by tuning decay rather than rejected.

**Implication.**
- The mechanism does fire. The earlier "publication-blocking" framing
  on threshold inertness was overstated.
- The mechanism's *tunability* claim is what's publication-blocking.
  If `archive_trigger_threshold ∈ [0.0, 0.9]` produces identical
  recall@k, the parameter is not tuning anything observable. Either
  remove it from the paper's "knobs we tuned" list, or lower
  `decay` so that the threshold actually discriminates between
  strongly-associated hop-1 neighbors and weak ones.
- **Defensible reframing for the paper:** the déjà-vu mechanism is
  a binary "if seed entity has archive memories, pull them" — not the
  graduated activation-strength gate the design implies. That's
  honest. Whether the binary version earns its complexity is a
  separate question (path-ablation showed graph path adds ~0
  on LongMemEval and -6.7pp on meridian_deep at k=10).
- **Investigative follow-up:** rerun the path ablation with decay
  values 0.3, 0.5, 0.7, 0.9 on meridian_deep. If decay=0.7+ makes
  hop-1 neighbors reachable at threshold 0.7, the threshold becomes
  a real discriminator and the parameter sweep should be re-run.

---

---

## 2026-04-25

### Path ablation on meridian_deep (n=15) — architecture earns its keep

**What.** Same 4-path leave-one-out ablation as the LongMemEval entry,
run against the aged-corpus regime: meridian_deep DB (~17k memories,
85% archived, 12 consolidation cycles deep, real 12-month timestamps),
15 hand-annotated probes.

**recall@5 (n=15):**

| config | any | all | frac | Δ any vs full | Δ frac vs full |
|---|---|---|---|---|---|
| `plain_rag` | 80.0% | 33.3% | 0.560 | 0.0pp | -0.047 |
| `no_vector` | 60.0% | 26.7% | 0.422 | -20.0pp | -0.184 |
| `no_keyword` | 73.3% | 33.3% | 0.518 | -6.7pp | -0.089 |
| `no_associative` | 86.7% | 40.0% | 0.640 | +6.7pp | +0.033 |
| `no_graph` | 86.7% | 40.0% | 0.627 | +6.7pp | +0.020 |
| `full` | 80.0% | **40.0%** | **0.607** | — | — |

**recall@10 (n=15):** *(this is where `full` wins decisively)*

| config | any | all | frac | Δ any vs full | Δ frac vs full |
|---|---|---|---|---|---|
| `plain_rag` | 86.7% | 40.0% | 0.627 | -6.7pp | -0.093 |
| `no_vector` | 73.3% | 33.3% | 0.538 | -20.0pp | -0.182 |
| `no_keyword` | 86.7% | 40.0% | 0.642 | -6.7pp | -0.078 |
| `no_associative` | 93.3% | 46.7% | 0.720 | 0.0pp | 0.000 |
| `no_graph` | 86.7% | 40.0% | 0.627 | -6.7pp | -0.093 |
| `full` | **93.3%** | **46.7%** | **0.720** | — | — |

**recall@25 (n=15):** all configs converge to 93.3% any / 53.3% all /
0.733 frac except `no_vector` (80.0% any). At generous k everyone finds
the same candidates; the architecture is doing **ranking**, not
candidate discovery.

**The story flips.** Where LongMemEval said "plain_rag dominates," the
aged personal corpus says "the architecture earns its keep at small-k":
- recall_any@10: `full` 93.3% vs `plain_rag` 86.7% (+6.7pp)
- recall_all@10: `full` 46.7% vs `plain_rag` 40.0% (+6.7pp)
- recall_frac@10: `full` 0.720 vs `plain_rag` 0.627 (+0.093)
- Both keyword and graph paths show real contribution (-6.7pp each
  when disabled)
- Vector path remains the load-bearer (-20pp when disabled)

The associative path is the one ambiguous case: it shows neutral or
slightly negative contribution at the meridian_deep slice (no_associative
matches full at k=10). That's worth investigating — the spreading-
activation path *should* be the differentiator on aged corpora where
vector similarity decays.

**How measured.**
```bash
# Host Postgres (no docker — meridian_deep DB lives on host)
python -m src.scripts.path_ablate \
    --corpus meridian_deep \
    --db postgresql://localhost:5432/meridian_deep \
    --annotations /Users/rileycoleman/meridian/probes/meridian_recall_gold.json \
    --k 5,10,25 \
    --output-dir ablation_results/meridian_deep
```

**Caveat.**
- n=15 is small. 1 question = 6.7pp, so the ±6.7pp deltas should be
  read as "worth investigating" not "statistically significant."
- Meridian_deep is single-user, hand-annotated by the author. Not blind.
- Convergence at k=25 means the value is a re-ranking effect, not novel
  candidate retrieval. That's the honest story to tell — not "we find
  more relevant memories" but "we surface them at higher rank."
- The associative path's neutral/negative result deserves a follow-up:
  is spreading activation actually firing on this query set? Most
  meridian_deep probes contain explicit person names, which the entity
  graph path also seeds on, so the marginal contribution may be
  redundant rather than absent.

**Implication.**
- Combined with LongMemEval result: **the architecture's value is
  benchmark-dependent.** On per-question fresh-haystack benchmarks
  (LongMemEval), it costs more than it earns. On aged-corpus retrieval
  (meridian_deep), it earns its complexity at small k.
- Defensible paper claim: **"Dendric matches plain RAG on short-horizon
  benchmarks at no quality cost; on aged-corpus retrieval where archive
  and lifecycle modulation are active, it gains +6.7pp recall_any@10
  over a vector-only baseline. Value is in re-ranking, not candidate
  discovery (gap closes at k=25)."**
- Direction for parameter sensitivity (Phase 2 #6): the small-k regime
  is where the architecture pays off. Sweep parameters at k=5,10
  specifically — not k=25 where everything converges.
- Direction for associative-path investigation: instrument which queries
  trigger spreading activation, and on what entities. If most queries
  share entities with the graph-path seeds, the two paths are
  duplicating coverage rather than complementing.

---

### Path ablation on LongMemEval (n=30 stratified) — uncomfortable result

**What.** 4-path leave-one-out ablation against the LongMemEval n=30
stratified slice (5 questions × 6 types, fresh DB per question, 1
consolidation cycle, full corpus ingest, retrieval-only swap):

**recall@5 (n=30):**

| config | any | all | frac | Δ any vs full |
|---|---|---|---|---|
| `plain_rag` (vector only) | **93.3%** | 56.7% | 0.731 | **+10.0pp** |
| `no_vector` | 63.3% | 40.0% | 0.506 | -20.0pp |
| `no_keyword` | 80.0% | 56.7% | 0.674 | -3.3pp |
| `no_associative` | 83.3% | 56.7% | 0.702 | 0.0pp |
| `no_graph` | 86.7% | 53.3% | 0.699 | +3.3pp |
| `full` | 83.3% | 56.7% | 0.686 | — |

**recall@10 (n=30):**

| config | any | all | frac | Δ any vs full |
|---|---|---|---|---|
| `plain_rag` | **93.3%** | **73.3%** | **0.844** | **+6.7pp** |
| `no_vector` | 66.7% | 50.0% | 0.572 | -20.0pp |
| `no_keyword` | 86.7% | 66.7% | 0.744 | 0.0pp |
| `no_associative` | 90.0% | 70.0% | 0.787 | +3.3pp |
| `no_graph` | 90.0% | 70.0% | 0.793 | +3.3pp |
| `full` | 86.7% | 70.0% | 0.769 | — |

**recall@25 (n=30):**

| config | any | all | frac | Δ any vs full |
|---|---|---|---|---|
| `plain_rag` | 93.3% | **86.7%** | **0.913** | 0.0pp |
| `no_vector` | 76.7% | 53.3% | 0.631 | -16.7pp |
| `no_keyword` | 90.0% | 70.0% | 0.797 | -3.3pp |
| `no_associative` | 93.3% | 76.7% | 0.842 | 0.0pp |
| `no_graph` | 93.3% | 76.7% | 0.842 | 0.0pp |
| `full` | 93.3% | 73.3% | 0.831 | — |

**The blunt read.** On LongMemEval, **plain RAG strictly dominates `full`**
on recall_any at k=5 (+10pp) and k=10 (+6.7pp), and ties at k=25. On
recall_all and recall_frac, plain_rag wins or ties at every k. The vector
path is doing nearly all the work; the other three paths are introducing
fusion noise that bumps good vector hits out of small-k results. Of the
non-vector paths, only keyword shows a real deficit when ablated (at
k=25, -3.3pp); associative and graph are zero-or-negative contributors
on this benchmark.

**How measured.**
```bash
docker compose run --rm dendric python -m src.scripts.path_ablate \
    --corpus longmemeval --per-type 5 \
    --db-prefix postgresql://postgres:postgres@db:5432/recall_per5 \
    --k 5,10,25 --output-dir ablation_results/per_type_5
```

Driver: `src/scripts/path_ablate.py`. Runs each config in a separate
subprocess (LongMemEval data file is 265 MB and parsing it inflates to
~2 GB; in-process looping OOMs the container at config 2). Cleans up
per-question DBs between configs to bound disk usage.

**Caveat.**
- This is a *retrieval* result on LongMemEval specifically. LongMemEval
  has properties that disadvantage Dendric's architecture:
  - Per-question haystacks: archive memories never accumulate, so
    déjà-vu can't fire (gold isn't in archive)
  - Single ingest + 1 cycle: lifecycle modulation has nothing to act on
  - Questions tend to cite specific sessions: trivial vector retrieval
- 1 question × 30 = 3.3pp granularity. The ±3.3pp deltas are noise.
- This benchmark mismatch is the most charitable read. The less charitable
  read: **the architecture's complexity is not earning its keep on the
  one external benchmark we can run.**
- Vector path is the genuine load-bearer (-20pp when disabled).

**Implication.**
- **Re-frame the architecture story.** Cannot honestly claim "Dendric beats
  RAG" based on LongMemEval. Defensible claim is "Dendric matches RAG
  on short-horizon single-user benchmarks at no quality cost; the
  architecture's value lies elsewhere."
- The "elsewhere" needs evidence. meridian_deep (aged corpus, archive
  retrieval, déjà-vu regime) is the natural place — running that next.
- If meridian_deep also shows full ≤ plain_rag, this is a real
  architectural problem, not a benchmark-mismatch. Don't write the paper
  before knowing that answer.
- Worth investigating *why* the non-vector paths add noise rather than
  signal at small k on LongMemEval. RRF k=60 may be too aggressive a
  flattener when paths disagree.

---

### Reproducibility package — Docker + synthetic corpus

**What.** Built a self-contained reproducibility stack:
- `Dockerfile` — Python 3.12-slim runtime with all deps
- `docker-compose.yml` — pgvector/pgvector:pg16 + dendric services
- `src/scripts/seed_synthetic_corpus.py` — deterministic 10-gold +
  80-distractor corpus generator
- `src/scripts/synthetic_gold.json` — generated annotations
- `REPRODUCE.md` — clone-to-results recipe

End-to-end validation: `docker compose build → seed → recall_at_k` runs
clean, all 10 gold queries hit recall_any@5 = 100% on the synthetic
corpus (expected — markers are distinctive; this is harness validation,
not a benchmark).

**How measured.**
```bash
export OPENAI_API_KEY=sk-...
docker compose build
docker compose run --rm dendric python -m src.scripts.seed_synthetic_corpus
docker compose run --rm dendric python -m src.scripts.recall_at_k \
    --corpus meridian_deep \
    --annotations src/scripts/synthetic_gold.json \
    --db postgresql://postgres:postgres@db:5432/dendric \
    --k 5,10,25
```

**Caveat.**
- Synthetic corpus is too easy (n=10, distinctive markers) to discriminate
  retrieval strategies. It's a *harness* test, not a benchmark.
- Image is 9.3 GB (sentence-transformers pulls torch). Worth slimming
  with CPU-only torch wheel if external reviewers will actually pull.
- `data/longmemeval/` gets baked into the image (no `.dockerignore`) —
  264 MB of waste. Fix before publishing widely.
- Reproducing the LongMemEval numbers requires gated dataset access;
  reproducing meridian_deep numbers is impossible (private corpus).

**Implication.**
- A reviewer with no Dendric context can clone, build, and run the
  synthetic eval in under 10 minutes. That's the table-stakes
  reproducibility threshold met.
- For real benchmarks, the README points at LongMemEval's HuggingFace
  download. meridian_deep is documented as private with no public
  substitute.

---

## 2026-04-24

### Variance / stability — zero drift under deterministic config

**What.** 5 repeated runs of meridian_deep recall@k (full config, n=15):
identical numbers every run. SD=0 across all k values.

| k | any | all | frac |
|---|---|---|---|
| 5 | 80.0% ± 0.0pp | 40.0% ± 0.0pp | 0.607 ± 0.000 |
| 10 | 93.3% ± 0.0pp | 46.7% ± 0.0pp | 0.720 ± 0.000 |
| 25 | 93.3% ± 0.0pp | 53.3% ± 0.0pp | 0.733 ± 0.000 |

**How measured.**
```bash
python -m src.scripts.recall_at_k --corpus meridian_deep --config full \
    --annotations /Users/rileycoleman/meridian/probes/meridian_recall_gold.json \
    --k 5,10,25 --repeats 5 --output /tmp/recall_variance_full.json
```

**Caveat.**
- Under `recall_mutates_state=False` only. Production config has
  reheat-on-access which SHOULD introduce drift as memories warm up
  with use. Separate measurement needed for the mutating case.
- Single-process. Multi-worker has shown drift in earlier bench runs
  (different workers embed in different orders, DB cursor order can
  diverge). Worth re-measuring at some point.

**Implication.**
- Paper-safe claim: **"under deterministic configuration, substrate
  retrieval has zero variance across repeated runs (n=5, n_queries=15,
  k ∈ {5, 10, 25})."**
- Validates that all other single-number results in this doc are
  stable point estimates, not noisy draws.
- Resolves open question from earlier multi-worker flakiness: that was
  scheduling-induced, not substrate-induced.

---

### LongMemEval stratified recall@k — first external number

**What.** Memory-level recall@k on 30 stratified LongMemEval questions
(5 per type × 6 types, per-question fresh DB + ingest + 1 consolidation
cycle + recall):

| k | recall_any | recall_all | recall_frac |
|---|---|---|---|
| 5 | 83.3% | 53.3% | 0.663 |
| 10 | 90.0% | 63.3% | 0.742 |
| 25 | 93.3% | 70.0% | 0.814 |

**By question type (at k=10):**

| type | any | all | frac |
|---|---|---|---|
| single-session-user | 100% | 100% | 1.000 |
| single-session-assistant | 100% | 100% | 1.000 |
| single-session-preference | 100% | 100% | 1.000 |
| knowledge-update | 100% | 80% | 0.900 |
| temporal-reasoning | 80% | 0% | 0.307 |
| multi-session | 60% | 0% | 0.247 |

**How measured.**
```bash
python -m src.scripts.recall_at_k --corpus longmemeval \
    --per-type 5 --k 5,10,25 --cycles 1 \
    --output /tmp/recall_at_k_stratified.json
```

Gold = turns with `has_answer: true`. Fresh DB per question (LongMemEval
haystacks are per-question). Full config.

**Caveat.**
- n=30 is small — n=5 per type is very small. Full LongMemEval (500)
  would give tighter numbers per type, especially for the 133-question
  temporal-reasoning and multi-session categories.
- The 100% on single-session categories is partially cheap: vector
  search alone solves these. Plain RAG baseline on same harness
  pending.
- Retrieval is cold (fresh DB, no history, no decay). Different regime
  from meridian_deep which tests AGED retrieval. Both numbers useful
  for different claims.

**Implication.**
- Multi-session and temporal-reasoning are where the substrate visibly
  struggles (60%/80% any@10, 0% all@10). Consistent with the broader
  pattern: these require evidence *aggregation* across turns, and our
  retrieval is strong at "find the relevant session" but weak at
  "surface every scattered mention."
- The 100% on 3/6 categories is a strong result — most RAG papers
  don't hit 100 on any slice.
- **Paper-safe headline:** "90% recall_any@10 overall; retrieval
  quality degrades sharply on queries requiring aggregation across
  sessions (60% any, 0% all on multi-session n=5)."
- **Open follow-up**: rerun LongMemEval recall@k with --config plain_rag
  for fair architecture-vs-baseline comparison. Same ingest, different
  retrieval stack.

---

### Plain RAG baseline vs full Dendric (meridian_deep, n=15)

**What.** The full Dendric architecture outperforms a vector-only baseline
on recall@k at k=10, but the gap closes to zero at k=25.

| k | metric | full | plain_rag | Δ |
|---|---|---|---|---|
| 5 | recall_any | 80.0% | 80.0% | 0.0pp |
| 5 | recall_all | 40.0% | 33.3% | +6.7pp |
| 5 | recall_frac | 0.607 | 0.560 | +0.047 |
| 10 | recall_any | 93.3% | 86.7% | **+6.7pp** |
| 10 | recall_all | 46.7% | 40.0% | **+6.7pp** |
| 10 | recall_frac | 0.720 | 0.627 | **+0.093** |
| 25 | recall_any | 93.3% | 93.3% | 0.0pp |
| 25 | recall_all | 53.3% | 53.3% | 0.0pp |
| 25 | recall_frac | 0.733 | 0.733 | 0.0pp |

**How measured.**
```bash
# Full
python -m src.scripts.recall_at_k --corpus meridian_deep --config full \
    --annotations /Users/rileycoleman/meridian/probes/meridian_recall_gold.json \
    --k 5,10,25 --output /tmp/recall_at_k_meridian_deep_v2.json

# Plain RAG (vector-only via config preset)
python -m src.scripts.recall_at_k --corpus meridian_deep --config plain_rag \
    --annotations /Users/rileycoleman/meridian/probes/meridian_recall_gold.json \
    --k 5,10,25 --output /tmp/recall_plain_rag_meridian.json
```

Gold annotations: `meridian/probes/meridian_recall_gold.json` (15 hand-
annotated probes from Riley's Claude-export corpus). Matching uses
`gold_context_matches` (substring hits on raw_content or context).

**Caveat.**
- n=15 is small. Need wider probe set or LongMemEval at scale before
  generalizing.
- Annotations are Riley-authored, not third-party. Valid but not blind.
- At k=25 both configs converge — same candidates surface, just at
  different ranks. The value is **ranking**, not candidate discovery.
- plain_rag shares the same underlying corpus (ingest is unchanged);
  we only swap the retrieval stack. Fair isolation of "retrieval
  architecture" from "storage quality."

**Implication.**
- Dendric's added complexity (keyword + associative + graph paths +
  lifecycle modulation + déjà-vu) earns its place *at tighter k* — i.e.
  when the downstream answer model has limited context budget.
- At generous k, plain RAG is competitive. Worth acknowledging in any
  paper — don't overclaim.
- Honest paper claim: **"+6.7pp recall_any@10 and +9.3pp recall_frac@10
  over vector-only baseline on long-term personal data. Delta narrows
  at k=25, indicating re-ranking (not expansion) is the source of gain."**

---

### recall@k metric design + first substrate-only number

**What.** Built `src/scripts/recall_at_k.py` — substrate-only retrieval
metric that doesn't depend on any answer model. Reports three variants:
- `recall_any@k` — ANY gold memory in top-k → 1 else 0
- `recall_all@k` — EVERY gold memory in top-k → 1 else 0
- `recall_frac@k` — (# gold found) / (# gold expected)

Matching is chunk-aware: if an answer-bearing turn got split into N
chunks during ingest, finding any 1 of the N counts (design decision:
retrieval served its purpose if it got to the right conversation).

**First number on meridian_deep** (full config, n=15, k=10):
- recall_any = 93.3%, recall_all = 46.7%, recall_frac = 0.720

**First number on LongMemEval** (n=3 smoke test, all single-session):
- recall_any@5 = 100%, recall_all@5 = 100%  (trivial — easy mode only)

Full stratified LongMemEval run (per-type 5, n=30) kicked off in
background; results pending.

**Caveat.**
- One of 15 meridian_deep probes ("what airline am I flying to cape town")
  fails cleanly: gold memories are in archive, query doesn't contain the
  seed entity "norse" needed to trigger déjà-vu via spreading activation.
  This is a documented architectural limitation: entity-level triggers
  require query to contain or associate to the right entity.
- One probe was initially annotated incorrectly (asked about events
  post-dating the corpus). Caught by inspecting a 0-hit failure and
  realizing the gold wasn't actually in the DB. **Methodology note:
  annotations must match what's actually in corpus.**

**Implication.**
- This metric is the foundation for all subsequent rigor work. Baselines,
  ablations, parameter sweeps — all cite recall@k.
- "norse failure" is a real architectural finding to flag: if your query
  doesn't seed entities the system has edges to, déjà-vu can't fire even
  when the gold is in archive. Worth calling out as a limitation section.

---

## 2026-04-21 — earlier sessions, retroactively logged

### Déjà-vu archive trigger was structurally unable to fire (pre-fix)

**What.** Against `meridian_deep` (17,822 memories, 85% in archive), 0
archive memories appeared in top-10 across 19 probes × 4 threshold
configs (0.7, 0.5, 0.3, 0.1). The threshold tuning did nothing.

Root cause: RRF math. Archive memories only reach the fused pool via
the `associative_archive` path (vector/keyword/graph all filter
`region != archive`). A single-path hit maxes out at
`associative_weight / (rrf_k + rank + 1) = 1.5/61 ≈ 0.0246`. A two-path
hit (vector + keyword) starts at `≈ 0.036`. **Archive always lost.**

Fix: `archive_rrf_boost` multiplier (1.8) applied to `associative_archive`
hits before fusion sum. Plus `archive_modulation_override` (1.3) to
bypass cold-temperature penalty on déjà-vu firings.

**How measured.** `server/ab_probe.py`. 4 configs × 19 queries.

**Caveat.**
- 1.8 tuned on this specific corpus. May need per-deployment calibration.
- Above 2.0 archive starts wrongly dominating on strong-entity queries
  (top-1 flipped to off-topic archive hits).

**Implication.**
- Any "dormant but reachable" mechanism that depends on RRF single-path
  scoring has this failure mode latent. Worth as a cautionary architectural
  finding in a paper.
- Dendric commit: `ed218d1`.

---

### Consolidation-time region migration under backdated timestamps

**What.** Imported 18,691 memories from 1 year of Claude chat with real
historical `created_at` timestamps. After 12 ACT-R consolidation cycles:

| region | count | % |
|---|---|---|
| hippocampus | 0 | 0% |
| neocortex | 2,692 | 15% |
| archive | 15,130 | 85% |

Avg temperature across all memories: 0.127 (very cold — expected).
Consolidation ran progressively: 317 pruned cycle 1, then 107, 82, 71,
50, 47, 45, 49, 23, 25, 28, 25 → asymptoting around ~25 prunes/cycle
by cycle 12.

**How measured.** `server/import_claude_export.py` output logs. See
`/tmp/meridian_import_full.log`.

**Caveat.**
- "12 cycles" is an arbitrary mapping to "elapsed time." Not validated
  against any ground-truth human memory decay curve.
- Pruning rate asymptote of ~25/cycle implies the system isn't drained —
  but didn't run to 100+ cycles yet to confirm long-horizon stability.

**Implication.**
- The lifecycle is doing real work — 85% migration is not a flat curve
  or a cliff. Worth the "region distribution over consolidation cycles"
  chart in a paper.
- **Open follow-up:** run 100+ cycles against the same import to confirm
  asymptote. If it drains to zero, there's a design bug.

---

## Methodology notes (reusable across findings)

### Annotation quality gotchas

- Must match corpus content — not aspirational content. If you're
  evaluating against data ingested through date X, don't annotate for
  things said after X.
- Substring matches are easier to author but coarser than memory-id
  matches. When a substring is common, consider requiring multiple.
- Hand-annotating your own corpus is biased by your own recall — a
  reviewer could argue you annotated things you *remember* and missed
  things the system might legitimately find. Mitigation for future:
  have a second person annotate the same queries independently.

### Why meridian_deep is a useful probe corpus despite being personal

- Real age distribution (12-month span, 85% archived)
- Real redundancy (topics repeated across conversations)
- Real noise (code dumps, one-off debugging, tangents)
- Real entity density (Mary, Cheryl, Pitchwits, Dendric recur)

Limitations: single user, English-only, software/life-admin domain.
**Good for characterization, bad for generalizable claims.** Pair with
LongMemEval for external validity.

### Fair-comparison baseline construction

When comparing Dendric variants (full / ablation / plain_rag):
- **Ingest uses the same config.** Same corpus in DB, same entities
  extracted, same embeddings. Only retrieval strategy differs.
- **Preset defined in `src/engine/config.py`** — new baselines are a
  function that returns an `EngineConfig`, not a separate codebase.
- **Run via `--config` flag in `recall_at_k.py`** — harness is identical
  across configs.

This ensures measured deltas are retrieval-architecture deltas, not
"different systems" comparisons.

---

## Status

### Done
- [x] Memory-level recall@k metric (`src/scripts/recall_at_k.py`)
- [x] Plain RAG baseline via `plain_rag_config` preset
- [x] Variance/stability measurement: SD=0 across 5 repeats on
      meridian_deep
- [x] LongMemEval stratified recall@k at n=30 (per-type 5)
- [x] LongMemEval recall@k with plain_rag — apples-to-apples
- [x] 4-path leave-one-out ablation on LongMemEval (n=30)
- [x] 4-path leave-one-out ablation on meridian_deep (n=15)
- [x] Dockerized reproducibility package (Dockerfile, compose, REPRODUCE.md,
      synthetic corpus generator)

### Pending (Phase 2)
- [ ] Parameter sensitivity sweep (archive_rrf_boost, trigger_threshold,
      mod_temp_lift) — focus at k=5,10 where architecture pays off
- [ ] Scale characterization: retrieval latency vs corpus size (1k →
      100k → 1M memories, synthetic)
- [ ] Long-horizon consolidation stability (100+ cycles)
- [ ] Investigate why associative path is neutral on meridian_deep —
      is spreading activation firing? on which queries / entities?

### Pending (Phase 3-4)
- [ ] Failure mode taxonomy on recall@k misses
- [ ] Second-person annotation pass on meridian_deep probes
- [ ] LongMemEval at full scale (500 questions, not just 30 stratified)
- [ ] At least one external dogfooder
- [ ] Statistical framing (CIs, significance tests on the LOO deltas)
- [ ] Architecture diagram
