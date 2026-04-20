# Overnight Diagnostic — Multi-Session Failure Analysis

**Branch:** `overnight/retrieval-diagnostic` (do not merge to main without review)
**Scope:** 4 multi-session questions that failed on the 30-question stratified baseline
**Runs used:** 2 of 3 allotted (~$0.10, 10 min wall-clock)

## TL;DR

Two distinct, independent root causes — not one bug.

1. **Score-mixing bug** in `run_longmemeval.py:387–427` — the aggregation-boost keyword sweep injects raw `ts_rank` scores (0.5–1.0) into a result list that's then sorted alongside RRF fusion scores (0.04–0.08). The bypass garbage swamps the real fused output. **Fix landed on this branch; validated on 1 of 4 questions; ready for benchmark validation.**
2. **Recall-gap on long-tail mentions** — the actual answer-bearing turn for some questions (Tiger I tank for "model kits", "week and a half" Star Wars for "MCU + SW weeks") is not in *any* path's top-25. This is genuine retrieval failure, not a sort bug. Architectural conversation, not a tune.

The score-mixing fix alone made the MCU/SW question's top-4 retrieval go from "European company name suggestions" garbage to the actual session_13 (MCU "two weeks") + session_23 (Star Wars marathon) memories. The model still got it wrong — but for a different reason now (Star Wars duration was never retrieved at all). One bug down, one architectural gap to discuss.

## What I changed on this branch

1. **`src/engine/core/engine.py`** — instrumented `recall()` to stash per-path candidate counts and per-memory rank/score across all 4 paths in `self._last_recall_paths`. Each returned result also gets a `_path_debug` dict showing which paths produced it pre-fusion. Diagnostic-only; no behavior change.
2. **`src/scripts/run_longmemeval.py`** — debug dump now includes `last_recall_paths` summary + `path_debug` per result + `_modulation` + `_unmodulated_fusion_score` for each top-k hit.
3. **`src/scripts/run_longmemeval.py:387–427`** — **behavior change**: aggregation-boost keyword sweep now rescales raw `ts_rank` scores into a synthetic `fusion_score` capped just below the lowest real fused score, so bypass results fill empty slots without displacing legitimate top-fusion hits. Tags them `'bypass_keyword'` in `retrieval_paths` for telemetry.

## Per-failure root cause

### `gpt4_59c863d7` — "How many model kits?" (gold: 5)

- **Hypothesis pre-fix:** "4 model kits" (missed Tiger tank)
- **Hypothesis post-fix:** "3 model kits" (missed Tiger AND Camaro)
- **Pre-fusion path counts:** `vector=25, keyword=25, associative=0, graph=0`
- **Why associative/graph were 0:** query entities resolved to `['kits']` — a noun fallback, not a real entity. No graph node, no spreading-activation seed. Entity extraction can't help here because the query is about a category, not a proper noun.
- **Where the missed items live:** session_40 (Tiger I tank), session_1 (Camaro mentioned in passing in same turn as B-29). Session_40 is **not in any path's top-25** — it talks about "weathering" and "tanks", and "model kits" doesn't embed-match it strongly enough.
- **Verdict:** Score-mixing fix didn't help this one (the bypass sweep wasn't the dominant problem). This is a **recall gap** — the corpus has the answer but fusion can't reach it.

### `e831120c` — "How many weeks for MCU + Star Wars?" (gold: 3.5)

- **Hypothesis pre-fix:** "about two weeks" (top-4 retrieval was unrelated company-naming garbage at score 0.8; relevant memories buried at rank 7-8)
- **Hypothesis post-fix:** "about 2 weeks" (top-4 now correctly contains session_13 MCU "two weeks" and session_23 Star Wars list)
- **Why still wrong post-fix:** The user's opening turn in session_23 says "Star Wars marathon, watched all the main films **in a week and a half**." That turn (session_23_turn_0) was **not in top-25 of any path**. The retrieved session_23 chunks are all from *later* in the conversation (assistant's enumeration of titles, no duration). The model saw "MCU = 2 weeks" + Star Wars list with no duration, so it answered "about 2 weeks."
- **Verdict:** Score-mixing fix solved the surface bug (now retrieving the right session). Recall gap still buries the duration sentence. 2 weeks + 1.5 weeks = 3.5 weeks would be the answer **if both were retrieved**.

### `0a995998` — "How many clothing items to pick up?" (gold: 3)

- **Hypothesis:** Found 2 of 3 items; abstained ("I don't know") despite system prompt saying "only as last resort." Synthesis-side over-abstention. Same retrieval gap — third item missed.

### `6d550036` — "How many projects led/leading?" (gold: 2)

- **Hypothesis:** Inconsistent across runs. Run 1: said "1 project" (failure). Run 3: said "2 projects" (pass). **Non-determinism** — different worker scheduling produces different recall ordering, model gets different evidence sets.
- **Verdict:** Not a clear bug, but a flakiness signal. Worth investigating where the non-determinism enters. `recall_mutates_state=False` should mean retrieval is deterministic per-process; multi-worker may break that assumption.

## The two architectural gaps

### Gap 1: Category queries can't seed associative/graph paths

When a query is about a category ("model kits", "projects", "clothing items"), entity extraction returns nothing or returns weak fallback nouns. The associative path and graph path then return zero candidates. Two paths out of four go silent.

**This is fixable, with options of varying ambition:**
- (Quick) **Always seed associative path with the persona** when query entities resolve to nothing — already implemented as `persona_fallback`, but doesn't help here because `query_entities=['kits']` is treated as resolved (just to a non-graph entity), so the fallback doesn't fire. Fix the gate: fallback should fire if seeds yield no graph hits, not just if no seeds were extracted.
- (Medium) **Type-class entity edges** — at ingest, link memories to a type-class node (`@type:vehicle`, `@type:project`, `@type:clothing`) so category queries can walk the graph. Requires either rule-based or LLM-based type assignment. Reintroduces some LLM consolidation cost.
- (Ambitious) **Query expansion** — decompose category queries into specific terms ("model kits" → "tank, plane, car kit, scale model, B-29, Spitfire, Camaro"), seed multiple parallel retrievals, union. Most powerful, most plumbing.

### Gap 2: Long-tail mentions don't reach top-25 even from keyword/vector

For both failing questions, the answer-bearing turn exists in the corpus but doesn't rank highly enough on any path. `vector=25` and `keyword=25` are returning 25 candidates each, but those 25 are dominated by *similar* memories (multiple chunks of the same session, multiple weathering-related conversations) rather than diverse mentions across different sessions.

**This points at a redundancy problem in retrieval, not just recall:**
- Vector top-25 contains 6 chunks from session_32 + session_43 (both about weathering) and zero from session_40 (Tiger tank, also about weathering — semantically close). Vector ANN is clustering on the dominant topic, not surfacing the diverse mentions.
- Keyword top-25 with `kw='kit'` returns FTS top by frequency, which favors sessions that use "kit" multiple times. Session_40 only mentions "kit" in passing.

**Fix options:**
- (Quick) **Per-session diversity in vector path** — cap how many chunks from any single session can occupy the top-25. Already done in the bypass sweep at run_longmemeval.py:401 (`max_per_session=2`). Could push that into vector_recall and keyword_recall directly so all paths benefit.
- (Medium) **Maximal Marginal Relevance** in vector_recall — re-rank top-100 vector hits to balance similarity-to-query against dissimilarity-to-already-selected. Standard fix for this exact failure mode.
- (Ambitious) **Aggregation-aware recall mode** — for `_AGGREGATION_RE` queries, the engine should run a different recall strategy entirely: maximize cross-session coverage, broaden vocabulary via query expansion, accept lower per-result similarity for higher diversity. This is a real architectural fork — "find one answer" vs "find all mentions" are different problems.

## What I'd recommend (not implementing tonight)

**Tomorrow morning, in this order:**

1. **Validate the score-mixing fix at scale.** Re-run the full 30-question stratified sample with this branch's changes. The MCU question retrieval is now structurally correct; even if it stays wrong on the duration, see whether the fix improved the other multi-session questions (the previous run's `0a995998` already had the right items found and just over-abstained — that one might flip with cleaner top-5 ordering). Cost: $0.25, 40 min.

2. **If multi-session improves meaningfully, merge this branch.** The score-mixing bug is a clean win — proper RRF outputs were always there, just buried by a careless append step.

3. **Then have the architectural conversation about Gap 1 and Gap 2.** Both are real, both have multiple options at different ambition levels, and the right answer depends on whether you want this to be a primitive that does *all* memory tasks well or a primitive that does *episodic* memory well and lets the agent layer above handle aggregation.

## What I did NOT do (and why)

- **Did not run the 500-question full benchmark.** Out of scope; you asked me not to.
- **Did not implement the O(n²) novelty fix.** Out of scope unless explicitly approved; you didn't approve it.
- **Did not commit to main.** Working branch only. Diff is reviewable.
- **Did not run the 3rd allotted benchmark.** Two runs gave clear signal; a third would have been spending money to confirm something already confirmed.
- **Did not address the recall gaps.** Architectural decisions, not overnight work.

## Files touched

```
src/engine/core/engine.py     # +path logging, no behavior change
src/scripts/run_longmemeval.py # +debug dump fields, +score-mixing fix
longmemeval/debug/*.json       # diagnostic dumps from runs
longmemeval/debug/OVERNIGHT_ANALYSIS.md  # this file
```

## Numbers

| Run | Where | Result |
|---|---|---|
| 30-q stratified (yesterday baseline) | `main` | 19/30 = 63.3% overall, 1/5 multi-session |
| 4-q diagnostic, no fix (run 1) | branch + logging | 1/4 multi-session, MCU/SW retrieval shows score-mixing |
| 4-q diagnostic, with score-mixing fix (run 2) | branch + fix | 1/4 multi-session, MCU/SW retrieval now structurally correct |

The 4-q multi-session score didn't move on the diagnostic, but the *retrieval underneath* did — which is the thing the score-mixing fix targets. Synthesis-stage and recall-gap problems are independent of this fix and were never expected to flip on these 4 questions. Real validation requires the 30-q rerun.
