# Where Dendric Sits — Honest Competitive Analysis (2026-04-20)

## The benchmark, characterized

LongMemEval-S is **not** "find the answer in 800 messages." It's six distinct tasks bundled into one benchmark, and they reward very different system properties.

| Type | n | Median sessions w/ answer | What it actually tests |
|---|---|---|---|
| single-session-user | 70 | 0.9 | Find the user's stated fact in one turn |
| single-session-assistant | 56 | 1.0 | Find the assistant's stated fact in one turn |
| single-session-preference | 30 | 1.0 | Recall a user preference exposed once |
| knowledge-update | 78 | 1.8 | Pick the *latest* of several conflicting facts |
| temporal-reasoning | 133 | 1.9 | Compute durations / orderings from dated mentions |
| **multi-session** | **133** | **2.3 (range 0–5)** | **Aggregate scattered mentions across sessions** |

Average haystack size: **48 sessions, 491 turns, ~800 chunked memories**. The corpus is the same shape across question types — what changes is how the answer is *distributed* through it.

**Critical finding from the dataset itself**: the multi-session category has 8 questions where **0 sessions are marked answer-bearing**. These are unanswerable-by-design (abstention checks), and any system that tries to answer them is wrong by definition. Of the remaining 125, the median answer is in **2 sessions**, but 19 questions have answers spread across **4–5 sessions**. Counting questions like "How many projects led" with 4 sessions contributing single mentions each are at the hardest end of the curve.

## What "good" looks like in 2026

Updated public LongMemEval scores (overall / multi-session):

| System | Overall | Multi-session | Approach |
|---|---|---|---|
| **Mastra Observational Memory** | 94.87% | 87.2% | Two LLM background agents (Observer + Reflector) maintain a dense, prompt-cacheable observation log. **LLM in the consolidation path.** |
| **OMEGA** | 95.4% | 83% | Proprietary; details light |
| **Emergence AI** | 86% | 81.2% | Cross-encoder reranking, *session-level* retrieval (match on turns, retrieve whole sessions), CoT prompting |
| **Supermemory** | ~? | 71.43% | Vector + reranking, also session-aware |
| **Zep** | 71.2% | ? | Knowledge graph + temporal extraction, LLM-heavy |
| **Plain RAG (Emergence's baseline)** | 52% | ? | Standard vector search |
| **Dendric (slim, 2026-04-20)** | 63.3%¹ | 20%² | Four parallel paths + RRF + lifecycle modulation, **no LLM in consolidation** |
| **Dendric (bloated, 2026-04-01)** | 61.6% | 62% | Same paths + scaffolding + tuned for bench |
| **Dendric (current architecture, 2026-04-06)** | 52.5% | 31.6% | Pre-slim |

¹ 30-question stratified sample, not full bench
² Same sample, n=5 multi-session — high variance

**Two facts to anchor on:**

1. **Plain RAG = 52%.** That's the floor. Anything below this means your sophisticated retrieval is *worse than vanilla vector search*. Dendric's 2026-04-06 full run was 52.5% — exactly at the plain-RAG floor.
2. **State-of-the-art = 95% overall, 87% multi-session.** And every public SOTA system **uses LLMs in the consolidation path** (Mastra's Observer/Reflector, Emergence's reranker, Zep's extraction, OMEGA's everything).

## The architectural fork you're standing on

You explicitly removed LLM consolidation. That was a real principled choice — deterministic compression, no extraction-cost-per-ingest, no provider lock-in. **It also matches none of the systems winning this benchmark.**

This is not a problem with your design. It's a clarity-of-purpose question.

### Two different bets you could be making

**Bet A — "biological memory primitive."** The thing you're building is a substrate for episodic memory in long-running agents, where:
- Every memory has a lifecycle (temperature, signals, decay, reheat)
- Consolidation is cheap, deterministic, and runs continuously
- The system is meant to be a *component* of a larger agent that does its own reasoning over recalled memories
- The benchmark question is "did the right memories surface?", not "did the system produce the right answer?"

If this is the bet, **LongMemEval is partially the wrong test.** It conflates retrieval with answer synthesis. A good Dendric result on this benchmark would be "retrieved set contains the answer-bearing memories at high rank" — which we can measure independently of whether Claude/GPT got the math right.

**Bet B — "memory system that produces correct answers."** The thing you're building competes with Mastra/Emergence/OMEGA. To win this, you need:
- An LLM (or strong rule-based system) in the consolidation path that can extract structured facts and compress aggressively
- Cross-encoder reranking
- Session-level retrieval semantics
- Probably more

If this is the bet, you've **deliberately handicapped yourself** by removing LLM consolidation, and your ceiling is roughly Plain RAG (~52%). The slim refactor moved you toward this ceiling cleanly, but it's still a ceiling.

**You can't win both bets with one system.** Mastra picked B and beat the bench. You picked A and the bench doesn't reward what you're optimizing for.

## What the multi-session failures actually tell us

From the OVERNIGHT_ANALYSIS dive into 4 failures, the underlying pattern was:
- Some failures were the score-mixing bug (now fixed) — **good news, real recall problem masked by an integration bug**
- Some failures were genuine recall gaps — answer-bearing turn doesn't reach top-25 on any path
- Some failures were synthesis arithmetic — model had two memories saying "2 weeks" and "1.5 weeks" but produced "about 2 weeks"

**The recall gaps are not unique to your system.** Mastra hits 87.2% on multi-session, not 95%. **Multi-session is the hardest category for every system.** Even the SOTA can only find the scattered evidence ~87% of the time.

But your 20-32% on multi-session is a *much* bigger gap than 87 → 95. So there's real headroom.

## Specific gaps worth naming

### Gap 1 (architectural): you have no session-level retrieval

Every winning system does **turn-match → session-retrieve.** They find the relevant turn, then pull the *entire session* it belongs to into context. That gives the answer model the surrounding conversation, which is critical for resolving ambiguous mentions.

You retrieve at the chunk/turn level only. So for the "model kits" question, even when you retrieve the session_1 turn that mentions B-29 and Camaro, you don't pull session_1's other turns where the user might have continued enumerating purchases.

**Cost to add:** moderate. Postgres can group by session. The retrieval path returns chunks; you'd add a "session expand" step that pulls all turns from the top-k sessions before passing context to the answer model. This is plausibly the single biggest win available to you within Bet A.

### Gap 2 (architectural): no reranker

Every winning system uses a cross-encoder reranker (e.g., bge-reranker, cohere-rerank). Vector search returns top-100; the reranker (small bidirectional model that scores query+candidate jointly) re-orders to top-10. This catches the cases where your vector embeddings produce 10× score inversions like the MCU/SW question we saw.

**Cost to add:** low-moderate. A reranker call adds ~200ms per query and the model can run locally (bge-reranker-base is ~278M params). It's mostly orthogonal to your existing architecture — drop it in between fusion and lifecycle modulation.

This breaks the "no LLM" purity but a reranker is *much* cheaper than LLM consolidation and runs at retrieval time, not ingest time. Different tradeoff.

### Gap 3 (algorithmic): aggregation bypass is brittle

The `_extract_subject_keywords` regex returning `['kits']` is too coarse. For "How many model kits", the right keyword set is something like `['model', 'kit', 'kits', 'tank', 'tanks', 'plane', 'B-29', 'Spitfire', 'Tiger']` — generated by either:
- (Quick) An LLM-based query expansion call (~1k tokens, cheap)
- (Medium) A learned query expansion from the corpus
- (Stupid-but-effective) Just call the answer model first to enumerate likely terms before retrieving

This crosses into Bet B territory (LLM in the loop), but at *query time* not ingest time, so the cost scales with query volume not corpus volume.

### Gap 4 (system): score determinism

Two runs of the same 4 questions returned different `correct/total` (1/4 vs 1/4 — but the *which one* passed varied). With `recall_mutates_state=False`, retrieval should be deterministic per-process. Multi-worker may break that assumption (different DB cursors, race conditions in batch ingest).

**Cost to investigate:** low. A run with `--workers 1` on the same 4 questions twice in a row tells you whether retrieval is deterministic when single-threaded. If yes, the non-determinism is in worker scheduling. If no, there's a deeper issue.

## Is the bench possible for what you've built?

**Honest answer: not at SOTA levels without changing your bet.**

But achievable benchmarks for the current architecture (Bet A, no LLM consolidation):

- **65–70% overall, 40–50% multi-session, 50–60% temporal-reasoning** — likely reachable with score-mixing fix + session-level retrieval + reranker. Roughly Zep-tier.
- **80%+** — probably requires LLM consolidation or extraction at ingest. Bet B.

If you want to win the benchmark, change the bet. If you want to keep the bet, redefine success as something narrower than "beat LongMemEval."

## Recommendation

**Don't optimize for LongMemEval as a goal.** Use it as a *diagnostic* — it's good at exposing recall gaps and synthesis failures. But the goal should be one of:

1. **A primitive that's good enough at retrieval that an agent layer above it can do well on workloads like LongMemEval.** Then the right test is "does retrieval surface the answer-bearing memories at high rank?" Independent of what the answer model does with them. Build that test instead of relying on the bundled benchmark.

2. **A real product that competes with Mastra etc.** Then accept LLM in consolidation and start there.

3. **An internal tool for your own agents** — what you originally said. Then the benchmark question is "does this make my own agents better at things I care about?" — not LongMemEval.

You said all three are on the table. Pick one for the next 2-4 weeks. The right answer to "is the benchmark possible" depends entirely on which one.

## What I'd do next

Independent of the bet, three things are unambiguously good moves:

1. **Add a retrieval-quality metric.** "Did the answer-bearing turn (we know which one — it's marked `has_answer: true` in the dataset) make it into top-k?" That's a number you can track that doesn't depend on the answer model. It tells you whether retrieval is actually getting better when you make changes. Cheap to build (~50 lines), no LLM cost. Should be in place before any further benchmark runs.

2. **Add session-level expansion as an experiment.** Even just on multi-session questions, pull the full session text for each top-k chunk and pass it to the answer model. That's bet-A-friendly (no LLM in consolidation), and it's the single biggest delta in the public results table.

3. **Stop running 30-question stratified samples for headline numbers.** They have too much variance — the same architecture has scored 70%, 73%, and 83% across three runs of n=30 on the same day. Run multi-session at n=30 (what you're doing now), use it for *category-specific* signal, but reserve overall accuracy claims for n≥100.

## Sources

- [Mastra Observational Memory architecture](https://mastra.ai/research/observational-memory)
- [Emergence AI on RAG approach to LongMemEval](https://www.emergence.ai/blog/sota-on-longmemeval-with-rag)
- [LongMemEval leaderboard (Mempalace)](https://www.mempalace.tech/benchmarks)
- [LongMemEval paper](https://arxiv.org/abs/2410.10813)
- [LongMemEval GitHub](https://github.com/xiaowu0162/LongMemEval)
