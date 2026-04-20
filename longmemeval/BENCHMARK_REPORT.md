# Dendric LongMemEval Benchmark Report
**Date:** 2026-04-06
**Model:** text-embedding-3-small (1536-dim) + Claude Haiku 4.5 (answer) + GPT-4o (eval)
**Config:** top_k=10, CoT=True, 1 consolidation cycle, novelty_gate=0.05

## Overall Results

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | **52.5% (262/499)** |
| **Task-Averaged Accuracy** | **61.6%** |
| Elapsed | 8733s (17.5s/question) |

## By Question Type

| Type | Correct | Total | Accuracy |
|------|---------|-------|----------|
| single-session-assistant | 48 | 56 | **85.7%** |
| single-session-preference | 24 | 30 | **80.0%** |
| single-session-user | 54 | 69 | **78.3%** |
| knowledge-update | 44 | 78 | 56.4% |
| temporal-reasoning | 50 | 133 | 37.6% |
| multi-session | 42 | 133 | 31.6% |

## Error Analysis

### Miss Breakdown (237 wrong, 1 error)

| Category | Count | % of Misses |
|----------|-------|-------------|
| Retrieval miss ("I don't know") | 155 | 65.4% |
| Wrong answer (retrieved but answered incorrectly) | 82 | 34.6% |
| Error (rate limit/crash) | 1 | - |

### By Type

| Type | Retrieval Miss | Wrong Answer | Total Misses |
|------|---------------|--------------|--------------|
| multi-session | 55 | 36 | 91 |
| temporal-reasoning | 60 | 23 | 83 |
| knowledge-update | 24 | 10 | 34 |
| single-session-user | 10 | 5 | 15 |
| single-session-preference | 1 | 5 | 6 |
| single-session-assistant | 5 | 3 | 8 |

## Root Causes

### 1. Retrieval Misses (155 questions, 65% of all misses)

The #1 problem. The correct memory exists in the database but doesn't appear in the top 10 results. This happens because:

- **Semantic gap:** The question phrasing is too different from the stored memory content. E.g., "How much did I spend on a designer handbag?" vs the memory saying "$800 on a Gucci bag" — vector similarity between the question and the memory's full context may not be high enough.
- **Needle in a haystack:** With ~800 memories per question and top_k=10, we're selecting 1.25% of content. Specific numbers ("17 skeins", "12 bass", "16GB RAM") are details buried in long conversations.
- **Session diversity penalty:** The fusion layer forces results from 3+ different sessions. If the answer is in one session and other sessions have higher-ranked irrelevant results, the answer gets pushed down.

### 2. Multi-Session Failures (91 misses, 31.6% accuracy)

The weakest category. These questions require aggregating information across multiple sessions (e.g., "How many model kits have I worked on?"). Failures split into:
- **55 retrieval misses:** Can't find all relevant sessions in top 10 results
- **36 wrong answers:** Found some mentions but couldn't count/aggregate correctly

This is a structural limitation: with top_k=10, you can't retrieve 5+ scattered mentions across 50+ sessions. The aggregation synthesis (`_synthesize_aggregates`) could help but runs during consolidation, which we only do once.

### 3. Temporal Reasoning Failures (83 misses, 37.6% accuracy)

Questions like "How many weeks ago did X happen?" or "Which happened first, A or B?" fail because:
- **60 retrieval misses:** The temporal decomposer doesn't always extract the right event queries
- **23 wrong answers:** Retrieved the right memories but the answer model miscalculated time differences
- Temporal questions require cross-referencing dates from multiple memories — hard with top_k=10

### 4. Wrong Answers (82 total)

The model retrieved relevant content but answered incorrectly. Common patterns:
- **Counting errors:** "How many X?" — found 4 of 5 instances, answered "4" instead of "5"
- **Recency bias:** Knowledge-update questions ask for the *latest* value, but the model sometimes picks an earlier one
- **Partial retrieval:** Found some relevant memories but missed key details in truncated content

## What's Working Well

- **Single-session recall (78-86%):** When the answer is in one conversation turn, vector similarity + keyword FTS finds it reliably
- **Embedding quality:** text-embedding-3-small gives strong semantic similarity (0.5+ for direct matches)
- **Fusion weights rebalance:** Vector=1.5, Keyword=0.8 dramatically improved results (from 15% to 85% on assistant type)
- **Speed:** 17.5s/question (was 522s before OpenAI embeddings)

## Recommendations for Improvement

### High Impact
1. **Increase top_k to 20-25 for multi-session/temporal questions** — these question types need more retrieved memories to aggregate across sessions
2. **Run more consolidation cycles** — the aggregate synthesis that builds cross-session summaries only fires during consolidation. More cycles = better aggregate memories
3. **Use a stronger answer model** — Claude Haiku makes counting mistakes; Claude Sonnet would do better on aggregation

### Medium Impact
4. **Reduce session diversity penalty for non-aggregation queries** — session_decay=0.85 is too aggressive for factual recall; consider 0.95
5. **Entity graph with link_entities=True** — currently disabled for benchmark speed, but would improve graph-path recall

### Lower Impact
6. **Better temporal decomposition** — the temporal decomposer could extract event-date pairs more reliably
7. **Chunk overlap tuning** — 300-char overlap may miss context boundaries
