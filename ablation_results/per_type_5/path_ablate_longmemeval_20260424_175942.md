# Path ablation — 2026-04-24T17:59:42

## recall@5

| config | any | all | frac | Δ any vs full | Δ frac vs full | verdict |
|---|---|---|---|---|---|---|
| `plain_rag` | 91.7% | 41.7% | 0.642 | +91.7pp | +0.642 | — (floor) |

## recall@10

| config | any | all | frac | Δ any vs full | Δ frac vs full | verdict |
|---|---|---|---|---|---|---|
| `plain_rag` | 91.7% | 58.3% | 0.769 | +91.7pp | +0.769 | — (floor) |

## recall@25

| config | any | all | frac | Δ any vs full | Δ frac vs full | verdict |
|---|---|---|---|---|---|---|
| `plain_rag` | 91.7% | 83.3% | 0.900 | +91.7pp | +0.900 | — (floor) |

## Reading the table

- **`no_X`** rows show recall when path X is disabled. A large **negative Δ** means that path was contributing substantially — disabling it hurt. A positive Δ means the path was net-harmful in this config (investigate, don't silently keep it).
- **`plain_rag`** is the vector-only floor. The gap between `plain_rag` and `full` is the *total* lift Dendric's architecture provides over standard RAG.
- **`full`** is the ceiling. All deltas are reported against it.