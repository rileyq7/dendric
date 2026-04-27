# Path ablation — 2026-04-24T17:46:23

## recall@5

| config | any | all | frac | Δ any vs full | Δ frac vs full | verdict |
|---|---|---|---|---|---|---|
| `plain_rag` | 91.7% | 66.7% | 0.806 | +0.0pp | +0.083 | — (floor) |
| `no_vector` | 58.3% | 41.7% | 0.486 | -33.3pp | -0.236 | **path helps** (+33.3pp any) |
| `no_keyword` | 83.3% | 58.3% | 0.681 | -8.3pp | -0.042 | **path helps** (+8.3pp any) |
| `no_associative` | 83.3% | 50.0% | 0.667 | -8.3pp | -0.056 | **path helps** (+8.3pp any) |
| `no_graph` | 91.7% | 58.3% | 0.750 | +0.0pp | +0.028 | neutral |
| `full` | 91.7% | 58.3% | 0.722 | +0.0pp | +0.000 | — (ceiling) |

## recall@10

| config | any | all | frac | Δ any vs full | Δ frac vs full | verdict |
|---|---|---|---|---|---|---|
| `plain_rag` | 91.7% | 75.0% | 0.847 | +0.0pp | +0.125 | — (floor) |
| `no_vector` | 58.3% | 41.7% | 0.486 | -33.3pp | -0.236 | **path helps** (+33.3pp any) |
| `no_keyword` | 100.0% | 66.7% | 0.785 | +8.3pp | +0.062 | path hurts (+8.3pp any) |
| `no_associative` | 91.7% | 75.0% | 0.819 | +0.0pp | +0.097 | neutral |
| `no_graph` | 91.7% | 66.7% | 0.792 | +0.0pp | +0.069 | neutral |
| `full` | 91.7% | 58.3% | 0.722 | +0.0pp | +0.000 | — (ceiling) |

## recall@25

| config | any | all | frac | Δ any vs full | Δ frac vs full | verdict |
|---|---|---|---|---|---|---|
| `plain_rag` | 91.7% | 91.7% | 0.917 | -8.3pp | +0.049 | — (floor) |
| `no_vector` | 83.3% | 41.7% | 0.576 | -16.7pp | -0.292 | **path helps** (+16.7pp any) |
| `no_keyword` | 100.0% | 83.3% | 0.910 | +0.0pp | +0.042 | neutral |
| `no_associative` | 100.0% | 83.3% | 0.896 | +0.0pp | +0.028 | neutral |
| `no_graph` | 100.0% | 83.3% | 0.896 | +0.0pp | +0.028 | neutral |
| `full` | 100.0% | 75.0% | 0.868 | +0.0pp | +0.000 | — (ceiling) |

## Reading the table

- **`no_X`** rows show recall when path X is disabled. A large **negative Δ** means that path was contributing substantially — disabling it hurt. A positive Δ means the path was net-harmful in this config (investigate, don't silently keep it).
- **`plain_rag`** is the vector-only floor. The gap between `plain_rag` and `full` is the *total* lift Dendric's architecture provides over standard RAG.
- **`full`** is the ceiling. All deltas are reported against it.