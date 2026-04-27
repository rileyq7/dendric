# Path ablation — 2026-04-27T11:24:58

## recall@5

| config | any | all | frac | Δ any vs full | Δ frac vs full | verdict |
|---|---|---|---|---|---|---|
| `plain_rag` | 80.0% | 33.3% | 0.560 | +0.0pp | -0.013 | — (floor) |
| `no_vector` | 46.7% | 13.3% | 0.267 | -33.3pp | -0.307 | **path helps** (+33.3pp any) |
| `no_keyword` | 73.3% | 26.7% | 0.473 | -6.7pp | -0.100 | **path helps** (+6.7pp any) |
| `no_associative` | 86.7% | 40.0% | 0.640 | +6.7pp | +0.067 | path hurts (+6.7pp any) |
| `no_graph` | 80.0% | 33.3% | 0.560 | +0.0pp | -0.013 | neutral |
| `full` | 80.0% | 33.3% | 0.573 | +0.0pp | +0.000 | — (ceiling) |

## recall@10

| config | any | all | frac | Δ any vs full | Δ frac vs full | verdict |
|---|---|---|---|---|---|---|
| `plain_rag` | 86.7% | 40.0% | 0.627 | -6.7pp | -0.093 | — (floor) |
| `no_vector` | 60.0% | 26.7% | 0.422 | -33.3pp | -0.298 | **path helps** (+33.3pp any) |
| `no_keyword` | 86.7% | 40.0% | 0.642 | -6.7pp | -0.078 | **path helps** (+6.7pp any) |
| `no_associative` | 93.3% | 46.7% | 0.720 | +0.0pp | +0.000 | neutral |
| `no_graph` | 86.7% | 40.0% | 0.627 | -6.7pp | -0.093 | **path helps** (+6.7pp any) |
| `full` | 93.3% | 46.7% | 0.720 | +0.0pp | +0.000 | — (ceiling) |

## recall@25

| config | any | all | frac | Δ any vs full | Δ frac vs full | verdict |
|---|---|---|---|---|---|---|
| `plain_rag` | 93.3% | 53.3% | 0.733 | +0.0pp | +0.000 | — (floor) |
| `no_vector` | 73.3% | 40.0% | 0.596 | -20.0pp | -0.138 | **path helps** (+20.0pp any) |
| `no_keyword` | 93.3% | 53.3% | 0.733 | +0.0pp | +0.000 | neutral |
| `no_associative` | 93.3% | 53.3% | 0.733 | +0.0pp | +0.000 | neutral |
| `no_graph` | 93.3% | 53.3% | 0.733 | +0.0pp | +0.000 | neutral |
| `full` | 93.3% | 53.3% | 0.733 | +0.0pp | +0.000 | — (ceiling) |

## Reading the table

- **`no_X`** rows show recall when path X is disabled. A large **negative Δ** means that path was contributing substantially — disabling it hurt. A positive Δ means the path was net-harmful in this config (investigate, don't silently keep it).
- **`plain_rag`** is the vector-only floor. The gap between `plain_rag` and `full` is the *total* lift Dendric's architecture provides over standard RAG.
- **`full`** is the ceiling. All deltas are reported against it.