# Path ablation — 2026-04-27T12:53:35

## recall@5

| config | any | all | frac | Δ any vs full | Δ frac vs full | verdict |
|---|---|---|---|---|---|---|
| `plain_rag` | 80.0% | 33.3% | 0.560 | -6.7pp | -0.080 | — (floor) |
| `no_vector` | 60.0% | 26.7% | 0.444 | -26.7pp | -0.196 | **path helps** (+26.7pp any) |
| `no_keyword` | 66.7% | 26.7% | 0.451 | -20.0pp | -0.189 | **path helps** (+20.0pp any) |
| `no_associative` | 86.7% | 40.0% | 0.640 | +0.0pp | +0.000 | neutral |
| `no_graph` | 86.7% | 40.0% | 0.627 | +0.0pp | -0.013 | neutral |
| `full` | 86.7% | 40.0% | 0.640 | +0.0pp | +0.000 | — (ceiling) |

## recall@10

| config | any | all | frac | Δ any vs full | Δ frac vs full | verdict |
|---|---|---|---|---|---|---|
| `plain_rag` | 86.7% | 40.0% | 0.627 | -6.7pp | -0.093 | — (floor) |
| `no_vector` | 73.3% | 33.3% | 0.538 | -20.0pp | -0.182 | **path helps** (+20.0pp any) |
| `no_keyword` | 80.0% | 33.3% | 0.576 | -13.3pp | -0.144 | **path helps** (+13.3pp any) |
| `no_associative` | 93.3% | 46.7% | 0.720 | +0.0pp | +0.000 | neutral |
| `no_graph` | 86.7% | 40.0% | 0.627 | -6.7pp | -0.093 | **path helps** (+6.7pp any) |
| `full` | 93.3% | 46.7% | 0.720 | +0.0pp | +0.000 | — (ceiling) |

## recall@25

| config | any | all | frac | Δ any vs full | Δ frac vs full | verdict |
|---|---|---|---|---|---|---|
| `plain_rag` | 93.3% | 53.3% | 0.733 | +0.0pp | +0.000 | — (floor) |
| `no_vector` | 86.7% | 46.7% | 0.684 | -6.7pp | -0.049 | **path helps** (+6.7pp any) |
| `no_keyword` | 93.3% | 53.3% | 0.733 | +0.0pp | +0.000 | neutral |
| `no_associative` | 93.3% | 53.3% | 0.733 | +0.0pp | +0.000 | neutral |
| `no_graph` | 93.3% | 53.3% | 0.733 | +0.0pp | +0.000 | neutral |
| `full` | 93.3% | 53.3% | 0.733 | +0.0pp | +0.000 | — (ceiling) |

## Reading the table

- **`no_X`** rows show recall when path X is disabled. A large **negative Δ** means that path was contributing substantially — disabling it hurt. A positive Δ means the path was net-harmful in this config (investigate, don't silently keep it).
- **`plain_rag`** is the vector-only floor. The gap between `plain_rag` and `full` is the *total* lift Dendric's architecture provides over standard RAG.
- **`full`** is the ceiling. All deltas are reported against it.