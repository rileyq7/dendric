# Path ablation — 2026-04-24T20:54:19

## recall@5

| config | any | all | frac | Δ any vs full | Δ frac vs full | verdict |
|---|---|---|---|---|---|---|
| `plain_rag` | 93.3% | 56.7% | 0.731 | +10.0pp | +0.046 | — (floor) |
| `no_vector` | 63.3% | 40.0% | 0.506 | -20.0pp | -0.180 | **path helps** (+20.0pp any) |
| `no_keyword` | 80.0% | 56.7% | 0.674 | -3.3pp | -0.011 | **path helps** (+3.3pp any) |
| `no_associative` | 83.3% | 56.7% | 0.702 | +0.0pp | +0.017 | neutral |
| `no_graph` | 86.7% | 53.3% | 0.699 | +3.3pp | +0.013 | path hurts (+3.3pp any) |
| `full` | 83.3% | 56.7% | 0.686 | +0.0pp | +0.000 | — (ceiling) |

## recall@10

| config | any | all | frac | Δ any vs full | Δ frac vs full | verdict |
|---|---|---|---|---|---|---|
| `plain_rag` | 93.3% | 73.3% | 0.844 | +6.7pp | +0.076 | — (floor) |
| `no_vector` | 66.7% | 50.0% | 0.572 | -20.0pp | -0.197 | **path helps** (+20.0pp any) |
| `no_keyword` | 86.7% | 66.7% | 0.744 | +0.0pp | -0.025 | neutral |
| `no_associative` | 90.0% | 70.0% | 0.787 | +3.3pp | +0.018 | path hurts (+3.3pp any) |
| `no_graph` | 90.0% | 70.0% | 0.793 | +3.3pp | +0.024 | path hurts (+3.3pp any) |
| `full` | 86.7% | 70.0% | 0.769 | +0.0pp | +0.000 | — (ceiling) |

## recall@25

| config | any | all | frac | Δ any vs full | Δ frac vs full | verdict |
|---|---|---|---|---|---|---|
| `plain_rag` | 93.3% | 86.7% | 0.913 | +0.0pp | +0.083 | — (floor) |
| `no_vector` | 76.7% | 53.3% | 0.631 | -16.7pp | -0.200 | **path helps** (+16.7pp any) |
| `no_keyword` | 90.0% | 70.0% | 0.797 | -3.3pp | -0.033 | **path helps** (+3.3pp any) |
| `no_associative` | 93.3% | 76.7% | 0.842 | +0.0pp | +0.011 | neutral |
| `no_graph` | 93.3% | 76.7% | 0.842 | +0.0pp | +0.011 | neutral |
| `full` | 93.3% | 73.3% | 0.831 | +0.0pp | +0.000 | — (ceiling) |

## Reading the table

- **`no_X`** rows show recall when path X is disabled. A large **negative Δ** means that path was contributing substantially — disabling it hurt. A positive Δ means the path was net-harmful in this config (investigate, don't silently keep it).
- **`plain_rag`** is the vector-only floor. The gap between `plain_rag` and `full` is the *total* lift Dendric's architecture provides over standard RAG.
- **`full`** is the ceiling. All deltas are reported against it.