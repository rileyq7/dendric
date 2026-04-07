"""
EWC-Inspired Retrieval Importance

Reference: Kirkpatrick et al. (2017) Elastic Weight Consolidation, DeepMind

Memories that frequently appear in top-k results have high importance.
High importance protects against compression and pruning.
"""


def update_retrieval_importance(
    was_in_top_k: bool,
    current_importance: float,
    learning_rate: float = 0.1,
    decay_rate: float = 0.01,
) -> float:
    if was_in_top_k:
        new_importance = current_importance + learning_rate * (1.0 - current_importance)
    else:
        new_importance = current_importance * (1.0 - decay_rate)

    return max(0.0, min(1.0, new_importance))


def compute_compression_resistance(
    retrieval_importance: float,
    uncertainty: float,
    temperature: float,
) -> float:
    resistance = retrieval_importance * (1.0 - uncertainty)

    # Warm memories get baseline resistance regardless
    if temperature > 0.6:
        resistance = max(resistance, 0.3)

    return resistance
