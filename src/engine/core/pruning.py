"""
MESU-Inspired Probabilistic Pruning

Reference: MESU (Metaplasticity from Synaptic Uncertainty), Nature Communications 2025

High signal variance = uncertain about this memory's value -> more prunable.
Low signal variance = confident -> protected.
"""

import random
from typing import List


def compute_signal_uncertainty(
    da_history: List[float],
    ne_history: List[float],
    usage_history: List[float],
    min_samples: int = 3,
) -> float:
    if len(da_history) < min_samples:
        return 0.5

    variances = []
    for history in [da_history, ne_history, usage_history]:
        if len(history) >= min_samples:
            window = history[-20:]
            mean = sum(window) / len(window)
            var = sum((x - mean) ** 2 for x in window) / len(window)
            variances.append(var)

    if not variances:
        return 0.5

    avg_variance = sum(variances) / len(variances)
    uncertainty = min(1.0, avg_variance / 0.1)
    return float(uncertainty)


def compute_prune_probability(
    temperature: float,
    uncertainty: float,
    retrieval_importance: float,
    access_count: int,
    cycles_since_last_access: int,
    base_prune_rate: float = 0.05,
) -> float:
    # Base vulnerability: cold + uncertain + unaccessed
    vulnerability = (1.0 - temperature) * uncertainty

    # Access recency factor
    recency_penalty = min(1.0, cycles_since_last_access / 50.0)

    # EWC protection
    protection = retrieval_importance * (1.0 - uncertainty)

    prob = base_prune_rate * vulnerability * recency_penalty * (1.0 - protection)

    # Hard floor: never prune above temperature 0.15
    if temperature > 0.15:
        prob = 0.0

    # Hard floor: never prune if accessed in last 3 cycles
    if cycles_since_last_access < 3:
        prob = 0.0

    return max(0.0, min(1.0, prob))


def should_prune(prune_probability: float) -> bool:
    return random.random() < prune_probability
