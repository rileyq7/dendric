"""
Token-level importance scoring and erosion.

Differential compression: high-importance tokens (entities, numbers, key terms)
survive consolidation better than stopwords and filler.

This enables memories to gracefully degrade rather than binary compression,
and provides temperature-dependent fidelity (read at different temperatures).
"""

import re
from typing import List, Dict, Optional

STOPWORDS = {
    'the', 'a', 'an', 'is', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'shall', 'can',
    'at', 'in', 'on', 'to', 'for', 'of', 'with', 'by', 'from',
    'and', 'but', 'or', 'so', 'yet', 'nor', 'not',
    'his', 'her', 'its', 'their', 'my', 'your', 'our',
    'this', 'that', 'these', 'those', 'it', 'he', 'she', 'they', 'we',
    'as', 'if', 'than', 'then', 'also', 'very', 'just', 'about',
    'which', 'what', 'when', 'where', 'who', 'why', 'how',
}

UNITS = {'kg', 'mg', 'ml', 'cm', 'mm', 'gb', 'mb', 'kb', 'tb', '%', 'hrs', 'min', 's'}


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization."""
    return text.split()


def score_token_importance(
    tokens: List[str],
    known_entities: Optional[List[str]] = None,
) -> List[Dict[str, any]]:
    """
    Score each token's importance (0.0-1.0).

    Signals:
    - Entity: Capitalized or in entity list → 0.95
    - Numeric: Contains digits or is a unit → 0.90
    - Rarity: Not a stopword, length > 3 → 0.5-0.8
    - Proximity: Near an entity or number → +0.4
    - Stopword: In stopword set → -0.3

    Args:
        tokens: List of word tokens
        known_entities: Optional list of known entity names

    Returns:
        List of {token, importance, weight} dicts
    """
    entity_set = {e.lower() for e in (known_entities or [])}
    results = []

    for i, token in enumerate(tokens):
        signals = []

        # Entity signal
        is_entity = (
            (i > 0 and token[0].isupper() and not token.isupper()) or
            (token.lower() in entity_set)
        )
        if is_entity:
            signals.append(0.95)

        # Numeric signal
        if re.search(r'\d', token):
            signals.append(0.90)
        elif token.lower() in UNITS:
            signals.append(0.85)

        # Rarity signal
        if token.lower() not in STOPWORDS and len(token) > 3:
            rarity = 0.5 + min(0.3, len(token) * 0.03)
            signals.append(rarity)

        # Proximity signal
        window_start = max(0, i - 2)
        window_end = min(len(tokens), i + 3)
        window = tokens[window_start:window_end]

        near_important = any(
            (j != i - window_start and (
                window[j][0].isupper() or re.search(r'\d', window[j])
            ))
            for j in range(len(window)) if j != (i - window_start)
        )
        if near_important:
            signals.append(0.4)

        # Stopword penalty
        if token.lower() in STOPWORDS:
            signals.append(-0.3)

        # Combine: max positive + sum negatives
        pos = [s for s in signals if s > 0]
        neg = [s for s in signals if s < 0]
        importance = (max(pos) if pos else 0.1) + sum(neg)
        importance = max(0.05, min(0.95, importance))

        results.append({
            'token': token,
            'importance': round(importance, 3),
            'weight': 1.0,  # Will be eroded over time
        })

    return results


def erode_tokens(
    token_weights: List[Dict],
    base_decay: float = 0.15,
) -> List[Dict]:
    """
    Erode tokens differentially based on importance.

    High-importance tokens decay slowly (survive longer).
    Low-importance tokens decay fast (erased first).

    Args:
        token_weights: List of {token, importance, weight} dicts
        base_decay: Base decay rate (0.0-1.0)

    Returns:
        Updated token_weights with weights decayed
    """
    for tw in token_weights:
        importance = tw.get('importance', 0.5)
        decay_rate = base_decay * (1.0 - importance)
        tw['weight'] = tw.get('weight', 1.0) * (1.0 - decay_rate)
    return token_weights


def read_at_temperature(
    token_weights: List[Dict],
    temperature: float,
) -> str:
    """
    Reconstruct memory content at a given temperature/fidelity.

    Higher temperature = full fidelity (all tokens visible).
    Lower temperature = eroded (only high-importance tokens visible).

    Threshold = 1.0 - temperature:
    - temp 1.0 → threshold 0.0 → all tokens (weight > 0.0)
    - temp 0.5 → threshold 0.5 → only weight > 0.5 (medium importance)
    - temp 0.1 → threshold 0.9 → only weight > 0.9 (very high importance)

    Args:
        token_weights: List of {token, importance, weight} dicts
        temperature: Temperature (0.0-1.0)

    Returns:
        Reconstructed text
    """
    if temperature >= 0.95:
        # Near full temperature: show all tokens
        return ' '.join(tw['token'] for tw in token_weights)

    threshold = 1.0 - temperature
    visible = [
        tw['token']
        for tw in token_weights
        if tw.get('weight', 1.0) > threshold
    ]
    return ' '.join(visible) if visible else "(eroded to empty)"


def measure_token_preservation(
    token_weights_original: List[Dict],
    token_weights_eroded: List[Dict],
) -> Dict[str, float]:
    """
    Measure what's preserved after erosion.

    Returns:
        {
            'tokens_original': count,
            'tokens_surviving': count,
            'importance_original': avg,
            'importance_surviving': avg,
            'preservation_rate': 0.0-1.0,
        }
    """
    original_count = len(token_weights_original)
    eroded_count = len([tw for tw in token_weights_eroded if tw['weight'] > 0.01])

    original_importance = sum(tw['importance'] for tw in token_weights_original) / original_count if original_count > 0 else 0
    eroded_importance = sum(tw['importance'] for tw in token_weights_eroded if tw['weight'] > 0.01) / eroded_count if eroded_count > 0 else 0

    return {
        'tokens_original': original_count,
        'tokens_surviving': eroded_count,
        'importance_original': round(original_importance, 3),
        'importance_surviving': round(eroded_importance, 3),
        'preservation_rate': round(eroded_count / original_count if original_count > 0 else 0, 3),
    }
