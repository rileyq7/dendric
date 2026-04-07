"""
Enhanced ACT-R Activation with Neuromodulators

Implements biologically-grounded activation combining:
- Base-level activation (power-law decay from access history)
- Spreading activation (placeholder for entity graph walk)
- Dopamine (multiplicative gain on base activation)
- Norepinephrine (inverted-U bonus for novelty)
- GABA (inhibition scaled by memory strength)

Reference: Anderson & Lebiere (1998), Krakauer et al. (2017)
"""

import math
import random
import time
from datetime import datetime, timezone
from typing import List, Optional


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    x = max(-10, min(10, x))
    return 1.0 / (1.0 + math.exp(-x))


def compute_activation_legacy(
    access_times: List[datetime],
    now: Optional[datetime] = None,
    decay_param: float = 0.5,
    relevance_boost: float = 0.0,
    noise_sigma: float = 0.25,
) -> float:
    """
    Original linear formula (preserved for comparison).

    Used in v0.1. Replaced by compute_activation_neuromodulated().
    """
    if now is None:
        now = datetime.now(timezone.utc)

    if not access_times:
        return -1.0 + relevance_boost

    total = 0.0
    for t_access in access_times:
        dt = (now - t_access).total_seconds()
        if dt < 1.0:
            dt = 1.0
        total += dt ** (-decay_param)

    if total <= 0:
        activation = -5.0
    else:
        activation = math.log(total)

    activation += relevance_boost

    noise = random.gauss(0, noise_sigma)
    activation += noise

    return activation


def compute_activation_neuromodulated(
    access_times: List[datetime],
    da_relevance: float = 0.5,
    ne_novelty: float = 0.5,
    gaba_inhibition: float = 0.0,
    now: Optional[datetime] = None,
    decay_param: float = 0.5,
    noise_sigma: float = 0.25,
) -> float:
    """
    Neuromodulated activation function combining:
    1. Base-level activation (ACT-R power-law)
    2. Spreading activation (placeholder)
    3. Dopamine as multiplicative gain
    4. Norepinephrine as inverted-U bonus
    5. GABA inhibition (weakness-scaled)
    6. Logistic noise (stochasticity)

    Args:
        access_times: List of datetime objects (when memory was accessed)
        da_relevance: Dopamine-related relevance signal (0.0-1.0)
        ne_novelty: Norepinephrine-related novelty signal (0.0-1.0)
        gaba_inhibition: GABA inhibition level (0.0-0.95)
        now: Current time
        decay_param: Power-law decay parameter (d=0.5 standard)
        noise_sigma: Logistic noise scale

    Returns:
        Activation level (pre-sigmoid, roughly -5 to +5)
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # ── 1. Base-level activation (ACT-R power-law decay) ──
    if not access_times:
        B = -1.0  # Never accessed penalty
    else:
        total = 0.0
        for t_access in access_times:
            dt = (now - t_access).total_seconds() / 86400.0  # Days
            if dt < 0.01:
                dt = 0.01
            total += dt ** (-decay_param)

        if total <= 0:
            B = -5.0
        else:
            B = math.log(total)

    # ── 2. Spreading activation (placeholder) ──
    # Will be replaced with entity graph walk once entity_edges are available
    SA = 0.2

    # ── 3. Dopamine as MULTIPLICATIVE gain ──
    # DA^1.5 makes high dopamine disproportionately powerful
    # Range: 1.0 (DA=0) to ~2.0 (DA=1.0)
    da_gain = 1.0 + (da_relevance ** 1.5)

    # ── 4. Norepinephrine as inverted-U bonus (Yerkes-Dodson law) ──
    # Peaks at NE=0.5, zero at extremes
    # 4*x*(1-x) is a parabola with maximum 1.0 at x=0.5
    ne_bonus = 4.0 * ne_novelty * (1.0 - ne_novelty)

    # ── 5. GABA inhibition scaled by weakness ──
    # Stronger memories (high B) feel less inhibition
    # Weak memories (low B) get suppressed more
    B_norm = sigmoid(B)  # Normalize to 0-1
    inhibition = gaba_inhibition * max(0.0, 1.0 - B_norm)

    # ── 6. Combine: (B + SA) * DA_gain + NE_bonus - inhibition ──
    A = (B + SA) * da_gain + ne_bonus - inhibition

    # ── 7. Add logistic noise (ACT-R stochasticity) ──
    # Logistic distribution: heavier tails than Gaussian
    u = random.random()
    u = max(0.001, min(0.999, u))  # Clamp to avoid log(0)
    logistic_noise = noise_sigma * math.log(u / (1.0 - u))
    A += logistic_noise

    return A


def activation_to_temperature(
    activation: float,
    min_temp: float = 0.0,
    max_temp: float = 1.0,
    midpoint: float = 0.0,
    steepness: float = 1.5,
) -> float:
    """
    Map activation to temperature via sigmoid.

    Args:
        activation: Raw activation (from compute_activation*)
        min_temp, max_temp: Temperature bounds
        midpoint: Activation value at which temp = 0.5
        steepness: Steepness of sigmoid

    Returns:
        Temperature (0.0-1.0)
    """
    t = sigmoid(steepness * (activation - midpoint))
    return min_temp + t * (max_temp - min_temp)


# Convenience function for backward compatibility
def compute_activation(
    access_times: List[datetime],
    now: Optional[datetime] = None,
    decay_param: float = 0.5,
    relevance_boost: float = 0.0,
    noise_sigma: float = 0.25,
    **neuromodulator_kwargs
) -> float:
    """
    Dispatch to neuromodulated activation if neuromodulators provided,
    else use legacy formula (for backward compatibility).
    """
    if neuromodulator_kwargs:
        return compute_activation_neuromodulated(
            access_times=access_times,
            da_relevance=neuromodulator_kwargs.get("da_relevance", 0.5),
            ne_novelty=neuromodulator_kwargs.get("ne_novelty", 0.5),
            gaba_inhibition=neuromodulator_kwargs.get("gaba_inhibition", 0.0),
            now=now,
            decay_param=decay_param,
            noise_sigma=noise_sigma,
        )
    else:
        return compute_activation_legacy(
            access_times=access_times,
            now=now,
            decay_param=decay_param,
            relevance_boost=relevance_boost,
            noise_sigma=noise_sigma,
        )
