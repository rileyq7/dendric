"""
Engine Configuration — All tuneable parameters in one place.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class EngineConfig:
    # Project
    project_name: str = "memory_engine"

    # Database
    db_url: str = field(default_factory=lambda: os.environ.get(
        "DATABASE_URL", "postgresql://localhost:5432/memory_engine"
    ))

    # Embedding model
    embed_model: str = "text-embedding-3-small"
    embed_dim: int = 1536

    # User goals (for DA scoring)
    goals: List[str] = field(default_factory=lambda: [
        "memory engine", "consolidation", "AI infrastructure"
    ])

    # ACT-R decay parameters
    decay_param: float = 0.5
    da_boost_factor: float = 0.5
    noise_sigma: float = 0.25
    activation_midpoint: float = 0.5  # Shift midpoint right for gentler mapping
    activation_steepness: float = 0.8  # Reduce steepness (was 1.5, too aggressive)

    # MESU pruning parameters
    base_prune_rate: float = 0.05
    min_prune_temp: float = 0.15
    min_prune_cycles: int = 3
    uncertainty_window: int = 20

    # EWC protection parameters
    importance_learning_rate: float = 0.1
    importance_decay_rate: float = 0.01

    # Compression thresholds
    summary_threshold: float = 0.70
    nugget_threshold: float = 0.40
    edges_threshold: float = 0.20
    archive_threshold: float = 0.10

    # Reheat parameters
    base_reheat: float = 0.15
    coldness_reheat_factor: float = 0.3

    # Retrieval fusion weights
    vector_weight: float = 1.5
    keyword_weight: float = 0.8
    associative_weight: float = 1.0
    graph_weight: float = 1.2
    temporal_weight: float = 2.5
    rrf_k: int = 60
    recency_bonus: float = 0.002
    session_decay: float = 0.85
    overfetch_multiplier: int = 3

    # Compression model
    compression_model: str = "claude-haiku-4-5-20251001"
    compression_api_key: str = field(default_factory=lambda: os.environ.get(
        "ANTHROPIC_API_KEY", ""
    ))

    # Spreading activation weight in temperature formula
    spreading_activation_weight: float = 0.3

    # Token erosion parameters
    erosion_base_decay: float = 0.15

    # Novelty gate threshold
    novelty_gate: float = 0.15
