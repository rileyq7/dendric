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

    # Spreading activation weight in temperature formula
    spreading_activation_weight: float = 0.3

    # Token erosion parameters
    erosion_base_decay: float = 0.15

    # Novelty gate threshold
    novelty_gate: float = 0.15

    # Reheat / EWC feedback during recall.
    # When False, recall is read-only with respect to memory state — required
    # for stable benchmark runs (otherwise query order and run count silently
    # mutate the importance/temperature landscape). Default True for normal use.
    recall_mutates_state: bool = field(default_factory=lambda: os.environ.get(
        "DENDRIC_RECALL_MUTATES_STATE", "1"
    ).lower() in ("1", "true", "yes"))

    # Cap on how many memories the per-query EWC update touches. The previous
    # implementation rewrote up to 500 memories per query (O(N) per recall),
    # which both bottlenecks throughput and creates query-throughput-coupled
    # decay. Now we only touch the top_k hits and an additional sample of
    # decay candidates capped here.
    ewc_update_max_decay: int = 0  # 0 = decay only via consolidation, not per-query

    # ── Déjà-vu archive trigger ────────────────────────────────────────
    # When spreading activation reaches a node strongly enough, also pull
    # archived memories linked to that node. Archived memories are
    # otherwise unreachable by normal retrieval (region != 'archive' filter
    # in vector/keyword paths), but a strong-enough association evokes them
    # back into awareness — déjà vu. Threshold is on per-entity activation
    # value after spreading; 0.0 disables.
    archive_trigger_threshold: float = 0.7

    # ── Persona ────────────────────────────────────────────────────────
    # Canonical name of the implicit owner of this memory stream. When set,
    # every ingested memory is automatically linked to this entity in the
    # graph — first-person episodic memory has an implicit owner by
    # construction (DA/NE/GABA only mean something relative to *whose*
    # goals/novelty/redundancy). For an agent's own memory, this is the
    # agent's identifier; for a chat persona, it's the persona name.
    # Empty string disables (default — back-compat with existing usage).
    persona: str = ""
    # Activation a query gets when seeded at the persona node. Lower than
    # other-entity seeds because "this is about me" is weaker evidence —
    # it's true of every memory.
    persona_seed_activation: float = 0.5
    # Whether spreading activation should fall back to the persona when no
    # query entities are in the graph. This is what makes queries like
    # "Has Jamie had X?" reach memories that say "I had X."
    persona_fallback_seed: bool = True

    # ── Lifecycle modulation at rank ──────────────────────────────────
    # Post-RRF, each candidate's fused_score is multiplied by a bounded
    # modulation function of its lifecycle state. This is what makes the
    # biological signals load-bearing at retrieval — without it, the
    # system stores temperature and DA/NE/GABA but never USES them at
    # query time. The bias is a tilt, not a filter: defaults below put
    # modulation in roughly [0.7, 1.5], so a strong RRF match on a cold
    # memory can still beat a weak match on a hot one.
    enable_lifecycle_modulation: bool = True
    mod_temp_lift: float = 0.4       # hot memories (temp=1) get +40% before clamp
    mod_da_lift: float = 0.3         # goal-relevant memories (DA=1) get +30%
    mod_ne_penalty: float = 0.3      # off-curve novelty penalized up to -30%
    mod_min: float = 0.7             # floor — strong RRF can still surface cold
    mod_max: float = 1.6             # ceiling — modulation tilts, doesn't dominate

    # ── Ablation flags ────────────────────────────────────────────────
    # All default True so production behavior is unchanged unless explicitly
    # ablated. Each flag is read at the call site of the corresponding
    # component. Used by the ablation harness (src/scripts/ablate.py).

    # Retrieval paths
    enable_vector_path: bool = True
    enable_keyword_path: bool = True
    enable_associative_path: bool = True
    enable_graph_path: bool = True

    # Temporal helpers
    enable_temporal_decomposition: bool = True   # decompose_temporal extra-vec/extra-kw
    enable_temporal_rerank: bool = True          # detect_temporal_query + temporal_rerank

    # Activation equation stages (in compute_temperature)
    activation_use_gane: bool = True             # stage 5: GANE feedback
    activation_use_noise: bool = True            # stage 7: arousal-modulated noise
