"""
Memory record dataclass and temperature/band helpers.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime, timezone
import uuid

from .erosion import read_at_temperature


BAND_THRESHOLDS = {
    "scorching": 0.85,
    "hot": 0.70,
    "warm": 0.55,
    "room": 0.40,
    "cool": 0.25,
    "cold": 0.10,
    "trace": 0.0,
}


def get_band(temperature: float) -> str:
    if temperature >= 0.85:
        return "scorching"
    if temperature >= 0.70:
        return "hot"
    if temperature >= 0.55:
        return "warm"
    if temperature >= 0.40:
        return "room"
    if temperature >= 0.25:
        return "cool"
    if temperature >= 0.10:
        return "cold"
    return "trace"


def get_format(temperature: float) -> str:
    if temperature >= 0.70:
        return "raw"
    if temperature >= 0.55:
        return "summary"
    if temperature >= 0.40:
        return "nugget"
    if temperature >= 0.10:
        return "edges"
    return "trace"


@dataclass
class Memory:
    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Content at each compression level
    raw_content: Optional[str] = None
    structured_summary: Optional[str] = None
    knowledge_nugget: Optional[str] = None
    entity_edges: Optional[str] = None

    # Lifecycle state
    temperature: float = 1.0
    region: str = "hippocampus"  # hippocampus | neocortex | archive

    # Signals (simplified to 3)
    da_relevance: float = 0.3
    ne_novelty: float = 0.5
    usage_score: float = 0.0

    # MESU uncertainty tracking
    da_history: List[float] = field(default_factory=list)
    ne_history: List[float] = field(default_factory=list)
    signal_variance: float = 0.5

    # EWC retrieval importance
    retrieval_hits: int = 0
    retrieval_importance: float = 0.0

    # Neuromodulators (for enhanced activation)
    gaba_inhibition: float = 0.0  # Inhibition level (0.0-0.95), increases with disuse

    # Embedding (stored in DB as vector, here as list for transport)
    embedding: Optional[List[float]] = None

    # Token-level erosion (experimental)
    token_weights: Optional[List[dict]] = None  # [{token, importance, weight}, ...]

    # Temporal
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_consolidated: Optional[datetime] = None
    access_times: List[datetime] = field(default_factory=list)
    access_count: int = 0

    # Context metadata
    source: str = "direct"
    context: str = ""
    location: str = ""
    time_of_day: str = ""
    day_of_week: str = ""
    co_entities: List[str] = field(default_factory=list)
    preceding_memory_id: Optional[str] = None

    # Compression metadata
    compression_level: str = "raw"
    tokens_original: Optional[int] = None
    tokens_current: Optional[int] = None

    # Compression outputs
    knowledge_nuggets: Optional[List[dict]] = None  # [{nugget, source_sentence, score, entities}]
    last_compressed_at: Optional[datetime] = None
    compression_temperature: Optional[float] = None

    metadata: dict = field(default_factory=dict)

    @property
    def band(self) -> str:
        return get_band(self.temperature)

    @property
    def format(self) -> str:
        return get_format(self.temperature)

    @property
    def best_content(self) -> str:
        if self.temperature >= 0.70 and self.raw_content:
            # Apply token erosion: at full temp show everything,
            # as temp drops toward 0.70 threshold, low-weight tokens fade
            if self.token_weights:
                return read_at_temperature(self.token_weights, self.temperature)
            return self.raw_content
        if self.temperature >= 0.40 and self.structured_summary:
            return self.structured_summary
        if self.temperature >= 0.10 and self.knowledge_nugget:
            return self.knowledge_nugget
        if self.entity_edges:
            return self.entity_edges
        return (
            self.raw_content
            or self.structured_summary
            or self.knowledge_nugget
            or self.entity_edges
            or ""
        )

    def to_dict(self) -> dict:
        """Serialize for API responses."""
        # Reconstruct eroded text at current temperature
        active_text = None
        if self.token_weights:
            active_text = read_at_temperature(self.token_weights, self.temperature)
        elif self.raw_content:
            active_text = self.raw_content

        return {
            "id": self.id,
            "raw_content": self.raw_content,
            "active_text": active_text,
            "structured_summary": self.structured_summary,
            "knowledge_nuggets": self.knowledge_nuggets or [],
            "entity_edges": self.entity_edges,
            "best_content": self.best_content,
            "temperature": self.temperature,
            "region": self.region,
            "band": self.band,
            "signals": {
                "DA": round(self.da_relevance, 4),
                "NE": round(self.ne_novelty, 4),
                "usage": round(self.usage_score, 4),
                "GABA": round(self.gaba_inhibition, 4),
            },
            "signal_variance": self.signal_variance,
            "retrieval_importance": self.retrieval_importance,
            "retrieval_hits": self.retrieval_hits,
            "source": self.source,
            "context": self.context,
            "access_count": self.access_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "compression_level": self.compression_level,
        }
