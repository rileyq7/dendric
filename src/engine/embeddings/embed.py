"""
Embedding wrapper — OpenAI text-embedding-3-small (1536-dim).

Fast, cheap ($0.02/1M tokens), no local GPU needed.

Hash-based pseudo-embeddings are available as an explicit opt-in
(MEMORY_ENGINE_ALLOW_HASH_EMBED=1) for offline tests. They produce vectors
of the right shape but are semantically meaningless — any benchmark or
production run with them silently becomes garbage. Default behavior raises
if no API key is set so misconfiguration fails loud.
"""

import os
import logging
import hashlib
import struct
import math

logger = logging.getLogger(__name__)


class EmbeddingConfigError(RuntimeError):
    """Raised when the embedding backend is misconfigured."""


_client = None
_embed_dim = 1536
_model_name = "text-embedding-3-small"
# text-embedding-3-small max input is 8192 tokens; ~4 chars/token is a safe estimate
_MAX_CHARS = 30000

# Map of supported model → expected dimension. Keeps schema/config/runtime in sync.
_MODEL_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def _hash_embed_allowed() -> bool:
    return os.environ.get("MEMORY_ENGINE_ALLOW_HASH_EMBED", "").lower() in ("1", "true", "yes")


def _get_client():
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        if _hash_embed_allowed():
            logger.warning(
                "OPENAI_API_KEY not set — using hash-based pseudo-embeddings "
                "(MEMORY_ENGINE_ALLOW_HASH_EMBED=1). Retrieval will be semantically meaningless."
            )
            return None
        raise EmbeddingConfigError(
            "OPENAI_API_KEY is not set and MEMORY_ENGINE_ALLOW_HASH_EMBED is not enabled. "
            "Set OPENAI_API_KEY for real embeddings, or set MEMORY_ENGINE_ALLOW_HASH_EMBED=1 "
            "to explicitly opt into pseudo-embeddings for offline tests."
        )

    from openai import OpenAI
    _client = OpenAI(api_key=api_key)
    logger.info(f"Initialized OpenAI embeddings client (model={_model_name}, dim={_embed_dim})")
    return _client


def assert_model_matches(expected_model: str, expected_dim: int) -> None:
    """Verify the loaded embedding model matches caller's expected config.

    Catches the silent class of bugs where config.embed_model and the actual
    runtime model drift apart, or where the schema dim doesn't match the model.
    Call once at engine startup.
    """
    if expected_model != _model_name:
        raise EmbeddingConfigError(
            f"Embedding model mismatch: config expects '{expected_model}' "
            f"but embed module is loaded with '{_model_name}'. "
            f"Vectors written by one will not be comparable to the other."
        )
    runtime_dim = _MODEL_DIMS.get(_model_name, _embed_dim)
    if runtime_dim != expected_dim:
        raise EmbeddingConfigError(
            f"Embedding dim mismatch: schema/config expects dim={expected_dim} "
            f"but model '{_model_name}' produces dim={runtime_dim}. "
            f"Run a migration or fix config.embed_dim before ingesting."
        )


def get_model(model_name: str = "text-embedding-3-small"):
    """Initialize client. Kept for API compat with callers that call get_model() at startup."""
    global _model_name
    _model_name = model_name
    return _get_client()


def get_embed_dim() -> int:
    return _embed_dim


def _truncate(text: str) -> str:
    """Truncate to stay within OpenAI's 8192 token limit."""
    return text[:_MAX_CHARS] if len(text) > _MAX_CHARS else text


def embed(text: str, model_name: str = "text-embedding-3-small") -> list:
    """Embed a single text."""
    client = _get_client()
    if client is None:
        return _hash_embed(text, _embed_dim)

    response = client.embeddings.create(model=model_name, input=[_truncate(text)])
    return response.data[0].embedding


def embed_batch(texts: list, model_name: str = "text-embedding-3-small", batch_size: int = 2048) -> list:
    """Embed multiple texts. OpenAI supports up to 2048 inputs per call."""
    client = _get_client()
    if client is None:
        return [_hash_embed(text, _embed_dim) for text in texts]

    truncated = [_truncate(t) for t in texts]
    all_embeddings = []
    for i in range(0, len(truncated), batch_size):
        chunk = truncated[i:i + batch_size]
        response = client.embeddings.create(model=model_name, input=chunk)
        all_embeddings.extend([item.embedding for item in response.data])

    return all_embeddings


def embed_query(text: str, model_name: str = "text-embedding-3-small") -> list:
    """Embed a query. Same as embed() — OpenAI doesn't distinguish doc/query."""
    return embed(text, model_name)


def _hash_embed(text: str, dim: int = 1536) -> list:
    """Hash-based pseudo-embedding for testing without API key."""
    words = text.lower().split()
    vec = [0.0] * dim

    for word in words:
        h = hashlib.sha256(word.encode()).digest()
        for i in range(min(dim, len(h) // 4)):
            val = struct.unpack("f", h[i * 4 : (i + 1) * 4])[0]
            val = max(-1.0, min(1.0, val / 1e30))
            vec[i] += val

    magnitude = math.sqrt(sum(v * v for v in vec)) or 1.0
    vec = [v / magnitude for v in vec]
    return vec
