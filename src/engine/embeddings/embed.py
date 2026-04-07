"""
Embedding wrapper — OpenAI text-embedding-3-small (1536-dim).

Fast, cheap ($0.02/1M tokens), no local GPU needed.
Fallback: hash-based pseudo-embedding for testing without API key.
"""

import os
import logging
import hashlib
import struct
import math

logger = logging.getLogger(__name__)

_client = None
_embed_dim = 1536
_model_name = "text-embedding-3-small"
# text-embedding-3-small max input is 8192 tokens; ~4 chars/token is a safe estimate
_MAX_CHARS = 30000


def _get_client():
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set — using hash-based pseudo-embeddings")
        return None

    from openai import OpenAI
    _client = OpenAI(api_key=api_key)
    logger.info(f"Initialized OpenAI embeddings client (model={_model_name}, dim={_embed_dim})")
    return _client


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
