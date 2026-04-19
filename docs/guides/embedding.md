# Embedding Layer

## Backend

Dendric embeds via OpenAI's `text-embedding-3-small` (1536-dim).

- **Model:** `text-embedding-3-small`
- **Dim:** 1536 (matches `vector(1536)` schema and `EngineConfig.embed_dim`)
- **Source:** `src/engine/embeddings/embed.py`
- **Cost:** $0.02 / 1M input tokens
- **Limits:** 8192 tokens per input; up to 2048 inputs per batch call

## API surface

```python
from engine.embeddings.embed import embed, embed_batch, embed_query

vec  = embed("some text")                 # single, returns list[float] of length 1536
vecs = embed_batch(["a", "b", "c"])       # batched, single OpenAI call (chunks of 2048)
qvec = embed_query("the question text")   # alias of embed(); kept for symmetry
```

All three require `OPENAI_API_KEY` in the environment.

## Configuration & safety

The embed module enforces three invariants at engine startup:

1. `OPENAI_API_KEY` must be set, **or** `MEMORY_ENGINE_ALLOW_HASH_EMBED=1` must be
   explicitly set to opt into hash-based pseudo-embeddings (offline tests only).
2. The runtime model name must match `EngineConfig.embed_model`.
3. The runtime model's dim must match `EngineConfig.embed_dim` and the pgvector schema.

Mismatches raise `EmbeddingConfigError` rather than silently producing garbage vectors.

## Hash fallback (offline tests only)

Setting `MEMORY_ENGINE_ALLOW_HASH_EMBED=1` enables a SHA-based pseudo-embedding when
no API key is available. **These vectors are semantically meaningless** — vector
recall will degrade to a hash-bag-of-words. Never use this for benchmarks or
anything you intend to measure.

## Switching models

To change embedding backends:

1. Pick a model and add it to `_MODEL_DIMS` in `embed.py`.
2. Update `EngineConfig.embed_model` and `embed_dim`.
3. Migrate the pgvector schema (`migrations.py`) to the new dim.
4. Re-embed existing memories — vectors from different models are not comparable.
