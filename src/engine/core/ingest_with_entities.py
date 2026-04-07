"""
Ingest pipeline that extracts and indexes entities during memory creation.
This is the integration point for the entity graph into the lifecycle.
"""

import logging
import uuid
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

from .memory import Memory
from .entity_extraction import extract_entities, extract_entities_with_metadata, compute_entity_salience
from .signals_enhanced import compute_da, compute_ne, compute_gaba
from .activation import compute_temperature
from ..embeddings.embed import embed
from ..storage.entity_graph import EntityGraphStore
from ..utils import extract_session_id

logger = logging.getLogger(__name__)


def ingest_memory_with_entities(
    content: str,
    store,  # In-memory store
    db_conn,  # PostgreSQL connection
    goals: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    source: str = "direct",
) -> Memory:
    """
    Ingest a memory with full entity graph integration.

    Flow:
    1. Create memory object
    2. Extract entities from content
    3. Build entity graph (upsert entities, create co-occurrence edges)
    4. Compute signals (DA, NE, GABA)
    5. Compute temperature
    6. Store in both in-memory and PostgreSQL

    Args:
        content: Memory content (raw text)
        store: In-memory MemoryStore
        db_conn: PostgreSQL connection
        goals: Current user goals (for DA computation)
        metadata: Additional metadata
        source: Where memory came from

    Returns:
        Memory object (stored in both backends)
    """
    memory_id = str(uuid.uuid4())
    now = datetime.now()

    logger.info(f"Ingesting memory {memory_id}: {content[:60]}...")

    # ── 1. Create memory object & store first (before entity linking) ──
    # Extract entities from text and metadata
    entity_graph = EntityGraphStore(db_conn)
    known_entity_names = entity_graph.get_entity_names()

    # Use metadata-enhanced extraction if available (authors, venue)
    authors = metadata.get("authors") if metadata else None
    venue = metadata.get("venue") if metadata else None
    extracted_entities = extract_entities_with_metadata(
        content,
        known_entities=known_entity_names,
        authors=authors,
        venue=venue
    )

    logger.debug(f"Extracted {len(extracted_entities)} entities from content")

    # ── 3. Compute embedding ──
    embedding = embed(content)

    # ── 4. Compute signals ──
    # For initial ingest, use baseline signals (will be updated during retrieval)
    da_relevance = compute_da(access_count=0, avg_outcome=0.5, user_rating=0.0)

    # NE and GABA: baseline (low novelty initially, moderate GABA)
    store_embeddings = [m.embedding for m in store.memories if m.embedding]
    ne_novelty = compute_ne(embedding, store_embeddings) if store_embeddings else 0.5
    gaba_inhibition = compute_gaba(embedding, store_embeddings) if store_embeddings else 0.3

    # ── 5. Compute temperature (without spreading activation at ingest) ──
    # At ingest, we don't have context for spreading activation yet
    # It will be computed during retrieval
    temperature = compute_temperature(
        accesses_days_ago=[0.001],  # Just ingested
        da_relevance=da_relevance,
        ne_novelty=ne_novelty,
        gaba_inhibition=gaba_inhibition,
        spreading_activation=0.0,  # Will be computed at retrieval time
    )

    # ── 6. Create memory object ──
    mem = Memory(
        id=memory_id,
        raw_content=content,
        temperature=temperature,
        region="hippocampus",
        da_relevance=da_relevance,
        ne_novelty=ne_novelty,
        usage_score=0.0,
        retrieval_hits=0,
        retrieval_importance=0.0,
        embedding=embedding,
        created_at=now,
        last_accessed=now,
        access_times=[now],
        access_count=1,
        source=source,
        context=metadata.get("context", "") if metadata else "",
        location=metadata.get("location", "") if metadata else "",
        co_entities=[name for name, _, _ in extracted_entities],
        compression_level="raw",
        tokens_original=len(content.split()),
        tokens_current=len(content.split()),
    )

    # ── 7. Insert memory row into DB first ──
    # Convert metadata to JSON string for JSONB column
    metadata_json = json.dumps(metadata) if metadata else '{}'

    with db_conn.cursor() as cur:
        cur.execute("""
            INSERT INTO memories (
                id, raw_content, temperature, region,
                da_relevance, ne_novelty, usage_score,
                retrieval_hits, retrieval_importance,
                embedding, created_at, last_accessed,
                access_times, access_count, source, context,
                co_entities, compression_level, tokens_original, tokens_current,
                metadata
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb
            )
        """, (
            memory_id, content, temperature, "hippocampus",
            da_relevance, ne_novelty, 0.0,
            0, 0.0,
            embedding, now, now,
            [now], 1, source, mem.context,
            [e[0] for e in extracted_entities], "raw",
            mem.tokens_original, mem.tokens_current,
            metadata_json,
        ))
    db_conn.commit()

    # ── 8. Build entity graph (now memory exists) ──
    session_id = extract_session_id(mem.context)
    entity_ids = []
    for name, etype, canonical in extracted_entities:
        # Upsert entity and get ID
        entity_id = entity_graph.upsert_entity(canonical, etype, name)
        entity_ids.append(entity_id)

        # Track which sessions mention this entity
        entity_graph.update_entity_session(entity_id, session_id)

        # Link memory to entity
        salience = compute_entity_salience(name, etype, content)
        entity_graph.insert_memory_entity(memory_id, entity_id, salience)

    # Create/strengthen co-occurrence edges for all entity pairs
    for i in range(len(entity_ids)):
        for j in range(i + 1, len(entity_ids)):
            a, b = sorted([entity_ids[i], entity_ids[j]])
            entity_graph.upsert_entity_edge(a, b)

    # ── 9. Store in memory store ──
    store.add(mem)

    logger.info(f"Ingested {memory_id} with {len(entity_ids)} entities")
    return mem


def batch_ingest_with_entities(
    papers: List[Dict[str, str]],
    store,
    db_conn,
    goals: Optional[List[str]] = None,
    batch_size: int = 32,
) -> List[Memory]:
    """
    Ingest multiple papers at once with batched embedding for efficiency.

    Args:
        papers: List of {raw_content, source, context, metadata}
        store: In-memory store
        db_conn: PostgreSQL connection
        goals: User goals
        batch_size: How many papers to embed in one batch (default 32)

    Returns:
        List of ingested memories
    """
    from ..embeddings.embed import get_embed_dim

    memories = []
    entity_graph = EntityGraphStore(db_conn)
    known_entity_names = entity_graph.get_entity_names()

    # Batch embeddings
    texts_to_embed = [p.get("raw_content", "") for p in papers]

    logger.info(f"Computing embeddings for {len(papers)} papers (batch_size={batch_size})...")
    embeddings_list = batch_embed_texts(texts_to_embed, batch_size=batch_size)

    for i, (paper, embedding) in enumerate(zip(papers, embeddings_list)):
        try:
            # Extract entities from text and metadata
            content = paper.get("raw_content", "")
            authors = paper.get("authors") if paper.get("authors") else None
            venue = paper.get("venue") if paper.get("venue") else None
            extracted_entities = extract_entities_with_metadata(
                content,
                known_entities=known_entity_names,
                authors=authors,
                venue=venue
            )

            logger.debug(f"Extracted {len(extracted_entities)} entities")

            memory_id = str(uuid.uuid4())
            now = datetime.now()

            # Compute signals
            goals = goals or []
            da_relevance = compute_da(access_count=0, avg_outcome=0.5, user_rating=0.0)
            store_embeddings = [m.embedding for m in store.memories if m.embedding]
            ne_novelty = compute_ne(embedding, store_embeddings) if store_embeddings else 0.5
            gaba_inhibition = compute_gaba(embedding, store_embeddings) if store_embeddings else 0.3

            # Compute temperature
            temperature = compute_temperature(
                accesses_days_ago=[0.001],
                da_relevance=da_relevance,
                ne_novelty=ne_novelty,
                gaba_inhibition=gaba_inhibition,
                spreading_activation=0.0,
            )

            # Create memory object
            mem = Memory(
                id=memory_id,
                raw_content=content,
                temperature=temperature,
                region="hippocampus",
                da_relevance=da_relevance,
                ne_novelty=ne_novelty,
                usage_score=0.0,
                retrieval_hits=0,
                retrieval_importance=0.0,
                embedding=embedding,
                created_at=now,
                last_accessed=now,
                access_times=[now],
                access_count=1,
                source=paper.get("source", "batch"),
                context=paper.get("metadata", {}).get("context", "") if paper.get("metadata") else "",
                location=paper.get("metadata", {}).get("location", "") if paper.get("metadata") else "",
                co_entities=[name for name, _, _ in extracted_entities],
                compression_level="raw",
                tokens_original=len(content.split()),
                tokens_current=len(content.split()),
            )

            # Convert metadata to JSON
            metadata_json = json.dumps(paper.get("metadata", {})) if paper.get("metadata") else '{}'

            # Insert memory row
            with db_conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO memories (
                        id, raw_content, temperature, region,
                        da_relevance, ne_novelty, usage_score,
                        retrieval_hits, retrieval_importance,
                        embedding, created_at, last_accessed,
                        access_times, access_count, source, context,
                        co_entities, compression_level, tokens_original, tokens_current,
                        metadata
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb
                    )
                """, (
                    memory_id, content, temperature, "hippocampus",
                    da_relevance, ne_novelty, 0.0,
                    0, 0.0,
                    embedding, now, now,
                    [now], 1, paper.get("source", "batch"), mem.context,
                    [e[0] for e in extracted_entities], "raw",
                    mem.tokens_original, mem.tokens_current,
                    metadata_json,
                ))
            db_conn.commit()

            # Build entity graph
            session_id = extract_session_id(mem.context)
            entity_ids = []
            for name, etype, canonical in extracted_entities:
                entity_id = entity_graph.upsert_entity(canonical, etype, name)
                entity_ids.append(entity_id)

                entity_graph.update_entity_session(entity_id, session_id)

                salience = compute_entity_salience(name, etype, content)
                entity_graph.insert_memory_entity(memory_id, entity_id, salience)

            # Create co-occurrence edges
            for j in range(len(entity_ids)):
                for k in range(j + 1, len(entity_ids)):
                    a, b = sorted([entity_ids[j], entity_ids[k]])
                    entity_graph.upsert_entity_edge(a, b)

            # Store in memory store
            store.add(mem)
            memories.append(mem)

            if (i + 1) % 100 == 0:
                logger.info(f"Ingested {i + 1} / {len(papers)} papers")

        except Exception as e:
            logger.error(f"Failed to ingest paper {i}: {e}")
            continue

    logger.info(f"Batch ingest complete: {len(memories)} / {len(papers)} papers")
    return memories


def batch_embed_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Compute embeddings for multiple texts using native model batching.

    This uses SentenceTransformer's efficient batching which:
    - Processes multiple texts in parallel on CPU/GPU
    - Caches the model between batches
    - Reduces overhead vs single-text embedding

    Args:
        texts: List of text strings to embed
        batch_size: How many to embed in parallel (tradeoff with memory)

    Returns:
        List of embedding vectors
    """
    from ..embeddings.embed import embed_batch

    if not texts:
        return []

    logger.info(f"Embedding {len(texts)} texts with batch_size={batch_size}")
    embeddings = embed_batch(texts, batch_size=batch_size)

    return embeddings
