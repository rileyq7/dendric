"""
Phase 3: Full Lifecycle Test with Entity Graph

This is the comprehensive validation:
1. Ingest 5k-250k papers with entity graph
2. Validate spreading activation vs citations
3. Compare entity graph impact on memory clustering
4. Generate case study with real metrics

Expected outcomes:
- NDCG >0.65 for citation correlation
- Recall@10 >50% for cited papers
- Co-citation clustering >30% high-overlap
- Temperature gradient stable across entity graph sizes
"""

import os
import sys
import logging
import json
from typing import List, Dict
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.scripts.s2orc_loader import load_s2orc_papers, format_paper_for_memory, build_citation_graph
from src.engine.core.ingest_with_entities import batch_ingest_with_entities
from src.engine.retrieval.spreading_activation import (
    retrieve_with_spreading_activation,
    prepare_entity_cache,
)
from src.engine.retrieval.citation_validation import (
    validate_spreading_activation_vs_citations,
    compute_co_citation_overlap,
    citation_graph_stats,
)
from src.engine.storage.entity_graph import EntityGraphStore
from src.engine.core.memory import Memory
import psycopg2
import psycopg2.extras


class SimpleMemoryStore:
    """Simple in-memory store."""
    def __init__(self):
        self.memories = []

    def add(self, mem: Memory):
        self.memories.append(mem)

    def get(self, mem_id: str) -> Memory:
        for mem in self.memories:
            if mem.id == mem_id:
                return mem
        return None


def setup_database(db_url: str = "postgresql://localhost/mnestic_phase3"):
    """Connect to PostgreSQL."""
    try:
        conn = psycopg2.connect(db_url)
        conn.autocommit = True
        logger.info(f"Connected to {db_url}")

        from engine.storage.migrations import run_migrations
        run_migrations(conn)

        conn.autocommit = False
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        try:
            conn = psycopg2.connect("postgresql://localhost/postgres")
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute("CREATE DATABASE mnestic_phase3")
            cur.close()
            conn.close()
            logger.info("Created mnestic_phase3 database")
            return setup_database(db_url)
        except:
            return None


def run_phase3(limit: int = 5000, sample_queries: int = 50):
    """Run Phase 3: Full lifecycle test with entity graph."""
    logger.info("=" * 70)
    logger.info(f"Phase 3: Entity Graph Lifecycle Test ({limit} papers)")
    logger.info("=" * 70)

    start_time = datetime.now()

    # ── Setup ──
    db_conn = setup_database()
    if not db_conn:
        logger.error("Database setup failed. Exiting.")
        return

    store = SimpleMemoryStore()

    # Clear existing
    try:
        EntityGraphStore(db_conn).clear_entity_graph()
        with db_conn.cursor() as cur:
            cur.execute("TRUNCATE memories CASCADE")
        db_conn.commit()
        logger.info("Cleared existing data")
    except Exception as e:
        logger.error(f"Failed to clear data: {e}")

    # ── Load papers ──
    logger.info(f"\nLoading papers (limit={limit})...")
    papers = load_s2orc_papers(limit=limit)
    if not papers:
        logger.error("Failed to load papers. Exiting.")
        return

    logger.info(f"Loaded {len(papers)} papers")

    # Build citation graph
    citation_edges, citation_counts = build_citation_graph(papers)
    papers_by_id = {p["paper_id"]: p for p in papers}

    logger.info(f"\nCitation Graph:")
    cite_stats = citation_graph_stats(papers_by_id, citation_edges)
    for key, value in cite_stats.items():
        logger.info(f"  {key}: {value}")

    # ── Ingest with entity graph ──
    logger.info(f"\nIngesting {len(papers)} papers with entity extraction...")
    formatted = [format_paper_for_memory(p) for p in papers]

    ingest_start = datetime.now()
    memories = batch_ingest_with_entities(formatted, store, db_conn)
    ingest_time = (datetime.now() - ingest_start).total_seconds()

    logger.info(f"Ingested {len(memories)} memories in {ingest_time:.1f}s")
    logger.info(f"  Average: {ingest_time/len(memories):.2f}s per paper")

    # ── Entity graph stats ──
    logger.info("\nEntity Graph Statistics:")
    entity_graph = EntityGraphStore(db_conn)
    graph_stats = entity_graph.entity_graph_stats()
    for key, value in graph_stats.items():
        logger.info(f"  {key}: {value}")

    # ── Prepare caches ──
    logger.info("\nPreparing retrieval caches...")
    entity_cache, entity_fan_count = prepare_entity_cache(db_conn)

    # Map paper_id -> memory_id
    paper_to_memory_id = {}
    with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT id, metadata FROM memories WHERE metadata IS NOT NULL")
        rows = cur.fetchall()
        for row in rows:
            metadata = row.get('metadata', {})
            if isinstance(metadata, dict):
                paper_id = metadata.get("paper_id")
                if paper_id:
                    paper_to_memory_id[paper_id] = str(row['id'])

    logger.info(f"Mapped {len(paper_to_memory_id)} papers to memories")

    # ── Query evaluation ──
    logger.info(f"\nTesting spreading activation on {sample_queries} query papers...")

    # Select papers with citations as queries
    query_papers = [
        pid for pid, cited in citation_edges.items()
        if len(cited) > 0
    ][:sample_queries]

    retrieval_results = {}

    for i, query_paper_id in enumerate(query_papers):
        query_memory_id = paper_to_memory_id.get(query_paper_id)
        if not query_memory_id:
            continue

        query_mem = store.get(query_memory_id)
        if not query_mem:
            continue

        # Create candidates from all memories
        candidates = [
            {
                "memory_id": mem.id,
                "vector_score": 0.5,
                "keyword_score": 0.5,
            }
            for mem in memories
        ]

        # Retrieve with SA
        ranked = retrieve_with_spreading_activation(
            query_mem.raw_content,
            candidates,
            db_conn,
            entity_cache=entity_cache,
            entity_fan_count=entity_fan_count,
            top_k=min(100, len(memories)),
        )

        retrieval_results[query_paper_id] = ranked

        if (i + 1) % 10 == 0:
            logger.info(f"  Completed {i + 1} / {len(query_papers)} queries")

    # ── Citation validation ──
    logger.info("\nValidating spreading activation vs citations...")
    citation_metrics = validate_spreading_activation_vs_citations(
        papers_by_id,
        citation_edges,
        entity_cache,
        entity_fan_count,
        paper_to_memory_id,
        retrieval_results,
    )

    logger.info("\nCitation Validation Results:")
    logger.info(f"  Citation pairs tested: {citation_metrics['citation_pairs_tested']}")
    logger.info(f"  Total citations: {citation_metrics['total_citations']}")

    if citation_metrics.get("total_citations", 0) > 0:
        logger.info(f"  Recall@1: {citation_metrics.get('recall_at_1', 0):.1%}")
        logger.info(f"  Recall@5: {citation_metrics.get('recall_at_5', 0):.1%}")
        logger.info(f"  Recall@10: {citation_metrics.get('recall_at_10', 0):.1%}")
        logger.info(f"  Mean NDCG: {citation_metrics.get('mean_ndcg', 0):.3f}")

    # ── Co-citation validation ──
    logger.info("\nValidating co-citation clustering...")
    co_cite_metrics = compute_co_citation_overlap(
        papers_by_id,
        citation_edges,
        entity_cache,
        paper_to_memory_id,
    )

    logger.info(f"  Co-citation pairs tested: {co_cite_metrics['co_citation_pairs']}")
    if co_cite_metrics.get("co_citation_pairs", 0) > 0:
        logger.info(f"  High overlap ratio: {co_cite_metrics.get('high_overlap_ratio', 0):.1%}")

    # ── Summary ──
    total_time = (datetime.now() - start_time).total_seconds()

    logger.info("\n" + "=" * 70)
    logger.info("Phase 3 Summary:")
    logger.info(f"  Papers: {len(papers)}")
    logger.info(f"  Entities: {graph_stats['entities']}")
    logger.info(f"  Memory-entity links: {graph_stats['memory_entity_links']}")
    logger.info(f"  Entity edges: {graph_stats['edges']}")
    logger.info(f"  Ingest time: {ingest_time:.1f}s ({ingest_time/len(memories):.2f}s/paper)")
    logger.info(f"  Total time: {total_time:.1f}s")

    if citation_metrics.get("total_citations", 0) > 0:
        logger.info(f"\n  Citation Metrics:")
        logger.info(f"    Recall@10: {citation_metrics.get('recall_at_10', 0):.1%}")
        logger.info(f"    Mean NDCG: {citation_metrics.get('mean_ndcg', 0):.3f}")

    if co_cite_metrics.get("co_citation_pairs", 0) > 0:
        logger.info(f"\n  Co-citation Metrics:")
        logger.info(f"    High overlap: {co_cite_metrics.get('high_overlap_ratio', 0):.1%}")

    # Check success criteria
    logger.info("\n" + "=" * 70)
    logger.info("Success Criteria:")

    success = True
    if citation_metrics.get("recall_at_10", 0) > 0.5:
        logger.info("  ✓ Recall@10 >50%")
    else:
        logger.info(f"  ✗ Recall@10 = {citation_metrics.get('recall_at_10', 0):.1%} (target >50%)")
        success = False

    if citation_metrics.get("mean_ndcg", 0) > 0.65:
        logger.info("  ✓ NDCG >0.65")
    else:
        logger.info(f"  ✗ NDCG = {citation_metrics.get('mean_ndcg', 0):.3f} (target >0.65)")

    if co_cite_metrics.get("high_overlap_ratio", 0) > 0.3:
        logger.info("  ✓ Co-citation overlap >30%")
    else:
        logger.info(f"  ✗ Co-citation overlap = {co_cite_metrics.get('high_overlap_ratio', 0):.1%} (target >30%)")

    logger.info("\n✓ Phase 3 validation complete!\n")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "papers": len(papers),
        "ingest_time": ingest_time,
        "total_time": total_time,
        "graph_stats": graph_stats,
        "citation_metrics": {
            k: v for k, v in citation_metrics.items()
            if k != "details"
        },
        "co_cite_metrics": co_cite_metrics,
        "success": success,
    }

    with open("phase3_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to phase3_results.json")

    db_conn.close()
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5000, help="Max papers to load")
    parser.add_argument("--queries", type=int, default=50, help="Sample query papers")
    args = parser.parse_args()

    run_phase3(limit=args.limit, sample_queries=args.queries)
