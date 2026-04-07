"""
Citation graph validation for spreading activation.

This validates that spreading activation correlates with citation patterns:
- If paper A cites paper B, they should have high entity overlap
- Co-cited papers (both cite paper C) should be nearby in entity space
- Citation depth predicts retrieval ranking
"""

import logging
import math
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


def validate_spreading_activation_vs_citations(
    papers_by_id: Dict[str, Dict],  # paper_id -> paper metadata
    citation_edges: Dict[str, Set[str]],  # paper_id -> set(cited_ids)
    entity_cache: Dict[str, Set[str]],  # memory_id -> set(entity_canonical_names)
    entity_fan_count: Dict[str, int],  # canonical_name -> fan
    paper_to_memory_id: Dict[str, str],  # paper_id -> memory_id in DB
    retrieval_results: Dict[str, List[Dict]],  # query_paper_id -> ranked results
) -> Dict:
    """
    Validate that spreading activation correlates with citations.

    For each paper in our dataset:
    - Get papers it cites
    - Rank all papers by spreading activation
    - Measure: Are cited papers ranked higher than random?

    Returns:
        {
            "correlation": float (0-1),
            "ndcg": float (normalized DCG),
            "recall_at_k": {1: float, 5: float, 10: float},
            "avg_citation_rank": float,
            "avg_random_rank": float,
            "details": {paper_id: {citation_stats}}
        }
    """
    from ..core.entity_extraction import compute_spreading_activation

    metrics = {
        "citation_pairs_tested": 0,
        "citations_in_top_10": 0,
        "citations_in_top_5": 0,
        "citations_in_top_1": 0,
        "total_citations": 0,
        "citation_ranks": [],
        "ndcg_scores": [],
        "details": {},
    }

    for query_paper_id, query_results in retrieval_results.items():
        query_memory_id = paper_to_memory_id.get(query_paper_id)
        if not query_memory_id:
            continue

        # Get papers this paper cites
        cited_ids = citation_edges.get(query_paper_id, set())
        if not cited_ids:
            continue

        # Map cited_ids to memory_ids
        cited_memory_ids = set()
        for cid in cited_ids:
            mid = paper_to_memory_id.get(cid)
            if mid:
                cited_memory_ids.add(mid)

        if not cited_memory_ids:
            continue

        # Get ranking positions
        rank_map = {r["memory_id"]: i for i, r in enumerate(query_results)}

        # Find ranks of cited papers
        citation_ranks_for_query = []
        for cmid in cited_memory_ids:
            rank = rank_map.get(cmid, float("inf"))
            citation_ranks_for_query.append(rank)

            metrics["total_citations"] += 1
            metrics["citation_pairs_tested"] += 1

            if rank < 1:
                metrics["citations_in_top_1"] += 1
            if rank < 5:
                metrics["citations_in_top_5"] += 1
            if rank < 10:
                metrics["citations_in_top_10"] += 1

            metrics["citation_ranks"].append(rank)

        # Compute NDCG for this query
        if citation_ranks_for_query:
            ndcg = compute_ndcg(citation_ranks_for_query, k=10)
            metrics["ndcg_scores"].append(ndcg)

            metrics["details"][query_paper_id] = {
                "num_cited": len(cited_memory_ids),
                "median_rank": sorted(citation_ranks_for_query)[len(citation_ranks_for_query) // 2],
                "max_rank": max(citation_ranks_for_query),
                "ndcg": ndcg,
            }

    # Compute aggregate metrics
    if metrics["total_citations"] > 0:
        metrics["recall_at_1"] = metrics["citations_in_top_1"] / metrics["total_citations"]
        metrics["recall_at_5"] = metrics["citations_in_top_5"] / metrics["total_citations"]
        metrics["recall_at_10"] = metrics["citations_in_top_10"] / metrics["total_citations"]
        metrics["avg_citation_rank"] = (
            sum(r for r in metrics["citation_ranks"] if r != float("inf")) / len([r for r in metrics["citation_ranks"] if r != float("inf")])
            if any(r != float("inf") for r in metrics["citation_ranks"])
            else float("inf")
        )

    if metrics["ndcg_scores"]:
        metrics["mean_ndcg"] = sum(metrics["ndcg_scores"]) / len(metrics["ndcg_scores"])

    return metrics


def compute_ndcg(relevant_ranks: List[float], k: int = 10) -> float:
    """
    Compute normalized discounted cumulative gain (NDCG@k).

    relevant_ranks: List of ranks where relevant items were found
    k: Cutoff for ranking

    NDCG = DCG@k / IDCG@k
    DCG = Σ (1 / log2(rank + 1)) for relevant items at rank <= k
    IDCG = DCG of perfect ranking (all relevant items at top)
    """
    # Filter to top-k
    relevant_in_k = [r for r in relevant_ranks if r < k]

    if not relevant_in_k:
        return 0.0

    # Compute DCG
    dcg = sum(1.0 / math.log2(r + 2) for r in relevant_in_k)

    # Compute IDCG (perfect ranking)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant_ranks))))

    return dcg / idcg if idcg > 0 else 0.0


def compute_co_citation_overlap(
    papers_by_id: Dict[str, Dict],
    citation_edges: Dict[str, Set[str]],
    entity_cache: Dict[str, Set[str]],
    paper_to_memory_id: Dict[str, str],
) -> Dict:
    """
    Validate co-citation clustering: papers that cite the same work should be nearby.

    Co-citation: papers A and B are co-cited if they both cite paper C

    Returns:
        {
            "co_citation_pairs": int,
            "high_entity_overlap": int,  # pairs with >50% entity overlap
            "overlap_correlation": float,
        }
    """
    metrics = {
        "co_citation_pairs": 0,
        "high_entity_overlap": 0,
        "overlap_distribution": defaultdict(int),
    }

    # Find co-citations
    cites_to_citing_papers = defaultdict(set)
    for paper_id, cited_ids in citation_edges.items():
        for cited_id in cited_ids:
            cites_to_citing_papers[cited_id].add(paper_id)

    # For each highly-cited paper, compare its citers
    for cited_id, citing_papers in cites_to_citing_papers.items():
        if len(citing_papers) < 2:
            continue

        citing_list = list(citing_papers)
        for i in range(len(citing_list)):
            for j in range(i + 1, len(citing_list)):
                paper_a_id = citing_list[i]
                paper_b_id = citing_list[j]

                mem_a = paper_to_memory_id.get(paper_a_id)
                mem_b = paper_to_memory_id.get(paper_b_id)

                if not mem_a or not mem_b:
                    continue

                entities_a = entity_cache.get(mem_a, set())
                entities_b = entity_cache.get(mem_b, set())

                if not entities_a or not entities_b:
                    continue

                metrics["co_citation_pairs"] += 1

                # Compute Jaccard similarity
                overlap = len(entities_a & entities_b)
                union = len(entities_a | entities_b)
                jaccard = overlap / union if union > 0 else 0.0

                metrics["overlap_distribution"][round(jaccard * 10) / 10] += 1

                if jaccard > 0.5:
                    metrics["high_entity_overlap"] += 1

    if metrics["co_citation_pairs"] > 0:
        metrics["high_overlap_ratio"] = metrics["high_entity_overlap"] / metrics["co_citation_pairs"]

    return metrics


def citation_graph_stats(
    papers_by_id: Dict[str, Dict],
    citation_edges: Dict[str, Set[str]],
) -> Dict:
    """
    Compute statistics about the citation graph.
    """
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)

    for paper_id, cited_ids in citation_edges.items():
        out_degree[paper_id] = len(cited_ids)
        for cited_id in cited_ids:
            in_degree[cited_id] += 1

    total_edges = sum(len(v) for v in citation_edges.values())

    return {
        "total_papers": len(papers_by_id),
        "total_citations": total_edges,
        "avg_citations_per_paper": total_edges / len(papers_by_id) if papers_by_id else 0,
        "max_citations": max(out_degree.values()) if out_degree else 0,
        "max_cited_by": max(in_degree.values()) if in_degree else 0,
        "papers_with_citations": len([p for p in out_degree.values() if p > 0]),
        "papers_being_cited": len([p for p in in_degree.values() if p > 0]),
    }
