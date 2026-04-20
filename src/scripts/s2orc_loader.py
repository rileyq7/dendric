"""
Load S2ORC (Semantic Scholar Open Research Corpus) for Phase 2 validation.

S2ORC is a large-scale citation graph with:
- 200M+ papers from Semantic Scholar
- Full citation metadata (paper A cites paper B)
- Structured abstracts and titles
- Citation context snippets

For Phase 2, we use:
- arxiv_2020_subset (easier to work with, still large)
- Or s2-corpus-[001-999] shards for full 250k dataset

Reference: https://github.com/allenai/s2orc
Dataset: https://huggingface.co/datasets/allenai/s2orc
"""

import logging
import json
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


def load_s2orc_papers(
    limit: int = 250000,
    min_abstract_length: int = 100,
    cache_file: str = "s2orc_papers.jsonl",
) -> List[Dict[str, str]]:
    """
    Load S2ORC papers with citation metadata.

    Args:
        limit: Max papers to load
        min_abstract_length: Filter out papers with shorter abstracts
        cache_file: Local cache file

    Returns:
        List of {title, abstract, paper_id, cited_by, cites, authors, year}
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Install datasets: pip install datasets")
        return load_s2orc_papers_local(limit, min_abstract_length, cache_file)

    logger.info(f"Loading S2ORC papers (limit={limit})...")

    try:
        # Stream arxiv subset (smaller, easier to work with)
        dataset = load_dataset("allenai/s2orc", "arxiv", split="train", streaming=True)

        papers = []
        citation_map = defaultdict(list)  # paper_id -> [cited_paper_ids]

        for i, sample in enumerate(dataset):
            if i >= limit:
                break

            abstract = sample.get("abstract", "").strip()
            if len(abstract) < min_abstract_length:
                continue

            title = sample.get("title", "").strip()
            if not title:
                continue

            paper_id = sample.get("paper_id", str(i))
            cited_by = sample.get("inbound_citations", [])  # Papers that cite this one
            cites = sample.get("outbound_citations", [])  # Papers this one cites

            papers.append({
                "title": title,
                "abstract": abstract,
                "paper_id": paper_id,
                "authors": sample.get("authors", ""),
                "year": sample.get("year", 0),
                "cited_by": cited_by,  # List of paper IDs
                "cites": cites,  # List of paper IDs
                "venue": sample.get("venue", ""),
            })

            # Build citation map for validation
            for cited_id in cites:
                citation_map[paper_id].append(cited_id)

        logger.info(f"Loaded {len(papers)} papers")
        logger.info(f"Citation edges: {sum(len(v) for v in citation_map.values())}")

        return papers

    except Exception as e:
        logger.error(f"Failed to load from HuggingFace: {e}")
        return load_s2orc_papers_local(limit, min_abstract_length, cache_file)


def load_s2orc_papers_local(
    limit: int = 250000,
    min_abstract_length: int = 100,
    cache_file: str = "s2orc_papers.jsonl",
) -> List[Dict[str, str]]:
    """
    Load from local JSONL cache (for testing without internet).
    Tries multiple cache file names in order: extended, regular, none.
    """
    # Try extended dataset first (Phase 3)
    cache_files = [
        "s2orc_extended.jsonl",
        "s2orc_papers.jsonl",
        cache_file,
    ]

    for cf in cache_files:
        try:
            with open(cf, "r") as f:
                papers = []
                for i, line in enumerate(f):
                    if i >= limit:
                        break
                    try:
                        paper = json.loads(line)
                        abstract = paper.get("abstract", "").strip()
                        if len(abstract) >= min_abstract_length:
                            papers.append({
                                "title": paper.get("title", ""),
                                "abstract": abstract,
                                "paper_id": paper.get("id", paper.get("paper_id", f"s2orc-{i}")),
                                "authors": paper.get("authors", ""),
                                "year": paper.get("year", 0),
                                "cited_by": paper.get("cited_by", []),
                                "cites": paper.get("cites", []),
                                "venue": paper.get("venue", ""),
                            })
                    except json.JSONDecodeError:
                        continue
                logger.info(f"Loaded {len(papers)} papers from {cf}")
                return papers
        except FileNotFoundError:
            continue

    logger.warning(f"No local cache found. Tried: {cache_files}. Returning empty list.")
    return []


def format_paper_for_memory(paper: Dict[str, str]) -> Dict[str, str]:
    """Format S2ORC paper for ingest."""
    content = f"""
Title: {paper['title']}

Authors: {paper.get('authors', 'Unknown')}

Year: {paper.get('year', 'Unknown')}

Venue: {paper.get('venue', 'Unknown')}

Abstract:
{paper['abstract']}
""".strip()

    return {
        "raw_content": content,
        "source": "s2orc",
        "context": f"Research paper {paper['paper_id']}",
        "metadata": {
            "paper_id": paper["paper_id"],
            "title": paper["title"],
            "authors": paper.get("authors", ""),
            "year": paper.get("year", 0),
            "cited_by": paper.get("cited_by", []),
            "cites": paper.get("cites", []),
            "venue": paper.get("venue", ""),
        }
    }


def build_citation_graph(papers: List[Dict[str, str]]) -> Tuple[Dict[str, set], Dict[str, int]]:
    """
    Build citation graph from papers.

    Returns:
        (citation_edges, citation_counts)
        - citation_edges: {paper_id -> set(cited_paper_ids)}
        - citation_counts: {paper_id -> in_degree + out_degree}
    """
    paper_ids = {p["paper_id"] for p in papers}
    citation_edges = defaultdict(set)
    citation_counts = defaultdict(int)

    for paper in papers:
        pid = paper["paper_id"]
        for cited_id in paper.get("cites", []):
            if cited_id in paper_ids:  # Only include edges within our set
                citation_edges[pid].add(cited_id)
                citation_counts[pid] += 1
                citation_counts[cited_id] += 1

    return dict(citation_edges), dict(citation_counts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    papers = load_s2orc_papers(limit=100)
    if papers:
        print(f"Loaded {len(papers)} papers")
        for p in papers[:3]:
            print(f"  {p['title'][:60]}... ({p['paper_id']})")
            print(f"    Cites: {len(p.get('cites', []))}, Cited by: {len(p.get('cited_by', []))}")
