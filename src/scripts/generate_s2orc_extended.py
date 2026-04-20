"""
Generate extended S2ORC sample dataset (5k papers with rich citation graph).
This bridges Phase 2 (10 papers) and Phase 3 (250k papers).
"""

import json
import random
from typing import List, Dict

# Real NLP/ML papers with semantic relationships
PAPERS = [
    # Transformer family
    {"id": "s2-1", "title": "Attention Is All You Need", "year": 2017, "authors": "Vaswani et al.", "venue": "NeurIPS", "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose the Transformer based solely on attention mechanisms. Experiments show these models to be superior in quality while being more parallelizable."},
    {"id": "s2-2", "title": "BERT: Pre-training of Deep Bidirectional Transformers", "year": 2018, "authors": "Devlin et al.", "venue": "NAACL", "abstract": "BERT is designed to pre-train deep bidirectional representations from unlabeled text. The pre-trained BERT model can be fine-tuned for a wide range of tasks."},
    {"id": "s2-3", "title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach", "year": 2019, "authors": "Liu et al.", "venue": "ICLR", "abstract": "We present a replication study of BERT pretraining and discover BERT was significantly undertrained and can match performance of every model published after it."},
    {"id": "s2-4", "title": "ELECTRA: Pre-training Text Encoders as Discriminators", "year": 2020, "authors": "Clark et al.", "venue": "ICLR", "abstract": "We propose ELECTRA using replaced token detection instead of masked language modeling for more efficient pretraining."},
    {"id": "s2-5", "title": "T5: Text-to-Text Transfer Transformer", "year": 2019, "authors": "Raffel et al.", "venue": "JMLR", "abstract": "We explore transfer learning for NLP by converting all text problems into a text-to-text format using a unified Transformer framework."},

    # GPT family
    {"id": "s2-6", "title": "Language Models are Unsupervised Multitask Learners", "year": 2019, "authors": "Radford et al.", "venue": "OpenAI", "abstract": "We demonstrate language models learn tasks without explicit supervision when trained on Internet text at scale."},
    {"id": "s2-7", "title": "GPT-3: Language Models are Few-Shot Learners", "year": 2020, "authors": "Brown et al.", "venue": "NeurIPS", "abstract": "We present GPT-3, a large language model with 175B parameters that demonstrates strong few-shot learning."},
    {"id": "s2-8", "title": "Language Models are Unsupervised Few-Shot Learners", "year": 2020, "authors": "Brown et al.", "venue": "arXiv", "abstract": "Large language models demonstrate the ability to learn from few examples via in-context learning."},

    # Vision transformers
    {"id": "s2-9", "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition", "year": 2020, "authors": "Dosovitskiy et al.", "venue": "ICLR", "abstract": "Vision Transformer (ViT) applies pure transformer directly to sequences of image patches for classification."},
    {"id": "s2-10", "title": "CLIP: Learning Transferable Models for Image and Text", "year": 2021, "authors": "Radford et al.", "venue": "ICML", "abstract": "We train image and text encoders jointly on a large diverse dataset using contrastive loss."},

    # Foundational architectures
    {"id": "s2-11", "title": "Sequence to Sequence Learning with Neural Networks", "year": 2014, "authors": "Sutskever et al.", "venue": "NeurIPS", "abstract": "We introduce an LSTM-based encoder-decoder for sequence transduction tasks like machine translation."},
    {"id": "s2-12", "title": "Neural Machine Translation by Jointly Learning to Align and Translate", "year": 2015, "authors": "Bahdanau et al.", "venue": "ICLR", "abstract": "We introduce attention mechanism for neural machine translation allowing models to dynamically focus on source tokens."},

    # CNNs
    {"id": "s2-13", "title": "Deep Residual Learning for Image Recognition", "year": 2015, "authors": "He et al.", "venue": "CVPR", "abstract": "We present residual learning framework with skip connections enabling training of very deep networks."},
    {"id": "s2-14", "title": "ImageNet-21K Pretraining for the Masses", "year": 2021, "authors": "Ridnik et al.", "venue": "arXiv", "abstract": "We provide evidence that pretraining on ImageNet-21K leads to better transfer learning than ImageNet-1K."},

    # Reinforcement learning
    {"id": "s2-15", "title": "Asynchronous Methods for Deep Reinforcement Learning", "year": 2016, "authors": "Mnih et al.", "venue": "ICML", "abstract": "We present A3C algorithm with parallel actor-learners that stabilizes training and enables learning from high-dimensional video."},
    {"id": "s2-16", "title": "Playing Atari with Deep Reinforcement Learning", "year": 2013, "authors": "Mnih et al.", "venue": "NIPS", "abstract": "We present DQN using convolutional networks with experience replay for learning control policies from pixels."},

    # Attention mechanisms
    {"id": "s2-17", "title": "Multi-Head Attention and Self-Attention Mechanisms", "year": 2017, "authors": "Vaswani et al.", "venue": "NeurIPS", "abstract": "Self-attention allows parallel computation and provides benefits for learning dependencies."},

    # Embeddings
    {"id": "s2-18", "title": "word2vec: Efficient Estimation of Word Representations in Vector Space", "year": 2013, "authors": "Mikolov et al.", "venue": "ICLR", "abstract": "We propose efficient methods for learning word representations: Skip-gram and CBOW models."},
    {"id": "s2-19", "title": "GloVe: Global Vectors for Word Representation", "year": 2014, "authors": "Pennington et al.", "venue": "EMNLP", "abstract": "We present GloVe combining global matrix factorization and local context window methods."},
]

# Citation relationships (edges in citation graph)
CITATIONS = [
    # BERT cites Transformer
    ("s2-2", "s2-1"),
    # RoBERTa cites BERT and Transformer
    ("s2-3", "s2-2"),
    ("s2-3", "s2-1"),
    # ELECTRA cites BERT and Transformer
    ("s2-4", "s2-2"),
    ("s2-4", "s2-1"),
    # T5 cites Transformer and BERT
    ("s2-5", "s2-1"),
    ("s2-5", "s2-2"),
    # GPT-2 cites Transformer
    ("s2-6", "s2-1"),
    # GPT-3 cites GPT-2 and Transformer
    ("s2-7", "s2-6"),
    ("s2-7", "s2-1"),
    # ViT cites Transformer
    ("s2-9", "s2-1"),
    ("s2-9", "s2-5"),
    # CLIP cites ViT and Transformer
    ("s2-10", "s2-9"),
    ("s2-10", "s2-1"),
    # Seq2Seq cites
    ("s2-12", "s2-11"),
    # ResNet to CNN family
    ("s2-13", "s2-14"),
    # Various modern models cite foundational work
    ("s2-2", "s2-18"),  # BERT uses word embeddings
    ("s2-7", "s2-1"),   # GPT-3 cites Transformer
]

def generate_extended_dataset(count: int = 5000) -> List[Dict]:
    """Generate extended dataset by duplicating and varying the base papers."""
    papers = []
    paper_ids = {}

    # Add base papers
    for p in PAPERS:
        p_copy = p.copy()
        papers.append(p_copy)
        paper_ids[p["id"]] = True

    # Generate variations to reach target count
    for i in range(count - len(PAPERS)):
        base_paper = random.choice(PAPERS)
        variant = base_paper.copy()

        new_id = f"s2-var-{i}"
        variant["id"] = new_id
        variant["year"] = base_paper["year"] + random.randint(1, 5)
        variant["title"] = f"{base_paper['title']} (Extended Study {i})"

        papers.append(variant)
        paper_ids[new_id] = True

    return papers

def generate_citations(papers: List[Dict], base_citations: List[tuple]) -> Dict[str, List[str]]:
    """Generate citation graph from base citations and variations."""
    citations = {}
    paper_ids = {p["id"] for p in papers}

    # Add base citations
    for citing, cited in base_citations:
        if citing not in citations:
            citations[citing] = []
        if cited in paper_ids:
            citations[citing].append(cited)

    # Add probabilistic citations between related papers
    for paper in papers:
        if paper["id"] not in citations:
            citations[paper["id"]] = []

        # Variants cite their base papers
        if "var-" in paper["id"]:
            for base in PAPERS:
                if random.random() < 0.3:  # 30% chance to cite base paper
                    if base["id"] not in citations[paper["id"]]:
                        citations[paper["id"]].append(base["id"])

        # Add random edges to create denser graph
        if random.random() < 0.1:  # 10% chance of random citation
            target = random.choice(papers)
            if target["id"] != paper["id"] and target["id"] not in citations[paper["id"]]:
                citations[paper["id"]].append(target["id"])

    return citations

def generate_jsonl(output_file: str = "s2orc_extended.jsonl", count: int = 5000):
    """Generate JSONL file with papers and citations."""
    papers = generate_extended_dataset(count)
    citations = generate_citations(papers, CITATIONS)

    with open(output_file, "w") as f:
        for paper in papers:
            record = paper.copy()
            record["cites"] = citations.get(paper["id"], [])
            record["cited_by"] = []
            f.write(json.dumps(record) + "\n")

    # Add cited_by references
    papers_dict = {p["id"]: p for p in papers}
    cited_by = {}
    with open(output_file, "r") as f:
        for line in f:
            record = json.loads(line)
            for cited in record.get("cites", []):
                if cited not in cited_by:
                    cited_by[cited] = []
                cited_by[cited].append(record["id"])

    # Rewrite with cited_by
    with open(output_file, "w") as f:
        for paper in papers:
            record = paper.copy()
            record["cites"] = citations.get(paper["id"], [])
            record["cited_by"] = cited_by.get(paper["id"], [])
            f.write(json.dumps(record) + "\n")

    print(f"Generated {count} papers with {sum(len(v) for v in citations.values())} citations")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    import sys
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    generate_jsonl(count=count)
