"""
Entity extraction from memory text.

Primary: GLiNER zero-shot NER (50M params, ~45ms/call on CPU).
Fallback: regex/heuristic extraction if GLiNER unavailable.
"""

import re
import math
import logging
from typing import List, Tuple, Dict, Set, Optional
from datetime import datetime

from .erosion import STOPWORDS

logger = logging.getLogger(__name__)

# ── GLiNER setup ────────────────────────────────────────────────────
_gliner_model = None
_gliner_available = None  # None = not checked yet

# Entity labels tuned for personal conversation data
_GLINER_LABELS = [
    "person name",
    "place name",
    "organization name",
    "brand or product",
    "event name",
    "title or creative work",
    "hobby or activity",
    "measurement or quantity",
]

# GLiNER false positives to filter
_GLINER_STOPWORDS = {
    "i", "i'm", "i've", "i'll", "i'd", "me", "my", "we", "you",
    "he", "she", "they", "it", "the", "a", "an", "this", "that",
    "also", "however", "but", "and", "or", "so", "then", "yes", "no",
    "okay", "ok", "sure", "well", "oh", "ah", "hey", "hi", "hello",
}


def _get_gliner():
    """Lazy-load GLiNER model. Returns None if unavailable."""
    global _gliner_model, _gliner_available
    if _gliner_available is False:
        return None
    if _gliner_model is not None:
        return _gliner_model
    try:
        from gliner import GLiNER
        _gliner_model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
        _gliner_available = True
        logger.info("Loaded GLiNER small-v2.1 for entity extraction")
        return _gliner_model
    except Exception as e:
        logger.warning(f"GLiNER not available, falling back to regex: {e}")
        _gliner_available = False
        return None


def _gliner_extract(text: str) -> List[Tuple[str, str, str]]:
    """Extract entities using GLiNER zero-shot NER."""
    model = _get_gliner()
    if model is None:
        return []

    # Truncate very long texts — GLiNER works on ~512 token windows
    truncated = text[:2000] if len(text) > 2000 else text

    raw = model.predict_entities(truncated, _GLINER_LABELS, threshold=0.3)

    entities = []
    for ent in raw:
        name = ent["text"].strip()
        label = ent["label"]

        # Filter stopword false positives
        if name.lower() in _GLINER_STOPWORDS:
            continue
        if len(name) < 2:
            continue

        # Map GLiNER labels to our entity types
        if label in ("person name", "place name", "organization name",
                     "brand or product", "event name", "title or creative work"):
            etype = "named"
        elif label == "measurement or quantity":
            etype = "numeric"
        elif label == "hobby or activity":
            etype = "named"
        else:
            etype = "named"

        canonical = name.lower().rstrip(".,;:!?").strip()
        if len(canonical) >= 2:
            entities.append((name, etype, canonical))

    return entities


# ── Gemma 4 E2B (offline / consolidation-time extractor) ────────────
_GEMMA_PROMPT = """Extract structured information from the text below.

Return ONLY valid JSON with this exact schema, nothing else:
{"entities": ["entity1", "entity2"], "relations": [["subject", "relation", "object"]]}

Rules:
- entities: proper nouns only (people, places, organizations, brands, works, events, products). NO common verbs, pronouns, or generic nouns.
- relations: factual triples grounded in the text.
- If none, return empty lists.

Text:
\"\"\"
{TEXT}
\"\"\"

JSON:"""


def extract_entities_gemma(
    text: str,
    ollama_url: str = "http://127.0.0.1:11434",
    model: str = "gemma4:e2b",
    timeout: int = 120,
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """
    High-quality offline entity + relation extraction using Gemma 4 E2B via Ollama.

    Slower than GLiNER (~20s/call vs 45ms) but produces:
    - Cleaner canonicals (1.4% dup rate vs 45% for GLiNER)
    - 2.5x more relations
    - Near-zero noise

    Use during consolidation, not live ingest.

    Returns:
        (entities, triples) where:
        - entities: list of (name, entity_type, canonical) — entity_type='named'
        - triples: list of (subject, relation, object)
    """
    import requests

    truncated = text[:2000]
    prompt = _GEMMA_PROMPT.replace("{TEXT}", truncated)

    try:
        r = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_ctx": 4096},
            },
            timeout=timeout,
        )
        resp = r.json().get("response", "")
    except Exception as e:
        logger.warning(f"Gemma extraction failed: {e}")
        return [], []

    # Parse fenced or bare JSON
    match = re.search(r"\{.*\}", resp, re.DOTALL)
    if not match:
        return [], []
    try:
        import json
        parsed = json.loads(match.group())
    except (ValueError, Exception):
        return [], []

    entities = []
    for ent in parsed.get("entities", []):
        if not isinstance(ent, str):
            continue
        name = ent.strip().rstrip(".,;:!?")
        if len(name) < 2:
            continue
        canonical = name.lower()
        if canonical in _GLINER_STOPWORDS:
            continue
        # Require at least 2 alphabetic chars
        if sum(1 for c in canonical if c.isalpha()) < 2:
            continue
        entities.append((name, "named", canonical))

    triples = []
    for rel in parsed.get("relations", []):
        if isinstance(rel, list) and len(rel) == 3 and all(isinstance(x, str) for x in rel):
            s, r_, o = (x.strip() for x in rel)
            if s and r_ and o:
                triples.append((s, r_, o))

    return entities, triples


# Academic domain terms that appear in nearly every paper (generic noise)
ACADEMIC_STOPWORDS = {
    'model', 'models', 'method', 'methods', 'approach', 'approaches',
    'algorithm', 'algorithms', 'system', 'systems', 'framework',
    'network', 'networks', 'neural', 'learning', 'training',
    'performance', 'results', 'evaluation', 'experiment', 'experiments',
    'analysis', 'data', 'dataset', 'datasets', 'task', 'tasks',
    'input', 'output', 'feature', 'features', 'representation',
    'representations', 'parameter', 'parameters', 'layer', 'layers',
    'architecture', 'proposed', 'existing', 'previous', 'recent',
    'baseline', 'baselines', 'benchmark', 'state-of-the-art',
    'loss', 'function', 'objective', 'optimization', 'gradient',
    'sample', 'samples', 'distribution', 'probability', 'inference',
    'prediction', 'predictions', 'accuracy', 'precision', 'recall',
    'process', 'technique', 'techniques', 'problem', 'solution',
    'information', 'knowledge', 'structure', 'component', 'components',
    'application', 'applications', 'domain', 'domains', 'space',
    'embedding', 'embeddings', 'vector', 'vectors', 'matrix',
    'mechanism', 'mechanisms', 'module', 'modules',
    'encoder', 'decoder', 'classification', 'regression',
    'generation', 'detection', 'segmentation', 'recognition',
    'transfer', 'generalization', 'convergence', 'batch', 'epoch',
    'weight', 'weights', 'bias', 'activation', 'softmax', 'dropout',
    'normalization', 'supervised', 'unsupervised', 'reinforcement',
    'image', 'images', 'text', 'token', 'tokens', 'sentence',
    'word', 'document', 'documents', 'label', 'labels', 'class',
    'annotation', 'annotations', 'corpus', 'corpora',
    'paper', 'papers', 'work', 'study', 'studies', 'research',
    'section', 'figure', 'table', 'appendix', 'abstract',
    'contribution', 'contributions', 'limitation', 'limitations',
    'show', 'demonstrate', 'achieve', 'improve', 'outperform',
    'effective', 'efficient', 'robust', 'novel', 'using', 'different',
    'large', 'small', 'high', 'low', 'new', 'first', 'proposed',
}

# Known ML models, datasets, venues — extract as high-salience named entities
KNOWN_MODELS = {
    'bert', 'gpt', 'gpt-2', 'gpt-3', 'gpt-4', 't5', 'bart', 'roberta',
    'xlnet', 'albert', 'electra', 'deberta', 'llama', 'mistral',
    'resnet', 'vgg', 'inception', 'efficientnet', 'vit', 'deit',
    'yolo', 'faster r-cnn', 'mask r-cnn', 'detr', 'sam',
    'transformer', 'lstm', 'gru', 'cnn', 'rnn', 'gan', 'vae',
    'diffusion', 'stable diffusion', 'dall-e',
    'clip', 'blip', 'flamingo', 'llava',
    'adam', 'sgd', 'adamw', 'lamb', 'lora', 'qlora', 'peft',
}

KNOWN_DATASETS = {
    'imagenet', 'cifar', 'cifar-10', 'cifar-100', 'mnist',
    'coco', 'pascal voc', 'ade20k', 'cityscapes',
    'glue', 'superglue', 'squad', 'mnli', 'sst',
    'wikitext', 'openwebtext', 'the pile', 'c4',
    'arxiv', 'pubmed', 's2orc', 'semantic scholar',
    'mmlu', 'hellaswag', 'humaneval', 'mbpp',
}

KNOWN_VENUES = {
    'neurips', 'nips', 'icml', 'iclr', 'aaai', 'ijcai',
    'cvpr', 'iccv', 'eccv', 'acl', 'emnlp', 'naacl',
    'kdd', 'www', 'sigir', 'cikm',
}


def extract_entities(text: str, known_entities: Optional[List[str]] = None) -> List[Tuple[str, str, str]]:
    """
    Extract entities from text.

    Primary: GLiNER zero-shot NER for high-quality named entity extraction.
    Fallback: regex/heuristic extraction if GLiNER unavailable.
    Both paths contribute; results are merged and deduplicated.

    Returns list of (name, entity_type, canonical_name) tuples.
    - name: Original form ("Theo", "28kg", "transformer")
    - entity_type: "named", "numeric", "concept", "temporal"
    - canonical_name: Lowercase normalized form for deduplication
    """
    entities = []

    # ── Primary: GLiNER extraction ──
    gliner_results = _gliner_extract(text)
    gliner_canonicals = {c for _, _, c in gliner_results}
    # Also block regex entries that are substrings of any GLiNER canonical
    # (so "Reading The Glass Menagerie" doesn't override "The Glass Menagerie")
    entities.extend(gliner_results)

    # ── Fallback / supplement: regex heuristics ──
    for name, etype, canonical in _regex_extract(text, known_entities):
        # Skip regex entries whose canonical contains a GLiNER canonical
        if any(gc in canonical and gc != canonical for gc in gliner_canonicals):
            continue
        entities.append((name, etype, canonical))

    # Deduplicate by canonical name (prefer longer original form)
    seen = {}  # canonical -> (name, entity_type, canonical_name)
    for name, etype, canonical in entities:
        if canonical not in seen or len(name) > len(seen[canonical][0]):
            seen[canonical] = (name, etype, canonical)

    return list(seen.values())


def _regex_extract(text: str, known_entities: Optional[List[str]] = None) -> List[Tuple[str, str, str]]:
    """Regex/heuristic entity extraction (fallback)."""
    entities = []
    tokens = text.split()

    # 1. Named entities: sequences of capitalized words (mid-sentence)
    i = 0
    while i < len(tokens):
        # Skip first token (sentence start capitalization)
        if i > 0 and tokens[i][0].isupper() and not tokens[i].isupper():
            # Collect consecutive capitalized tokens
            name_parts = [tokens[i]]
            j = i + 1
            # Break the run if previous token ended with sentence/clause punctuation
            while j < len(tokens) and tokens[j][0].isupper() and not tokens[j-1][-1] in ".,;:!?":
                name_parts.append(tokens[j])
                j += 1
            # Skip single-word captures right after sentence-ending punctuation
            # (almost always sentence-starter noise: "Can", "If", "Remember")
            prev_ended_sentence = tokens[i-1][-1] in ".!?"
            if len(name_parts) > 1 or not prev_ended_sentence:
                name = ' '.join(name_parts).rstrip(".,;:!?")
                canonical = name.lower().strip()
                # Require at least 2 alphabetic chars to filter list markers like "**2", "1-2"
                alpha_count = sum(1 for c in canonical if c.isalpha())
                if (canonical and canonical not in _GLINER_STOPWORDS
                        and len(canonical) >= 2 and alpha_count >= 2):
                    entities.append((name, 'named', canonical))
            i = j
        else:
            i += 1

    # 1b. Known entities: models, datasets, venues (high-salience named entities)
    text_lower = text.lower()
    for entity_list in [KNOWN_MODELS, KNOWN_DATASETS, KNOWN_VENUES]:
        for known_entity in entity_list:
            pattern = r'\b' + re.escape(known_entity) + r'\b'
            for match in re.finditer(pattern, text_lower):
                start, end = match.span()
                original = text[start:end]
                canonical = known_entity.lower()
                if not any(e[2] == canonical for e in entities):
                    entities.append((original, 'named', canonical))

    # 2. Numeric entities: only meaningful quantities (number + unit, or currency).
    # Bare numbers are NOT entities — they create graph pollution like "100", "**2", "1-2".
    unit_tokens = {'kg', 'mg', 'ml', 'cm', 'mm', 'gb', 'mb', 'mph', 'km', 'hrs', 'mins', '%'}
    for i, token in enumerate(tokens):
        clean = token.strip(".,;:!?()[]")
        if not re.match(r'^[\d.]+$', clean) and not clean.startswith(('£', '$', '€', '¥')):
            continue
        if i + 1 < len(tokens) and tokens[i + 1].strip(".,;:!?").lower() in unit_tokens:
            numeric_str = f"{clean} {tokens[i + 1].strip('.,;:!?')}"
            entities.append((numeric_str, 'numeric', numeric_str.lower()))
        elif clean.startswith(('£', '$', '€', '¥')) and len(clean) > 1:
            entities.append((clean, 'numeric', clean.lower()))

    # 3. Temporal entities: days, months, ordinals, plausible years (1900-2099 only)
    temporal_patterns = [
        r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b',
        r'\b\d{1,2}(st|nd|rd|th)\b',
        r'\b(19|20)\d{2}\b',  # Plausible years only
    ]
    for pattern in temporal_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            entities.append((match.group(), 'temporal', match.group().lower()))

    # 4. Match against known entities (from existing graph)
    if known_entities:
        text_lower = text.lower()
        for known in known_entities:
            if known.lower() in text_lower:
                if not any(e[2] == known.lower() for e in entities):
                    entities.append((known, 'named', known.lower()))

    return entities


def extract_entities_with_metadata(
    text: str,
    known_entities: Optional[List[str]] = None,
    authors: Optional[List[str]] = None,
    venue: Optional[str] = None
) -> List[Tuple[str, str, str]]:
    """
    Extract entities from text + metadata (authors, venue).

    This enhances extract_entities() by incorporating structured metadata
    from S2ORC: author names and publication venue.

    Args:
        text: Paper content/abstract
        known_entities: Existing entity names from graph
        authors: Author names from metadata (e.g., ['John Smith', 'Jane Doe'])
        venue: Publication venue (e.g., 'NeurIPS', 'ICCV')

    Returns:
        List of (name, entity_type, canonical_name) tuples, including
        author and venue entities
    """
    # Start with text-based extraction
    entities = extract_entities(text, known_entities=known_entities)

    # Add author entities (if metadata provided)
    if authors:
        for author in authors:
            if author and len(author.strip()) > 0:
                canonical = author.lower().strip()
                # Only add if not already extracted (prefer already-found version)
                if not any(e[2] == canonical for e in entities):
                    entities.append((author, 'named', canonical))

    # Add venue entity (if metadata provided)
    if venue and len(venue.strip()) > 0:
        venue_clean = venue.strip()
        canonical = venue_clean.lower()
        # Check if already captured by known venues
        if not any(e[2] == canonical for e in entities):
            # Higher salience for explicitly labeled venues
            entities.append((venue_clean, 'named', canonical))

    # Deduplicate by canonical name (same logic as extract_entities)
    seen = {}
    for name, etype, canonical in entities:
        if canonical not in seen or len(name) > len(seen[canonical][0]):
            seen[canonical] = (name, etype, canonical)

    return list(seen.values())


def compute_entity_salience(name: str, entity_type: str, text: str) -> float:
    """
    Compute salience (0.0-1.0) indicating how central this entity is to this memory.

    - Named entities: 0.9 (high importance)
    - Numeric: 0.8
    - Temporal: 0.5
    - Concept: based on frequency in this text
    """
    if entity_type == 'named':
        return 0.9
    if entity_type == 'numeric':
        return 0.8
    if entity_type == 'temporal':
        return 0.5

    # Concept salience based on frequency in this specific text
    count = text.lower().count(name.lower())
    return min(0.8, 0.3 + count * 0.15)


def get_entity_fan(entity_count: Dict[str, int], canonical_name: str) -> int:
    """
    Get fan count for an entity (how many memories mention it).
    Used in spreading activation formula.
    """
    return entity_count.get(canonical_name, 1)


# ── ACT-R Spreading Activation ──────────────────────────────────────

def compute_spreading_activation(
    memory_id: str,
    query_entities: List[str],
    entity_cache: Dict[str, Set[str]],  # memory_id -> set of entity canonical names
    entity_fan_count: Dict[str, int],   # canonical_name -> fan count
    W: float = 1.0,
    S_max: float = 3.5
) -> float:
    """
    ACT-R spreading activation for a memory given query entities.

    SA = Σ_j (W / n_sources) * max(0, S_max - ln(fan_j))

    Where:
    - j iterates over entities in query that also appear in this memory
    - W = total source activation (divided among query entities)
    - fan_j = how many memories contain entity j (more = weaker individual connections)
    - S_max = maximum associative strength (increased to 3.5 to reduce fan effect impact
              and improve discrimination on sparse citation graphs)

    Args:
        memory_id: UUID of memory to evaluate
        query_entities: List of canonical entity names from query
        entity_cache: Precomputed {memory_id -> set(canonical_names)}
        entity_fan_count: Precomputed {canonical_name -> fan_count}
        W: Source activation weight
        S_max: Maximum associative strength (default 3.5, tuned for academic citation graphs)

    Returns:
        Spreading activation score (0.0+)
    """
    # Get entities for this memory
    mem_entities = entity_cache.get(memory_id, set())
    if not mem_entities:
        return 0.0

    # Find overlap: which query entities appear in this memory?
    query_set = set(query_entities)
    overlap = query_set.intersection(mem_entities)
    if not overlap:
        return 0.0

    n_sources = max(1, len(query_set))
    source_weight = W / n_sources  # Divide attention across query entities

    total_SA = 0.0
    for entity_name in overlap:
        # Fan effect: more widespread entities contribute less
        fan = entity_fan_count.get(entity_name, 1)

        # Associative strength: S = S_max - ln(fan)
        # Popular entity (fan=1000) → S ≈ 0, weak
        # Rare entity (fan=5) → S ≈ 2.0 - 1.6 ≈ 0.4, strong
        S_ji = max(0, S_max - math.log(max(1, fan)))

        total_SA += source_weight * S_ji

    return total_SA
