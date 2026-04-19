"""
Entity extraction from memory text.

Pure regex/heuristic extraction. No LLM dependencies.
"""

import re
import logging
from typing import List, Tuple, Dict, Optional

from .erosion import STOPWORDS

logger = logging.getLogger(__name__)


# Pronouns, discourse markers, common false positives
_BASE_STOPWORDS = {
    "i", "i'm", "i've", "i'll", "i'd", "me", "my", "we", "you",
    "he", "she", "they", "it", "the", "a", "an", "this", "that",
    "also", "however", "but", "and", "or", "so", "then", "yes", "no",
    "okay", "ok", "sure", "well", "oh", "ah", "hey", "hi", "hello",
}

# Common words misidentified as named entities (capitalized at sentence
# starts, after [Assistant]:, etc.).
_ENGINE_ENTITY_STOPWORDS = {
    "here", "the", "this", "that", "it's", "i'm", "i'll", "i've", "i'd",
    "you", "you're", "you'll", "we", "we're", "they", "there", "these",
    "those", "what", "when", "where", "which", "who", "how", "why",
    "yes", "no", "not", "but", "and", "or", "so", "if", "then",
    "also", "just", "very", "really", "well", "sure", "great", "good",
    "some", "many", "much", "more", "most", "other", "another", "such",
    "like", "want", "need", "can", "will", "would", "could", "should",
    "let", "make", "get", "take", "give", "have", "had", "has", "was",
    "were", "been", "being", "are", "did", "does", "done",
    "user", "assistant", "session context", "user: i'm", "user: i",
    "here are", "one", "two", "three", "four", "five",
    "remember", "think", "know", "see", "look", "try", "keep",
    "first", "last", "new", "old", "next", "start", "end",
    "absolutely", "definitely", "certainly", "exactly", "actually",
    "however", "although", "while", "since", "because", "therefore",
    "sure", "sure,", "okay", "right", "thanks", "thank", "please",
    "maybe", "perhaps", "probably", "might", "today", "tomorrow",
    "based", "here's", "that's", "there's", "what's", "it",
}

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

# Common English verbs that frequently appear sentence-initial as imperatives
# ("Hang the painting", "Store the leftovers", "Plan ahead"). These have no
# verb-shape suffix to catch them with morphology, so they need to be listed.
# Kept small and tied directly to the extractor — NOT a 100+ word catch-all.
_COMMON_IMPERATIVE_VERBS = {
    "hang", "store", "plan", "make", "take", "give", "find", "get", "got",
    "put", "set", "use", "add", "open", "close", "stop", "start",
    "keep", "let", "leave", "move", "turn", "show", "tell", "ask",
    "buy", "sell", "send", "bring", "call", "pick", "pull", "push",
    "drop", "pass", "wait", "look", "watch", "check", "clean",
    "cook", "wash", "read", "write", "draw", "paint", "build", "fix",
    "try", "see", "hear", "feel", "think", "know", "want", "need",
    "love", "hate", "hope", "wish", "had", "have", "did", "do", "say",
    "said", "tell", "told", "let", "made", "took", "gave", "found",
    "saw", "seen", "went", "going", "came", "back", "yes", "well",
}

# Authoritative merged stopword set used to filter all entity extraction.
ENTITY_STOPWORDS = (
    _BASE_STOPWORDS
    | ACADEMIC_STOPWORDS
    | STOPWORDS
    | _ENGINE_ENTITY_STOPWORDS
    | _COMMON_IMPERATIVE_VERBS
)


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
    Extract entities from text via regex/heuristics.

    Returns list of (name, entity_type, canonical_name) tuples.
    - name: Original form ("Theo", "28kg", "transformer")
    - entity_type: "named", "numeric", "temporal"
    - canonical_name: Lowercase normalized form for deduplication
    """
    entities = _regex_extract(text, known_entities)

    # Deduplicate by canonical name (prefer longer original form)
    seen: Dict[str, Tuple[str, str, str]] = {}
    for name, etype, canonical in entities:
        if canonical not in seen or len(name) > len(seen[canonical][0]):
            seen[canonical] = (name, etype, canonical)

    return list(seen.values())


def _regex_extract(text: str, known_entities: Optional[List[str]] = None) -> List[Tuple[str, str, str]]:
    """Regex/heuristic entity extraction.

    Sentence-initial capitalization is grammatical, not semantic. A single-token
    capitalized word at sentence start ("Hang", "Plan", "Store") is almost
    always a regular verb/noun, not a named entity. We previously caught this
    with a 100+ word blocklist in engine.py — that's a band-aid that rots.

    Real fix here: track sentence-initial positions, then for single-token
    capitalized matches that ARE sentence-initial, require independent evidence
    of proper-noun status (appears non-sentence-initially elsewhere in the same
    text, or matches a known proper-noun set).
    """
    entities = []
    tokens = text.split()

    # Identify sentence-initial token indices: index 0, plus any token whose
    # predecessor ended in . ! or ? (period as full-stop, not abbreviation).
    sentence_initial: set = {0}
    for k in range(1, len(tokens)):
        prev = tokens[k - 1].rstrip(")'\"")
        if prev and prev[-1] in ".!?":
            sentence_initial.add(k)

    # Pre-compute non-sentence-initial capitalized tokens (lowercased, stripped
    # of punctuation). These are reliable proper-noun seeds — a word that's
    # capitalized mid-sentence is almost always a name.
    confirmed_proper: set = set()
    for k, tok in enumerate(tokens):
        if k in sentence_initial:
            continue
        if tok and tok[0].isupper() and not tok.isupper():
            confirmed_proper.add(tok.lower().strip(".,;:!?'\""))

    _CONNECTORS = {"of", "the", "and", "for", "at", "in", "on", "de", "la", "von", "van", "di", "du"}
    _LEADING_NOISE = {"the", "a", "an", "besides", "also", "here", "there"}
    _KNOWN_PROPER = KNOWN_MODELS | KNOWN_DATASETS | KNOWN_VENUES

    i = 0
    while i < len(tokens):
        if tokens[i] and tokens[i][0].isupper() and not tokens[i].isupper():
            start_index = i
            # If sentence-initial token is a verb/stopword that happens to be
            # capitalized by grammar, don't include it in the name chain — but
            # do allow the chain to continue from the next token. This catches
            # "Visited the Museum of Modern Art" → "Museum of Modern Art".
            head_lower = tokens[i].lower().strip(".,;:!?'\"")
            head_is_grammar = (
                start_index in sentence_initial
                and (head_lower in ENTITY_STOPWORDS
                     or head_lower.endswith(("ing", "ed", "ly", "ize", "ise")))
            )
            if head_is_grammar:
                # Skip the head; continue scanning from i+1 if it could start a
                # new chain. But since the next token is rarely capitalized
                # (after a verb), just advance one and let the outer loop re-scan.
                i += 1
                continue

            name_parts = [tokens[i]]
            j = i + 1
            while j < len(tokens) and tokens[j] and tokens[j-1][-1] not in ".,;:!?":
                tok = tokens[j]
                if tok[0].isupper() and not tok.isupper():
                    name_parts.append(tok)
                    j += 1
                elif tok.lower().strip(".,;:!?") in _CONNECTORS and j + 1 < len(tokens) \
                        and tokens[j + 1] and tokens[j + 1][0].isupper() and not tokens[j + 1].isupper():
                    name_parts.append(tok)
                    j += 1
                else:
                    break
            name = ' '.join(name_parts).rstrip(".,;:!?")
            canonical = name.lower().strip().strip("'\"")
            canonical_tokens = canonical.split()
            while canonical_tokens and canonical_tokens[-1] in _CONNECTORS:
                canonical_tokens.pop()
            while canonical_tokens and canonical_tokens[0] in _LEADING_NOISE:
                canonical_tokens.pop(0)
            canonical = " ".join(canonical_tokens)
            alpha_count = sum(1 for c in canonical if c.isalpha())

            # Single-token sentence-initial capitalization needs to clear an
            # extra bar — sentence-start capitalization is grammar, not
            # semantics. Accept if any of these hold:
            #   (a) the token appears capitalized again non-sentence-initially
            #       elsewhere in the same text (real proper nouns repeat),
            #   (b) it's in our known proper-noun set (models/datasets/venues),
            #   (c) it doesn't look verb-shaped AND isn't a stopword
            #       (this is the lenient default — real names like "Mango",
            #       "Tamiya", "Vallejo" pass; verbs like "Hang", "Plan",
            #       "Store", "Make" get caught by the verb-shape and stopword
            #       filters above).
            is_single_token_sentence_initial = (
                start_index in sentence_initial
                and len(canonical_tokens) == 1
            )
            if is_single_token_sentence_initial:
                head = canonical_tokens[0] if canonical_tokens else ""
                if head in confirmed_proper or head in _KNOWN_PROPER:
                    pass  # confirmed
                elif head in ENTITY_STOPWORDS:
                    i = j
                    continue
                elif head.endswith(("ing", "ed", "ly", "ize", "ise")):
                    # verb-shaped; skip
                    i = j
                    continue
                # else: lenient accept — the stopword set is the actual filter

            if (canonical and canonical not in ENTITY_STOPWORDS
                    and len(canonical) >= 2 and alpha_count >= 2):
                entities.append((name, 'named', canonical))
            i = j
        else:
            i += 1

    # 1b. Known entities: models, datasets, venues
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

    # 3. Temporal entities
    temporal_patterns = [
        r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b',
        r'\b\d{1,2}(st|nd|rd|th)\b',
        r'\b(19|20)\d{2}\b',
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
    """
    entities = extract_entities(text, known_entities=known_entities)

    if authors:
        for author in authors:
            if author and len(author.strip()) > 0:
                canonical = author.lower().strip()
                if not any(e[2] == canonical for e in entities):
                    entities.append((author, 'named', canonical))

    if venue and len(venue.strip()) > 0:
        venue_clean = venue.strip()
        canonical = venue_clean.lower()
        if not any(e[2] == canonical for e in entities):
            entities.append((venue_clean, 'named', canonical))

    seen: Dict[str, Tuple[str, str, str]] = {}
    for name, etype, canonical in entities:
        if canonical not in seen or len(name) > len(seen[canonical][0]):
            seen[canonical] = (name, etype, canonical)

    return list(seen.values())


def compute_entity_salience(name: str, entity_type: str, text: str) -> float:
    """
    Compute salience (0.0-1.0) indicating how central this entity is to this memory.
    """
    if entity_type == 'named':
        return 0.9
    if entity_type == 'numeric':
        return 0.8
    if entity_type == 'temporal':
        return 0.5
    count = text.lower().count(name.lower())
    return min(0.8, 0.3 + count * 0.15)


def get_entity_fan(entity_count: Dict[str, int], canonical_name: str) -> int:
    """Get fan count for an entity (how many memories mention it)."""
    return entity_count.get(canonical_name, 1)
