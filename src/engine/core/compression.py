"""
Deterministic memory compression. No LLM calls.

Uses token importance scores from ingest for all ranking decisions.
Produces three outputs per memory:
  1. structured_summary — extractive summary ranked by token importance
  2. knowledge_nuggets — fact-dense sentences containing entity+number/date patterns
  3. entity_edges — typed entities with verb-mediated relationship predicates
"""

import json
import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set

logger = logging.getLogger(__name__)


# ── Compression thresholds (unchanged) ──────────────────────────────

COMPRESSION_THRESHOLDS = {
    "structured_summary": 0.70,
    "knowledge_nugget": 0.40,
    "entity_edges": 0.20,
    "archive": 0.10,
}


def should_compress(
    memory: dict,
    target_level: str,
    compression_resistance: float,
) -> bool:
    import random
    threshold = COMPRESSION_THRESHOLDS.get(target_level, 0.5)
    if memory["temperature"] >= threshold:
        return False
    if memory.get(target_level):
        return False
    if random.random() < compression_resistance:
        return False
    return True


# ── Data classes ────────────────────────────────────────────────────

@dataclass
class Entity:
    text: str           # "Battersea Vets", "March 15", "0.548"
    entity_type: str    # "named", "numeric", "concept", "temporal"
    start: int          # char offset in original text
    end: int            # char offset end
    importance: float   # from token importance scores


@dataclass
class ScoredSentence:
    text: str
    index: int              # original position (0-based)
    char_start: int         # char offset start in original text
    char_end: int           # char offset end
    score: float            # aggregate importance
    entity_count: int       # number of entities found in this sentence
    has_numeric: bool       # contains a numeric entity
    has_temporal: bool      # contains a temporal entity


@dataclass
class EntityEdge:
    source: str             # "Theo"
    source_type: str        # "named"
    target: str             # "Battersea Vets"
    target_type: str        # "named"
    predicate: str          # "treated_at" or "co_occurs"
    weight: float           # from token importance


# ── Sentence-initial detection ──────────────────────────────────────

_SENTENCE_BOUNDARY = re.compile(r'(?:^|[.!?]\s+)')


def _is_sentence_initial(text: str, pos: int) -> bool:
    """Check if position is at the start of a sentence."""
    if pos == 0:
        return True
    before = text[:pos]
    # Check if preceded by sentence-ending punctuation + whitespace
    stripped = before.rstrip()
    if stripped and stripped[-1] in '.!?':
        return True
    # Check if preceded by newline
    if before.endswith('\n'):
        return True
    return False


# ── 1. Entity Extraction with Spans ────────────────────────────────

# Capitalized words to never extract as named entities
_ENTITY_STOPWORDS = {
    'i', 'the', 'this', 'that', 'it', 'he', 'she', 'they', 'we',
    'my', 'a', 'an', 'his', 'her', 'its', 'our', 'your', 'their',
    'if', 'but', 'and', 'or', 'so', 'yet', 'for', 'nor', 'not',
    'at', 'in', 'on', 'to', 'of', 'by', 'from', 'with', 'as',
    'is', 'was', 'are', 'were', 'be', 'been', 'being',
    'has', 'had', 'have', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'shall', 'can',
    'what', 'when', 'where', 'who', 'why', 'how', 'which',
    'there', 'here', 'then', 'than', 'also', 'very', 'just', 'about',
    'after', 'before', 'during', 'between', 'through', 'into',
    'each', 'every', 'some', 'any', 'all', 'both', 'few', 'more',
    'most', 'other', 'no', 'only', 'same', 'such',
}

_MONTHS = (
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December',
)
_MONTHS_PATTERN = '|'.join(_MONTHS)

_DAYS = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
_DAYS_PATTERN = '|'.join(_DAYS)

# Temporal patterns (order matters — checked first)
_TEMPORAL_PATTERNS = [
    # ISO dates
    re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),
    # Month Day, Year or Month Day
    re.compile(rf'\b({_MONTHS_PATTERN})\s+\d{{1,2}}(?:,?\s+\d{{4}})?\b'),
    # Ordinal day + month
    re.compile(rf'\b\d{{1,2}}(?:st|nd|rd|th)\s+(?:{_MONTHS_PATTERN})\b'),
    # Short date format
    re.compile(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'),
    # Day names
    re.compile(rf'\b({_DAYS_PATTERN})\b'),
    # Relative time
    re.compile(r'\b(?:yesterday|today|tomorrow|last\s+week|next\s+month|last\s+month|next\s+week)\b', re.IGNORECASE),
]

# Numeric patterns
_NUMERIC_PATTERNS = [
    # Percentages
    re.compile(r'\b\d+\.?\d*%'),
    # Currency
    re.compile(r'[$\u00a3\u20ac]\d[\d,.]*'),
    # Counts with units
    re.compile(r'\b\d+\.?\d*\s*(?:mg|kg|ml|hours?|days?|weeks?|months?|years?|sessions?|times?|km|mi|lbs?|oz|gb|mb|kb|tb)\b', re.IGNORECASE),
    # Decimals (scores, metrics)
    re.compile(r'\b\d+\.\d+\b'),
    # Significant numbers (2+ digits)
    re.compile(r'\b\d{2,}[kmbt]?\b', re.IGNORECASE),
]

# Multi-word proper nouns
_PROPER_NOUN_PATTERN = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')

# Single capitalized word (not sentence-initial)
_SINGLE_CAP_PATTERN = re.compile(r'\b([A-Z][a-z]{2,})\b')

# Quoted strings
_QUOTED_PATTERN = re.compile(r'["\u201c]([^"\u201d]{2,50})["\u201d]')


def extract_compression_entities(
    text: str,
    token_scores: Optional[List[dict]] = None,
) -> List[Entity]:
    """
    Extract entities with character spans and importance scores.

    Priority order: temporal > numeric > named > concept.
    Earlier matches claim their spans; later passes skip occupied chars.
    """
    if not text:
        return []

    occupied: Set[int] = set()
    entities: List[Entity] = []

    def _spans_free(start: int, end: int) -> bool:
        return not any(i in occupied for i in range(start, end))

    def _claim(start: int, end: int):
        occupied.update(range(start, end))

    def _importance_at(start: int, end: int) -> float:
        """Look up average token importance for a character span."""
        if not token_scores:
            return 0.5
        # Map char span to token indices (approximate via cumulative char offsets)
        span_text = text[start:end].split()
        importances = []
        char_pos = 0
        for ts in token_scores:
            tok = ts.get('token', '')
            tok_start = text.find(tok, char_pos)
            if tok_start == -1:
                char_pos += len(tok) + 1
                continue
            tok_end = tok_start + len(tok)
            # Check overlap
            if tok_end > start and tok_start < end:
                importances.append(ts.get('importance', 0.5))
            char_pos = tok_end
        return sum(importances) / max(len(importances), 1)

    # Pass 1: Temporal
    for pattern in _TEMPORAL_PATTERNS:
        for match in pattern.finditer(text):
            s, e = match.start(), match.end()
            if _spans_free(s, e):
                _claim(s, e)
                entities.append(Entity(
                    text=match.group().strip(),
                    entity_type='temporal',
                    start=s, end=e,
                    importance=_importance_at(s, e),
                ))

    # Pass 2: Numeric
    for pattern in _NUMERIC_PATTERNS:
        for match in pattern.finditer(text):
            s, e = match.start(), match.end()
            if _spans_free(s, e):
                _claim(s, e)
                entities.append(Entity(
                    text=match.group().strip(),
                    entity_type='numeric',
                    start=s, end=e,
                    importance=_importance_at(s, e),
                ))

    # Pass 3: Named — multi-word proper nouns first
    for match in _PROPER_NOUN_PATTERN.finditer(text):
        s, e = match.start(), match.end()
        if _spans_free(s, e):
            name = match.group()
            # Filter stopword-only matches
            if any(w.lower() not in _ENTITY_STOPWORDS for w in name.split()):
                _claim(s, e)
                entities.append(Entity(
                    text=name,
                    entity_type='named',
                    start=s, end=e,
                    importance=_importance_at(s, e),
                ))

    # Pass 3b: Named — single capitalized words (not sentence-initial)
    # Collect all mid-sentence capitalized words
    mid_sentence_caps = set()
    for match in _SINGLE_CAP_PATTERN.finditer(text):
        word = match.group()
        if word.lower() in _ENTITY_STOPWORDS:
            continue
        if not _is_sentence_initial(text, match.start()):
            mid_sentence_caps.add(word.lower())

    for match in _SINGLE_CAP_PATTERN.finditer(text):
        s, e = match.start(), match.end()
        word = match.group()
        if word.lower() in _ENTITY_STOPWORDS:
            continue
        if not _spans_free(s, e):
            continue
        # Accept if: not sentence-initial, OR appears mid-sentence elsewhere
        if not _is_sentence_initial(text, s) or word.lower() in mid_sentence_caps:
            _claim(s, e)
            entities.append(Entity(
                text=word,
                entity_type='named',
                start=s, end=e,
                importance=_importance_at(s, e),
            ))

    # Pass 4: Concept — quoted strings
    for match in _QUOTED_PATTERN.finditer(text):
        s, e = match.start(1), match.end(1)
        if _spans_free(s, e):
            _claim(s, e)
            entities.append(Entity(
                text=match.group(1),
                entity_type='concept',
                start=s, end=e,
                importance=_importance_at(s, e),
            ))

    # Sort by position
    entities.sort(key=lambda e: e.start)
    return entities


# ── 2. Sentence Splitting + Scoring ────────────────────────────────

# Abbreviations to avoid splitting on
_ABBREV_LOOKBEHIND = r'(?<!\bMr)(?<!\bMrs)(?<!\bDr)(?<!\bSt)(?<!\bvs)(?<!\betc)(?<!\bJr)(?<!\bSr)'
_SENT_SPLIT = re.compile(_ABBREV_LOOKBEHIND + r'[.!?]\s+(?=[A-Z])')


def split_sentences(text: str) -> List[dict]:
    """Split text into sentences with char offsets."""
    if not text or not text.strip():
        return []

    parts = []
    last = 0
    for match in _SENT_SPLIT.finditer(text):
        # The split point is after the punctuation + space
        end = match.end()
        sentence_text = text[last:match.start() + 1].strip()  # include the punctuation
        if sentence_text:
            parts.append({
                'text': sentence_text,
                'char_start': last,
                'char_end': last + len(text[last:match.start() + 1].rstrip()),
            })
        last = end

    # Remaining text after last split
    remainder = text[last:].strip()
    if remainder:
        parts.append({
            'text': remainder.rstrip('.!? ') + ('.' if remainder.rstrip()[-1:] not in '.!?' else ''),
            'char_start': last,
            'char_end': len(text),
        })

    return parts


def score_sentences(
    sentences: List[dict],
    token_scores: List[dict],
    entities: List[Entity],
) -> List[ScoredSentence]:
    """Score each sentence using token importance and entity density."""
    if not sentences:
        return []

    # Build a char-offset -> token importance map
    # We'll match tokens to sentences by char range
    scored = []

    for idx, sent in enumerate(sentences):
        cs, ce = sent['char_start'], sent['char_end']

        # Find token scores that fall within this sentence
        sent_token_importances = []
        char_pos = 0
        for ts in token_scores:
            tok = ts.get('token', '')
            # Advance char_pos to find this token
            tok_idx = ts.get('_char_start')
            if tok_idx is None:
                # Approximate: tokens are whitespace-split from original text
                importance = ts.get('importance', 0.5)
                sent_token_importances.append(importance)
                continue

        # Fallback: just use all token scores proportionally
        if not sent_token_importances:
            # Split sentence text and match against token scores by position
            sent_words = sent['text'].split()
            # Estimate which token_scores correspond to this sentence
            # by counting words before this sentence
            words_before = len(sentences[0]['text'].split()) if idx > 0 else 0
            for i in range(idx):
                if i < idx:
                    words_before = sum(len(sentences[j]['text'].split()) for j in range(idx))
                    break

            for wi in range(len(sent_words)):
                ti = words_before + wi
                if ti < len(token_scores):
                    sent_token_importances.append(
                        token_scores[ti].get('importance', 0.5)
                    )
                else:
                    sent_token_importances.append(0.5)

        # Average token importance
        avg_importance = (
            sum(sent_token_importances) / max(len(sent_token_importances), 1)
        )

        # Entity density boost
        sent_entities = [
            e for e in entities
            if e.start >= cs and e.end <= ce
        ]
        entity_count = len(sent_entities)
        has_numeric = any(e.entity_type == 'numeric' for e in sent_entities)
        has_temporal = any(e.entity_type == 'temporal' for e in sent_entities)

        entity_density_boost = 1.0 + (0.3 * min(entity_count, 4))

        score = avg_importance * entity_density_boost

        scored.append(ScoredSentence(
            text=sent['text'],
            index=idx,
            char_start=cs,
            char_end=ce,
            score=score,
            entity_count=entity_count,
            has_numeric=has_numeric,
            has_temporal=has_temporal,
        ))

    return scored


# ── 3. Structured Summary ──────────────────────────────────────────

def extract_summary(
    scored_sentences: List[ScoredSentence],
    temperature: float,
) -> str:
    """
    Extractive summary: keep top-K sentences ranked by importance,
    restored to original order for coherence. K scales with temperature.
    """
    total = len(scored_sentences)
    if total == 0:
        return ''
    if total <= 2:
        return ' '.join(s.text for s in scored_sentences)

    keep_ratio = max(0.2, min(0.8, temperature))
    k = max(2, int(total * keep_ratio))

    ranked = sorted(scored_sentences, key=lambda s: s.score, reverse=True)[:k]
    ordered = sorted(ranked, key=lambda s: s.index)

    return ' '.join(s.text for s in ordered)


# ── 4. Knowledge Nuggets ───────────────────────────────────────────

def extract_nuggets(
    scored_sentences: List[ScoredSentence],
    entities: List[Entity],
) -> List[dict]:
    """
    Extract fact-dense nuggets: sentences with entity + number,
    entity + date, or entity + entity patterns.
    """
    nuggets = []

    for sent in scored_sentences:
        sent_entities = [
            e for e in entities
            if e.start >= sent.char_start and e.end <= sent.char_end
        ]
        types = {e.entity_type for e in sent_entities}

        is_fact = (
            len(sent_entities) >= 2
            or ('numeric' in types and len(sent_entities) >= 1)
            or ('temporal' in types and len(sent_entities) >= 1)
        )

        if is_fact:
            parts = [e.text for e in sent_entities]
            nuggets.append({
                'nugget': ' \u00b7 '.join(parts),
                'source_sentence': sent.text,
                'score': round(sent.score, 4),
                'entities': [
                    {'text': e.text, 'type': e.entity_type}
                    for e in sent_entities
                ],
            })

    nuggets.sort(key=lambda n: n['score'], reverse=True)
    return nuggets[:5]


# ── 5. Entity Edges (verb-mediated relationships) ──────────────────

COMMON_VERBS = {
    'is', 'was', 'are', 'were', 'has', 'had', 'have',
    'got', 'get', 'gets',
    'went', 'go', 'goes',
    'ran', 'run', 'runs',
    'made', 'make', 'makes',
    'said', 'told', 'asked',
    'gave', 'give', 'gives',
    'took', 'take', 'takes',
    'came', 'come', 'comes',
    'saw', 'see', 'sees',
    'found', 'find', 'finds',
    'called', 'named',
    'scored', 'achieved', 'reached', 'hit',
    'started', 'began', 'finished', 'completed',
    'bought', 'sold', 'paid', 'cost',
    'lives', 'lived', 'works', 'worked',
    'moved', 'visited', 'attended',
    'received', 'sent', 'delivered',
    'built', 'created', 'designed', 'developed',
    'joined', 'left', 'quit',
    'likes', 'loves', 'hates', 'prefers', 'enjoys',
    'needs', 'wants', 'requires',
    'uses', 'used',
    'met', 'meet', 'meets',
    'born', 'died',
    'ate', 'eat', 'eats', 'drank', 'drinks',
    'improved', 'jumped', 'dropped', 'increased', 'decreased',
}

VERB_TO_PREDICATE = {
    'is': 'is_a', 'was': 'is_a', 'are': 'is_a', 'were': 'is_a',
    'has': 'has', 'had': 'had', 'have': 'has',
    'got': 'received', 'get': 'received',
    'went': 'went_to', 'go': 'goes_to',
    'lives': 'lives_in', 'lived': 'lived_in',
    'works': 'works_at', 'worked': 'worked_at',
    'visited': 'visited', 'attended': 'attended',
    'born': 'born_in', 'died': 'died_in',
    'met': 'met', 'meet': 'met',
    'uses': 'uses', 'used': 'used',
    'likes': 'likes', 'loves': 'loves', 'prefers': 'prefers',
    'scored': 'scored', 'achieved': 'achieved',
    'bought': 'bought', 'sold': 'sold',
    'joined': 'joined', 'left': 'left',
    'started': 'started', 'began': 'started',
    'created': 'created', 'built': 'built', 'developed': 'developed',
    'improved': 'improved', 'jumped': 'jumped_to',
    'dropped': 'dropped_to', 'increased': 'increased_to', 'decreased': 'decreased_to',
}


def find_verb_between(text: str, entity1_end: int, entity2_start: int) -> Optional[str]:
    """Find the first common verb in the text between two entity spans."""
    between = text[entity1_end:entity2_start].strip().lower()
    words = between.split()
    for word in words:
        cleaned = word.rstrip('.,;:!?')
        if cleaned in COMMON_VERBS:
            return cleaned
    return None


def verb_to_predicate(verb: Optional[str]) -> str:
    """Map a verb to a relationship predicate."""
    if verb is None:
        return 'co_occurs'
    return VERB_TO_PREDICATE.get(verb, verb + '_with')


def compute_edge_weight(
    source_entity: Entity,
    target_entity: Entity,
    sentence_score: float,
) -> float:
    """Edge weight from entity importance and sentence score."""
    avg_importance = (source_entity.importance + target_entity.importance) / 2
    return round(avg_importance * sentence_score, 4)


def extract_edges(
    text: str,
    sentences: List[ScoredSentence],
    entities: List[Entity],
) -> List[EntityEdge]:
    """Extract verb-mediated entity relationships per sentence."""
    edges = []

    for sent in sentences:
        sent_entities = [
            e for e in entities
            if e.start >= sent.char_start and e.end <= sent.char_end
        ]

        for i, e1 in enumerate(sent_entities):
            for e2 in sent_entities[i + 1:]:
                verb = find_verb_between(text, e1.end, e2.start)
                predicate = verb_to_predicate(verb)
                weight = compute_edge_weight(e1, e2, sent.score)

                edges.append(EntityEdge(
                    source=e1.text,
                    source_type=e1.entity_type,
                    target=e2.text,
                    target_type=e2.entity_type,
                    predicate=predicate,
                    weight=weight,
                ))

    return deduplicate_edges(edges)


def deduplicate_edges(edges: List[EntityEdge]) -> List[EntityEdge]:
    """Deduplicate edges, preferring verb-mediated over co_occurs, and higher weight."""
    seen: Dict[tuple, EntityEdge] = {}

    for edge in edges:
        key = (edge.source.lower(), edge.target.lower())
        if key not in seen:
            seen[key] = edge
        else:
            existing = seen[key]
            if existing.predicate == 'co_occurs' and edge.predicate != 'co_occurs':
                # Prefer verb-mediated predicate
                seen[key] = edge
            elif edge.weight > existing.weight:
                # Keep higher weight, but preserve verb predicate if existing has one
                if existing.predicate != 'co_occurs':
                    edge.predicate = existing.predicate
                seen[key] = edge

    return list(seen.values())


# ── 6. Main compress() function ────────────────────────────────────

def compress(
    memory_text: str,
    token_scores: List[dict],
    temperature: float = 0.5,
) -> dict:
    """
    Deterministic memory compression. No LLM calls.
    Uses token importance scores from ingest for all ranking decisions.

    Args:
        memory_text: Raw memory text to compress
        token_scores: Token importance scores from ingest
                      (list of {token, importance, weight})
        temperature: Current memory temperature (0.0-1.0),
                     controls how many sentences to keep in summary

    Returns:
        dict with keys:
            structured_summary: str
            knowledge_nuggets: list[dict]
            entity_edges: list[dict]
    """
    if not memory_text or not memory_text.strip():
        return {
            'structured_summary': '',
            'knowledge_nuggets': [],
            'entity_edges': [],
        }

    # 1. Extract entities with spans and importance
    entities = extract_compression_entities(memory_text, token_scores)

    # 2. Split into sentences and score each one
    raw_sentences = split_sentences(memory_text)
    scored_sentences = score_sentences(raw_sentences, token_scores or [], entities)

    # 3. Build the three outputs
    summary = extract_summary(scored_sentences, temperature)
    nuggets = extract_nuggets(scored_sentences, entities)
    edges = extract_edges(memory_text, scored_sentences, entities)

    result = {
        'structured_summary': summary,
        'knowledge_nuggets': nuggets,
        'entity_edges': [
            {
                'source': e.source,
                'source_type': e.source_type,
                'target': e.target,
                'target_type': e.target_type,
                'predicate': e.predicate,
                'weight': e.weight,
            }
            for e in edges
        ],
    }

    _log_compression(memory_text, result)

    return result


# ── Logging ─────────────────────────────────────────────────────────

def _log_compression(input_text: str, result: dict):
    """Log compression input/output for quality evaluation."""
    pair = {
        'input': input_text,
        'structured_summary': result['structured_summary'],
        'knowledge_nuggets': result['knowledge_nuggets'],
        'entity_edges': result['entity_edges'],
    }
    try:
        with open('compression_training_data.jsonl', 'a') as f:
            f.write(json.dumps(pair) + '\n')
    except Exception:
        pass
