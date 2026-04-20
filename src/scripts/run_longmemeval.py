#!/usr/bin/env python3
"""
LongMemEval Benchmark — Dendric Memory Engine

Runs the official LongMemEval benchmark (ICLR 2025, Wu et al.) against Dendric.
500 questions testing 5 memory abilities across multi-session chat histories.

Two-phase approach:
  Phase 1: Generate hypothesis file (Dendric ingest + retrieve + Claude answer)
  Phase 2: Evaluate with official evaluate_qa.py (GPT-4o judge, exact prompts per type)

Usage:
    python run_longmemeval.py --questions 10 --top-k 10 --cot
    python run_longmemeval.py --questions 500 --workers 4
"""

import os
import re
import sys
import json
import time
import logging
import argparse
import threading
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce noise from HTTP libraries when running parallel
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from anthropic import Anthropic
except ImportError:
    logger.error("anthropic not installed. Install with: pip install anthropic")
    sys.exit(1)

try:
    import openai
    from openai import OpenAI
except ImportError:
    logger.error("openai not installed. Install with: pip install openai")
    sys.exit(1)

try:
    import backoff
    HAS_BACKOFF = True
except ImportError:
    HAS_BACKOFF = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from src.engine.core.engine import MemoryEngine
from src.engine.retrieval.temporal import detect_temporal_query

# Thread-safe print lock
_print_lock = threading.Lock()

def tprint(*args, **kwargs):
    """Thread-safe print."""
    with _print_lock:
        print(*args, **kwargs, flush=True)


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_longmemeval_data(data_path: Optional[str] = None) -> List[Dict]:
    """Load LongMemEval benchmark data from local file or HuggingFace."""
    if data_path and os.path.exists(data_path):
        logger.info(f"Loading from local file: {data_path}")
        with open(data_path) as f:
            return json.load(f)

    os.makedirs("data/longmemeval", exist_ok=True)
    target = "data/longmemeval/longmemeval_s_cleaned.json"
    if os.path.exists(target):
        logger.info(f"Loading from cached file: {target}")
        with open(target) as f:
            return json.load(f)

    import urllib.request
    url = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
    logger.info(f"Downloading {url}...")
    urllib.request.urlretrieve(url, target)
    logger.info(f"Downloaded to {target}")
    with open(target) as f:
        return json.load(f)


# ── Memory Ingestion ─────────────────────────────────────────────────────────

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def ingest_longmemeval_item(
    item: Dict,
    engine: MemoryEngine,
    consolidation_cycles: int = 1,
) -> List[str]:
    """
    Ingest one LongMemEval item's chat sessions into Dendric.

    Collects all content upfront across three strategies, then batch-ingests
    with a single embedding call for 10-50x speedup over per-item remember().

    Three-level ingestion strategy:
    1. Per-turn memories: each turn gets its own memory
    2. Chunked session memories: overlapping chunks for broader context
    3. Session summaries: short sessions get a combined memory

    Returns list of memory IDs.
    """
    sessions = item.get("haystack_sessions", [])
    dates = item.get("haystack_dates", [])

    # ── Collect all items upfront ──
    batch_items = []

    for session_idx, turns in enumerate(sessions):
        session_date = dates[session_idx] if session_idx < len(dates) else None

        # Strategy 1: Per-turn ingestion
        for turn_idx, turn in enumerate(turns):
            role = turn.get('role', 'unknown')
            content = turn.get('content', '')
            if not content.strip() or len(content.strip()) < 10:
                continue

            ctx = f"session_{session_idx}_turn_{turn_idx}"
            if session_date:
                ctx = f"{session_date} | {ctx}"

            batch_items.append({
                "content": f"[{role.capitalize()}]: {content}",
                "source": "longmemeval_chat",
                "context": ctx,
            })

        # Strategy 2: Chunked session memories
        parts = []
        for turn in turns:
            role = turn.get('role', 'unknown')
            content = turn.get('content', '')
            parts.append(f"{role.capitalize()}: {content}")

        session_content = "\n".join(parts)
        if len(session_content.strip()) < 10:
            continue

        chunks = _chunk_text(session_content)
        session_topic = session_content[:200].replace('\n', ' ').strip()

        for chunk_idx, chunk in enumerate(chunks):
            ctx = f"session_{session_idx}"
            if len(chunks) > 1:
                ctx += f"_chunk_{chunk_idx}"
            if session_date:
                ctx = f"{session_date} | {ctx}"

            if chunk_idx > 0 and len(chunks) > 1:
                chunk = f"[Session context: {session_topic}...]\n\n{chunk}"

            batch_items.append({
                "content": chunk,
                "source": "longmemeval_chat",
                "context": ctx,
            })

        # Strategy 3: Session-level summary memory
        word_count = len(session_content.split())
        if len(turns) >= 2 and word_count < 500:
            ctx = f"session_{session_idx}_summary"
            if session_date:
                ctx = f"{session_date} | {ctx}"

            batch_items.append({
                "content": session_content,
                "source": "longmemeval_chat",
                "context": ctx,
            })

    # ── Batch ingest all at once ──
    # Entity linking enabled: GLiNER provides high-quality entities
    # which let consolidation cycles build meaningful aggregates.
    results = engine.remember_batch(batch_items, link_entities=True)
    memory_ids = [r.get('id', '') for r in results]

    for cycle in range(consolidation_cycles):
        try:
            engine.consolidate()
        except Exception as e:
            logger.warning(f"Consolidation cycle {cycle+1} failed: {e}")

    return memory_ids


# ── Answer Generation ────────────────────────────────────────────────────────

_AGGREGATION_RE = re.compile(
    r"\b(?:how many|how much|total|in total|all the|combined|sum of|altogether)\b",
    re.IGNORECASE,
)

# Stopwords for subject extraction
_SUBJECT_STOPWORDS = {
    "i", "me", "my", "we", "our", "you", "your", "the", "a", "an",
    "did", "do", "does", "have", "has", "had", "was", "were", "been",
    "is", "am", "are", "will", "would", "could", "should", "can",
    "to", "of", "in", "on", "at", "for", "with", "from", "by",
    "and", "or", "but", "not", "no", "so", "if", "than", "that",
    "this", "these", "those", "it", "its",
    "how", "many", "much", "what", "which", "where", "when", "who",
    "total", "all", "any", "some", "each", "every",
    "since", "start", "year", "month", "week", "day", "time", "days",
    "currently", "recently", "ago", "last", "past", "first",
    "spent", "made", "taken", "done", "gone", "been",
    "worked", "bought", "used", "attended", "visited", "played",
    "watched", "tried", "completed", "finished", "started",
    "different", "types", "various", "kind", "kinds", "type",
    "need", "want", "get", "got", "go", "went", "come", "came",
    "spend", "pick", "up", "return", "take", "give",
    "items", "things", "number", "amount", "hours", "money",
    "three", "two", "four", "five", "six", "seven", "eight",
    "weeks", "months", "years", "watch", "drive", "driving",
    "combined", "including", "related",
}


def _extract_subject_keywords(question: str) -> List[str]:
    """Extract meaningful subject nouns/phrases from an aggregation query.

    "How many model kits have I worked on or bought?" -> ["model kits", "model", "kits"]
    "How much money did I spend on bike-related expenses?" -> ["bike-related expenses", "bike", "expenses"]
    """
    # Strip the aggregation prefix to get the subject
    q = re.sub(r"^(?:how many|how much|what is the total number of|what are all the)\s+", "",
               question.lower().strip().rstrip("?."))
    # Strip trailing verb phrases
    q = re.sub(r"\s+(?:did|have|do|was|were|am|is)\s+(?:I|we|you)\b.*$", "", q)
    q = re.sub(r"\s+(?:since|in|during|from|this|last)\s+.*$", "", q)
    q = re.sub(r"\s+(?:that I|I have|I've|I did)\b.*$", "", q)

    words = q.split()
    # Filter stopwords
    content_words = [w for w in words if w.lower() not in _SUBJECT_STOPWORDS and len(w) > 1]

    results = []
    # Full phrase first (most specific)
    if len(content_words) >= 2:
        results.append(" ".join(content_words))
    # Individual words
    results.extend(content_words)
    return results


def _generate_answer(model: str, prompt: str, max_tokens: int = 500) -> str:
    """Route answer generation to the right provider based on model name.

    Model-name prefix decides the provider:
      gpt-*    → OpenAI (e.g. gpt-5.4-nano)
      claude-* → Anthropic
    Clients are constructed per-call; both SDKs are cheap to instantiate and
    this keeps the path stateless under parallel workers.
    """
    name = (model or "").lower()
    if name.startswith("gpt-"):
        client = OpenAI()
        # GPT-5.x family (incl. nano) rejects `max_tokens` — requires
        # `max_completion_tokens` in chat.completions. Internal reasoning
        # tokens count against this budget, so give nano some headroom.
        is_gpt5 = name.startswith("gpt-5")
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if is_gpt5:
            kwargs["max_completion_tokens"] = max(max_tokens, 2000)
        else:
            kwargs["max_tokens"] = max_tokens
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
    # Default to Anthropic
    client = Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def answer_question(
    question: str,
    engine: MemoryEngine,
    top_k: int = 10,
    use_cot: bool = False,
    question_timestamp: str = None,
    return_retrieved: bool = False,
    **kwargs,
):
    """Retrieve memories with Dendric using multi-query + temporal awareness, then answer with Claude.

    If return_retrieved=True, returns (hypothesis, retrieved_memories_list)."""
    seen_ids = set()
    all_results = []

    # Detect aggregation queries (counting/summing across sessions)
    is_aggregation = bool(_AGGREGATION_RE.search(question))
    retrieve_k = top_k
    # Counting questions need broader recall — bump final cap so scattered items aren't truncated
    effective_top_k = max(top_k, 25) if is_aggregation else top_k

    # Detect temporal signals for context-aware prompting
    temporal_info = detect_temporal_query(question, question_timestamp)

    try:
        results = engine.recall(
            query=question, top_k=retrieve_k, reheat=False,
            question_timestamp=question_timestamp,
        )
        for r in results:
            if r.get('id') not in seen_ids:
                seen_ids.add(r['id'])
                all_results.append(r)
    except Exception as e:
        logger.warning(f"Retrieval failed: {e}")

    rephrase_map = {
        "where did I": "I went to",
        "when did I": "I did it on",
        "what did I": "I",
        "how long": "it takes",
        "what is the name": "the name is",
        "what was my": "my",
        "how many days": "the number of days is",
        "how many weeks": "the number of weeks is",
        "how many months": "the number of months is",
    }
    alt_query = question
    for pattern, replacement in rephrase_map.items():
        if question.lower().startswith(pattern):
            alt_query = question.lower().replace(pattern, replacement, 1)
            break

    if alt_query != question:
        try:
            results2 = engine.recall(
                query=alt_query, top_k=retrieve_k, reheat=False,
                question_timestamp=question_timestamp,
            )
            for r in results2:
                if r.get('id') not in seen_ids:
                    seen_ids.add(r['id'])
                    all_results.append(r)
        except Exception as e:
            logger.warning(f"Secondary retrieval failed: {e}")

    # Aggregation boost: targeted keyword searches for subject nouns
    # This catches mentions scattered across sessions that vector similarity misses.
    if is_aggregation:
        subject_keywords = _extract_subject_keywords(question)
        for kw in subject_keywords[:3]:
            try:
                kw_results = engine.store.keyword_search(kw, top_k=top_k * 3, min_temp=0.0)
                for r in kw_results:
                    if r.get('id') not in seen_ids:
                        seen_ids.add(r['id'])
                        all_results.append(r)
            except Exception as e:
                logger.warning(f"Keyword search for '{kw}' failed: {e}")

    # For aggregation queries: limit per-session representation to maximize session coverage
    # Multiple chunks from the same session waste slots — keep max 2 per session
    if is_aggregation and len(all_results) > effective_top_k:
        session_results = {}  # session_id -> list of results (max 2)
        no_session = []
        max_per_session = 2
        for r in all_results:
            ctx = r.get('context', '')
            sess = 'unknown'
            if '|' in ctx:
                sess_part = ctx.split('|')[-1].strip()
                m = re.match(r'(session_\d+)', sess_part)
                if m:
                    sess = m.group(1)
            if sess == 'unknown':
                no_session.append(r)
            else:
                if sess not in session_results:
                    session_results[sess] = []
                if len(session_results[sess]) < max_per_session:
                    session_results[sess].append(r)

        # Rebuild: up to 2 per session, sorted by score
        deduped = []
        for results_list in session_results.values():
            deduped.extend(results_list)
        deduped.sort(key=lambda r: r.get('fusion_score', r.get('score', 0)), reverse=True)
        deduped.extend(no_session)
        all_results = deduped

    all_results = all_results[:effective_top_k]

    context_parts = []
    for i, result in enumerate(all_results):
        mem_id = result.get('id', '')
        if not mem_id:
            continue
        mem = engine.store.get(mem_id)
        content = mem.raw_content if mem else result.get('content', result.get('raw_content', ''))
        temp = mem.temperature if mem else result.get('temperature', 0.5)
        score = result.get('fusion_score', result.get('similarity', 0.5))
        mem_context = result.get('context', '')

        # Always include date — needed for knowledge-update recency and temporal reasoning
        date_part = ''
        if mem_context and '|' in mem_context:
            date_part = mem_context.split('|')[0].strip()
        if date_part:
            context_parts.append(
                f"[Memory {i+1}, date={date_part}, score={score:.2f}]\n{content}"
            )
        else:
            context_parts.append(f"[Memory {i+1}, score={score:.2f}]\n{content}")

    if not context_parts:
        if return_retrieved:
            return "I don't know.", all_results
        return "I don't know."

    # Prepend temporal computation hint if available
    temporal_hint = None
    for r in all_results:
        if r.get("_temporal_hint"):
            temporal_hint = r["_temporal_hint"]
            break

    context = "\n\n".join(context_parts)
    if temporal_hint:
        context = temporal_hint + "\n\n" + context

    # Build prompt — temporal queries get timestamp-aware instructions
    if temporal_info['is_temporal'] and use_cot:
        prompt = (
            f"You are answering a question about a user's past conversations. "
            f"Each memory has a date stamp — pay close attention to WHEN things were said.\n\n"
            f"The question is being asked on: {question_timestamp or 'unknown date'}\n\n"
            f"Retrieved memories:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Instructions:\n"
            f"1. Read every memory carefully, paying attention to the DATE of each memory.\n"
            f"2. If the question asks about a specific time period, prioritize memories from that period.\n"
            f"3. If the question asks about time differences (days/weeks/months between events), "
            f"calculate from the dates shown.\n"
            f"4. If the question asks about ordering, use the dates to determine chronological order.\n"
            f"5. If the question asks about changes over time, compare earlier and later memories.\n"
            f"6. Give a short, direct answer. Do not hedge or qualify.\n"
            f"7. Only say \"I don't know\" as an absolute last resort.\n\nAnswer:"
        )
    elif is_aggregation and use_cot:
        prompt = (
            f"You are answering a counting/aggregation question about a user's past conversations. "
            f"The items may be scattered across many different conversations.\n\n"
            f"Retrieved memories:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Instructions:\n"
            f"1. Search ALL memories. Each may contain a unique item to count.\n"
            f"2. Do not double-count items mentioned in multiple memories.\n"
            f"3. When memories show conflicting counts, prefer the MOST RECENT (latest date).\n"
            f"4. Answer with the count and a brief list. Be concise.\n"
            f"5. Only say \"I don't know\" as an absolute last resort.\n\nAnswer:"
        )
    elif use_cot:
        prompt = (
            f"You are answering a question about a user's past conversations. "
            f"Search through ALL of the following retrieved memories carefully — "
            f"the answer may be a small detail mentioned in passing.\n\n"
            f"Retrieved memories:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Instructions:\n"
            f"1. Read every memory carefully, even ones that seem unrelated — the answer is often a brief mention.\n"
            f"2. Use context clues from surrounding conversation to infer answers.\n"
            f"3. IMPORTANT: When memories contain conflicting information about the same topic "
            f"(e.g., different values for a count, location, preference, or schedule), "
            f"ALWAYS use the information from the MOST RECENT memory (latest date). "
            f"The user may have updated or corrected this information over time.\n"
            f"4. Give a short, direct answer. Do not hedge or qualify.\n"
            f"5. Only say \"I don't know\" as an absolute last resort.\n\nAnswer:"
        )
    else:
        prompt = (
            f"You are answering a question about a user's past conversations. "
            f"Search through ALL memories carefully — the answer may be a small detail.\n\n"
            f"IMPORTANT: When memories contain conflicting information about the same topic, "
            f"ALWAYS prefer the MOST RECENT memory (latest date). The user may have updated their information.\n\n"
            f"Retrieved memories:\n{context}\n\n"
            f"Question: {question}\n\nAnswer concisely and directly:"
        )

    answer_model = kwargs.get("answer_model", "claude-haiku-4-5-20251001")
    try:
        text = _generate_answer(answer_model, prompt)
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        text = "I don't know."
    if return_retrieved:
        return text, all_results
    return text


# ── Official Evaluation (exact from xiaowu0162/LongMemEval) ──────────────────

def get_anscheck_prompt(task, question, answer, response, abstention=False):
    """
    Exact function from official LongMemEval evaluate_qa.py.
    Source: https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py
    """
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError(f"Unknown question type: {task}")
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response)
    return prompt


def evaluate_with_gpt4o(question_id, question, answer, hypothesis, question_type):
    """Run GPT-4o evaluation using the exact official protocol."""
    client = OpenAI()
    abstention = '_abs' in str(question_id)
    prompt = get_anscheck_prompt(question_type, question, answer, hypothesis, abstention=abstention)

    kwargs = {
        'model': 'gpt-4o-2024-08-06',
        'messages': [{"role": "user", "content": prompt}],
        'n': 1,
        'temperature': 0,
        'max_tokens': 10,
    }

    if HAS_BACKOFF:
        @backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIError))
        def _call():
            return client.chat.completions.create(**kwargs)
        completion = _call()
    else:
        completion = client.chat.completions.create(**kwargs)

    eval_response = completion.choices[0].message.content.strip()
    label = 'yes' in eval_response.lower()
    return label


# ── Single Question Worker ──────────────────────────────────────────────────

def _create_temp_db(db_name: str):
    """Create a temporary database for worker isolation with pgvector."""
    import psycopg2
    conn = psycopg2.connect("postgresql://localhost:5432/postgres")
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"DROP DATABASE IF EXISTS {db_name}")
        cur.execute(f"CREATE DATABASE {db_name}")
    conn.close()
    # Enable pgvector extension in the new database
    conn2 = psycopg2.connect(f"postgresql://localhost:5432/{db_name}")
    conn2.autocommit = True
    with conn2.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn2.close()


def _drop_temp_db(db_name: str):
    """Drop a temporary worker database."""
    import psycopg2
    conn = psycopg2.connect("postgresql://localhost:5432/postgres")
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"DROP DATABASE IF EXISTS {db_name}")
    conn.close()


def process_one_question(
    idx: int,
    item: Dict,
    n: int,
    top_k: int,
    use_cot: bool,
    consolidation_cycles: int,
    use_temp_db: bool = False,
    answer_model: str = "claude-haiku-4-5-20251001",
    db_prefix: str = "lme",
    debug_questions: Optional[set] = None,
) -> Dict:
    """
    Process a single LongMemEval question end-to-end.
    When use_temp_db=True, creates an isolated database per question (for parallel mode).
    """
    question_id = item['question_id']
    question = item['question']
    answer = str(item['answer'])
    question_type = item['question_type']

    tprint(f"[{idx+1}/{n}] {question_type}: {question[:70]}...")

    db_name = None
    try:
        from src.engine.config import EngineConfig
        if use_temp_db:
            db_name = f"mnestic_{db_prefix}_worker_{idx}"
            _create_temp_db(db_name)
            config = EngineConfig(db_url=f"postgresql://localhost:5432/{db_name}", novelty_gate=0.05, overfetch_multiplier=5)
            engine = MemoryEngine(config=config)
        else:
            config = EngineConfig(novelty_gate=0.05, overfetch_multiplier=5)
            engine = MemoryEngine(config=config)
            engine.store.clear_all()
            engine._cycle_count = 0

        # Phase 1: Ingest + Retrieve + Answer
        memory_ids = ingest_longmemeval_item(item, engine, consolidation_cycles)
        question_timestamp = item.get('question_date', None)
        debug_this = debug_questions and question_id in debug_questions
        if debug_this:
            hypothesis, retrieved = answer_question(
                question, engine, top_k, use_cot, question_timestamp,
                answer_model=answer_model, return_retrieved=True,
            )
            try:
                os.makedirs("longmemeval/debug", exist_ok=True)
                dbg = {
                    'question_id': question_id,
                    'question': question,
                    'gold': answer,
                    'hypothesis': hypothesis,
                    'top_k': top_k,
                    'last_recall_paths': getattr(engine, '_last_recall_paths', None),
                    'retrieved': [
                        {
                            'rank': i + 1,
                            'id': r.get('id'),
                            'score': r.get('fusion_score', r.get('similarity', r.get('score'))),
                            'retrieval_paths': r.get('retrieval_paths'),
                            'path_debug': r.get('_path_debug'),
                            'temperature': r.get('temperature'),
                            'modulation': r.get('_modulation'),
                            'unmodulated_fusion_score': r.get('_unmodulated_fusion_score'),
                            'context': r.get('context'),
                            'content': (engine.store.get(r['id']).raw_content
                                        if r.get('id') and engine.store.get(r['id'])
                                        else r.get('content', r.get('raw_content', ''))),
                        }
                        for i, r in enumerate(retrieved)
                    ],
                }
                with open(f"longmemeval/debug/{question_id}.json", 'w') as f:
                    json.dump(dbg, f, indent=2, default=str)
                tprint(f"  [debug] wrote longmemeval/debug/{question_id}.json")
            except Exception as e:
                logger.warning(f"Debug dump failed for {question_id}: {e}")
        else:
            hypothesis = answer_question(question, engine, top_k, use_cot, question_timestamp,
                                         answer_model=answer_model)

        # Phase 2: Official GPT-4o evaluation
        label = evaluate_with_gpt4o(question_id, question, answer, hypothesis, question_type)

        engine.store.close()
        if db_name:
            _drop_temp_db(db_name)

        result = {
            'idx': idx,
            'question_id': question_id,
            'question': question[:200],
            'gold': answer[:200],
            'predicted': hypothesis[:200],
            'correct': label,
            'type': question_type,
            'memories_ingested': len(memory_ids),
            'hypothesis_full': hypothesis,
        }

        status = "✓" if label else "✗"
        tprint(f"  [{idx+1}/{n}] {status} {hypothesis[:70]}...")

        return result

    except Exception as e:
        logger.error(f"Error on question {idx}: {e}")
        try:
            engine.store.close()
        except Exception:
            pass
        if db_name:
            try:
                _drop_temp_db(db_name)
            except Exception:
                pass
        return {
            'idx': idx, 'question_id': question_id,
            'question': question[:200], 'gold': answer[:200],
            'error': str(e), 'type': question_type,
        }


# ── Main Benchmark ───────────────────────────────────────────────────────────

def _flush_incremental(results_detail, hypotheses, type2acc, start_time, config):
    """Flush current results to disk so partial runs are recoverable."""
    elapsed = time.time() - start_time
    total_correct = sum(1 for r in results_detail if r.get('correct'))
    total = sum(1 for r in results_detail if 'correct' in r)

    sorted_results = sorted(results_detail, key=lambda r: r.get('idx', 0))
    sorted_hyps = sorted(hypotheses, key=lambda h: h.get('question_id', ''))

    # Write hypothesis file
    hyp_path = "longmemeval/longmemeval_hypothesis_incremental.jsonl"
    with open(hyp_path, 'w') as f:
        for h in sorted_hyps:
            f.write(json.dumps(h) + '\n')

    # Write detailed results
    results_path = "longmemeval/longmemeval_results_incremental.json"
    results = {
        'overall_accuracy': total_correct / total if total else 0,
        'task_averaged_accuracy': 0,
        'total': total,
        'correct': total_correct,
        'by_type': {k: {'n': len(v), 'correct': sum(v), 'accuracy': sum(v)/len(v) if v else 0}
                    for k, v in type2acc.items()},
        'per_question': sorted_results,
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'status': 'in_progress',
    }
    # Compute task-averaged accuracy
    task_accs = [v['accuracy'] for v in results['by_type'].values() if v['n'] > 0]
    results['task_averaged_accuracy'] = sum(task_accs) / len(task_accs) if task_accs else 0

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def run_longmemeval(
    data_path: Optional[str] = None,
    max_questions: int = 500,
    top_k: int = 10,
    use_cot: bool = True,
    consolidation_cycles: int = 1,
    workers: int = 1,
    answer_model: str = "claude-haiku-4-5-20251001",
    question_types: Optional[List[str]] = None,
    db_prefix: str = "lme",
    debug_questions: Optional[set] = None,
    per_type: Optional[int] = None,
    only_ids: Optional[set] = None,
) -> Dict:
    """Run the full LongMemEval benchmark with optional parallelism."""

    # Ensure output directory exists
    os.makedirs("longmemeval", exist_ok=True)

    # Pre-warm the embedding model before spawning workers
    from src.engine.embeddings.embed import get_model
    logger.info("Pre-warming embedding model...")
    get_model("nomic-ai/nomic-embed-text-v1.5")

    logger.info("Loading LongMemEval data...")
    data = load_longmemeval_data(data_path)
    logger.info(f"Loaded {len(data)} questions")

    # Filter by question type if specified
    if question_types:
        data = [d for d in data if d.get('question_type') in question_types]
        logger.info(f"Filtered to {len(data)} questions of types: {question_types}")

    # Filter to specific question IDs if requested
    if only_ids:
        data = [d for d in data if d.get('question_id') in only_ids]
        logger.info(f"Filtered to {len(data)} questions matching --only-ids")

    # Stratified sampling: take first `per_type` questions from each category
    if per_type:
        by_type = defaultdict(list)
        for item in data:
            by_type[item.get('question_type', 'unknown')].append(item)
        sampled = []
        for qtype in sorted(by_type.keys()):
            sampled.extend(by_type[qtype][:per_type])
        data = sampled
        logger.info(f"Stratified to {len(data)} questions ({per_type} per type across {len(by_type)} types)")

    # --per-type and --only-ids express a deliberate sample size via filtering.
    # Don't let the --questions default silently truncate them.
    if per_type or only_ids:
        n = len(data)
    else:
        n = min(max_questions, len(data))
    config = {
        'questions': n, 'top_k': top_k, 'cot': use_cot,
        'consolidation_cycles': consolidation_cycles,
        'workers': workers,
        'answer_model': answer_model,
        'eval_model': 'gpt-4o-2024-08-06',
        'question_types': question_types,
    }

    print(f"\n{'='*70}")
    print(f"LONGMEMEVAL BENCHMARK — Dendric Memory Engine")
    print(f"{'='*70}")
    print(f"Questions: {n}  |  Top-k: {top_k}  |  CoT: {use_cot}")
    print(f"Cycles: {consolidation_cycles}  |  Workers: {workers}")
    print(f"Answer model: {answer_model}")
    if question_types:
        print(f"Types: {', '.join(question_types)}")
    print(f"{'='*70}\n")

    start_time = time.time()

    # Results tracking
    type2acc = defaultdict(list)
    hypotheses = []
    results_detail = []

    # Incremental flush every N questions
    flush_interval = 5

    def _record_result(result):
        results_detail.append(result)
        if 'correct' in result:
            type2acc[result['type']].append(1 if result['correct'] else 0)
            hypotheses.append({
                'question_id': result['question_id'],
                'hypothesis': result.get('hypothesis_full', result.get('predicted', '')),
                'autoeval_label': {'model': 'gpt-4o-2024-08-06', 'label': result['correct']},
            })
        # Flush to disk periodically
        if len(results_detail) % flush_interval == 0:
            _flush_incremental(results_detail, hypotheses, type2acc, start_time, config)

    if workers <= 1:
        # Sequential mode (original behavior)
        for idx, item in enumerate(data[:n]):
            result = process_one_question(idx, item, n, top_k, use_cot, consolidation_cycles,
                                          use_temp_db=False, answer_model=answer_model,
                                          db_prefix=db_prefix, debug_questions=debug_questions)
            _record_result(result)
            if 'correct' in result:
                total_correct = sum(1 for r in results_detail if r.get('correct'))
                total = sum(1 for r in results_detail if 'correct' in r)
                tprint(f"  Running: {total_correct}/{total} ({100*total_correct/total:.1f}%)")
    else:
        # Parallel mode — each worker gets its own temporary database
        tprint(f"Running {workers} workers in parallel (temp DB per question)...\n")
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for idx, item in enumerate(data[:n]):
                future = executor.submit(
                    process_one_question, idx, item, n, top_k, use_cot, consolidation_cycles,
                    use_temp_db=True, answer_model=answer_model, db_prefix=db_prefix,
                    debug_questions=debug_questions,
                )
                futures[future] = idx

            for future in as_completed(futures):
                result = future.result()
                _record_result(result)

                completed += 1
                total_correct = sum(1 for r in results_detail if r.get('correct'))
                total = sum(1 for r in results_detail if 'correct' in r)
                if total > 0:
                    tprint(f"  Progress: {completed}/{n} done | {total_correct}/{total} correct ({100*total_correct/total:.1f}%)")

    # Final flush
    _flush_incremental(results_detail, hypotheses, type2acc, start_time, config)

    # Sort results by original index for consistent output
    results_detail.sort(key=lambda r: r.get('idx', 0))
    hypotheses.sort(key=lambda h: h.get('question_id', ''))

    # ── Report ────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    total_correct = sum(1 for r in results_detail if r.get('correct'))
    total = sum(1 for r in results_detail if 'correct' in r)

    print(f"\n{'='*70}")
    print(f"LONGMEMEVAL RESULTS — Dendric Memory Engine")
    print(f"{'='*70}")

    task_accs = []
    print(f"\nAccuracy by question type:")
    for qtype, accs in sorted(type2acc.items()):
        mean_acc = sum(accs) / len(accs) if accs else 0
        task_accs.append(mean_acc)
        print(f"  {qtype}: {sum(accs)}/{len(accs)} ({100*mean_acc:.1f}%)")

    overall = total_correct / total if total else 0
    task_avg = sum(task_accs) / len(task_accs) if task_accs else 0
    print(f"\nOverall Accuracy: {100*overall:.2f}% ({total_correct}/{total})")
    print(f"Task-averaged Accuracy: {100*task_avg:.2f}%")
    print(f"Elapsed: {elapsed:.0f}s ({elapsed/max(1,total):.1f}s/question)")
    if workers > 1:
        print(f"Effective throughput: {elapsed/max(1,total)/workers:.1f}s/question/worker")
    print(f"{'='*70}")

    # Save hypothesis file in official format
    ts = int(time.time())
    hyp_path = f"longmemeval/longmemeval_hypothesis_{ts}.jsonl"
    with open(hyp_path, 'w') as f:
        for h in hypotheses:
            f.write(json.dumps(h) + '\n')

    # Save detailed results
    results_path = f"longmemeval/longmemeval_results_{ts}.json"
    results = {
        'overall_accuracy': overall,
        'task_averaged_accuracy': task_avg,
        'total': total,
        'correct': total_correct,
        'by_type': {k: {'n': len(v), 'correct': sum(v), 'accuracy': sum(v)/len(v) if v else 0}
                    for k, v in type2acc.items()},
        'per_question': results_detail,
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
    }
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nHypothesis file: {hyp_path}")
    print(f"Results file:    {results_path}")
    print(f"\nTo re-evaluate with official script:")
    print(f"  python /tmp/LongMemEval/src/evaluation/evaluate_qa.py gpt-4o {hyp_path} data/longmemeval/longmemeval_s_cleaned.json")
    print(f"  python /tmp/LongMemEval/src/evaluation/print_qa_metrics.py {hyp_path}.eval-results-gpt-4o data/longmemeval/longmemeval_s_cleaned.json")
    print()

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run LongMemEval benchmark on Dendric")
    parser.add_argument("--data", default=None, help="Path to longmemeval_s_cleaned.json")
    parser.add_argument("--questions", type=int, default=10, help="Number of questions (default: 10)")
    parser.add_argument("--top-k", type=int, default=10, help="Retrieval top-k (default: 10)")
    parser.add_argument("--cot", action="store_true", default=True, help="Chain-of-thought (default: True)")
    parser.add_argument("--no-cot", action="store_false", dest="cot")
    parser.add_argument("--cycles", type=int, default=1, help="Consolidation cycles (default: 1)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default: 1)")
    parser.add_argument("--answer-model", default="claude-haiku-4-5-20251001",
                        help="Model for answering (default: claude-haiku-4-5-20251001)")
    parser.add_argument("--types", nargs="+", default=None,
                        help="Question types to run (e.g., knowledge-update temporal-reasoning multi-session)")
    parser.add_argument("--db-prefix", default="lme",
                        help="DB name prefix for worker isolation (default: lme)")
    parser.add_argument("--debug-questions", default=None,
                        help="Comma-separated question_ids to dump retrieved memories for (writes longmemeval/debug/<id>.json)")
    parser.add_argument("--per-type", type=int, default=None,
                        help="Stratified sampling: take this many questions from each category (overrides --questions for selection)")
    parser.add_argument("--only-ids", default=None,
                        help="Comma-separated question_ids to run (filters dataset to exactly these)")
    args = parser.parse_args()
    debug_set = set(s.strip() for s in args.debug_questions.split(",")) if args.debug_questions else None
    only_ids_set = set(s.strip() for s in args.only_ids.split(",")) if args.only_ids else None

    try:
        run_longmemeval(
            data_path=args.data, max_questions=args.questions,
            top_k=args.top_k, use_cot=args.cot, consolidation_cycles=args.cycles,
            workers=args.workers, answer_model=args.answer_model,
            question_types=args.types, db_prefix=args.db_prefix,
            debug_questions=debug_set, per_type=args.per_type,
            only_ids=only_ids_set,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
