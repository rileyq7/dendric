# Dendric Project Structure

## Directory Layout

```
dendric/
├── src/                               # All core code
│   ├── engine/                        # Core system (DO NOT MODIFY WITHOUT PLAN)
│   │   ├── __init__.py
│   │   ├── core/                      # Memory core
│   │   │   ├── __init__.py
│   │   │   ├── memory.py
│   │   │   ├── entity_extraction.py   # ✓ OPTIMIZED (entity fixes)
│   │   │   ├── signals_enhanced.py
│   │   │   ├── activation.py
│   │   │   ├── ingest_with_entities.py    # ✓ OPTIMIZED (batch ingest)
│   │   │   └── erosion.py
│   │   │
│   │   ├── embeddings/                # Embedding layer
│   │   │   ├── __init__.py
│   │   │   └── embed.py               # ✓ OPTIMIZED (batch embedding)
│   │   │
│   │   ├── storage/                   # Data persistence
│   │   │   ├── __init__.py
│   │   │   ├── entity_graph.py
│   │   │   └── memory_store.py
│   │   │
│   │   ├── retrieval/                 # Query & retrieval
│   │   │   ├── __init__.py
│   │   │   ├── spreading_activation.py
│   │   │   └── citation_validation.py
│   │   │
│   │   └── models/                    # Data models
│   │       ├── __init__.py
│   │       └── schemas.py
│   │
│   ├── scripts/                       # Utility scripts
│   │   ├── validate_phase3_lifecycle.py   # Main validation runner
│   │   ├── s2orc_loader.py                # S2ORC data loader
│   │   ├── generate_s2orc_extended.py     # Generate test data
│   │   └── setup_db.py                    # Database setup
│   │
│   └── web/                           # Web UI (optional)
│       └── [web components]
│
├── tests/                             # Test suite
│   ├── __init__.py
│   ├── test_entity_extraction.py      # ✓ 4/4 PASS
│   ├── test_entity_improvements.py    # ✓ PASS
│   ├── test_activation.py
│   └── integration/
│       ├── __init__.py
│       └── test_phase3_lifecycle.py   # ✓ PASS (266-paper, 3.7k-paper)
│
├── docs/                              # Documentation
│   ├── QUICKSTART.md                  # Getting started (5 min)
│   ├── ARCHITECTURE.md                # System architecture
│   ├── ENTITY_EXTRACTION.md           # Entity extraction guide
│   ├── PERFORMANCE.md                 # Performance analysis
│   ├── DEPLOYMENT.md                  # Production deployment
│   └── guides/
│       ├── entity-extraction.md
│       ├── embedding.md
│       └── deployment.md
│
├── data/                              # Data files
│   ├── s2orc_papers.jsonl             # Sample papers (~10 papers)
│   ├── s2orc_extended.jsonl           # Extended test set (~3.7k papers)
│   └── arxiv_papers_sample.jsonl      # ArXiv samples (if available)
│
├── results/                           # Experiment results & analysis
│   ├── README.md                      # Results guide
│   ├── phase3_results.json            # 3,669-paper validation
│   ├── 5k_validation.log              # Ingest logs
│   ├── 5K_RESULTS.md                  # Scale test analysis
│   ├── CASE_STUDY_RESULTS.md          # Research methodology
│   ├── 250K_LIFECYCLE_RESULTS.md      # Deployment plan
│   └── analysis/
│       ├── entity_extraction_analysis.json
│       └── performance_metrics.json
│
├── docs_archive/                      # Reference documentation
│   ├── SESSION_SUMMARY.md
│   ├── ENTITY_EXTRACTION_SUMMARY.md
│   ├── CHANGES.md
│   └── [other historical docs]
│
├── legacy/                            # Old code (do not use)
│   └── [previous implementations]
│
├── README.md                          # Main project overview
├── GETTING_STARTED.md                 # Getting started guide
├── PROJECT_STRUCTURE.md               # This file
├── CLAUDE.md                          # Claude Code instructions (if needed)
├── .env                               # Environment variables (git-ignored)
├── .gitignore
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package setup
└── VERSION                            # Version file
```

## File Organization Rules

### ✅ DO
- Keep `src/engine/` stable and well-tested
- Document all changes to core code
- Put all scripts in `src/scripts/`
- Save experiment results in `results/`
- Keep `docs/` current with architecture changes
- Put Python packages/modules in `src/`

### ❌ DON'T
- Add random .md files to project root (use docs/ or results/)
- Create test files outside tests/
- Put utility scripts at root level (use src/scripts/)
- Add source code outside src/ (except tests/ and docs/)
- Keep results/analysis in docs_archive/ (move to results/)

## Production Code

**Stable and tested:**
- `src/engine/core/entity_extraction.py` — Entity extraction + 4 fixes
- `src/engine/core/ingest_with_entities.py` — Batch ingest + entity graph
- `src/engine/embeddings/embed.py` — Batch embedding (70+ papers/sec)
- `src/engine/retrieval/spreading_activation.py` — ACT-R spreading activation

**All other code** in src/engine/ is production-ready but may be modified with EnterPlanMode.

## Test Suite

- `tests/test_entity_extraction_fixes.py` — 4 unit tests (all PASS)
- `tests/test_entity_improvements.py` — Analysis tests
- `tests/test_activation.py` — Activation function tests
- Run: `pytest tests/ -v`

## Validation Scripts

- `src/scripts/validate_phase3_lifecycle.py` — Main validation runner
  - Supports `--limit` (number of papers) and `--queries` (number of citation tests)
  - Current validated: 3,669 papers, Recall@10 54.3%

## Key Metrics

| Component | Status | Metric |
|-----------|--------|--------|
| Entity Extraction | ✓ Optimized | 140% NDCG improvement |
| Batch Embedding | ✓ Optimized | 70+ papers/sec, 9x faster |
| Spreading Activation | ✓ Validated | 54.3% Recall@10 at scale |
| Tests | ✓ Passing | 4/4 unit tests |
| Validation | ✓ Complete | 3,669 papers tested |
