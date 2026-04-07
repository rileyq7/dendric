# Dendric — Memory Consolidation System

> Academic paper retrieval using ACT-R spreading activation and entity graphs.

## Quick Start

```bash
# Setup
python -m pip install -r requirements.txt
python src/scripts/setup_db.py

# Run validation
python src/scripts/validate_phase3_lifecycle.py --limit 266 --queries 50
```

## Project Structure

```
dendric/
├── src/
│   ├── engine/         # Core memory system
│   │   ├── core/       # Activation, signals, entity extraction, ingest
│   │   ├── embeddings/ # Batch embedding
│   │   ├── retrieval/  # Spreading activation, citation validation
│   │   ├── storage/    # Entity graph, memory store
│   │   └── models/     # Schemas
│   ├── scripts/        # Validation & benchmark runners
│   └── web/            # Web UI
└── docs/               # Architecture, deployment, performance
```

## Key Components

### Entity Extraction
Signal-based (no LLM), academic stopword filter, known-entity recognition for models/datasets/venues, metadata extraction from S2ORC.

### Batch Embedding
Native SentenceTransformer batching, 768-dim `nomic-embed-text` model.

### Spreading Activation
ACT-R fan effect with parameter tuning, entity co-occurrence edges, citation ranking validation.

### Activation Equation
7-stage Dendric activation model grounded in ACT-R, GANE, dual-receptor pharmacology, and hybrid GABA inhibition.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — System architecture
- [docs/ENTITY_EXTRACTION.md](docs/ENTITY_EXTRACTION.md) — Entity extraction guide
- [docs/PERFORMANCE.md](docs/PERFORMANCE.md) — Performance analysis
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) — Production deployment
- [docs/QUICKSTART.md](docs/QUICKSTART.md) — 5-minute quick start
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) — Full directory layout

## License

MIT
