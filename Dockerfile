# Dendric runtime image — lean enough to run the eval harness.
# We don't embed secrets; OPENAI_API_KEY is passed at runtime via env_file.
# Use the local_* stage chain so the final image is small, reproducible,
# and doesn't include build-time intermediates.

FROM python:3.12-slim AS base

# System deps psycopg2-binary can sometimes still want at link time, plus
# git for editable installs, plus build-essential for any wheel-less deps
# we pick up transitively.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        postgresql-client \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so code changes don't invalidate the dep layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir openai python-dotenv fastapi "uvicorn[standard]"

# Copy source last — most-frequently-changed layer
COPY src/ ./src/
COPY data/ ./data/

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DENDRIC_PATH=/app

# Default command: print the reproduction quickstart
CMD ["python", "-c", "print('Dendric container ready.\\n\\nQuickstart:\\n  docker compose run dendric python -m src.scripts.seed_synthetic_corpus\\n  docker compose run dendric python -m src.scripts.recall_at_k --corpus meridian_deep --db postgresql://postgres:postgres@db:5432/synthetic --annotations data/synthetic_gold.json --k 5,10,25\\n\\nSee REPRODUCE.md for the full recipe.')"]
