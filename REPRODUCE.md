# Reproducing Dendric's recall@k numbers

This recipe gets you from a clean clone to a recall@k result in under ten
minutes, without needing LongMemEval access or any personal data. If you
can't reproduce the synthetic-corpus numbers, something is broken in the
harness — please file an issue.

## What you need

- Docker + Docker Compose
- An OpenAI API key (used for embeddings during ingest)
- ~2 GB free disk for the image + pgvector volume

## Steps

```bash
export OPENAI_API_KEY=sk-...          # must be set; never baked into image

docker compose build                  # builds the Dendric runtime

docker compose run --rm dendric \
    python -m src.scripts.seed_synthetic_corpus

docker compose run --rm dendric \
    python -m src.scripts.recall_at_k \
        --corpus meridian_deep \
        --annotations src/scripts/synthetic_gold.json \
        --db postgresql://postgres:postgres@db:5432/dendric \
        --k 5,10,25
```

You should see `recall_any@10` around 1.0 on the synthetic corpus —
every gold memory carries a distinctive, rarely-collided marker, so
retrieval is expected to nail it. The purpose of the synthetic run is
*harness validation*, not benchmarking: if synthetic comes back at
chance-level, the wiring is broken.

## Reproducing the real benchmark numbers

The numbers in `docs/RIGOR_FINDINGS.md` use two corpora we can't ship:

- **LongMemEval** — gated academic dataset. Request access via the
  LongMemEval repo, drop the archive into `data/longmemeval/`, then run
  `src/scripts/setup_longmemeval.sh`.
- **meridian_deep** — the author's personal Meridian corpus plus a
  hand-annotated probe file. Not shareable.

Once you have LongMemEval:

```bash
docker compose run --rm dendric \
    python -m src.scripts.recall_at_k \
        --corpus longmemeval \
        --per-type 5 \
        --config full \
        --k 5,10,25
```

For the plain-RAG baseline, swap `--config full` for `--config plain_rag`.

## Troubleshooting

- **`could not connect to server: Connection refused`** — the `db`
  service hadn't finished starting. `docker compose run` waits on the
  healthcheck, so this usually means the healthcheck is failing. Check
  `docker compose logs db`.
- **`extension "vector" is not available`** — you're pointing at a
  non-pgvector Postgres. The compose stack uses `pgvector/pgvector:pg16`
  which has it preinstalled; if you pointed `DATABASE_URL` at an
  external DB, install pgvector there first.
- **OpenAI 401** — `OPENAI_API_KEY` not exported in the host shell
  before `docker compose run`. The compose file passes it through but
  won't invent one.

## What this does and doesn't prove

Reproducing the synthetic numbers proves the harness runs end to end
against a real Postgres+pgvector stack and that retrieval returns
something sensible on a tiny corpus. It does **not** prove anything
about Dendric's performance versus other systems — for that you need
LongMemEval.
