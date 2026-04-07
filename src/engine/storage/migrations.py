"""
Schema setup and migrations for Postgres + pgvector.
"""

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
    -- Identity
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Content at each compression level
    raw_content     TEXT,
    structured_summary TEXT,
    knowledge_nugget TEXT,
    entity_edges    TEXT,

    -- Lifecycle state
    temperature     FLOAT NOT NULL DEFAULT 1.0,
    region          TEXT NOT NULL DEFAULT 'hippocampus',

    -- Signals (simplified to 3)
    da_relevance    FLOAT NOT NULL DEFAULT 0.3,
    ne_novelty      FLOAT NOT NULL DEFAULT 0.5,
    usage_score     FLOAT NOT NULL DEFAULT 0.0,

    -- MESU uncertainty tracking
    da_history      FLOAT[] DEFAULT '{}',
    ne_history      FLOAT[] DEFAULT '{}',
    signal_variance FLOAT DEFAULT 0.5,

    -- EWC retrieval importance
    retrieval_hits  INTEGER DEFAULT 0,
    retrieval_importance FLOAT DEFAULT 0.0,

    -- Retrieval indexes
    embedding       vector(1536),
    content_tsv     tsvector,

    -- Temporal
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_accessed   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_consolidated TIMESTAMPTZ,
    access_times    TIMESTAMPTZ[] DEFAULT '{}',
    access_count    INTEGER DEFAULT 0,

    -- Context metadata
    source          TEXT DEFAULT 'direct',
    context         TEXT DEFAULT '',
    location        TEXT DEFAULT '',
    time_of_day     TEXT DEFAULT '',
    day_of_week     TEXT DEFAULT '',
    co_entities     TEXT[] DEFAULT '{}',
    preceding_memory_id UUID,

    -- Compression metadata
    compression_level TEXT DEFAULT 'raw',
    tokens_original INTEGER,
    tokens_current  INTEGER,

    -- Token-level erosion weights
    token_weights   JSONB,

    metadata        JSONB DEFAULT '{}'
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_temp ON memories(temperature);
CREATE INDEX IF NOT EXISTS idx_region ON memories(region);
CREATE INDEX IF NOT EXISTS idx_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_source ON memories(source);
CREATE INDEX IF NOT EXISTS idx_entities ON memories USING GIN (co_entities);
CREATE INDEX IF NOT EXISTS idx_content_fts ON memories USING GIN (content_tsv);

-- Archive table (cold storage for raw content)
CREATE TABLE IF NOT EXISTS archive (
    memory_id       UUID PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    raw_content     TEXT NOT NULL,
    original_embedding vector(1536),
    archived_at     TIMESTAMPTZ DEFAULT now(),
    encoding_context JSONB DEFAULT '{}'
);

-- Retrieval log (for EWC importance tracking)
CREATE TABLE IF NOT EXISTS retrieval_log (
    id              SERIAL PRIMARY KEY,
    query_text      TEXT NOT NULL,
    query_embedding vector(1536),
    results         UUID[] NOT NULL,
    top_k_ids       UUID[] NOT NULL,
    timestamp       TIMESTAMPTZ DEFAULT now()
);

-- Procedural memory (extracted patterns)
CREATE TABLE IF NOT EXISTS procedures (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern         TEXT NOT NULL,
    source_memories UUID[] NOT NULL,
    confidence      FLOAT DEFAULT 0.5,
    created_at      TIMESTAMPTZ DEFAULT now(),
    last_reinforced TIMESTAMPTZ DEFAULT now(),
    reinforcement_count INTEGER DEFAULT 1
);

-- Entity graph tables
CREATE TABLE IF NOT EXISTS entities (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL,
    entity_type     TEXT NOT NULL,
    canonical_name  TEXT NOT NULL,
    first_seen      TIMESTAMPTZ DEFAULT now(),
    last_seen       TIMESTAMPTZ,
    mention_count   INTEGER DEFAULT 1,
    session_ids     TEXT[] DEFAULT '{}',
    UNIQUE(canonical_name, entity_type)
);

CREATE INDEX IF NOT EXISTS idx_entities_canonical ON entities(canonical_name);

CREATE TABLE IF NOT EXISTS memory_entities (
    memory_id       UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    entity_id       UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    salience        FLOAT DEFAULT 0.5,
    position        INTEGER,
    PRIMARY KEY (memory_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_me_memory ON memory_entities(memory_id);
CREATE INDEX IF NOT EXISTS idx_me_entity ON memory_entities(entity_id);

CREATE TABLE IF NOT EXISTS entity_edges (
    entity_a            UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    entity_b            UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    weight              FLOAT DEFAULT 1.0,
    co_occurrence_count INTEGER DEFAULT 1,
    last_reinforced     TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (entity_a, entity_b),
    CHECK (entity_a < entity_b)
);

CREATE INDEX IF NOT EXISTS idx_ee_a ON entity_edges(entity_a);
CREATE INDEX IF NOT EXISTS idx_ee_b ON entity_edges(entity_b);

-- tsvector auto-update trigger
CREATE OR REPLACE FUNCTION memories_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.content_tsv :=
        setweight(to_tsvector('english', coalesce(NEW.raw_content, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(NEW.structured_summary, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(NEW.knowledge_nugget, '')), 'C') ||
        setweight(to_tsvector('english', coalesce(NEW.entity_edges, '')), 'D');
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tsvector_update ON memories;
CREATE TRIGGER tsvector_update BEFORE INSERT OR UPDATE
    ON memories FOR EACH ROW EXECUTE FUNCTION memories_tsv_trigger();
"""

# ivfflat index requires data to exist first, so create it separately
IVFFLAT_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_embedding ON memories
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"""


def run_migrations(conn):
    """Run all schema migrations."""
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
        # Add token_weights column to existing tables
        cur.execute("""
            ALTER TABLE memories ADD COLUMN IF NOT EXISTS token_weights JSONB;
        """)
        # Add session tracking columns to entities
        cur.execute("""
            ALTER TABLE entities ADD COLUMN IF NOT EXISTS session_ids TEXT[] DEFAULT '{}';
        """)
        cur.execute("""
            ALTER TABLE entities ADD COLUMN IF NOT EXISTS last_seen TIMESTAMPTZ;
        """)
        # Add predicate column to entity_edges (for verb-mediated relationships)
        cur.execute("""
            ALTER TABLE entity_edges ADD COLUMN IF NOT EXISTS predicate TEXT DEFAULT 'co_occurs';
        """)
        # Compression output storage
        cur.execute("""
            ALTER TABLE memories ADD COLUMN IF NOT EXISTS knowledge_nuggets JSONB DEFAULT '[]'::jsonb;
        """)
        cur.execute("""
            ALTER TABLE memories ADD COLUMN IF NOT EXISTS last_compressed_at TIMESTAMPTZ;
        """)
        cur.execute("""
            ALTER TABLE memories ADD COLUMN IF NOT EXISTS compression_temperature REAL;
        """)
        cur.execute("""
            ALTER TABLE memories ADD COLUMN IF NOT EXISTS gemma_reextracted BOOLEAN DEFAULT FALSE;
        """)
        # Migrate vector columns to 1536-dim (from old 768-dim nomic model).
        # Old embeddings are incompatible, so truncate and drop+recreate columns.
        cur.execute("""
            DO $$ BEGIN
                IF EXISTS (
                    SELECT 1 FROM pg_attribute a
                    JOIN pg_class c ON a.attrelid = c.oid
                    WHERE c.relname = 'memories' AND a.attname = 'embedding'
                    AND a.atttypmod != -1 AND a.atttypmod != 1540
                ) THEN
                    TRUNCATE memories CASCADE;
                    TRUNCATE archive;
                    TRUNCATE retrieval_log;
                    DROP INDEX IF EXISTS idx_embedding;
                    ALTER TABLE memories DROP COLUMN embedding;
                    ALTER TABLE memories ADD COLUMN embedding vector(1536);
                    ALTER TABLE archive DROP COLUMN original_embedding;
                    ALTER TABLE archive ADD COLUMN original_embedding vector(1536);
                    ALTER TABLE retrieval_log DROP COLUMN query_embedding;
                    ALTER TABLE retrieval_log ADD COLUMN query_embedding vector(1536);
                END IF;
            END $$;
        """)
        # Performance indexes for graph retrieval
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_entity_edges_weight ON entity_edges(weight);
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed);
        """)
    conn.commit()


def create_vector_index(conn):
    """Create ivfflat index (call after inserting initial data)."""
    with conn.cursor() as cur:
        cur.execute(IVFFLAT_INDEX_SQL)
    conn.commit()
