"""
Database operations for entity graph: entities, memory_entities, entity_edges.
"""

import logging
import psycopg2.extras
from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime
from uuid import UUID

logger = logging.getLogger(__name__)


class EntityGraphStore:
    """Handles all entity graph operations."""

    def __init__(self, conn, autocommit: bool = True):
        self.conn = conn
        self.autocommit = autocommit

    def _commit(self):
        if self.autocommit:
            self.conn.commit()

    # ── Entity CRUD ──────────────────────────────────────────────────

    def upsert_entity(self, canonical_name: str, entity_type: str, original_name: str) -> UUID:
        """
        Create or get entity. Returns entity ID.
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                INSERT INTO entities (name, entity_type, canonical_name)
                VALUES (%s, %s, %s)
                ON CONFLICT (canonical_name, entity_type)
                DO UPDATE SET mention_count = entities.mention_count + 1, name = EXCLUDED.name
                RETURNING id
            """, (original_name, entity_type, canonical_name))
            row = cur.fetchone()
        self._commit()
        return row['id']

    def get_entity(self, canonical_name: str, entity_type: str) -> Optional[UUID]:
        """Get entity ID by canonical name and type."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id FROM entities
                WHERE canonical_name = %s AND entity_type = %s
            """, (canonical_name, entity_type))
            row = cur.fetchone()
        return row['id'] if row else None

    def get_entity_names(self) -> List[str]:
        """Get all canonical entity names (for known entity matching at ingest)."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT DISTINCT canonical_name FROM entities")
            rows = cur.fetchall()
        return [row['canonical_name'] for row in rows]

    def update_entity_session(self, entity_id: UUID, session_id: str):
        """Track which sessions mention this entity. Appends session_id if not already present."""
        if not session_id or session_id == "unknown":
            return
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE entities
                SET session_ids = CASE
                        WHEN %s = ANY(session_ids) THEN session_ids
                        ELSE array_append(session_ids, %s)
                    END,
                    last_seen = NOW()
                WHERE id = %s
            """, (session_id, session_id, entity_id))
        self._commit()

    def get_high_fan_entities(self, min_sessions: int = 3) -> List[Dict]:
        """Get entities mentioned across many sessions (candidates for aggregation)."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT e.id, e.canonical_name, e.entity_type, e.mention_count,
                       e.session_ids, COALESCE(array_length(e.session_ids, 1), 0) as session_count
                FROM entities e
                WHERE COALESCE(array_length(e.session_ids, 1), 0) >= %s
                ORDER BY COALESCE(array_length(e.session_ids, 1), 0) DESC
            """, (min_sessions,))
            rows = cur.fetchall()
        return [dict(row) for row in rows]

    def get_memories_for_entity(self, entity_id: UUID) -> List[UUID]:
        """Reverse lookup: get all memory IDs linked to an entity."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT me.memory_id FROM memory_entities me
                WHERE me.entity_id = %s
            """, (entity_id,))
            rows = cur.fetchall()
        return [row['memory_id'] for row in rows]

    def get_all_entities(self) -> Dict[str, int]:
        """
        Get all entities and their mention counts.
        Returns {canonical_name -> mention_count}.
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT canonical_name, mention_count FROM entities")
            rows = cur.fetchall()
        return {row['canonical_name']: row['mention_count'] for row in rows}

    # ── Memory-Entity Links ──────────────────────────────────────────

    def insert_memory_entity(self, memory_id: UUID, entity_id: UUID, salience: float, position: Optional[int] = None):
        """Link a memory to an entity."""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO memory_entities (memory_id, entity_id, salience, position)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (memory_id, entity_id)
                DO UPDATE SET salience = EXCLUDED.salience
            """, (memory_id, entity_id, salience, position))
        self._commit()

    def get_entity_ids_for_memory(self, memory_id: UUID) -> List[UUID]:
        """Get all entity IDs for a memory."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT entity_id FROM memory_entities
                WHERE memory_id = %s
            """, (memory_id,))
            rows = cur.fetchall()
        return [row['entity_id'] for row in rows]

    def delete_memory_entities(self, memory_id: UUID):
        """Remove all entity links for a memory (used before re-extraction)."""
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM memory_entities WHERE memory_id = %s", (memory_id,))
        self._commit()

    def get_entity_canonical_names_for_memory(self, memory_id: UUID) -> Set[str]:
        """Get canonical names of all entities in a memory."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT DISTINCT e.canonical_name
                FROM memory_entities me
                JOIN entities e ON me.entity_id = e.id
                WHERE me.memory_id = %s
            """, (memory_id,))
            rows = cur.fetchall()
        return set(row['canonical_name'] for row in rows)

    def get_entity_fan_counts(self) -> Dict[str, int]:
        """
        Get fan count for all entities.
        Returns {canonical_name -> count of memories}.
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT e.canonical_name, COUNT(DISTINCT me.memory_id) as fan
                FROM entities e
                LEFT JOIN memory_entities me ON e.id = me.entity_id
                GROUP BY e.id, e.canonical_name
            """)
            rows = cur.fetchall()
        return {row['canonical_name']: row['fan'] for row in rows}

    # ── Entity Edges (Co-occurrence Graph) ────────────────────────────

    def upsert_entity_edge(self, entity_a: UUID, entity_b: UUID):
        """Create or strengthen an edge between two entities."""
        # Skip self-edges (same entity extracted under different surface forms)
        if entity_a == entity_b:
            return

        # Ensure ordering: entity_a < entity_b
        if entity_a > entity_b:
            entity_a, entity_b = entity_b, entity_a

        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO entity_edges (entity_a, entity_b, weight, co_occurrence_count)
                VALUES (%s, %s, 1.0, 1)
                ON CONFLICT (entity_a, entity_b)
                DO UPDATE SET
                    co_occurrence_count = entity_edges.co_occurrence_count + 1,
                    weight = LEAST(10.0, LOG(entity_edges.co_occurrence_count + 2)),
                    last_reinforced = NOW()
            """, (entity_a, entity_b))
        self._commit()

    def get_edges_for_entity(self, entity_id: UUID) -> List[Dict]:
        """Get all edges involving this entity."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM entity_edges
                WHERE entity_a = %s OR entity_b = %s
            """, (entity_id, entity_id))
            rows = cur.fetchall()
        return rows

    def decay_entity_edges(self, decay_rate: float = 0.02):
        """
        Weaken edges that haven't been reinforced recently.
        Prune edges that decay below threshold.
        """
        with self.conn.cursor() as cur:
            # Update weights
            cur.execute("""
                UPDATE entity_edges
                SET weight = weight * POWER(1.0 - %s, EXTRACT(DAY FROM (NOW() - last_reinforced)))
                WHERE last_reinforced < NOW() - INTERVAL '1 day'
            """, (decay_rate,))

            # Delete dead edges
            cur.execute("DELETE FROM entity_edges WHERE weight < 0.01")

        self._commit()

    def upsert_compression_edge(
        self, source_name: str, source_type: str,
        target_name: str, target_type: str,
        predicate: str, weight: float,
    ):
        """
        Insert or update an entity edge from compression output.
        Prefers verb-mediated predicates over co_occurs, and higher weights.
        """
        source_id = self.upsert_entity(source_name.lower(), source_type, source_name)
        target_id = self.upsert_entity(target_name.lower(), target_type, target_name)

        # Skip self-edges (same entity)
        if source_id == target_id:
            return

        # Ensure ordering for PK constraint (entity_a < entity_b)
        if source_id > target_id:
            source_id, target_id = target_id, source_id

        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO entity_edges (entity_a, entity_b, weight, predicate, co_occurrence_count)
                VALUES (%s, %s, %s, %s, 1)
                ON CONFLICT (entity_a, entity_b)
                DO UPDATE SET
                    predicate = CASE
                        WHEN entity_edges.predicate = 'co_occurs' AND EXCLUDED.predicate != 'co_occurs'
                            THEN EXCLUDED.predicate
                        WHEN EXCLUDED.weight > entity_edges.weight
                            THEN EXCLUDED.predicate
                        ELSE entity_edges.predicate
                    END,
                    weight = GREATEST(entity_edges.weight, EXCLUDED.weight),
                    co_occurrence_count = entity_edges.co_occurrence_count + 1,
                    last_reinforced = NOW()
            """, (source_id, target_id, weight, predicate))
        self._commit()

    def reinforce_edge(self, entity_a: UUID, entity_b: UUID, boost: float = 1.05):
        """Strengthen an edge (e.g., after co-retrieval)."""
        if entity_a > entity_b:
            entity_a, entity_b = entity_b, entity_a

        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE entity_edges
                SET weight = weight * %s, last_reinforced = NOW()
                WHERE (entity_a = %s AND entity_b = %s) OR (entity_a = %s AND entity_b = %s)
            """, (boost, entity_a, entity_b, entity_b, entity_a))
        self._commit()

    # ── Spreading Activation ──────────────────────────────────────────

    def compute_spreading_activation(self, memory_id: UUID) -> float:
        """
        Compute spreading activation for a memory based on entity graph connectivity.
        Returns 0.0-1.0 (tanh-normalized).

        SA(m) = sum over entities in m of: (entity_edge_weight_sum / fan(entity))
        """
        import math

        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    COALESCE(SUM(edge_strength / GREATEST(fan_count, 1)), 0) as raw_activation
                FROM (
                    SELECT
                        me.entity_id,
                        COALESCE((
                            SELECT SUM(ee.weight) FROM entity_edges ee
                            WHERE ee.entity_a = me.entity_id OR ee.entity_b = me.entity_id
                        ), 0) as edge_strength,
                        COALESCE((
                            SELECT COUNT(DISTINCT me2.memory_id) FROM memory_entities me2
                            WHERE me2.entity_id = me.entity_id
                        ), 1) as fan_count
                    FROM memory_entities me
                    WHERE me.memory_id = %s
                ) entity_activations
            """, (memory_id,))
            row = cur.fetchone()

        raw = float(row['raw_activation']) if row else 0.0
        # Normalize with tanh: raw=2.0 → ~0.76, raw=1.0 → ~0.46
        return math.tanh(raw * 0.5)

    # ── Consolidation ────────────────────────────────────────────────

    def consolidate_entity_graph(self, last_consolidation_time=None):
        """
        Update entity graph edge weights based on access patterns.
        1. Strengthen edges between entities co-accessed in recent queries
        2. Weaken all edges via fan-effect decay
        3. Prune dead edges
        """
        import math

        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Step 1: Strengthen co-accessed edges
            # Find entity pairs in memories accessed since last consolidation
            if last_consolidation_time:
                cur.execute("""
                    SELECT e1.entity_id as entity_a_id, e2.entity_id as entity_b_id,
                           COUNT(*) as co_access_count
                    FROM memory_entities e1
                    JOIN memory_entities e2
                        ON e1.memory_id = e2.memory_id AND e1.entity_id < e2.entity_id
                    JOIN memories m ON m.id = e1.memory_id
                    WHERE m.last_accessed > %s
                    GROUP BY e1.entity_id, e2.entity_id
                """, (last_consolidation_time,))
                co_accessed = cur.fetchall()

                for row in co_accessed:
                    boost = math.log1p(row['co_access_count']) * 0.1
                    entity_a = row['entity_a_id']
                    entity_b = row['entity_b_id']
                    cur.execute("""
                        UPDATE entity_edges
                        SET weight = LEAST(weight + %s, 1.0),
                            last_reinforced = NOW()
                        WHERE (entity_a = %s AND entity_b = %s)
                    """, (boost, entity_a, entity_b))

            # Step 2: Fan-effect decay
            # Edges involving high-fan entities decay faster
            cur.execute("""
                WITH fan_counts AS (
                    SELECT entity_id, COUNT(DISTINCT memory_id) as fan
                    FROM memory_entities
                    GROUP BY entity_id
                )
                UPDATE entity_edges ee
                SET weight = ee.weight * (1.0 / (1.0 + 0.05 * GREATEST(
                    COALESCE(fa.fan, 1), COALESCE(fb.fan, 1)
                )))
                FROM fan_counts fa, fan_counts fb
                WHERE ee.entity_a = fa.entity_id
                  AND ee.entity_b = fb.entity_id
            """)

            # Step 3: Prune dead edges
            cur.execute("DELETE FROM entity_edges WHERE weight < 0.01")
            pruned = cur.rowcount

        self._commit()
        return pruned

    # ── Analytics ────────────────────────────────────────────────────

    def entity_graph_stats(self) -> Dict:
        """Get statistics about the entity graph."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT COUNT(*) as count FROM entities")
            entity_count = cur.fetchone()['count']

            cur.execute("SELECT COUNT(*) as count FROM memory_entities")
            me_count = cur.fetchone()['count']

            cur.execute("SELECT COUNT(*) as count FROM entity_edges")
            edge_count = cur.fetchone()['count']

            cur.execute("""
                SELECT AVG(mention_count) as avg_mentions
                FROM entities
            """)
            avg_mentions = cur.fetchone()['avg_mentions'] or 0

        return {
            'entities': entity_count,
            'memory_entity_links': me_count,
            'edges': edge_count,
            'avg_entity_mentions': float(avg_mentions),
        }

    def clear_entity_graph(self):
        """Delete all entity graph data (for testing/reset)."""
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM entity_edges")
            cur.execute("DELETE FROM memory_entities")
            cur.execute("DELETE FROM entities")
        self._commit()
