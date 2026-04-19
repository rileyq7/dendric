"""
Path 4: Entity Graph Walk Retrieval.

Walks the entity graph starting from query entities to find connected memories.
This is the only retrieval path that reliably surfaces memories connected by
shared entities regardless of semantic similarity or temperature.

Algorithm:
1. Extract entities from query
2. Find matching entity nodes in graph
3. Walk 1-2 hops along weighted edges
4. Collect and score memories attached to discovered entities
"""

import logging
from typing import List, Dict, Any, Optional

import psycopg2.extras

from ..storage.entity_graph import EntityGraphStore

logger = logging.getLogger(__name__)


def graph_recall(
    query_entities: List[str],
    store,
    top_k: int = 20,
    min_edge_weight: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Walk the entity graph from query entities and return connected memories.

    Args:
        query_entities: List of canonical entity names from the query
        store: PostgresStore with active connection
        top_k: Max results to return
        min_edge_weight: Minimum edge weight to follow

    Returns:
        List of memory dicts with 'graph_score' field added
    """
    if not query_entities:
        return []

    conn = store.conn

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        # Step 1: Find matching entity IDs (exact + fuzzy)
        entity_names_lower = [n.lower() for n in query_entities]

        cur.execute("""
            SELECT id, canonical_name FROM entities
            WHERE LOWER(canonical_name) = ANY(%s)
        """, (entity_names_lower,))
        seed_rows = cur.fetchall()

        if not seed_rows:
            # Fuzzy fallback: partial match
            patterns = ['%' + n + '%' for n in entity_names_lower]
            cur.execute("""
                SELECT id, canonical_name FROM entities
                WHERE LOWER(canonical_name) LIKE ANY(%s)
                LIMIT 20
            """, (patterns,))
            seed_rows = cur.fetchall()

        if not seed_rows:
            return []

        seed_ids = [str(r['id']) for r in seed_rows]
        edge_weights = {sid: 1.0 for sid in seed_ids}  # direct match = 1.0

        # Step 2: Walk 1 hop — direct entity connections
        cur.execute("""
            SELECT
                CASE WHEN ee.entity_a::text = ANY(%s) THEN ee.entity_b::text
                     ELSE ee.entity_a::text END as connected_id,
                ee.weight,
                ee.predicate
            FROM entity_edges ee
            WHERE (ee.entity_a::text = ANY(%s) OR ee.entity_b::text = ANY(%s))
              AND ee.weight >= %s
            ORDER BY ee.weight DESC
            LIMIT 50
        """, (seed_ids, seed_ids, seed_ids, min_edge_weight))
        hop1 = cur.fetchall()

        reached_ids = set(seed_ids)
        for hop in hop1:
            eid = hop['connected_id']
            reached_ids.add(eid)
            edge_weights[eid] = max(edge_weights.get(eid, 0), float(hop['weight']))

        # Step 3: Optional hop 2 — only for strong edges (weight > 0.3)
        hop1_strong = [h['connected_id'] for h in hop1 if float(h['weight']) > 0.3]
        if hop1_strong:
            cur.execute("""
                SELECT
                    CASE WHEN ee.entity_a::text = ANY(%s) THEN ee.entity_b::text
                         ELSE ee.entity_a::text END as connected_id,
                    ee.weight
                FROM entity_edges ee
                WHERE (ee.entity_a::text = ANY(%s) OR ee.entity_b::text = ANY(%s))
                  AND ee.weight >= %s
                  AND ee.entity_a::text != ALL(%s) AND ee.entity_b::text != ALL(%s)
                ORDER BY ee.weight DESC
                LIMIT 30
            """, (hop1_strong, hop1_strong, hop1_strong,
                  min_edge_weight * 2, seed_ids, seed_ids))
            hop2 = cur.fetchall()

            for hop in hop2:
                eid = hop['connected_id']
                reached_ids.add(eid)
                # Hop-2 weight is discounted
                edge_weights[eid] = max(
                    edge_weights.get(eid, 0),
                    float(hop['weight']) * 0.5
                )

        # Step 4: Find memories attached to reached entities
        reached_list = list(reached_ids)
        # Archive excluded — only the spreading-activation déjà-vu trigger
        # is allowed to surface archived memories.
        cur.execute("""
            SELECT
                m.*,
                ARRAY_AGG(DISTINCT me.entity_id::text) as matched_entity_ids
            FROM memories m
            JOIN memory_entities me ON me.memory_id = m.id
            WHERE me.entity_id::text = ANY(%s)
              AND m.region != 'archive'
            GROUP BY m.id
            ORDER BY m.temperature DESC
            LIMIT %s
        """, (reached_list, top_k * 2))
        mem_rows = cur.fetchall()

    # Step 5: Score each memory by sum of path weights to its entities
    from ..storage.postgres import _row_to_mem
    results = []
    for row in mem_rows:
        matched_eids = row.get('matched_entity_ids', [])
        graph_score = sum(edge_weights.get(eid, 0) for eid in matched_eids)

        mem = _row_to_mem(row)
        d = mem.to_dict()
        d['graph_score'] = round(graph_score, 4)
        d['graph_hops'] = 1 if any(eid in seed_ids for eid in matched_eids) else 2
        results.append(d)

    results.sort(key=lambda x: x['graph_score'], reverse=True)
    return results[:top_k]

