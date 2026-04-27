"""Diagnostic: does the déjà-vu archive trigger ever fire on real probes?

Background. The parameter sweep found that `archive_trigger_threshold`
produces *byte-identical* recall@k results for every value 0.0–0.9 on
both meridian_deep and LongMemEval. That implies the threshold gate
never engages, which would mean the déjà-vu mechanism is non-functional
on these probe sets.

Three possible explanations:
  (a) Spreading activation never produces an entity activation ≥ 0.7
      on any real probe — gate is unreachable.
  (b) Activation crosses the gate, but no archive memories are linked
      to the activated entity — gate fires but pulls nothing.
  (c) Activation crosses, archive memories are pulled, but they're
      already in the active-path candidate set, so the gate is
      structurally redundant rather than dormant.

This script runs each probe through `spreading_activation_recall`
manually and reports:
  - Which seed entities the query resolved to
  - The post-spread activation distribution
  - Per-threshold count of entities that would have crossed
  - Whether each crossing entity has any archive-region memories
  - Whether those archive memories appeared in the final top-10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.engine.config import EngineConfig  # noqa: E402
from src.engine.core.engine import MemoryEngine, _extract_query_concept_terms  # noqa: E402
from src.engine.core.entity_extraction import extract_entities  # noqa: E402
from src.engine.storage.entity_graph import EntityGraphStore  # noqa: E402
from src.engine.retrieval.associative import spreading_activation_recall  # noqa: E402


THRESHOLDS_TO_REPORT = [0.0, 0.3, 0.5, 0.7, 0.9]


def _resolve_seeds(query: str, eg: EntityGraphStore, persona: str) -> tuple[list[str], list[str], list[str]]:
    """Replicate the engine.recall() seed-resolution logic so we know which
    entities the spreading-activation path actually started from."""
    query_entity_tuples = extract_entities(query)
    query_entity_names = [c for _, _, c in query_entity_tuples]
    if not query_entity_names:
        query_entity_names = _extract_query_concept_terms(query)

    seeds_resolved = []
    seeds_unresolved = []
    for name in query_entity_names:
        canonical = (name or "").strip().lower()
        eid = eg.get_entity_by_name(canonical)
        if eid is not None:
            seeds_resolved.append(canonical)
        else:
            seeds_unresolved.append(canonical)

    persona_will_seed = (
        persona and not seeds_resolved
    )
    return seeds_resolved, seeds_unresolved, [persona] if persona_will_seed else []


def _compute_activations_only(
    seeds: list[str], eg: EntityGraphStore, persona: str = "",
    persona_seed_activation: float = 0.5,
    persona_fallback: bool = True,
    max_hops: int = 2, decay: float = 0.5, activation_threshold: float = 0.05,
) -> dict[str, float]:
    """Re-run the BFS spread step from associative.py, return final
    {entity_id_str: activation} dict. Mirrors the inner loop exactly.
    No memory fetch — we only want activation values."""
    persona_lc = (persona or "").strip().lower()

    non_persona_seeds = []
    persona_in_query = False
    for name in seeds:
        canonical = (name or "").strip().lower()
        if canonical == persona_lc:
            persona_in_query = True
            continue
        eid = eg.get_entity_by_name(canonical)
        if eid is not None:
            non_persona_seeds.append((eid, 1.0))

    seed_ids_with_act = list(non_persona_seeds)

    needs_persona = (
        persona_lc and not non_persona_seeds
        and (persona_in_query or persona_fallback)
    )
    if needs_persona:
        persona_eid = eg.get_entity_by_name(persona_lc)
        if persona_eid is not None:
            seed_ids_with_act.append((persona_eid, persona_seed_activation))

    if not seed_ids_with_act:
        return {}

    activation = {eid: act for eid, act in seed_ids_with_act}
    frontier = [eid for eid, _ in seed_ids_with_act]
    for hop in range(1, max_hops + 1):
        next_frontier = []
        hop_factor = decay ** hop
        for eid in frontier:
            current_act = activation.get(eid, 0.0)
            if current_act < activation_threshold:
                continue
            for edge in eg.get_edges_for_entity(eid):
                neighbor = edge["entity_b"] if edge["entity_a"] == eid else edge["entity_a"]
                edge_weight = float(edge.get("weight", 1.0))
                spread = current_act * edge_weight * hop_factor
                if spread > activation.get(neighbor, 0.0):
                    activation[neighbor] = spread
                    next_frontier.append(neighbor)
        frontier = next_frontier
        if not frontier:
            break
    return activation


def _archive_memory_count_for_entity(eg: EntityGraphStore, store, eid) -> int:
    """How many archive-region memories are linked to this entity?
    eid may be a UUID, a UUID string, or whatever the activation dict
    stores — we coerce to whatever get_memories_for_entity wants."""
    from uuid import UUID
    if isinstance(eid, str):
        try:
            eid = UUID(eid)
        except ValueError:
            return 0
    mids = list(eg.get_memories_for_entity(eid))
    if not mids:
        return 0
    with store.conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM memories WHERE id = ANY(%s::uuid[]) AND region = 'archive'",
            ([str(m) for m in mids],),
        )
        return cur.fetchone()[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decay-values", default="0.5",
        help="Comma-separated decay values to test (default: 0.5, the production default). "
             "Pass e.g. 0.3,0.5,0.7,0.9 to compare. Larger decay means activation falls "
             "off less per hop, so hop-1 neighbors can reach the threshold.",
    )
    parser.add_argument(
        "--max-hops", type=int, default=2,
        help="Spreading-activation hop limit (default: 2, matches production).",
    )
    args = parser.parse_args()
    decay_values = [float(x.strip()) for x in args.decay_values.split(",") if x.strip()]

    db_url = os.environ.get("DATABASE_URL", "postgresql://localhost:5432/meridian_deep")
    annotations_path = "/Users/rileycoleman/meridian/probes/meridian_recall_gold.json"
    persona = os.environ.get("PERSONA", "riley")

    print(f"DB:          {db_url}")
    print(f"Annotations: {annotations_path}")
    print(f"Persona:     {persona}")
    print(f"Decay sweep: {decay_values}  (max_hops={args.max_hops})")
    print()

    with open(annotations_path) as f:
        annotations = json.load(f)

    cfg = EngineConfig(db_url=db_url, persona=persona, recall_mutates_state=False)
    engine = MemoryEngine(config=cfg)
    eg = EntityGraphStore(engine.store.conn)

    n_probes = len(annotations)

    for decay in decay_values:
        print(f"\n{'='*120}")
        print(f"  decay = {decay}  (hop-1 cap = {decay}, hop-2 cap = {decay**2:.3f})")
        print(f"{'='*120}")
        print(f"{'#':<3} {'query':<55} {'seeds':<30} {'max_act':>8} "
              + " ".join(f">={t}".rjust(6) for t in THRESHOLDS_TO_REPORT)
              + f" {'arch@.7':>8}")
        print("-" * 130)

        # Per-decay aggregates
        probe_crossing_counts = {t: 0 for t in THRESHOLDS_TO_REPORT}
        probe_archive_counts = {t: 0 for t in THRESHOLDS_TO_REPORT}
        # Track hop-1 reach: how many probes have at least one *non-seed* (hop-1+)
        # entity reaching the 0.7 threshold? That's the test for whether the
        # threshold is graduated rather than binary.
        probes_with_nonseed_at_07 = 0

        for i, ann in enumerate(annotations):
            query = ann["query"]
            seeds_resolved, _, persona_seeds = _resolve_seeds(query, eg, persona)
            all_seeds = seeds_resolved + persona_seeds

            if not all_seeds:
                print(f"{i+1:<3} {query[:53]:<55} (no seeds)")
                continue

            acts = _compute_activations_only(
                all_seeds, eg, persona=persona,
                max_hops=args.max_hops, decay=decay,
            )
            if not acts:
                continue

            # Identify seed entity IDs so we can ask "does any *non-seed* cross 0.7?"
            seed_ids = set()
            for name in all_seeds:
                eid = eg.get_entity_by_name((name or "").strip().lower())
                if eid is not None:
                    seed_ids.add(eid)

            max_act = max(acts.values())
            crossings = [sum(1 for v in acts.values() if v >= t) for t in THRESHOLDS_TO_REPORT]
            archive_hits_07 = sum(
                _archive_memory_count_for_entity(eg, engine.store, eid)
                for eid, act in acts.items() if act >= 0.7
            )

            nonseed_at_07 = any(
                act >= 0.7 and eid not in seed_ids
                for eid, act in acts.items()
            )
            if nonseed_at_07:
                probes_with_nonseed_at_07 += 1

            for t, n in zip(THRESHOLDS_TO_REPORT, crossings):
                if n > 0:
                    probe_crossing_counts[t] += 1
                    arch_total = sum(
                        _archive_memory_count_for_entity(eg, engine.store, eid)
                        for eid, act in acts.items() if act >= t
                    )
                    if arch_total > 0:
                        probe_archive_counts[t] += 1

            seed_str = ",".join(seeds_resolved[:2]) or ("[fb:" + (persona_seeds[0] if persona_seeds else "") + "]")
            crossings_str = " ".join(f"{n:>6d}" for n in crossings)
            print(f"{i+1:<3} {query[:53]:<55} {seed_str[:28]:<30} {max_act:>8.3f} {crossings_str} {archive_hits_07:>8d}")

        print()
        print(f"  Probes with non-seed entity ≥0.7: {probes_with_nonseed_at_07}/{n_probes} "
              f"  (= threshold is graduated if >0, binary if 0)")
        print(f"  {'thresh':<8} {'probes_crossing':>15} {'with_archive':>14}")
        for t in THRESHOLDS_TO_REPORT:
            print(f"  {t:<8.1f} {probe_crossing_counts[t]:>15d} {probe_archive_counts[t]:>14d}")


if __name__ == "__main__":
    main()
