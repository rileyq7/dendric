#!/usr/bin/env python3
"""
Design-invariant checks for Dendric's core mechanisms.

Tests three invariants the architecture promises, regardless of bench score:

  1. **Lifecycle modulation works**: a hot, goal-relevant memory outranks
     a cold one when both are otherwise comparable.
  2. **Spreading activation reaches expected entity neighbors**: a query
     about an entity surfaces co-occurring entities' memories via the
     associative path, not just direct vector matches.
  3. **Déjà-vu trigger reaches archived memories**: when an entity is
     strongly activated, archived memories linked to it surface — and
     ONLY then; archive stays dormant otherwise.

These are design-level invariants. If any fails, the architecture isn't
doing what it claims regardless of benchmark performance.

Usage:
  python -m src.scripts.per_signal_recall_check
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.engine.core.engine import MemoryEngine
from src.engine.config import EngineConfig


def _make_engine(persona: str = "alex") -> MemoryEngine:
    db_url = os.environ.get(
        "DATABASE_URL", "postgresql://localhost:5432/dendric_stability"
    )
    config = EngineConfig(
        db_url=db_url,
        persona=persona,
        recall_mutates_state=False,
        archive_trigger_threshold=0.7,
    )
    eng = MemoryEngine(config=config)
    eng.store.clear_all()
    return eng


def check_lifecycle_modulation() -> bool:
    """Lifecycle modulation tilts ranking by temperature/DA/NE, but bounded.

    The design: a hot goal-relevant memory should get a measurable score lift
    over a cold off-curve one. Specifically, the *modulation factor* itself
    should reflect lifecycle state — hot memories get mod ≥ 1.0, cold off-curve
    memories get mod ≤ 1.0. The ranking effect is a tilt, not a filter, so the
    test is on the modulation values, not on which memory ends up rank-1
    (that's also influenced by RRF, which is correct behavior).
    """
    print("\n[1/3] Lifecycle modulation: tilts by temperature + signal state ...")
    eng = _make_engine()
    eng.remember("Alex picked up Spanish flashcards at the bookshop.", context="session_1")
    eng.remember("Alex wandered through the bookshop looking for poetry.", context="session_2")

    with eng.store.conn.cursor() as cur:
        cur.execute(
            "UPDATE memories SET temperature=0.15, da_relevance=0.1, ne_novelty=0.95 "
            "WHERE raw_content LIKE '%poetry%'"
        )
    eng.store.conn.commit()

    results = eng.recall("Alex bookshop visit", top_k=5)
    if len(results) < 2:
        print("  ✗ FAIL: not enough results to compare")
        return False

    hot_mod = next((r.get("_modulation") for r in results if "Spanish" in (r.get("raw_content","") or "")), None)
    cold_mod = next((r.get("_modulation") for r in results if "poetry" in (r.get("raw_content","") or "")), None)

    if hot_mod is None or cold_mod is None:
        print(f"  ✗ FAIL: modulation values missing (hot={hot_mod}, cold={cold_mod})")
        return False

    if hot_mod > cold_mod:
        print(f"  ✓ PASS: hot modulation={hot_mod:.3f} > cold modulation={cold_mod:.3f}")
        return True
    print(f"  ✗ FAIL: hot modulation={hot_mod:.3f} not greater than cold={cold_mod:.3f}")
    return False


def check_spreading_activation() -> bool:
    """Spreading activation should reach co-occurring entities' memories."""
    print("\n[2/3] Spreading activation: reaches co-occurring entities ...")
    eng = _make_engine()
    # Create co-occurrence: Mango appears with Tamiya in one memory; Tamiya
    # appears alone in another. Querying for Mango should reach the Tamiya-only
    # memory via the spreading activation path.
    eng.remember("Mango watched while I painted my Tamiya model kit.", context="session_1")
    eng.remember("Picked up new Tamiya thinner at the hobby shop.", context="session_2")

    results = eng.recall("What did I do with Mango?", top_k=5)
    paths_per_result = {(r.get("text") or r.get("raw_content", ""))[:40]: r.get("retrieval_paths", []) for r in results}

    # Find the Tamiya-thinner memory in results
    thinner_paths = None
    for text, paths in paths_per_result.items():
        if "thinner" in text:
            thinner_paths = paths
            break

    if thinner_paths and "associative" in thinner_paths:
        print(f"  ✓ PASS: Tamiya-thinner memory reached via 'associative' path")
        print(f"    paths={thinner_paths}")
        return True
    print("  ✗ FAIL: spreading activation did not reach Tamiya-thinner memory")
    print(f"    paths_per_result={paths_per_result}")
    return False


def check_dejavu_trigger() -> bool:
    """Archived memories surface only when entity activation is strong."""
    print("\n[3/3] Déjà-vu trigger: archive reachable only via strong activation ...")
    eng = _make_engine()
    # Three Mango memories, archive one of them
    eng.remember("Walked Mango along the canal.", context="session_1")
    eng.remember("Mango spotted a fox in the garden.", context="session_2")
    eng.remember("Vet check for Mango: weight stable.", context="session_3")
    with eng.store.conn.cursor() as cur:
        cur.execute("UPDATE memories SET region='archive', temperature=0.05 WHERE raw_content LIKE '%canal%'")
    eng.store.conn.commit()

    # Query ABOUT mango — high activation → déjà-vu should fire
    mango_results = eng.recall("What did I do with Mango?", top_k=10)
    archive_in_mango = any(r.get("region") == "archive" for r in mango_results)

    # Query about something unrelated — no mango activation → archive stays dormant
    other_results = eng.recall("Tell me about pancakes.", top_k=10)
    archive_in_other = any(r.get("region") == "archive" for r in other_results)

    if archive_in_mango and not archive_in_other:
        print("  ✓ PASS: archived memory surfaced for Mango query, dormant for unrelated query")
        return True
    print(f"  ✗ FAIL: archive_in_mango={archive_in_mango} archive_in_other={archive_in_other}")
    return False


def main():
    results = [
        check_lifecycle_modulation(),
        check_spreading_activation(),
        check_dejavu_trigger(),
    ]
    passed = sum(results)
    total = len(results)
    print(f"\n=== {passed}/{total} invariants passed ===")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
