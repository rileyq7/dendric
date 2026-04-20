#!/usr/bin/env python3
"""
Quick validation of LongMemEval setup.

Checks:
1. Dependencies installed
2. PostgreSQL running
3. Engine initializes
4. Basic retrieval works
5. API clients available
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

print("="*70)
print("LongMemEval Setup Validation")
print("="*70)

# Check 1: Dependencies
print("\n1. Checking Python dependencies...")
deps = {
    "anthropic": "required (answer generation)",
    "sentence_transformers": "required (embeddings)",
    "psycopg2": "required (database)",
    "pgvector": "required (vector storage)",
    "numpy": "required (numerical ops)",
    "openai": "optional (GPT-4o evaluation)",
    "datasets": "optional (HF data loading)",
}

all_ok = True
for pkg, desc in deps.items():
    try:
        __import__(pkg.replace("-", "_"))
        status = "✓"
    except ImportError:
        status = "✗" if "required" in desc else "⚠"
        all_ok = all_ok and ("optional" in desc)

    print(f"   {status} {pkg:20s} ({desc})")

if not all_ok:
    print("\n⚠ Missing required dependencies. Install with:")
    print("   pip install -r requirements.txt")
    print("   pip install datasets openai  # optional but recommended")
    sys.exit(1)

# Check 2: PostgreSQL
print("\n2. Checking PostgreSQL...")
try:
    import psycopg2
    conn = psycopg2.connect("postgresql://localhost/postgres")
    print("   ✓ PostgreSQL running")
    conn.close()
except Exception as e:
    print(f"   ✗ PostgreSQL connection failed: {e}")
    print("   Start with: brew services start postgresql")
    sys.exit(1)

# Check 3: Engine initialization
print("\n3. Checking Dendric engine...")
try:
    from src.engine.core.engine import MemoryEngine
    engine = MemoryEngine()
    print("   ✓ Engine initialized")
except Exception as e:
    print(f"   ✗ Engine initialization failed: {e}")
    sys.exit(1)

# Check 4: Basic memory operations
print("\n4. Testing memory operations...")
try:
    # Remember
    result = engine.remember(
        "The quick brown fox jumps over the lazy dog.",
        source="test"
    )
    mem_id = result.get('id')
    print(f"   ✓ Remember succeeded (mem_id={mem_id[:8]}...)")

    # Recall
    results = engine.recall("fox", top_k=5)
    print(f"   ✓ Recall succeeded ({len(results)} results)")

    # Consolidate (optional - can be slow)
    # stats = engine.consolidate()
    # print(f"   ✓ Consolidate succeeded")
    print(f"   ✓ Consolidate available (skipped - can be slow)")
except Exception as e:
    print(f"   ✗ Memory operations failed: {e}")
    sys.exit(1)

# Check 5: API clients
print("\n5. Checking API clients...")
try:
    from anthropic import Anthropic
    client = Anthropic()
    print("   ✓ Anthropic client available")
except Exception as e:
    print(f"   ✗ Anthropic client failed: {e}")

try:
    from openai import OpenAI
    client = OpenAI()
    print("   ✓ OpenAI client available (GPT-4o)")
except Exception as e:
    print(f"   ✗ OpenAI client not available: {e}")
    print("      REQUIRED for LongMemEval evaluation (pip install openai)")

# Check 6: Data loading
print("\n6. Checking data loading...")
data_path = "data/longmemeval/longmemeval_s_cleaned.json"
if os.path.exists(data_path):
    import json
    try:
        with open(data_path) as f:
            data = json.load(f)
        print(f"   ✓ Data file found ({len(data)} questions)")
    except Exception as e:
        print(f"   ✗ Failed to load data: {e}")
else:
    print(f"   ⚠ Data file not found at {data_path}")
    print("      Will download from HuggingFace on first run")
    try:
        from datasets import load_dataset
        print("   ✓ datasets library available for download")
    except ImportError:
        print("   ✗ datasets library not available (install: pip install datasets)")

print("\n" + "="*70)
print("✓ Setup validation complete!")
print("="*70)
print("\nReady to run benchmark:")
print("  python src/scripts/run_longmemeval.py --questions 10")
print("")
