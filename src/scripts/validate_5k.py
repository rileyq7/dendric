#!/usr/bin/env python3
"""
Quick 5k-paper validation to confirm entity extraction improvements scale.
"""

import subprocess
import sys

print("="*70)
print("Phase 3 Validation: 5,000 Papers")
print("="*70)
print("\nRunning with improved entity extraction...")
print("Expected: NDCG ≥ 0.45, Recall@10 ≥ 50%")
print("\nThis will take ~10 minutes...\n")

# Run validation
result = subprocess.run(
    [sys.executable, "validate_phase3_lifecycle.py", "--limit", "5000", "--queries", "200"],
    capture_output=False
)

sys.exit(result.returncode)
