#!/bin/bash
# Setup script for LongMemEval benchmark

set -e

echo "======================================================================"
echo "LongMemEval Benchmark Setup — Dendric Memory Engine"
echo "======================================================================"

# Check Python version
echo ""
echo "Checking Python version..."
python3 --version

# Install dependencies
echo ""
echo "Installing required packages..."
pip install -q datasets urllib3

# Check for optional dependencies
echo ""
echo "Checking optional dependencies..."
python3 -c "import anthropic; print('✓ anthropic installed')" 2>/dev/null || echo "⚠ anthropic not installed (required for answer generation)"
python3 -c "import openai; print('✓ openai installed')" 2>/dev/null || echo "⚠ openai not installed (will fall back to Claude for evaluation)"

# Create data directory
echo ""
echo "Creating data directory..."
mkdir -p data/longmemeval

# Check PostgreSQL
echo ""
echo "Checking PostgreSQL..."
if psql --version > /dev/null 2>&1; then
    echo "✓ PostgreSQL installed"

    # Try to check if mnestic database exists
    if psql -l 2>/dev/null | grep -q mnestic; then
        echo "✓ mnestic database exists"
    else
        echo "⚠ mnestic database not found — will be created automatically"
    fi
else
    echo "⚠ PostgreSQL not found — make sure it's running before starting benchmark"
fi

echo ""
echo "======================================================================"
echo "Setup complete!"
echo "======================================================================"
echo ""
echo "Quick test (10 questions, ~5 minutes):"
echo "  python src/scripts/run_longmemeval.py --questions 10"
echo ""
echo "Medium test (50 questions, ~25 minutes):"
echo "  python src/scripts/run_longmemeval.py --questions 50"
echo ""
echo "Full benchmark (500 questions, ~4 hours):"
echo "  python src/scripts/run_longmemeval.py --questions 500"
echo ""
echo "With custom data file:"
echo "  python src/scripts/run_longmemeval.py --data path/to/longmemeval_s_cleaned.json --questions 10"
echo ""
