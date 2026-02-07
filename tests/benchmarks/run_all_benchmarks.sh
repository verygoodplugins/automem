#!/bin/bash
# Run all LoCoMo benchmark configurations for comprehensive evaluation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "AutoMem LoCoMo Benchmark Suite"
echo "=============================================="
echo "Timestamp: $TIMESTAMP"
echo "Output: $OUTPUT_DIR"
echo ""

# Check if AutoMem is running
if ! curl -s http://localhost:8001/health > /dev/null; then
    echo "❌ AutoMem is not running. Start with: make dev"
    exit 1
fi
echo "✅ AutoMem is healthy"
echo ""

# Run Python benchmarks
echo "=============================================="
echo "Running Python Benchmarks"
echo "=============================================="

echo ""
echo "1️⃣  Default mode (retrieval + official F1 + evidence hints)"
python "$SCRIPT_DIR/test_locomo.py" \
    --output "$OUTPUT_DIR/results_default_$TIMESTAMP.json"

echo ""
echo "2️⃣  Strict mode (no evidence hints)"
python "$SCRIPT_DIR/test_locomo.py" \
    --no-evidence-hints \
    --output "$OUTPUT_DIR/results_strict_$TIMESTAMP.json"

echo ""
echo "3️⃣  E2E mode (LLM generates answers)"
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Skipping E2E mode (OPENAI_API_KEY not set)"
else
    python "$SCRIPT_DIR/test_locomo.py" \
        --eval-mode e2e \
        --output "$OUTPUT_DIR/results_e2e_$TIMESTAMP.json"
fi

# Run CORE-compatible benchmark if Node.js available
echo ""
echo "=============================================="
echo "Running CORE-Compatible Benchmark"
echo "=============================================="

if command -v node &> /dev/null && [ -n "$OPENAI_API_KEY" ]; then
    cd "$SCRIPT_DIR/core_adapter"
    if [ ! -d "node_modules" ]; then
        npm install
    fi
    node evaluate.js
    mv evaluation_results.json "$OUTPUT_DIR/results_core_compat_$TIMESTAMP.json"
    cd -
else
    echo "⚠️  Skipping CORE adapter (requires Node.js and OPENAI_API_KEY)"
fi

# Summary
echo ""
echo "=============================================="
echo "📊 SUMMARY"
echo "=============================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""

for f in "$OUTPUT_DIR"/results_*_$TIMESTAMP.json; do
    if [ -f "$f" ]; then
        name=$(basename "$f" | sed "s/_$TIMESTAMP.json//")
        accuracy=$(python -c "import json; d=json.load(open('$f')); print(f\"{d.get('overall', {}).get('accuracy', 0)*100:.2f}%\")" 2>/dev/null || echo "N/A")
        echo "  $name: $accuracy"
    fi
done

echo ""
echo "✅ Benchmark suite complete!"
