#!/bin/bash
#
# Run LoCoMo benchmark with LENIENT evaluation (CORE-compatible)
# Uses GPT-5.1 for best reasoning quality
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
RESULTS_DIR="$SCRIPT_DIR/results"

mkdir -p "$RESULTS_DIR"

cd "$PROJECT_ROOT"
source venv/bin/activate

echo ""
echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}  LoCoMo Benchmark - LENIENT Mode (GPT-5.1) ${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""
echo "This uses CORE-compatible methodology:"
echo "  - GPT-5.1 for answer generation"
echo "  - GPT-5.1 for semantic evaluation"
echo "  - Lenient grading: 'touches on same topic' = CORRECT"
echo ""

# Check if AutoMem is running
if ! curl -s http://localhost:8001/health > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  AutoMem not running. Starting with 'make dev'...${NC}"
    make dev &
    sleep 10
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Run with lenient evaluation
echo -e "${GREEN}▶️  Running E2E + Lenient benchmark...${NC}"
python tests/benchmarks/test_locomo.py \
    --eval-mode e2e \
    --lenient \
    --e2e-model gpt-5.1 \
    --eval-judge-model gpt-5.1 \
    2>&1 | tee "$RESULTS_DIR/lenient_benchmark_${TIMESTAMP}.log"

echo ""
echo -e "${GREEN}✅ Benchmark complete!${NC}"
echo "Results saved to: $RESULTS_DIR/lenient_benchmark_${TIMESTAMP}.log"


