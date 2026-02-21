#!/usr/bin/env bash
# Compare two scoring configs against the same snapshot.
# Usage: ./scripts/bench/compare_configs.sh <benchmark> <baseline_config> <test_config>
set -uo pipefail

BENCH="${1:-locomo}"
BASELINE="${2:-baseline}"
TEST_CONFIG="${3:?Usage: compare_configs.sh <benchmark> <baseline> <test_config>}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

BLUE='\033[0;34m'; GREEN='\033[0;32m'; NC='\033[0m'

echo -e "${BLUE}=== Comparing: ${BASELINE} vs ${TEST_CONFIG} (${BENCH}) ===${NC}"

# Run baseline
echo -e "${BLUE}--- Running BASELINE: ${BASELINE} ---${NC}"
"${REPO_ROOT}/scripts/bench/restore_and_eval.sh" "$BENCH" "$BASELINE"
BASELINE_RESULT=$(ls -t "${REPO_ROOT}/benchmarks/results/${BENCH}_${BASELINE}_"*.json 2>/dev/null | head -1)

# Run test config
echo -e "${BLUE}--- Running TEST: ${TEST_CONFIG} ---${NC}"
"${REPO_ROOT}/scripts/bench/restore_and_eval.sh" "$BENCH" "$TEST_CONFIG"
TEST_RESULT=$(ls -t "${REPO_ROOT}/benchmarks/results/${BENCH}_${TEST_CONFIG}_"*.json 2>/dev/null | head -1)

# Compare
if [[ -n "$BASELINE_RESULT" ]] && [[ -n "$TEST_RESULT" ]]; then
    echo ""
    echo -e "${GREEN}=== Comparison ===${NC}"
    python3 "${REPO_ROOT}/scripts/bench/compare_results.py" \
        --baseline "$BASELINE_RESULT" \
        --test "$TEST_RESULT" \
        --output "${REPO_ROOT}/benchmarks/results/compare_${BASELINE}_vs_${TEST_CONFIG}_${TIMESTAMP}.json"
else
    echo "ERROR: Could not find result files for comparison"
    exit 1
fi
