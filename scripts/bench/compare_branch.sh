#!/usr/bin/env bash
# Compare a branch against main using a pre-loaded snapshot.
# Usage: ./scripts/bench/compare_branch.sh <branch> [config] [benchmark]
#
# Example:
#   ./scripts/bench/compare_branch.sh feat/enhanced-recall baseline locomo
#   make bench-compare-branch BRANCH=feat/enhanced-recall
set -uo pipefail

BRANCH="${1:?Usage: compare_branch.sh <branch> [config] [benchmark]}"
CONFIG="${2:-baseline}"
BENCH="${3:-locomo}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${REPO_ROOT}/benchmarks/results"

BLUE='\033[0;34m'; GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'

CURRENT_BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"

echo -e "${BLUE}=== Branch Comparison ===${NC}"
echo "  Current: ${CURRENT_BRANCH}"
echo "  Compare: ${BRANCH}"
echo "  Config:  ${CONFIG}"
echo "  Bench:   ${BENCH}"
echo ""

# 1. Evaluate current branch
echo -e "${BLUE}--- Evaluating current branch: ${CURRENT_BRANCH} ---${NC}"
"${REPO_ROOT}/scripts/bench/restore_and_eval.sh" "$BENCH" "$CONFIG"
CURRENT_RESULT=$(ls -t "${RESULTS_DIR}/${BENCH}_${CONFIG}_"*.json 2>/dev/null | head -1)

if [[ -z "$CURRENT_RESULT" ]]; then
    echo -e "${RED}No result file found for current branch${NC}"
    exit 1
fi
# Copy to stable name
cp "$CURRENT_RESULT" "${RESULTS_DIR}/${BENCH}_${CONFIG}_current_${TIMESTAMP}.json"
CURRENT_RESULT="${RESULTS_DIR}/${BENCH}_${CONFIG}_current_${TIMESTAMP}.json"

# 2. Stash, switch, rebuild, evaluate
echo -e "${BLUE}--- Switching to branch: ${BRANCH} ---${NC}"
cd "$REPO_ROOT"
git stash --include-untracked 2>/dev/null || true
git checkout "$BRANCH"

echo -e "${BLUE}Rebuilding flask-api for ${BRANCH}...${NC}"
docker compose build flask-api

"${REPO_ROOT}/scripts/bench/restore_and_eval.sh" "$BENCH" "$CONFIG"
BRANCH_RESULT=$(ls -t "${RESULTS_DIR}/${BENCH}_${CONFIG}_"*.json 2>/dev/null | head -1)

if [[ -z "$BRANCH_RESULT" ]]; then
    echo -e "${RED}No result file found for branch ${BRANCH}${NC}"
    git checkout "$CURRENT_BRANCH"
    git stash pop 2>/dev/null || true
    exit 1
fi
cp "$BRANCH_RESULT" "${RESULTS_DIR}/${BENCH}_${CONFIG}_${BRANCH//\//_}_${TIMESTAMP}.json"
BRANCH_RESULT="${RESULTS_DIR}/${BENCH}_${CONFIG}_${BRANCH//\//_}_${TIMESTAMP}.json"

# 3. Restore original branch
echo -e "${BLUE}Restoring ${CURRENT_BRANCH}...${NC}"
git checkout "$CURRENT_BRANCH"
git stash pop 2>/dev/null || true

# Rebuild original branch API
docker compose build flask-api

# 4. Compare
echo ""
echo -e "${GREEN}=== Comparison Results ===${NC}"
python3 "${REPO_ROOT}/scripts/bench/compare_results.py" \
    --baseline "$CURRENT_RESULT" \
    --test "$BRANCH_RESULT" \
    --output "${RESULTS_DIR}/compare_${CURRENT_BRANCH}_vs_${BRANCH//\//_}_${TIMESTAMP}.json"
