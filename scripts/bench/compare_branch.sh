#!/usr/bin/env bash
# Compare a branch against main using a pre-loaded snapshot.
# Usage: ./scripts/bench/compare_branch.sh <branch> [config] [benchmark]
#
# Example:
#   ./scripts/bench/compare_branch.sh feat/enhanced-recall baseline locomo
#   make bench-compare-branch BRANCH=feat/enhanced-recall
set -euo pipefail

BRANCH="${1:?Usage: compare_branch.sh <branch> [config] [benchmark]}"
CONFIG="${2:-baseline}"
BENCH="${3:-locomo}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${REPO_ROOT}/benchmarks/results"

# Shared utilities (colors + wait_for_api)
source "$(dirname "$0")/../lib/common.sh"

CURRENT_BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
STASH_CREATED=false

sanitize_filename_component() {
    printf '%s' "$1" | sed 's/[^A-Za-z0-9._-]/_/g'
}

CURRENT_BRANCH_SAFE="$(sanitize_filename_component "$CURRENT_BRANCH")"
BRANCH_SAFE="$(sanitize_filename_component "$BRANCH")"

# Cleanup trap: restore original branch on exit
cleanup() {
    local current
    current="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
    if [[ "$current" != "$CURRENT_BRANCH" ]]; then
        echo -e "${BLUE}Restoring branch ${CURRENT_BRANCH}...${NC}"
        git -C "$REPO_ROOT" checkout "$CURRENT_BRANCH" 2>/dev/null || true
    fi
    if [[ "$STASH_CREATED" == true ]]; then
        git -C "$REPO_ROOT" stash pop 2>/dev/null || true
    fi
}
trap cleanup EXIT

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
cd "$REPO_ROOT" || { echo -e "${RED}Failed to cd to ${REPO_ROOT}${NC}"; exit 1; }
if ! git diff --quiet || ! git diff --cached --quiet || \
   [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
    git stash push --include-untracked -m "bench-compare-${TIMESTAMP}" >/dev/null
    STASH_CREATED=true
fi
git checkout "$BRANCH"

echo -e "${BLUE}Rebuilding flask-api for ${BRANCH}...${NC}"
docker compose build flask-api

"${REPO_ROOT}/scripts/bench/restore_and_eval.sh" "$BENCH" "$CONFIG"
BRANCH_RESULT=$(ls -t "${RESULTS_DIR}/${BENCH}_${CONFIG}_"*.json 2>/dev/null | head -1)

if [[ -z "$BRANCH_RESULT" ]]; then
    echo -e "${RED}No result file found for branch ${BRANCH}${NC}"
    git checkout "$CURRENT_BRANCH"
    if [[ "$STASH_CREATED" == true ]]; then
        git stash pop 2>/dev/null || true
    fi
    exit 1
fi
cp "$BRANCH_RESULT" "${RESULTS_DIR}/${BENCH}_${CONFIG}_${BRANCH_SAFE}_${TIMESTAMP}.json"
BRANCH_RESULT="${RESULTS_DIR}/${BENCH}_${CONFIG}_${BRANCH_SAFE}_${TIMESTAMP}.json"

# 3. Restore original branch (trap handles cleanup on failure too)
echo -e "${BLUE}Restoring ${CURRENT_BRANCH}...${NC}"
git checkout "$CURRENT_BRANCH"
if [[ "$STASH_CREATED" == true ]]; then
    git stash pop 2>/dev/null || true
fi
trap - EXIT  # Disable cleanup trap since we restored manually

# Rebuild original branch API
docker compose build flask-api

# 4. Compare
echo ""
echo -e "${GREEN}=== Comparison Results ===${NC}"
python3 "${REPO_ROOT}/scripts/bench/compare_results.py" \
    --baseline "$CURRENT_RESULT" \
    --test "$BRANCH_RESULT" \
    --output "${RESULTS_DIR}/compare_${CURRENT_BRANCH_SAFE}_vs_${BRANCH_SAFE}_${TIMESTAMP}.json"
