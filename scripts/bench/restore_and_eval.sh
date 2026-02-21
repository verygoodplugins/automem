#!/usr/bin/env bash
# Restore a benchmark snapshot and run evaluation (no re-ingestion).
# Usage: ./scripts/bench/restore_and_eval.sh [benchmark_name] [config_name]
set -uo pipefail

BENCH_NAME="${1:-locomo}"
CONFIG="${2:-baseline}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SNAPSHOT_DIR="${REPO_ROOT}/benchmarks/snapshots/${BENCH_NAME}"
COMPOSE_PROJECT="automem"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

RED='\033[0;31m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; NC='\033[0m'

wait_for_api() {
    local url="$1" max="${2:-60}" attempt=0
    echo -e "${BLUE}Waiting for API at ${url}...${NC}"
    while [ $attempt -lt $max ]; do
        if curl -fsS "${url}/health" > /dev/null 2>&1; then
            echo -e "${GREEN}API ready after ${attempt}s${NC}"
            return 0
        fi
        sleep 1; attempt=$((attempt + 1))
        [ $((attempt % 10)) -eq 0 ] && echo -e "${YELLOW}  Still waiting... (${attempt}s)${NC}"
    done
    echo -e "${RED}ERROR: API not ready after ${max}s${NC}"
    return 1
}

# Verify snapshot exists
if [[ ! -f "${SNAPSHOT_DIR}/falkordb.tar.gz" ]] || [[ ! -f "${SNAPSHOT_DIR}/qdrant.tar.gz" ]]; then
    echo -e "${RED}Snapshot not found: ${SNAPSHOT_DIR}${NC}"
    echo "Run first: make bench-ingest BENCH=${BENCH_NAME}"
    exit 1
fi

echo -e "${BLUE}=== Restore & Eval: ${BENCH_NAME} (config: ${CONFIG}) ===${NC}"

# 1. Stop all containers
echo -e "${BLUE}Stopping containers...${NC}"
cd "$REPO_ROOT"
docker compose down 2>/dev/null || true

# 2. Recreate volumes from snapshot
echo -e "${BLUE}Restoring volumes from snapshot...${NC}"
docker volume rm "${COMPOSE_PROJECT}_falkordb_data" "${COMPOSE_PROJECT}_qdrant_data" 2>/dev/null || true
docker volume create "${COMPOSE_PROJECT}_falkordb_data"
docker volume create "${COMPOSE_PROJECT}_qdrant_data"

docker run --rm \
    -v "${COMPOSE_PROJECT}_falkordb_data:/data" \
    -v "${SNAPSHOT_DIR}:/backup:ro" \
    alpine sh -c "cd /data && tar xzf /backup/falkordb.tar.gz"

docker run --rm \
    -v "${COMPOSE_PROJECT}_qdrant_data:/qdrant_storage" \
    -v "${SNAPSHOT_DIR}:/backup:ro" \
    alpine sh -c "cd /qdrant_storage && tar xzf /backup/qdrant.tar.gz"

# 3. Start services with optional config overrides
if [[ "${CONFIG}" != "baseline" ]] && [[ -f "${REPO_ROOT}/scripts/lab/configs/${CONFIG}.json" ]]; then
    echo -e "${BLUE}Applying config: ${CONFIG}${NC}"
    # Convert JSON config to .env.bench format
    python3 -c "
import json, sys
with open('${REPO_ROOT}/scripts/lab/configs/${CONFIG}.json') as f:
    config = json.load(f)
for k, v in config.items():
    print(f'{k}={v}')
" > "${REPO_ROOT}/.env.bench"
    docker compose --env-file .env --env-file .env.bench up -d
else
    docker compose up -d
fi

wait_for_api "http://localhost:8001" 60 || exit 1

# 4. Run evaluation only
echo -e "${BLUE}Running evaluation...${NC}"
export AUTOMEM_TEST_BASE_URL="http://localhost:8001"
export AUTOMEM_TEST_API_TOKEN="test-token"

RESULTS_DIR="${REPO_ROOT}/benchmarks/results"
mkdir -p "${RESULTS_DIR}"
OUTPUT="${RESULTS_DIR}/${BENCH_NAME}_${CONFIG}_${TIMESTAMP}"

if [[ "$BENCH_NAME" == locomo* ]]; then
    EVAL_ARGS="--eval-only --no-cleanup --output ${OUTPUT}"
    if [[ "$BENCH_NAME" == "locomo-mini" ]]; then
        EVAL_ARGS="--conversations 0,1 ${EVAL_ARGS}"
    fi
    python3 tests/benchmarks/test_locomo.py \
        --base-url "$AUTOMEM_TEST_BASE_URL" \
        --api-token "$AUTOMEM_TEST_API_TOKEN" \
        ${EVAL_ARGS}
elif [[ "$BENCH_NAME" == longmemeval* ]]; then
    python3 tests/benchmarks/longmemeval/test_longmemeval.py \
        --base-url "$AUTOMEM_TEST_BASE_URL" \
        --api-token "$AUTOMEM_TEST_API_TOKEN" \
        --no-cleanup --output "${OUTPUT}"
fi

echo ""
echo -e "${GREEN}=== Evaluation complete ===${NC}"
echo -e "${GREEN}  Results: ${OUTPUT}.json${NC}"
