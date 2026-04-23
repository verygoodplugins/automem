#!/usr/bin/env bash
# Restore a benchmark snapshot and run evaluation (no re-ingestion).
# Usage: ./scripts/bench/restore_and_eval.sh [benchmark_name] [config_name]
set -euo pipefail

BENCH_NAME="${1:-locomo}"
CONFIG="${2:-baseline}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python3"
fi

# Shared utilities (colors + wait_for_api)
source "$(dirname "$0")/../lib/common.sh"

if [[ ! "$BENCH_NAME" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo -e "${RED}Invalid benchmark name: ${BENCH_NAME}${NC}"
    exit 1
fi

SNAPSHOT_DIR="${REPO_ROOT}/benchmarks/snapshots/${BENCH_NAME}"
COMPOSE_PROJECT="automem"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# Verify snapshot exists
if [[ ! -f "${SNAPSHOT_DIR}/falkordb.tar.gz" ]] || [[ ! -f "${SNAPSHOT_DIR}/qdrant.tar.gz" ]]; then
    echo -e "${RED}Snapshot not found: ${SNAPSHOT_DIR}${NC}"
    echo "Run first: make bench-ingest BENCH=${BENCH_NAME}"
    exit 1
fi

echo -e "${BLUE}=== Restore & Eval: ${BENCH_NAME} (config: ${CONFIG}) ===${NC}"

# 1. Stop all containers
echo -e "${BLUE}Stopping containers...${NC}"
cd "$REPO_ROOT" || { echo -e "${RED}Failed to cd to ${REPO_ROOT}${NC}"; exit 1; }
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
if [[ ! "$CONFIG" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo -e "${RED}Invalid config name: ${CONFIG}${NC}"
    exit 1
fi
if [[ "${CONFIG}" != "baseline" ]]; then
    CONFIG_FILE="${REPO_ROOT}/scripts/lab/configs/${CONFIG}.json"
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo -e "${RED}Config file not found: ${CONFIG_FILE}${NC}"
        exit 1
    fi
    echo -e "${BLUE}Applying config: ${CONFIG}${NC}"
    # Convert JSON config to .env.bench format
    python3 -c "
import json, sys
with open('${CONFIG_FILE}') as f:
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
OUTPUT="${RESULTS_DIR}/${BENCH_NAME}_${CONFIG}_${TIMESTAMP}.json"

if [[ "$BENCH_NAME" == locomo* ]]; then
    EVAL_ARGS="--eval-only --no-cleanup --output ${OUTPUT}"
    if [[ "$BENCH_NAME" == "locomo-mini" ]]; then
        EVAL_ARGS="--conversations 0,1 ${EVAL_ARGS}"
    fi
    "$PYTHON_BIN" tests/benchmarks/test_locomo.py \
        --base-url "$AUTOMEM_TEST_BASE_URL" \
        --api-token "$AUTOMEM_TEST_API_TOKEN" \
        ${EVAL_ARGS}
elif [[ "$BENCH_NAME" == longmemeval* ]]; then
    LONGMEM_ARGS=(--config "$CONFIG" --no-cleanup --output "${OUTPUT}")
    if [[ "$BENCH_NAME" == "longmemeval-mini" ]]; then
        LONGMEM_ARGS+=(--max-questions 20)
    fi
    "$PYTHON_BIN" tests/benchmarks/longmemeval/test_longmemeval.py \
        --base-url "$AUTOMEM_TEST_BASE_URL" \
        --api-token "$AUTOMEM_TEST_API_TOKEN" \
        "${LONGMEM_ARGS[@]}"
else
    echo -e "${RED}Unknown benchmark: ${BENCH_NAME}${NC}"
    echo "Supported: locomo, locomo-mini, longmemeval, longmemeval-mini"
    exit 1
fi

echo ""
echo -e "${GREEN}=== Evaluation complete ===${NC}"
echo -e "${GREEN}  Results: ${OUTPUT}${NC}"
