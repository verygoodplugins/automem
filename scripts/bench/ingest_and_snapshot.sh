#!/usr/bin/env bash
# Ingest benchmark dataset into AutoMem and snapshot the Docker volumes.
# Usage: ./scripts/bench/ingest_and_snapshot.sh [locomo|longmemeval-mini]
set -uo pipefail

BENCH_NAME="${1:-locomo}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SNAPSHOT_DIR="${REPO_ROOT}/benchmarks/snapshots/${BENCH_NAME}"
COMPOSE_PROJECT="automem"

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

echo -e "${BLUE}=== Ingest & Snapshot: ${BENCH_NAME} ===${NC}"

# 1. Start services with benchmark-friendly env
echo -e "${BLUE}Starting services...${NC}"
cd "$REPO_ROOT"
MEMORY_CONTENT_HARD_LIMIT=50000 MEMORY_AUTO_SUMMARIZE=false docker compose up -d
wait_for_api "http://localhost:8001" 60 || exit 1

# 2. Run ingestion only
echo -e "${BLUE}Ingesting ${BENCH_NAME} data...${NC}"
export AUTOMEM_TEST_BASE_URL="http://localhost:8001"
export AUTOMEM_TEST_API_TOKEN="test-token"

if [[ "$BENCH_NAME" == "locomo" ]]; then
    python3 tests/benchmarks/test_locomo.py \
        --base-url "$AUTOMEM_TEST_BASE_URL" \
        --api-token "$AUTOMEM_TEST_API_TOKEN" \
        --ingest-only --no-cleanup
elif [[ "$BENCH_NAME" == locomo-mini ]]; then
    python3 tests/benchmarks/test_locomo.py \
        --base-url "$AUTOMEM_TEST_BASE_URL" \
        --api-token "$AUTOMEM_TEST_API_TOKEN" \
        --conversations 0,1 --ingest-only --no-cleanup
elif [[ "$BENCH_NAME" == longmemeval-mini ]]; then
    python3 tests/benchmarks/longmemeval/test_longmemeval.py \
        --base-url "$AUTOMEM_TEST_BASE_URL" \
        --api-token "$AUTOMEM_TEST_API_TOKEN" \
        --max-questions 20 --no-cleanup
else
    echo -e "${RED}Unknown benchmark: ${BENCH_NAME}${NC}"
    echo "Supported: locomo, locomo-mini, longmemeval-mini"
    exit 1
fi

# 3. Stop containers to flush all data to volumes
echo -e "${BLUE}Stopping containers to flush data...${NC}"
docker compose stop flask-api
sleep 2
docker compose stop falkordb qdrant
sleep 1

# 4. Snapshot the volumes
echo -e "${BLUE}Creating snapshot...${NC}"
mkdir -p "${SNAPSHOT_DIR}"

docker run --rm \
    -v "${COMPOSE_PROJECT}_falkordb_data:/data:ro" \
    -v "${SNAPSHOT_DIR}:/backup" \
    alpine tar czf /backup/falkordb.tar.gz -C /data .

docker run --rm \
    -v "${COMPOSE_PROJECT}_qdrant_data:/qdrant_storage:ro" \
    -v "${SNAPSHOT_DIR}:/backup" \
    alpine tar czf /backup/qdrant.tar.gz -C /qdrant_storage .

# 5. Restart services
echo -e "${BLUE}Restarting services...${NC}"
docker compose up -d

echo ""
echo -e "${GREEN}=== Snapshot complete ===${NC}"
echo -e "${GREEN}  Location: ${SNAPSHOT_DIR}${NC}"
ls -lh "${SNAPSHOT_DIR}"/*.tar.gz
echo ""
echo -e "${BLUE}To evaluate against this snapshot:${NC}"
echo "  make bench-eval BENCH=${BENCH_NAME} CONFIG=baseline"
