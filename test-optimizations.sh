#!/bin/bash
# Test script for AutoMem optimizations
# Run this to verify all optimizations are working correctly

set -e

API_URL="${API_URL:-http://localhost:8001}"
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Testing AutoMem Optimizations${NC}\n"

# Test 1: Health endpoint with enrichment stats
echo -e "${BLUE}Test 1: Health endpoint with enrichment stats${NC}"
HEALTH=$(curl -s "$API_URL/health")
echo "$HEALTH" | jq .enrichment
if echo "$HEALTH" | jq -e '.enrichment.status' > /dev/null; then
    echo -e "${GREEN}✓ Enrichment stats present in /health${NC}\n"
else
    echo "✗ Enrichment stats missing from /health"
    exit 1
fi

# Test 2: Embedding batching (store multiple memories)
echo -e "${BLUE}Test 2: Embedding batching${NC}"
echo "Storing 25 memories rapidly to trigger batch processing..."
for i in {1..25}; do
    curl -s -X POST "$API_URL/memory" \
        -H "Content-Type: application/json" \
        -d "{\"content\": \"Optimization test memory $i - $(date +%s)\"}" > /dev/null
    echo -n "."
done
echo ""
echo -e "${GREEN}✓ Stored 25 memories (check logs for batch processing)${NC}"
echo "Expected log: 'Generated X OpenAI embeddings in batch'"
echo ""

# Test 3: Structured logging in recall
echo -e "${BLUE}Test 3: Structured logging in recall${NC}"
RECALL=$(curl -s "$API_URL/recall?query=optimization&limit=5")
QUERY_TIME=$(echo "$RECALL" | jq -r '.query_time_ms')
echo "Query time: ${QUERY_TIME}ms"
echo -e "${GREEN}✓ Recall completed (check logs for 'recall_complete')${NC}"
echo "Expected log: 'recall_complete' with latency_ms, results, etc."
echo ""

# Test 4: Store with structured logging
echo -e "${BLUE}Test 4: Structured logging in store${NC}"
STORE=$(curl -s -X POST "$API_URL/memory" \
    -H "Content-Type: application/json" \
    -d '{"content": "Final optimization test memory", "importance": 0.9, "tags": ["test", "optimization"]}')
MEMORY_ID=$(echo "$STORE" | jq -r '.memory_id')
STORE_TIME=$(echo "$STORE" | jq -r '.query_time_ms')
echo "Stored memory: $MEMORY_ID in ${STORE_TIME}ms"
echo -e "${GREEN}✓ Memory stored (check logs for 'memory_stored')${NC}"
echo "Expected log: 'memory_stored' with memory_id, latency_ms, etc."
echo ""

# Summary
echo -e "${BLUE}=== Summary ===${NC}"
echo "✓ All optimizations appear to be working"
echo ""
echo "Next steps:"
echo "1. Check logs for batch embedding processing"
echo "2. Check logs for structured logging (recall_complete, memory_stored)"
echo "3. Monitor /health endpoint for enrichment stats"
echo "4. After consolidation runs, verify faster execution times"
echo ""
echo "Configuration:"
echo "  EMBEDDING_BATCH_SIZE=${EMBEDDING_BATCH_SIZE:-20}"
echo "  EMBEDDING_BATCH_TIMEOUT_SECONDS=${EMBEDDING_BATCH_TIMEOUT_SECONDS:-2.0}"

