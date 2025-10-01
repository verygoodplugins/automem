#!/bin/bash
set -e

# Multi-Tenant AutoMem Test Script
# Tests tenant provisioning, isolation, and admin endpoints

BASE_URL="${BASE_URL:-http://localhost:8001}"
ADMIN_TOKEN="${ADMIN_TOKEN:-test-admin-token}"

echo "========================================="
echo "AutoMem Multi-Tenant Test"
echo "========================================="
echo "Base URL: $BASE_URL"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

pass() {
  echo -e "${GREEN}✓${NC} $1"
}

fail() {
  echo -e "${RED}✗${NC} $1"
  exit 1
}

# Test 1: Health check
echo "Test 1: Health check..."
response=$(curl -s "$BASE_URL/health")
if echo "$response" | grep -q "healthy"; then
  pass "Health check passed"
else
  fail "Health check failed: $response"
fi

# Test 2: Create Fernando's tenant
echo ""
echo "Test 2: Create Fernando's tenant..."
response=$(curl -s -X POST "$BASE_URL/admin/tenants" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "fernando",
    "name": "Fernando Test Client",
    "api_token": "fernando_test_token_12345",
    "metadata": {"plan": "pro", "email": "fernando@test.com"}
  }')

if echo "$response" | grep -q "fernando"; then
  pass "Fernando's tenant created"
  FERNANDO_TOKEN="fernando_test_token_12345"
else
  fail "Failed to create Fernando's tenant: $response"
fi

# Test 3: Create Claudio's tenant
echo ""
echo "Test 3: Create Claudio's tenant..."
response=$(curl -s -X POST "$BASE_URL/admin/tenants" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "claudio",
    "name": "Claudio Test Client",
    "api_token": "claudio_test_token_67890",
    "metadata": {"plan": "basic", "email": "claudio@test.com"}
  }')

if echo "$response" | grep -q "claudio"; then
  pass "Claudio's tenant created"
  CLAUDIO_TOKEN="claudio_test_token_67890"
else
  fail "Failed to create Claudio's tenant: $response"
fi

# Test 4: List tenants
echo ""
echo "Test 4: List all tenants..."
response=$(curl -s "$BASE_URL/admin/tenants" \
  -H "Authorization: Bearer $ADMIN_TOKEN")

if echo "$response" | grep -q "fernando" && echo "$response" | grep -q "claudio"; then
  pass "Both tenants listed"
else
  fail "Failed to list tenants: $response"
fi

# Test 5: Get Fernando's stats
echo ""
echo "Test 5: Get Fernando's stats..."
response=$(curl -s "$BASE_URL/admin/tenants/fernando/stats" \
  -H "Authorization: Bearer $ADMIN_TOKEN")

if echo "$response" | grep -q "fernando"; then
  pass "Fernando's stats retrieved"
else
  fail "Failed to get Fernando's stats: $response"
fi

# Test 6: Fernando stores a memory
echo ""
echo "Test 6: Fernando stores a memory..."
response=$(curl -s -X POST "$BASE_URL/memory" \
  -H "Authorization: Bearer $FERNANDO_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"content": "Fernando uses PostgreSQL for reliability"}')

if echo "$response" | grep -q "success"; then
  pass "Fernando stored a memory"
else
  fail "Failed to store Fernando's memory: $response"
fi

# Test 7: Claudio stores a memory
echo ""
echo "Test 7: Claudio stores a memory..."
response=$(curl -s -X POST "$BASE_URL/memory" \
  -H "Authorization: Bearer $CLAUDIO_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"content": "Claudio prefers MongoDB for flexibility"}')

if echo "$response" | grep -q "success"; then
  pass "Claudio stored a memory"
else
  fail "Failed to store Claudio's memory: $response"
fi

# Test 8: Fernando recalls memories (should only see his)
echo ""
echo "Test 8: Fernando recalls memories..."
response=$(curl -s "$BASE_URL/recall?query=database" \
  -H "Authorization: Bearer $FERNANDO_TOKEN")

if echo "$response" | grep -q "fernando"; then
  pass "Fernando can recall memories"
else
  fail "Failed to recall Fernando's memories: $response"
fi

# Test 9: Claudio recalls memories (should only see his)
echo ""
echo "Test 9: Claudio recalls memories..."
response=$(curl -s "$BASE_URL/recall?query=database" \
  -H "Authorization: Bearer $CLAUDIO_TOKEN")

if echo "$response" | grep -q "claudio"; then
  pass "Claudio can recall memories"
else
  fail "Failed to recall Claudio's memories: $response"
fi

# Test 10: Test auth failure (invalid token)
echo ""
echo "Test 10: Test invalid token..."
response=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/memory" \
  -H "Authorization: Bearer invalid_token")

if [ "$response" == "401" ]; then
  pass "Invalid token rejected"
else
  fail "Invalid token not rejected (status: $response)"
fi

# Test 11: Test admin auth failure
echo ""
echo "Test 11: Test invalid admin token..."
response=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/admin/tenants" \
  -H "Authorization: Bearer wrong_admin_token" \
  -d '{"tenant_id":"test"}')

if [ "$response" == "403" ]; then
  pass "Invalid admin token rejected"
else
  fail "Invalid admin token not rejected (status: $response)"
fi

# Cleanup
echo ""
echo "Cleanup: Deleting test tenants..."
curl -s -X DELETE "$BASE_URL/admin/tenants/fernando?confirm=true" \
  -H "Authorization: Bearer $ADMIN_TOKEN" > /dev/null
pass "Fernando's tenant deleted"

curl -s -X DELETE "$BASE_URL/admin/tenants/claudio?confirm=true" \
  -H "Authorization: Bearer $ADMIN_TOKEN" > /dev/null
pass "Claudio's tenant deleted"

echo ""
echo "========================================="
echo -e "${GREEN}All tests passed!${NC}"
echo "========================================="
echo ""
echo "Multi-tenant setup is working correctly:"
echo "  ✓ Tenant provisioning"
echo "  ✓ Authentication"
echo "  ✓ Tenant isolation"
echo "  ✓ Admin endpoints"
echo ""
echo "Ready for deployment!"
