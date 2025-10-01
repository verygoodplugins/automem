#!/bin/bash
# Script to run integration tests against the live Railway deployment (non-interactive)
# Use this for automated testing/CI

set -e

# Ensure we're in the project directory
cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Get Railway environment variables
LIVE_URL=$(railway variables --json | jq -r '.RAILWAY_PUBLIC_DOMAIN // empty' | sed 's/^/https:\/\//')
LIVE_API_TOKEN=$(railway variables --json | jq -r '.AUTOMEM_API_TOKEN // empty')
LIVE_ADMIN_TOKEN=$(railway variables --json | jq -r '.ADMIN_API_TOKEN // empty')

if [ -z "$LIVE_URL" ] || [ -z "$LIVE_API_TOKEN" ]; then
    echo "❌ Error: Could not fetch Railway configuration"
    echo "   Make sure you're linked to the Railway project: railway link"
    exit 1
fi

echo "🌐 Testing against: $LIVE_URL"

# Set required environment variables
export AUTOMEM_RUN_INTEGRATION_TESTS=1
export AUTOMEM_TEST_BASE_URL="$LIVE_URL"
export AUTOMEM_TEST_API_TOKEN="$LIVE_API_TOKEN"
export AUTOMEM_TEST_ADMIN_TOKEN="$LIVE_ADMIN_TOKEN"
export AUTOMEM_ALLOW_LIVE=1

# Run the tests
echo "🧪 Running integration tests..."
python -m pytest tests/test_integration.py -v "$@"

echo "✅ Live server tests completed!"

