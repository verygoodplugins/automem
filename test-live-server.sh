#!/bin/bash
# Script to run integration tests against the live Railway deployment

set -e

NON_INTERACTIVE=0
PYTEST_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --non-interactive)
      NON_INTERACTIVE=1
      ;;
    *)
      PYTEST_ARGS+=("$arg")
      ;;
  esac
done

# Ensure we're in the project directory
cd "$(dirname "$0")"

# Activate virtual environment if present
if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
fi

# Get Railway environment variables
echo "🔍 Fetching Railway configuration..."
LIVE_URL=$(railway variables --json | jq -r '.RAILWAY_PUBLIC_DOMAIN // empty' | sed 's/^/https:\/\//')
LIVE_API_TOKEN=$(railway variables --json | jq -r '.AUTOMEM_API_TOKEN // empty')
LIVE_ADMIN_TOKEN=$(railway variables --json | jq -r '.ADMIN_API_TOKEN // empty')

if [ -z "$LIVE_URL" ] || [ -z "$LIVE_API_TOKEN" ]; then
    echo "❌ Error: Could not fetch Railway configuration"
    echo "   Make sure you're linked to the Railway project: railway link"
    exit 1
fi

echo "🌐 Live server URL: $LIVE_URL"
echo ""

if [ "$NON_INTERACTIVE" -eq 0 ]; then
  # Confirm before running against live
  echo "⚠️  WARNING: This will run integration tests against the LIVE production server!"
  echo "   The tests will create and delete test memories tagged with 'test' and 'integration'."
  echo ""
  read -p "Are you sure you want to continue? (y/N) " -n 1 -r
  echo ""
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      echo "❌ Aborted"
      exit 1
  fi
fi

# Set required environment variables
export AUTOMEM_RUN_INTEGRATION_TESTS=1
export AUTOMEM_TEST_BASE_URL="$LIVE_URL"
export AUTOMEM_TEST_API_TOKEN="$LIVE_API_TOKEN"
export AUTOMEM_TEST_ADMIN_TOKEN="$LIVE_ADMIN_TOKEN"
export AUTOMEM_ALLOW_LIVE=1

# Run the tests
echo ""
echo "🧪 Running integration tests against live server..."
python -m pytest tests/test_integration.py -v "${PYTEST_ARGS[@]}"

echo ""
echo "✅ Live server tests completed!"
