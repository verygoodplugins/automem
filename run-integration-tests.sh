#!/bin/bash
# Script to run integration tests with proper environment setup

set -e

# Ensure we're in the project directory
cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Set required environment variables
export AUTOMEM_RUN_INTEGRATION_TESTS=1
export AUTOMEM_TEST_API_TOKEN=test-token
export AUTOMEM_TEST_ADMIN_TOKEN=test-admin-token

# Start Docker services with proper tokens
echo "🐳 Starting Docker services..."
AUTOMEM_API_TOKEN=test-token ADMIN_API_TOKEN=test-admin-token docker compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 5

# Run the tests
echo "🧪 Running integration tests..."
python -m pytest tests/test_integration.py -v "$@"

echo "✅ Integration tests completed!"
