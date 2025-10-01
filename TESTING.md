# Testing Guide

This document describes the testing setup for AutoMem and how to run tests against different environments.

## Test Suite Overview

AutoMem has a comprehensive test suite with 62 tests covering:
- API endpoints (36 tests)
- Core functionality (8 tests)
- Consolidation engine (6 tests)
- Enrichment pipeline (2 tests)
- Integration tests (8 tests)

## Quick Commands

```bash
# Unit tests only (fast, no services required)
make test

# Integration tests (local Docker)
make test-integration

# Integration tests (live Railway server)
make test-live
```

## Test Types

### 1. Unit Tests
**Command**: `make test`

- Fast, isolated tests using mock/stub implementations
- No external services required
- Tests API logic, validation, edge cases
- Safe to run anytime

### 2. Integration Tests (Local)
**Command**: `make test-integration`

- Tests against real Docker services (FalkorDB + Qdrant + API)
- Automatically starts services with test credentials
- Creates test memories tagged with `["test", "integration"]`
- Cleans up all test data after completion
- Requires: Docker, Docker Compose

**What it does:**
1. Starts Docker services with `AUTOMEM_API_TOKEN=test-token`
2. Waits for services to be ready (5s)
3. Runs full integration test suite
4. Tests real database operations, embeddings, associations

### 3. Live Server Tests (Railway)
**Command**: `make test-live`

- Tests against the production Railway deployment
- Verifies local and live environments have matching behavior
- Prompts for confirmation before running (safety measure)
- Automatically fetches Railway credentials
- Requires: Railway CLI, linked project (`railway link`)

**Safety features:**
- Interactive confirmation required
- Only creates/modifies test memories with unique UUIDs
- All test data is cleaned up immediately
- Read-only operations for health checks and recalls

## Test Scripts

### Interactive Live Testing
```bash
./test-live-server.sh
```
Prompts for confirmation before running against production.

### Automated Live Testing
```bash
./test-live-server-auto.sh
```
Non-interactive version for CI/automation.

### Manual Integration Testing
```bash
./run-integration-tests.sh
```
Runs integration tests with proper environment setup.

## Environment Variables

### Required for Integration Tests
- `AUTOMEM_RUN_INTEGRATION_TESTS=1` - enables integration tests
- `AUTOMEM_TEST_API_TOKEN` - API authentication token
- `AUTOMEM_TEST_ADMIN_TOKEN` - admin authentication token (optional for some tests)

### Optional Configuration
- `AUTOMEM_TEST_BASE_URL` - override API endpoint (default: `http://localhost:8001`)
- `AUTOMEM_ALLOW_LIVE=1` - required to test against non-localhost URLs
- `AUTOMEM_START_DOCKER=1` - auto-start Docker services
- `AUTOMEM_STOP_DOCKER=1` - auto-stop Docker after tests (default)

## Test Results

All tests pass cleanly with no warnings (filtered via `pytest.ini`):
- ✅ 61 passed
- ⏭️ 1 skipped (rate limiting not implemented)
- ⚠️ 0 warnings

## Comparing Local vs Live

To verify local Docker environment matches production:

```bash
# Run tests locally
make test-integration

# Run same tests against live
make test-live
```

Both should produce identical results, confirming:
- API responses match
- Authentication works correctly
- Database operations behave the same
- Embeddings are generated consistently

## Troubleshooting

### "API not available" error
The integration tests wait up to 10 seconds for the API to be ready. If services take longer:
- Check `docker compose ps` to see service status
- Check `docker compose logs flask-api` for startup errors
- Manually verify health: `curl http://localhost:8001/health`

### "Unauthorized" errors (401)
Ensure environment variables match:
- Local: `AUTOMEM_API_TOKEN=test-token`
- Docker: Set via `docker-compose.yml` environment section
- Railway: Check with `railway variables`

### Railway CLI issues
```bash
# Install Railway CLI
npm install -g @railway/cli

# Link to project
railway link

# Verify connection
railway status
```

## CI/CD Integration

For automated testing in CI:

```bash
# Unit tests (always safe)
make test

# Integration tests (if Docker available)
make test-integration

# Live tests (if Railway credentials available)
./test-live-server-auto.sh
```

## Best Practices

1. **Always run unit tests** before committing
2. **Run integration tests** when changing API logic or database operations
3. **Run live tests** before deploying to verify no regressions
4. **Check test coverage** with `pytest --cov` (requires pytest-cov)
5. **Review test output** - integration tests show actual API responses

