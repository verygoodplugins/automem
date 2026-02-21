# Makefile - Development commands
.PHONY: help install dev test fmt lint test-integration test-live test-locomo test-locomo-live test-longmemeval test-longmemeval-live test-longmemeval-watch clean logs deploy

# Default target
help:
	@echo "🧠 FalkorDB Memory System - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install    - Set up virtual environment and dependencies"
	@echo "  make dev        - Start local development environment"
	@echo ""
	@echo "Development:"
	@echo "  make test       - Run unit tests only"
	@echo "  make fmt        - Format code (black + isort)"
	@echo "  make lint       - Lint code (flake8)"
	@echo "  make test-integration - Run all tests including integration tests"
	@echo "  make test-live  - Run integration tests against live Railway server"
	@echo "  make logs       - Show development logs"
	@echo "  make clean      - Clean up containers and volumes"
	@echo ""
	@echo "Benchmarks:"
	@echo "  make bench-mini-locomo    - Mini LoCoMo (2 conversations, <5 min)"
	@echo "  make bench-mini-longmemeval - Mini LongMemEval (20 questions)"
	@echo "  make bench-mini           - Both mini benchmarks"
	@echo "  make bench-ingest BENCH=locomo - Ingest + snapshot (run once)"
	@echo "  make bench-eval BENCH=locomo CONFIG=baseline - Eval from snapshot (~2 min)"
	@echo "  make bench-compare BENCH=locomo CONFIG=bm25 BASELINE=baseline - A/B compare"
	@echo "  make test-locomo          - Full LoCoMo benchmark (local)"
	@echo "  make test-locomo-live     - Full LoCoMo benchmark (Railway)"
	@echo "  make test-longmemeval     - Full LongMemEval benchmark (local)"
	@echo "  make test-longmemeval-live - Full LongMemEval benchmark (Railway)"
	@echo "  make test-longmemeval-watch - Full LongMemEval with notifications"
	@echo ""
	@echo "Deployment:"
	@echo "  make deploy     - Deploy to Railway"
	@echo "  make status     - Check deployment status"

# Set up development environment
install:
	@echo "🔧 Setting up development environment..."
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements-dev.txt
	@echo "✅ Virtual environment ready!"
	@echo "💡 Run 'source venv/bin/activate' to activate"

# Start local development
dev:
	@echo "🚀 Starting local development environment..."
	docker compose up --build

# Run tests
test:
	@echo "🧪 Running unit tests..."
	@if [ ! -x "./venv/bin/pytest" ]; then \
		echo "🔧 ./venv/bin/pytest not found; bootstrapping with 'make install'..."; \
		$(MAKE) install; \
	fi
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/pytest -rs -m unit

# Format code
fmt:
	@echo "✨ Formatting code (black + isort) ..."
	./venv/bin/black .
	./venv/bin/isort .

# Lint code
lint:
	@echo "🔍 Linting (flake8) ..."
	./venv/bin/flake8 .

# Run integration tests (requires Docker services)
test-integration:
	@echo "🧪 Running integration tests..."
	@echo "🐳 Starting Docker services..."
	@AUTOMEM_API_TOKEN=test-token ADMIN_API_TOKEN=test-admin-token docker compose up -d
	@echo "⏳ Waiting for services to be ready..."
	@sleep 5
	@echo "🧪 Running tests..."
	@AUTOMEM_RUN_INTEGRATION_TESTS=1 AUTOMEM_TEST_API_TOKEN=test-token AUTOMEM_TEST_ADMIN_TOKEN=test-admin-token PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/pytest -rs -m integration

# Run integration tests against live Railway server
test-live:
	@./test-live-server.sh

# Show logs
logs:
	docker compose logs -f flask-api

# Clean up
clean:
	@echo "🧹 Cleaning up..."
	docker compose down -v || true

# Deploy to Railway
deploy:
	@echo "🚀 Deploying to Railway..."
	railway up

# Check deployment status
status:
	@echo "📊 Checking deployment status..."
	railway status || railway logs

# Run LoCoMo benchmark (local)
test-locomo:
	@./test-locomo-benchmark.sh

# Run LoCoMo benchmark (Railway)
test-locomo-live:
	@./test-locomo-benchmark.sh --live

# Run LongMemEval benchmark (local)
test-longmemeval:
	@./test-longmemeval-benchmark.sh

# Run LongMemEval benchmark (Railway)
test-longmemeval-live:
	@./test-longmemeval-benchmark.sh --live

# Run LongMemEval benchmark with persistent log + local notifications
test-longmemeval-watch:
	@./scripts/run_longmemeval_watch.sh

# Mini benchmarks (fast iteration, <5 minutes)
bench-mini-locomo:
	@./test-locomo-benchmark.sh --conversations 0,1

bench-mini-longmemeval:
	@./test-longmemeval-benchmark.sh --max-questions 20

bench-mini: bench-mini-locomo bench-mini-longmemeval

# Snapshot-based benchmarks (ingest once, eval many)
bench-ingest:
	@scripts/bench/ingest_and_snapshot.sh $(or $(BENCH),locomo)

bench-eval:
	@scripts/bench/restore_and_eval.sh $(or $(BENCH),locomo) $(or $(CONFIG),baseline)

bench-compare:
	@scripts/bench/compare_configs.sh $(or $(BENCH),locomo) $(or $(BASELINE),baseline) $(CONFIG)

bench-compare-branch:
	@scripts/bench/compare_branch.sh $(BRANCH) $(or $(CONFIG),baseline) $(or $(BENCH),locomo)

bench-snapshots:
	@ls -la benchmarks/snapshots/ 2>/dev/null || echo "No snapshots yet. Run: make bench-ingest BENCH=locomo"

# Recall Quality Lab
lab-clone:
	@echo "🔬 Cloning production data to local Docker..."
	@bash scripts/lab/clone_production.sh

lab-queries:
	@echo "🔬 Generating test queries from local data..."
	@python3 scripts/lab/create_test_queries.py

lab-test:
	@echo "🔬 Running recall quality test..."
	@python3 scripts/lab/run_recall_test.py --config $(or $(CONFIG),baseline)

lab-compare:
	@echo "🔬 Comparing configs: $(BASELINE) vs $(CONFIG)..."
	@python3 scripts/lab/run_recall_test.py --config $(CONFIG) --compare $(or $(BASELINE),baseline)

lab-sweep:
	@echo "🔬 Sweeping $(PARAM)..."
	@python3 scripts/lab/run_recall_test.py --sweep $(PARAM) $(VALUES)
