# Makefile - Development commands
.PHONY: help install dev test test-integration test-live test-locomo test-locomo-live clean logs deploy

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
	@echo "  make test-integration - Run all tests including integration tests"
	@echo "  make test-live  - Run integration tests against live Railway server"
	@echo "  make logs       - Show development logs"
	@echo "  make clean      - Clean up containers and volumes"
	@echo ""
	@echo "Benchmarks:"
	@echo "  make test-locomo      - Run LoCoMo benchmark (local)"
	@echo "  make test-locomo-live - Run LoCoMo benchmark (Railway)"
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
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/pytest -rs

# Run all tests including integration tests
test-integration:
	@echo "🧪 Running all tests including integration tests..."
	@echo "🐳 Starting Docker services..."
	@AUTOMEM_API_TOKEN=test-token ADMIN_API_TOKEN=test-admin-token docker compose up -d
	@echo "⏳ Waiting for services to be ready..."
	@sleep 5
	@echo "🧪 Running tests..."
	@AUTOMEM_RUN_INTEGRATION_TESTS=1 AUTOMEM_TEST_API_TOKEN=test-token AUTOMEM_TEST_ADMIN_TOKEN=test-admin-token PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/pytest -rs

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
