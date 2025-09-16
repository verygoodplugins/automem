# Makefile - Development commands
.PHONY: help install dev test clean logs deploy

# Default target
help:
	@echo "🧠 FalkorDB Memory System - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install    - Set up virtual environment and dependencies"
	@echo "  make dev        - Start local development environment"
	@echo ""
	@echo "Development:"
	@echo "  make test       - Run tests against local/remote FalkorDB"
	@echo "  make logs       - Show development logs"
	@echo "  make clean      - Clean up containers and volumes"
	@echo ""
	@echo "Deployment:"
	@echo "  make deploy     - Deploy to Railway"
	@echo "  make status     - Check deployment status"

# Set up development environment
install:
	@echo "🔧 Setting up development environment..."
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	./venv/bin/pip install pytest requests python-dotenv
	@echo "✅ Virtual environment ready!"
	@echo "💡 Run 'source venv/bin/activate' to activate"

# Start local development
dev:
	@echo "🚀 Starting local development environment..."
	docker-compose up --build

# Run tests
test:
	@echo "🧪 Running tests..."
	./venv/bin/python test_falkordb.py

# Show logs
logs:
	docker-compose logs -f flask-api

# Clean up
clean:
	@echo "🧹 Cleaning up..."
	docker-compose down -v
	docker system prune -f

# Deploy to Railway
deploy:
	@echo "🚀 Deploying to Railway..."
	railway up

# Check deployment status
status:
	@echo "📊 Checking deployment status..."
	railway logs
	@echo ""
	@echo "🌐 Your services:"
	@echo "  FalkorDB Browser: https://flask-production-5fcd.up.railway.app:3000"
	@echo "  Memory API: https://flask-production-5fcd.up.railway.app:8000"
