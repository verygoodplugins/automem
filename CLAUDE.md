# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoMem is a Flask-based memory service that provides durable memory storage for AI assistants using FalkorDB (graph database) for relationships and Qdrant (vector database) for semantic search. The service gracefully degrades when Qdrant is unavailable, ensuring graph operations always succeed.

## Development Commands

```bash
# Setup environment
make install          # Create venv and install dependencies
source venv/bin/activate

# Development
make dev             # Start full stack (FalkorDB + Qdrant + API) via Docker
make test            # Run pytest test suite
make logs            # Follow Flask API logs
make clean           # Clean up Docker containers/volumes

# Code quality
black .              # Format Python code
flake8               # Lint Python code

# Deployment
make deploy          # Deploy to Railway
make status          # Check deployment status
```

## Architecture

The API (`app.py`) provides three core endpoints:
- `POST /memory` - Store a memory with content, tags, importance, and optional embedding
- `GET /recall` - Recall memories via text search or vector similarity
- `POST /associate` - Create relationships between memories (RELATES_TO, LEADS_TO, OCCURRED_BEFORE)

Data flows through:
1. **Flask API** (port 8001) - Request validation, orchestration
2. **FalkorDB** (port 6379) - Graph storage for Memory nodes and relationship edges
3. **Qdrant** (optional) - 768-dimensional vector search for semantic similarity

Key implementation patterns:
- Memory IDs are UUIDs stored in both databases for cross-referencing
- When embeddings aren't provided, deterministic placeholder vectors are generated using content hash
- Graph writes always succeed even if vector storage fails (graceful degradation)
- Timestamps are normalized to UTC ISO format

## Testing

Tests use pytest with a DummyGraph fixture to mock FalkorDB operations:

```bash
pytest                        # Run all tests
pytest tests/test_app.py -v   # Run with verbose output
pytest -k test_store_memory   # Run specific test
```

## Environment Configuration

Key variables (see `.env.example`):
- `FALKORDB_HOST` / `FALKORDB_PORT` - Graph database connection
- `QDRANT_URL` / `QDRANT_API_KEY` - Vector database (optional)
- `VECTOR_SIZE` - Embedding dimensions (default: 768)

Local config can be placed in `~/.config/automem/.env` for persistent settings.

## Migration Tool

The `scripts/migrate_legacy_memories.py` script migrates memories from the legacy MCP memory service's SQLite database to AutoMem, preserving content, tags, and importance scores.