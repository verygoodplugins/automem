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

# Testing specific features
pytest                               # Run all tests
pytest tests/test_app.py -v         # Run with verbose output
pytest -k test_store_memory          # Run specific test by name
pytest tests/test_consolidation_engine.py::TestMemoryConsolidator  # Run test class

# Deployment
make deploy          # Deploy to Railway
make status          # Check deployment status
```

## API Endpoints

The API (`app.py`) provides 13 endpoints:

### Core Memory Operations
- `POST /memory` - Store a memory with content, tags, importance, metadata, and optional embedding
- `GET /recall` - Recall memories via text search, vector similarity, time range, and tags
- `PATCH /memory/<id>` - Update existing memory (content, tags, importance, metadata)
- `DELETE /memory/<id>` - Remove memory from both graph and vector stores
- `GET /memory/by-tag` - Filter memories by tags with importance/recency scoring

### Relationship Management
- `POST /associate` - Create relationships between memories (11 types available)

### Consolidation & Analysis
- `POST /consolidate` - Trigger memory consolidation tasks (decay, creative, cluster, forget, full)
- `GET /consolidate/status` - Check consolidation scheduler status and last run times
- `GET /startup-recall` - Retrieve memories for startup context
- `GET /analyze` - Analyze graph statistics and memory patterns

### Enrichment
- `GET /enrichment/status` - Inspect queue depth, worker state, and throughput metrics
- `POST /enrichment/reprocess` - Force reprocessing of specific memories (requires `X-Admin-Token`)

### Health
- `GET /health` - Service health check with database connectivity status

## Architecture

### Data Flow
1. **Flask API** (port 8001) - Request validation, orchestration, authentication
2. **FalkorDB** (port 6379) - Graph storage for Memory nodes and relationship edges
3. **Qdrant** (optional, port 6333) - 768-dimensional vector search for semantic similarity
4. **Consolidation Engine** - Background processing for memory maintenance

### Memory Consolidation Engine

The `consolidation.py` module implements biological memory patterns:
- **Decay** - Hourly exponential relevance updates (fractional-day decay keeps quick passes meaningful)
- **Creative** - Discovers hidden associations during "REM-like" processing (hourly)
- **Clustering** - Semantic grouping to compress related memories (every 6 hours)
- **Forgetting** - Archives low-importance memories (daily)

Scheduling is managed by `ConsolidationScheduler` with configurable intervals via environment variables.

### Enrichment Pipeline

- Queue-backed worker consumes `EnrichmentJob`s created on each memory write and optional reprocess calls.
- Extracts entities (tools/projects/people/organisations/concepts) using spaCy when available, otherwise regex heuristics, and writes them to metadata plus entity tags (`entity:<type>:<slug>`).
- Adds short summaries, timestamps (`enriched_at`), and per-run metrics under `metadata.enrichment`.
- Establishes temporal (`PRECEDED_BY`) and semantic (`SIMILAR_TO`) edges, including symmetric cosine scores from Qdrant.
- Detects recurring patterns per memory type, strengthens shared `Pattern` nodes, and links memories via `EXEMPLIFIES` relationships with key terms.
- Metrics exposed at `GET /enrichment/status` include processed counts, last success/error, queue depth, and inflight jobs.

### Relationship Types

AutoMem supports 11 relationship types with optional properties:
```python
# Original core relationships
RELATES_TO         # General relationship
LEADS_TO           # Causal relationship
OCCURRED_BEFORE    # Temporal relationship

# Enhanced PKG relationships
PREFERS_OVER       # Preference relationship (context, strength, reason)
EXEMPLIFIES        # Pattern example (pattern_type, confidence)
CONTRADICTS        # Conflicting information (resolution, reason)
REINFORCES         # Strengthens pattern (strength, observations)
INVALIDATED_BY     # Superseded information (reason, timestamp)
EVOLVED_INTO       # Evolution of knowledge (confidence, reason)
DERIVED_FROM       # Derived knowledge (transformation, confidence)
PART_OF            # Hierarchical relationship (role, context)
```

### Memory Type Classification

Memories are classified into types for better organization:
- `Decision` - Strategic choices and rationales
- `Pattern` - Recurring behaviors and approaches
- `Preference` - User preferences and settings
- `Style` - Coding/writing style patterns
- `Habit` - Regular practices and workflows
- `Insight` - Learned insights and discoveries
- `Context` - Environmental and project context
- `Memory` - Default base type

### Embedding Generation

When embeddings aren't provided:
1. If `OPENAI_API_KEY` is set → generates real embeddings using `text-embedding-3-small` model
2. Otherwise → creates deterministic placeholder vectors using content hash
3. Graph writes always succeed even if vector storage fails (graceful degradation)

## Testing

Tests use pytest with a `DummyGraph` fixture to mock FalkorDB operations:

```bash
# Set environment variable to disable conflicting plugins
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

# Run tests
pytest                                      # All tests
pytest tests/test_app.py::test_recall -v   # Single test with verbose
pytest -k "consolidat"                     # Tests matching pattern
pytest --tb=short                          # Shorter traceback format
```

## Environment Configuration

Key variables (create `.env` or `~/.config/automem/.env`):
```bash
# Core services
FALKORDB_HOST=localhost       # Graph database host
FALKORDB_PORT=6379           # Graph database port
FALKORDB_GRAPH=memories      # Graph name
QDRANT_URL=                  # Vector database URL (optional)
QDRANT_API_KEY=              # Qdrant cloud API key (optional)
QDRANT_COLLECTION=memories   # Collection name
VECTOR_SIZE=768              # Embedding dimensions

# API configuration
PORT=8001                    # API port
AUTOMEM_API_TOKEN=           # Required for authentication
ADMIN_API_TOKEN=             # For admin endpoints

# OpenAI (optional)
OPENAI_API_KEY=              # For real embeddings

# Consolidation intervals (seconds)
CONSOLIDATION_DECAY_INTERVAL_SECONDS=3600     # 1 hour
CONSOLIDATION_DECAY_IMPORTANCE_THRESHOLD=0.3  # Only skip truly low-importance items
CONSOLIDATION_CREATIVE_INTERVAL_SECONDS=3600  # 1 hour
CONSOLIDATION_CLUSTER_INTERVAL_SECONDS=21600  # 6 hours
CONSOLIDATION_FORGET_INTERVAL_SECONDS=86400   # 24 hours

# Enrichment controls
ENRICHMENT_MAX_ATTEMPTS=3                     # Retry attempts before giving up
ENRICHMENT_SIMILARITY_LIMIT=5                 # Neighbour links via Qdrant
ENRICHMENT_SIMILARITY_THRESHOLD=0.8           # Minimum cosine to link memories
ENRICHMENT_IDLE_SLEEP_SECONDS=2               # Worker sleep when idle
ENRICHMENT_FAILURE_BACKOFF_SECONDS=5          # Backoff between retries
ENRICHMENT_ENABLE_SUMMARIES=true              # Toggle automatic summary creation
ENRICHMENT_SPACY_MODEL=en_core_web_sm         # spaCy model for entity extraction
```

Install spaCy locally to improve entity extraction:

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

## Migration Tools

Several migration scripts help transition from legacy systems:

```bash
# Migrate from MCP memory service SQLite database
python scripts/migrate_legacy_memories.py

# Migrate from extracted memory dumps
python migrate_extracted.py

# Full MCP to PKG migration with relationships
python migrate_mcp_to_pkg.py

# Migrate memory project with validation
python migrate_memory_project.py

# Re-embed existing memories with OpenAI
python scripts/reembed_embeddings.py
```

## Key Implementation Patterns

- Memory IDs are UUIDs stored in both databases for cross-referencing
- Timestamps are normalized to UTC ISO format
- Recall scoring combines vector similarity, keyword match, tag overlap, and recency
- Authentication supports Bearer token, X-API-Key header, or query parameter
- Graph operations are atomic with automatic rollback on errors
- Vector store errors are logged but don't block graph writes
- Consolidation runs in background threads without blocking API requests
