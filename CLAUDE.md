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
make test            # Run pytest test suite (unit tests only)
make test-integration # Run all tests including integration tests (starts Docker)
make test-live       # Run integration tests against live Railway server
make logs            # Follow Flask API logs
make clean           # Clean up Docker containers/volumes

# Benchmarking
make test-locomo      # Run LoCoMo benchmark against local server
make test-locomo-live # Run LoCoMo benchmark against Railway server

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

The `automem/api` module provides **28 endpoints** (admin: 2, memory: 10, recall: 4, graph: 5, health: 1, enrichment: 2, consolidation: 2, stream: 2). Additionally, 12 legacy routes remain in `app.py` for backward compatibility—combined total of 40 if both sets are active.

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
5. **FalkorDB Browser** (optional, port 3001) - Web UI for graph visualization (start with `docker compose --profile browser up`)

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

AutoMem uses a provider pattern with three embedding backends:

#### Provider Priority (Auto-Selection)
1. **OpenAI** (`openai:text-embedding-3-large`) - If `OPENAI_API_KEY` is set
   - High-quality semantic embeddings via API
   - Requires network and API costs
   - 3072 dimensions by default (configurable via `EMBEDDING_MODEL` and `VECTOR_SIZE`)

2. **FastEmbed** (`fastembed:BAAI/bge-base-en-v1.5`) - Local ONNX model
   - Good quality semantic embeddings
   - No API key or internet required (after first download)
   - Downloads ~210MB model to `~/.config/automem/models/` on first use
   - 768 dimensions (default), also supports 384 and 1024 dim models
   - Note: Pin `onnxruntime<1.20` to avoid compatibility issues with fastembed 0.4.x

3. **Placeholder** (`placeholder`) - Hash-based fallback
   - Deterministic vectors from content hash
   - No semantic meaning, last resort only

**Upgrade safety:** If your existing Qdrant collection is 768d, keep `VECTOR_SIZE=768` (and `text-embedding-3-small`) until you re-embed. The server fails fast on a dimension mismatch to avoid corrupting data.

#### Provider Configuration

Control via `EMBEDDING_PROVIDER` environment variable:
- `auto` (default): Try OpenAI → FastEmbed → Placeholder
- `openai`: Use OpenAI only (fail if unavailable)
- `local`: Use FastEmbed only (fail if unavailable)
- `placeholder`: Use placeholder embeddings

Graph writes always succeed even if vector storage fails (graceful degradation).

**Module:** `automem/embedding/` provides `EmbeddingProvider` abstraction with three implementations: `OpenAIEmbeddingProvider`, `FastEmbedProvider`, `PlaceholderEmbeddingProvider`.

## Testing

Tests use pytest with a `DummyGraph` fixture to mock FalkorDB operations:

```bash
# Set environment variable to disable conflicting plugins
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

# Run tests
pytest                                      # All unit tests
pytest tests/test_app.py::test_recall -v   # Single test with verbose
pytest -k "consolidat"                     # Tests matching pattern
pytest --tb=short                          # Shorter traceback format

# Integration tests (requires Docker services)
make test-integration                       # Starts services and runs integration tests
AUTOMEM_RUN_INTEGRATION_TESTS=1 pytest tests/test_integration.py -v

# Live server testing (requires Railway deployment)
./test-live-server.sh                       # Tests against deployed Railway instance
make test-live                              # Same as above via Makefile
```

Test files:
- `tests/test_app.py` - Core API endpoint tests
- `tests/test_consolidation_engine.py` - Memory consolidation logic tests
- `tests/test_enrichment.py` - Entity extraction and enrichment tests
- `tests/test_integration.py` - Full stack integration tests (requires Docker)
- `tests/test_api_endpoints.py` - Comprehensive API endpoint tests
- `tests/benchmarks/locomo/` - LoCoMo benchmark suite for long-term memory evaluation

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
VECTOR_SIZE=3072             # Embedding dimensions (3072 for large, 768 for small)

# API configuration
PORT=8001                    # API port
AUTOMEM_API_TOKEN=           # Required for authentication
ADMIN_API_TOKEN=             # For admin endpoints

# Embedding configuration
EMBEDDING_PROVIDER=auto      # auto|openai|local|placeholder
OPENAI_API_KEY=              # For OpenAI embeddings (optional)

# Consolidation intervals (seconds)
CONSOLIDATION_DECAY_INTERVAL_SECONDS=86400    # 1 day (default)
CONSOLIDATION_DECAY_IMPORTANCE_THRESHOLD=0.3  # Only skip truly low-importance items
CONSOLIDATION_CREATIVE_INTERVAL_SECONDS=604800  # 1 week (default)
CONSOLIDATION_CLUSTER_INTERVAL_SECONDS=2592000  # 1 month (default)
CONSOLIDATION_FORGET_INTERVAL_SECONDS=0       # Disabled by default (set to enable)

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

Use the consolidated helper to migrate from the legacy MCP SQLite store, then
optionally re-embed:

```bash
# Preview what will be imported
python scripts/migrate_mcp_sqlite.py --dry-run

# Run migration against a deployed instance
python scripts/migrate_mcp_sqlite.py \
  --db /path/to/sqlite_vec.db \
  --automem-url https://automem.example.com \
  --api-token $AUTOMEM_API_TOKEN

# Refresh embeddings after the migration
python scripts/reembed_embeddings.py --limit 200
```

## Utility Scripts

The `scripts/` directory contains maintenance and recovery tools:

### Backup & Recovery
- **backup_automem.py** - Creates backups of FalkorDB and Qdrant data
- **recover_from_qdrant.py** - Recovers graph data from Qdrant vector store

### Data Management
- **cleanup_memory_types.py** - Cleans up memory type classifications
- **reclassify_with_llm.py** - Uses LLM to reclassify memory types
- **deduplicate_qdrant.py** - Removes duplicate vectors from Qdrant
- **reembed_embeddings.py** - Regenerates embeddings for existing memories
- **reenrich_batch.py** - Batch re-enrichment of memories

### Monitoring
- **health_monitor.py** - Health monitoring service for production deployments

All scripts support `--help` for detailed usage information.

## Local vs Railway Workflow

### Typical Development Flow
1. **Local Development** - Make changes and test with `make dev`
2. **Unit Tests** - Verify with `make test`
3. **Integration Tests** - Validate with `make test-integration`
4. **Deploy to Railway** - Push changes with `make deploy`
5. **Live Validation** - Test deployed instance with `make test-live`
6. **Benchmarking** - Validate performance with `make test-locomo-live`

### When to Use Each Environment

**Local (Docker Compose)**:
- Feature development and debugging
- Rapid iteration without deployment delays
- Testing consolidation/enrichment behavior
- Privacy-focused work (data stays local)
- Cost-free development

**Railway (Cloud)**:
- Production deployment for 24/7 availability
- Multi-device access (laptop, desktop, mobile)
- Team collaboration with shared memory
- Testing real-world latency and performance
- Integration with remote AI tools

### Testing Against Railway
Before deploying breaking changes, test against your Railway instance:
```bash
# Set Railway environment variables
export AUTOMEM_TEST_URL=https://your-app.railway.app
export AUTOMEM_TEST_API_TOKEN=your_token
export AUTOMEM_TEST_ADMIN_TOKEN=your_admin_token

# Run integration tests against Railway
./test-live-server.sh
```

## Key Implementation Patterns

- Memory IDs are UUIDs stored in both databases for cross-referencing
- Timestamps are normalized to UTC ISO format
- Recall scoring combines vector similarity, keyword match, tag overlap, and recency
- Authentication supports Bearer token, X-API-Key header, or query parameter
- Graph operations are atomic with automatic rollback on errors
- Vector store errors are logged but don't block graph writes
- Consolidation runs in background threads without blocking API requests
- Enrichment pipeline processes memories asynchronously with automatic retries
