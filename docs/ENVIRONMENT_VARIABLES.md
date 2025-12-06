# Environment Variables Reference

Complete reference for all AutoMem environment variables.

## Quick Start

```bash
# Copy example and customize
cp .env.example .env
nano .env
```

---

## Required Variables

### Core Services

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `FALKORDB_HOST` | FalkorDB hostname | `localhost` | `falkordb.railway.internal` |
| `FALKORDB_PORT` | FalkorDB port | `6379` | `6379` |
| `FALKORDB_PASSWORD` | FalkorDB password (optional) | - | `your-secure-password` |
| `FALKORDB_GRAPH` | Graph database name | `memories` | `memories` |

### Authentication

| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `AUTOMEM_API_TOKEN` | API authentication token | ✅ Yes | Generate: `openssl rand -hex 32` |
| `ADMIN_API_TOKEN` | Admin endpoint token | ✅ Yes | Generate: `openssl rand -hex 32` |

**⚠️ Important: Admin Endpoints Require BOTH Tokens**

Admin endpoints (like `/enrichment/reprocess`, `/admin/reembed`) require **two-level authentication**:

1. **`Authorization: Bearer <AUTOMEM_API_TOKEN>`** - For general API access
2. **`X-Admin-Token: <ADMIN_API_TOKEN>`** - For admin-level operations

Example:
```bash
curl -X POST \
  -H "Authorization: Bearer ${AUTOMEM_API_TOKEN}" \
  -H "X-Admin-Token: ${ADMIN_API_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"ids": ["memory-id"]}' \
  https://automem.up.railway.app/enrichment/reprocess
```

### Embedding Providers

AutoMem supports three embedding backends with automatic fallback.

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `EMBEDDING_PROVIDER` | Embedding backend selection | `auto` | `auto`, `openai`, `local`, `placeholder` |
| `OPENAI_API_KEY` | OpenAI API key (for OpenAI provider) | - | `sk-proj-...` |

**Provider Options:**
- `auto` (default): Try OpenAI → FastEmbed local model → Placeholder
- `openai`: Use OpenAI API only (requires `OPENAI_API_KEY`)
- `local`: Use FastEmbed local model only (~210MB download on first use)
- `placeholder`: Use hash-based embeddings (no semantic search)

**Local Model Details:**
- Model: `BAAI/bge-base-en-v1.5` (768 dimensions)
- Size: ~210MB (cached to `~/.config/automem/models/`)
- No API key or internet required after first download
- Good semantic quality, faster than API calls
- Recommended: pin `onnxruntime<1.20` with `fastembed` 0.4.x to avoid runtime issues
- Docker: persist model cache with a volume (see docker-compose.yml)
- Ensure `VECTOR_SIZE` equals the selected model's dimension (default 768)

**Additional Configuration:**
| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `AUTOMEM_MODELS_DIR` | Override model cache directory | `~/.config/automem/models/` | Useful in containers |

---

## Optional Variables

### Qdrant (Vector Database)

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `QDRANT_URL` | Qdrant endpoint URL | `http://localhost:6333` | `https://xyz.qdrant.io` |
| `QDRANT_API_KEY` | Qdrant API key | - | `your-qdrant-key` |
| `QDRANT_COLLECTION` | Collection name | `memories` | `memories` |
| `VECTOR_SIZE` | Embedding dimension | `3072` | `3072` (large), `768` (small) |

**Note**: Without Qdrant, AutoMem uses deterministic placeholder embeddings (for testing only).

### API Server

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `PORT` | Flask server port | `8001` | ✅ **Yes** (Railway) |

**⚠️ Railway Deployment**: `PORT` **must** be explicitly set to `8001` in Railway. Without it, Flask defaults to port 5000, causing service connection failures. This is **required** for Railway deployments, even though it has a default in local development.

### Scripts Only

| Variable | Description | Default | Used By |
|----------|-------------|---------|---------|
| `AUTOMEM_API_URL` | AutoMem API endpoint | `http://localhost:8001` | `recover_from_qdrant.py`, `health_monitor.py` |

**Backward Compatibility**: `MCP_MEMORY_HTTP_ENDPOINT` is deprecated but still supported (falls back to this if `AUTOMEM_API_URL` not set).

### Health Monitor

| Variable | Description | Default |
|----------|-------------|---------|
| `HEALTH_MONITOR_WEBHOOK` | Webhook URL for alerts (e.g., Slack) | - |
| `HEALTH_MONITOR_EMAIL` | Email address for alerts | - |
| `HEALTH_MONITOR_DRIFT_THRESHOLD` | Warning threshold (%) | `5` |
| `HEALTH_MONITOR_CRITICAL_THRESHOLD` | Critical threshold (%) for recovery | `50` |

**Note**: Auto-recovery is **disabled by default**. Use `--auto-recover` flag to enable (not recommended without testing).

---

## Advanced Configuration

### Consolidation Engine

Controls memory merging, pattern detection, and decay.

| Variable | Description | Default | Unit |
|----------|-------------|---------|------|
| `CONSOLIDATION_TICK_SECONDS` | Check interval | `60` | seconds |
| `CONSOLIDATION_DECAY_INTERVAL_SECONDS` | Decay check interval | `3600` | seconds |
| `CONSOLIDATION_CREATIVE_INTERVAL_SECONDS` | Pattern detection interval | `3600` | seconds |
| `CONSOLIDATION_CLUSTER_INTERVAL_SECONDS` | Clustering interval | `21600` | seconds |
| `CONSOLIDATION_FORGET_INTERVAL_SECONDS` | Forget interval | `86400` | seconds |
| `CONSOLIDATION_DECAY_IMPORTANCE_THRESHOLD` | Min importance to keep | `0.3` | 0-1 |
| `CONSOLIDATION_HISTORY_LIMIT` | Max consolidation history | `20` | count |
| `CONSOLIDATION_CONTROL_NODE_ID` | Control node identifier | `global` | string |

### Model Configuration

Controls which OpenAI models are used for embeddings and classification.

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `EMBEDDING_MODEL` | OpenAI embedding model | `text-embedding-3-large` | `text-embedding-3-large` (3072d), `text-embedding-3-small` (768d) |
| `VECTOR_SIZE` | Embedding dimension | `3072` | Must match embedding model |
| `CLASSIFICATION_MODEL` | LLM for memory type classification | `gpt-4o-mini` | `gpt-4o-mini`, `gpt-4.1`, `gpt-5.1` |

**Embedding Model Comparison:**
| Model | Dimensions | Cost/1M tokens | Quality | Use Case |
|-------|-----------|----------------|---------|----------|
| `text-embedding-3-large` | 3072 | $0.13 | Excellent | **Default** - Better semantic precision |
| `text-embedding-3-small` | 768 | $0.02 | Good | Cost-sensitive, high-volume deployments |

To use small embeddings (saves ~$0.11/1M tokens):
```bash
EMBEDDING_MODEL=text-embedding-3-small
VECTOR_SIZE=768
```

**Classification Model Pricing (Dec 2025):**
| Model | Input | Output | Notes |
|-------|-------|--------|-------|
| `gpt-4o-mini` | $0.15/1M | $0.60/1M | **Default** - Good enough for classification |
| `gpt-4.1` | ~$2/1M | ~$8/1M | Better reasoning |
| `gpt-5.1` | $1.25/1M | $10/1M | Best reasoning, use for benchmarks |

**⚠️ Changing embedding models requires re-embedding all memories.** See [Re-embedding Guide](#re-embedding-memories) below.

### Enrichment Engine

Controls entity extraction and relationship linking.

| Variable | Description | Default |
|----------|-------------|---------|
| `ENRICHMENT_MAX_ATTEMPTS` | Max retry attempts | `3` |
| `ENRICHMENT_SIMILARITY_LIMIT` | Max similar memories to link | `5` |
| `ENRICHMENT_SIMILARITY_THRESHOLD` | Min similarity for linking | `0.8` |
| `ENRICHMENT_IDLE_SLEEP_SECONDS` | Sleep when queue empty | `2` |
| `ENRICHMENT_FAILURE_BACKOFF_SECONDS` | Backoff on failure | `5` |
| `ENRICHMENT_ENABLE_SUMMARIES` | Enable summarization | `true` |
| `ENRICHMENT_SPACY_MODEL` | spaCy model name | `en_core_web_sm` |

**Note**: Enrichment requires spaCy: `pip install spacy && python -m spacy download en_core_web_sm`

### Search Weights

Controls how different factors are weighted in memory recall.

| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `SEARCH_WEIGHT_VECTOR` | Semantic similarity | `0.35` | Vector search |
| `SEARCH_WEIGHT_KEYWORD` | Keyword matching | `0.35` | TF-IDF |
| `SEARCH_WEIGHT_TAG` | Tag matching | `0.15` | Exact tag match |
| `SEARCH_WEIGHT_IMPORTANCE` | Memory importance | `0.10` | User/system defined |
| `SEARCH_WEIGHT_CONFIDENCE` | Confidence score | `0.05` | Memory reliability |
| `SEARCH_WEIGHT_RECENCY` | Recent memories | `0.10` | Time-based boost |
| `SEARCH_WEIGHT_EXACT` | Exact phrase match | `0.15` | Full text match |

These act as **relative weights** in the scoring formula. Keeping them roughly normalized (summing to ~1.0) is recommended for interpretability, but the service does not auto-normalize them.

### Recall Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `RECALL_RELATION_LIMIT` | Max related memories per seed in graph expansion | `5` |
| `RECALL_EXPANSION_LIMIT` | Total max expansion results (relations + entities) | `25` |

---

## Railway Auto-Populated

Railway automatically injects these variables in production. **Do not set manually.**

| Variable | Description | Example |
|----------|-------------|---------|
| `RAILWAY_PUBLIC_DOMAIN` | Public app URL | `automem.up.railway.app` |
| `RAILWAY_PRIVATE_DOMAIN` | Internal service URL | `automem.railway.internal` |
| `RAILWAY_ENVIRONMENT` | Environment name | `production` |
| `RAILWAY_PROJECT_ID` | Project UUID | `abc123...` |
| `RAILWAY_SERVICE_ID` | Service UUID | `def456...` |

**Usage in AutoMem**: `app.py` falls back to `RAILWAY_PRIVATE_DOMAIN` if `FALKORDB_HOST` not set.

**Railway Networking Notes**:
- Railway's internal networking uses **IPv6**. AutoMem binds to `::` (IPv6 dual-stack) to accept connections from other services.
- `RAILWAY_PRIVATE_DOMAIN` resolves to IPv6 addresses (e.g., `fd12:ca03:42be:0:1000:50:1079:5b6c`).
- This is handled automatically - no configuration needed.

---

## Testing Only

These variables are only used by test suites.

| Variable | Description | Default |
|----------|-------------|---------|
| `AUTOMEM_RUN_INTEGRATION_TESTS` | Enable integration tests | `0` |
| `AUTOMEM_START_DOCKER` | Auto-start Docker in tests | `0` |

---

## Variable Priority & Fallbacks

AutoMem loads environment variables from multiple sources with this priority:

1. **Process environment** (highest priority)
2. **`.env` in project root**
3. **`~/.config/automem/.env`** (global config)
4. **Defaults in code** (lowest priority)

### Example Fallback Chain

```python
# FalkorDB host resolution
FALKORDB_HOST = (
    os.getenv("FALKORDB_HOST")                  # 1. Explicit setting
    or os.getenv("RAILWAY_PRIVATE_DOMAIN")      # 2. Railway internal domain
    or os.getenv("RAILWAY_PUBLIC_DOMAIN")       # 3. Railway public domain
    or "localhost"                              # 4. Default
)
```

---

## Security Best Practices

### ✅ Do

- Use Railway's secret generation for tokens
- Rotate `AUTOMEM_API_TOKEN` and `ADMIN_API_TOKEN` regularly
- Keep `.env` out of version control (already in `.gitignore`)
- Use Railway's private domains for service-to-service communication
- Set `FALKORDB_PASSWORD` in production

### ❌ Don't

- Commit `.env` to Git
- Share API tokens in public channels
- Use weak passwords for `FALKORDB_PASSWORD`
- Expose FalkorDB publicly (use `RAILWAY_PRIVATE_DOMAIN`)
- Hardcode credentials in code

---

## Troubleshooting

### "FalkorDB connection failed"

**Check**:
1. `FALKORDB_HOST` is correct (Railway: use `${{FalkorDB.RAILWAY_PRIVATE_DOMAIN}}`)
2. `FALKORDB_PORT` matches service port
3. `FALKORDB_PASSWORD` matches FalkorDB's `REDIS_PASSWORD`
4. FalkorDB service is running and healthy

### "Qdrant is not available"

**Check**:
1. `QDRANT_URL` is reachable
2. `QDRANT_API_KEY` is correct (if using Qdrant Cloud)
3. Collection exists: `curl $QDRANT_URL/collections/memories`

**Note**: AutoMem works without Qdrant (graph-only mode) but semantic search is disabled.

### "401 Unauthorized"

**Check**:
1. `AUTOMEM_API_TOKEN` is set and matches request token
2. Token is passed correctly: `Authorization: Bearer $TOKEN`
3. For admin endpoints: `X-Admin-Token` header also required

---

## Migration Guide

### From Old Variable Names

| Old Name | New Name | Status |
|----------|----------|--------|
| `MCP_MEMORY_HTTP_ENDPOINT` | `AUTOMEM_API_URL` | Deprecated, use new name |
| `MCP_MEMORY_AUTO_DISCOVER` | - | Removed (unused) |
| `DEVELOPMENT` | - | Removed (unused) |

**Backward compatibility**: Old names still work but will show deprecation warnings.

---

## Re-embedding Memories

When changing `EMBEDDING_MODEL` or `VECTOR_SIZE`, you must re-embed all existing memories:

### 1. Backup First
```bash
python scripts/backup_automem.py
```

### 2. Set New Environment Variables
```bash
export EMBEDDING_MODEL=text-embedding-3-large  # or text-embedding-3-small
export VECTOR_SIZE=3072  # 3072 for large, 768 for small
```

### 3. Recreate Qdrant Collection
The collection must be recreated with the new dimension:

```bash
# Delete old collection
curl -X DELETE "$QDRANT_URL/collections/memories" \
  -H "api-key: $QDRANT_API_KEY"

# Create new collection with correct dimension
curl -X PUT "$QDRANT_URL/collections/memories" \
  -H "api-key: $QDRANT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"vectors": {"size": 3072, "distance": "Cosine"}}'  # Use 768 for small
```

### 4. Re-embed All Memories
```bash
python scripts/reembed_embeddings.py --batch-size 32
```

The script reads from FalkorDB (which retains all memory data) and re-creates the vector embeddings in Qdrant.

**Cost estimate:** ~$0.15 for 6000 memories with large embeddings, ~$0.02 with small.

---

## See Also

- [Railway Deployment Guide](./RAILWAY_DEPLOYMENT.md)
- [Deployment Checklist](./DEPLOYMENT_CHECKLIST.md)
- [Installation Guide](../INSTALLATION.md)
