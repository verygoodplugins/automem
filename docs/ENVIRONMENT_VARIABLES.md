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

AutoMem supports five embedding backends with automatic fallback.

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `EMBEDDING_PROVIDER` | Embedding backend selection | `auto` | `auto`, `voyage`, `openai`, `local`, `ollama`, `placeholder` |
| `VOYAGE_API_KEY` | Voyage API key (for Voyage provider) | - | `pa-...` |
| `VOYAGE_MODEL` | Voyage embedding model | `voyage-4` | `voyage-4`, `voyage-4-large`, `voyage-4-lite` |
| `OPENAI_API_KEY` | API key (OpenAI or compatible provider) | - | `sk-proj-...` |
| `OPENAI_BASE_URL` | Custom base URL for OpenAI-compatible APIs | - | `https://openrouter.ai/api/v1` |
| `OLLAMA_BASE_URL` | Ollama API base URL | `http://localhost:11434` | `http://localhost:11434` |
| `OLLAMA_MODEL` | Ollama embedding model | `nomic-embed-text` | `nomic-embed-text` |
| `OLLAMA_TIMEOUT` | Ollama request timeout (seconds) | `30` | `10`, `30`, `60` |
| `OLLAMA_MAX_RETRIES` | Ollama retry count | `2` | `0`, `1`, `2` |

**Provider Options:**
- `auto` (default): Try Voyage → OpenAI → Ollama (if configured) → FastEmbed local model → Placeholder
- `voyage`: Use Voyage API only (requires `VOYAGE_API_KEY`)
- `openai`: Use OpenAI API only (requires `OPENAI_API_KEY`)
- `local`: Use FastEmbed local model only (~210MB download on first use)
- `ollama`: Use Ollama only (requires `OLLAMA_BASE_URL` and a pulled model)
- `placeholder`: Use hash-based embeddings (no semantic search)

**Local Model Details:**
- Models: `BAAI/bge-small-en-v1.5` (384d), `BAAI/bge-base-en-v1.5` (768d), `BAAI/bge-large-en-v1.5` (1024d)
- Size: ~67MB / ~210MB / ~1.2GB (cached to `~/.config/automem/models/`)
- No API key or internet required after first download
- Good semantic quality, faster than API calls
- Recommended: pin `onnxruntime<1.20` with `fastembed` 0.4.x to avoid runtime issues
- Docker: persist model cache with a volume (see docker-compose.yml)
- Ensure `VECTOR_SIZE` equals the selected model's dimension (default 768)

**Ollama Details:**
- Requires a running Ollama server (default `http://localhost:11434`)
- Pull the embedding model before use: `ollama pull nomic-embed-text`
- Embedding dimensions vary by model; set `VECTOR_SIZE` to match the model output
- For `auto` mode, Ollama is only attempted if `OLLAMA_BASE_URL` or `OLLAMA_MODEL` is set

**Voyage Details:**
- Requires `VOYAGE_API_KEY`
- Optional model override via `VOYAGE_MODEL` (default `voyage-4`)
- Voyage 4 family supports output dimensions `256`, `512`, `1024`, or `2048`
- Set `VECTOR_SIZE` to one of the supported Voyage dimensions

**OpenAI-Compatible Providers (OpenRouter, LiteLLM, Azure, vLLM, etc.):**

The `openai` provider works with any service that exposes an OpenAI-compatible `/v1/embeddings` endpoint. Set `OPENAI_BASE_URL` to point at the provider and `OPENAI_API_KEY` to your provider's key:

```bash
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-or-v1-your-openrouter-key
OPENAI_BASE_URL=https://openrouter.ai/api/v1
EMBEDDING_MODEL=openai/text-embedding-3-small   # model name varies by provider
VECTOR_SIZE=768                                  # must match the model's output dimension
```

**Important notes for non-OpenAI providers:**
- The `dimensions` parameter (which lets OpenAI truncate embeddings) is **only sent to OpenAI's own API** (`api.openai.com`). For all other base URLs it is omitted because most compatible providers don't support it.
- You **must** set `VECTOR_SIZE` to match the model's native output dimension since dimension truncation is unavailable.
- Model names may differ per provider (e.g. `openai/text-embedding-3-small` on OpenRouter vs `text-embedding-3-small` on OpenAI).

**Additional Configuration:**
| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `AUTOMEM_MODELS_DIR` | Override model cache directory | `~/.config/automem/models/` | Useful in containers |

---

## Embedding Provider Tradeoffs

| Provider | Cost | Pros | Cons | Best for |
|----------|------|------|------|----------|
| Voyage | API usage | Strong quality, flexible model family, simple API | Requires outbound network and matching `VECTOR_SIZE` | Teams using Voyage for quality/cost balance |
| OpenAI | ~$0.13/1M tokens (large), ~$0.02/1M tokens (small) | Highest semantic quality, zero infra | Recurring API cost, outbound network required | Production accuracy with minimal ops |
| FastEmbed (local) | Hardware-only (CPU/GPU) | Offline after first download, consistent latency | Model download size, quality tied to model size | Self-hosted + cost-sensitive environments |
| Ollama | Hardware-only (CPU/GPU) | Fully local, easy model swapping | Requires running Ollama service, model dims vary | Self-hosted deployments with Ollama already in stack |
| Placeholder | Free | Always available, deterministic | No semantic search quality | Development/testing without vectors |

**Dimension matching tip:** If embeddings fail with a size mismatch, confirm your model's output length and set `VECTOR_SIZE` accordingly (or use `VECTOR_SIZE_AUTODETECT=true` if you accept adopting the existing collection size).

## Hosting Considerations (Railway vs Self-Hosted)

- **Railway / managed PaaS:** Voyage and OpenAI are the simplest choices (no local model downloads). FastEmbed works but increases image size and cold-start time; use a persistent volume for `AUTOMEM_MODELS_DIR` if supported. Ollama typically requires a **separate service** (Railway does not ship Ollama by default), so you'll need to deploy Ollama elsewhere and set `OLLAMA_BASE_URL` to that service.
- **Self-hosted Docker/VPS:** FastEmbed and Ollama are straightforward and avoid API costs. Ollama benefits from GPU acceleration if available; otherwise expect higher latency on CPU. Ensure the Ollama base URL is reachable from the AutoMem container (`OLLAMA_BASE_URL=http://ollama:11434` in docker-compose setups).
- **Dimension consistency:** Regardless of host, make sure `VECTOR_SIZE` matches the embedding model output. Changing models requires re-embedding existing memories.

## Optional Variables

### Qdrant (Vector Database)

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `QDRANT_URL` | Qdrant endpoint URL | `http://localhost:6333` | `https://xyz.qdrant.io` |
| `QDRANT_API_KEY` | Qdrant API key | - | `your-qdrant-key` |
| `QDRANT_COLLECTION` | Collection name | `memories` | `memories` |
| `VECTOR_SIZE` | Embedding dimension | `3072` | `3072` (large), `768` (small) |
| `VECTOR_SIZE_AUTODETECT` | Adopt existing collection dimension instead of failing on mismatch | `false` | `true` |

👉 **New to Qdrant?** See the [Qdrant Setup Guide](QDRANT_SETUP.md) for step-by-step instructions on creating a collection with the right settings.

**Notes**:
- Without Qdrant, AutoMem uses deterministic placeholder embeddings (for testing only).
- **Existing deployments on 768d**: set `VECTOR_SIZE=768` (and `EMBEDDING_MODEL=text-embedding-3-small`) until you run the migration script.
- By default the service fails fast if the configured vector size does not match the Qdrant collection to prevent silent corruption.
- To opt into legacy auto-detection (use the existing collection dimension), set `VECTOR_SIZE_AUTODETECT=true`.

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
| `CONSOLIDATION_DECAY_INTERVAL_SECONDS` | Decay check interval | `86400` | seconds |
| `CONSOLIDATION_CREATIVE_INTERVAL_SECONDS` | Pattern detection interval | `604800` | seconds |
| `CONSOLIDATION_CLUSTER_INTERVAL_SECONDS` | Clustering interval | `2592000` | seconds |
| `CONSOLIDATION_FORGET_INTERVAL_SECONDS` | Forget interval (`0` disables) | `0` | seconds |
| `CONSOLIDATION_DECAY_IMPORTANCE_THRESHOLD` | Min importance to keep | `0.3` | 0-1 |
| `CONSOLIDATION_HISTORY_LIMIT` | Max consolidation history | `20` | count |
| `CONSOLIDATION_CONTROL_NODE_ID` | Control node identifier | `global` | string |
| `CONSOLIDATION_DELETE_THRESHOLD` | Delete threshold (`0` disables) | `0.0` | 0-1 |
| `CONSOLIDATION_ARCHIVE_THRESHOLD` | Archive threshold (`0` disables) | `0.0` | 0-1 |
| `CONSOLIDATION_GRACE_PERIOD_DAYS` | Protect recent memories | `90` | days |
| `CONSOLIDATION_IMPORTANCE_PROTECTION_THRESHOLD` | Protect high-importance memories | `0.7` | 0-1 |
| `CONSOLIDATION_PROTECTED_TYPES` | Comma-separated protected types | `Decision,Insight` | string |

### Model Configuration

Controls which OpenAI models are used for embeddings and classification.

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `EMBEDDING_MODEL` | OpenAI embedding model | `text-embedding-3-large` | `text-embedding-3-large` (3072d), `text-embedding-3-small` (768d) |
| `VECTOR_SIZE` | Embedding dimension | `3072` | Must match embedding model |
| `VECTOR_SIZE_AUTODETECT` | Adopt existing collection dimension instead of failing on mismatch | `false` | `true` |
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

**Upgrade safety**: Changing embedding dimensions requires a full re-embed. AutoMem refuses to start if `VECTOR_SIZE` does not match the existing Qdrant collection; set the value to your current dimension (usually `768`) before migrating, then switch to `3072` after running `scripts/reembed_embeddings.py`. To override strict mode and adopt the existing collection dimension, set `VECTOR_SIZE_AUTODETECT=true` (use only if you understand the risk of dimension drift).

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

Controls how different factors are weighted in memory recall scoring.

| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `SEARCH_WEIGHT_VECTOR` | Semantic similarity | `0.35` | Vector search via Qdrant |
| `SEARCH_WEIGHT_KEYWORD` | Keyword matching | `0.35` | TF-IDF style |
| `SEARCH_WEIGHT_RELATION` | Graph relationship boost | `0.25` | Memories connected via edges |
| `SEARCH_WEIGHT_TAG` | Tag matching | `0.20` | Tag overlap scoring |
| `SEARCH_WEIGHT_EXACT` | Exact phrase match | `0.20` | Full query in metadata |
| `SEARCH_WEIGHT_IMPORTANCE` | Memory importance | `0.10` | User/system defined |
| `SEARCH_WEIGHT_RECENCY` | Recent memories | `0.10` | Linear decay over 180 days |
| `SEARCH_WEIGHT_CONFIDENCE` | Confidence score | `0.05` | Memory reliability |
| `SEARCH_WEIGHT_RELEVANCE` | Consolidation relevance | `0.0` | Decay-derived score (see below) |

These act as **relative weights** in the scoring formula. Keeping them roughly normalized (summing to ~1.0) is recommended for interpretability, but the service does not auto-normalize them.

**`SEARCH_WEIGHT_RELEVANCE` (new):** This weight incorporates `relevance_score`, a value maintained by the consolidation decay engine that reflects access patterns and age. It's synced to both FalkorDB and Qdrant payloads. Default is `0.0` (disabled) — set to e.g. `0.15` to boost frequently-accessed memories. Use the Recall Quality Lab to test different values before changing production.

### Memory Content Limits

Controls auto-summarization and content size validation on store.

| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `MEMORY_CONTENT_SOFT_LIMIT` | Char limit before auto-summarization triggers | `500` | Content above this is summarized |
| `MEMORY_CONTENT_HARD_LIMIT` | Char limit before rejection | `2000` | Content above this is rejected |
| `MEMORY_AUTO_SUMMARIZE` | Enable/disable auto-summarization | `true` | `false` stores as-is |
| `MEMORY_SUMMARY_TARGET_LENGTH` | Target length for summarized content | `300` | Characters |

### Sync Configuration

Background worker that checks FalkorDB ↔ Qdrant consistency.

| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `SYNC_CHECK_INTERVAL_SECONDS` | How often to check for drift | `3600` | 1 hour |
| `SYNC_AUTO_REPAIR` | Auto-fix inconsistencies | `true` | Set `false` for dry-run mode |

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
| `AUTOMEM_TEST_BASE_URL` | Base URL for Recall Quality Lab | `http://localhost:8001` |

### Recall Quality Lab

The `scripts/lab/` directory provides a data-driven framework for testing and optimizing recall scoring. It uses IR metrics (Recall@K, MRR, NDCG) and statistical comparison to evaluate config changes.

**Makefile targets:**

| Target | Description | Example |
|--------|-------------|---------|
| `make lab-clone` | Clone production data to local Docker | `make lab-clone` |
| `make lab-queries` | Generate test queries from local data | `make lab-queries` |
| `make lab-test` | Run recall test with a config | `make lab-test CONFIG=baseline` |
| `make lab-compare` | A/B compare two configs | `make lab-compare CONFIG=fix_v1 BASELINE=baseline` |
| `make lab-sweep` | Sweep a parameter across values | `make lab-sweep PARAM=SEARCH_WEIGHT_VECTOR VALUES=0.20,0.30,0.40,0.50` |

**Config files** live in `scripts/lab/configs/` as JSON. Each config maps env var names to values that override the server's search weights for that test run. See `baseline.json` for an example.

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
| `AUTOMEM_ENDPOINT` | `AUTOMEM_API_URL` | Deprecated, use new name |
| `MCP_MEMORY_HTTP_ENDPOINT` | `AUTOMEM_API_URL` | Deprecated, use new name |
| `MCP_MEMORY_AUTO_DISCOVER` | - | Removed (unused) |
| `DEVELOPMENT` | - | Removed (unused) |

**Backward compatibility**: Old names still work as fallbacks.

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
