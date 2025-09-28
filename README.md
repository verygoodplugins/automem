# AutoMem

AutoMem is a Flask service that gives AI assistants durable memory. It keeps a
canonical record in FalkorDB (graph) and mirrors semantic vectors in Qdrant so
callers can mix structured lookups, relationship traversal, and semantic recall.

## Features

- REST API for the full memory lifecycle: store, recall (hybrid vector/keyword
  search with metadata scoring), filter by time and tags, update, delete, and
  create graph associations.
- FalkorDB powers rich relationships and consolidation workflows (decay,
  creative association discovery, clustering, controlled forgetting).
- Background enrichment pipeline extracts entities, writes summaries, links
  temporal/semantic neighbors, and queues retries with health reporting.
- Qdrant integration for semantic recall. The service falls back gracefully when
  Qdrant is unavailable and can regenerate embeddings on demand.
- Deterministic placeholder embeddings when none are supplied, making local
testing easy.
- Built-in admin endpoint to re-embed existing data—no more manual tunnelling to
  the database.
- Containerised development environment with FalkorDB, Qdrant, and the API.

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (for the bundle stack)

### Local development

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

# Spin up FalkorDB + Qdrant + API
make dev
```

The API listens on `http://localhost:8001`, FalkorDB on `6379`, and Qdrant on
`6333`.

Optional: install spaCy for richer entity extraction inside the enrichment
pipeline:

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

### Bare API (no Docker)

```bash
source venv/bin/activate
PORT=8001 python app.py
```

Set `FALKORDB_HOST` / `FALKORDB_PORT` so the service can reach the graph. Qdrant
params are optional if you only want graph operations.

### MCP bridge

```bash
MCP_MEMORY_HTTP_ENDPOINT=http://127.0.0.1:8001 \
  node scripts/http-bridge.js
```

Or point `MCP_MEMORY_HTTP_ENDPOINT` at your hosted deployment. The bridge exposes
synchronised tools for AutoMem (store/recall/update/delete/etc.).

## Documentation

- `docs/ONBOARDING.md` – project overview, environment setup, and service access
- `docs/archive/` – historical documents (PKG feature deep dives, architecture notes, ideas)

## Authentication

Set an API token on the server:

```bash
export AUTOMEM_API_TOKEN=super-secret
```

Provide it in one of the following ways (required for every endpoint except
`/health`):

- `Authorization: Bearer <token>` header (preferred)
- `X-API-Key: <token>` header
- `?api_key=<token>` query parameter (fallback)

Administrative operations (e.g. re-embedding) require an additional
`ADMIN_API_TOKEN` supplied via the `X-Admin-Token` header.

## API Overview

### Store

`POST /memory`

```json
{
  "content": "Finished integrating FalkorDB",
  "tags": ["deployment", "success"],
  "importance": 0.9,
  "metadata": {
    "source": "slack",
    "entities": { "people": ["vikas singhal"] }
  },
  "timestamp": "2025-09-16T12:37:21Z",
  "embedding": [0.12, 0.56, ...]  // optional
}
```

Returns `201 Created` with a deterministic ID. When Qdrant is configured but no
embedding is supplied, the service generates a temporary placeholder.

### Recall

`GET /recall`

Query parameters:

- `query`: full-text search string.
- `embedding`: comma-separated 768-d vector for direct semantic lookup.
- `limit`: number of results (default 5, max 50).
- `time_query`: human phrases (`today`, `yesterday`, `last week`, `last 7 days`,
  `this month`, etc.).
- `start`, `end`: explicit ISO timestamps (override `time_query`).
- `tags`: one or more tags (pass multiple `tags` params or a comma-separated
  value).

Responses include merged vector/keyword hits with scoring details:

```json
{
  "status": "success",
  "results": [
    {
      "id": "...",
      "match_type": "vector",
      "final_score": 0.82,
      "score_components": {
        "vector": 0.64,
        "tag": 0.50,
        "recency": 0.90,
        "exact": 1.00
      },
      "memory": { ... }
    }
  ],
  "time_window": { "start": "2025-09-01T00:00:00+00:00", "end": "2025-09-30T23:59:59+00:00" },
  "tags": ["vikas singhal"],
  "count": 5
}
```

### Update

`PATCH /memory/<id>` mirrors the POST payload (content, tags, importance,
metadata, timestamps, etc.). New embeddings are generated automatically when
the content changes.

### Delete

`DELETE /memory/<id>` removes the record from FalkorDB and its vector from
Qdrant.

### Filter by tag

`GET /memory/by-tag?tags=foo&tags=bar&limit=50` returns the most recent/important
memories matching any requested tag.

### Create an association

`POST /associate`

```json
{
  "memory1_id": "uuid-of-source",
  "memory2_id": "uuid-of-target",
  "type": "RELATES_TO",
  "strength": 0.8
}
```

Associations prevent self-links and validate relationship types.

### Admin: re-embed

`POST /admin/reembed`

```bash
curl -X POST https://.../admin/reembed \
  -H 'Authorization: Bearer ${AUTOMEM_API_TOKEN}' \
  -H 'X-Admin-Token: ${ADMIN_API_TOKEN}' \
  -d '{"batch_size": 32, "limit": 100}'
```

Regenerates embeddings in controlled batches—perfect for migrations.

## Enrichment Pipeline

The enrichment worker runs asynchronously to augment each memory after it is
stored:

- Extracts entities (tools, projects, people, organisations, concepts) and
  stores them in metadata while tagging memories (`entity:<type>:<slug>`).
- Generates lightweight summaries (first-sentence snippets) and timestamps the
  enrichment run.
- Adds temporal links (`PRECEDED_BY`) to the most recent predecessors, keeping a
  `count` of reinforcement.
- Detects emerging patterns for each memory type and strengthens shared
  `Pattern` nodes with key terms.
- Uses Qdrant similarity search to connect close neighbours via
  `SIMILAR_TO` relationships (symmetric) with stored cosine scores.
- Exposes metrics and queue health at `GET /enrichment/status`, including
  pending/in-flight counts and last success/error details.
- Supports forced reprocessing through `POST /enrichment/reprocess` (requires an
  admin token).

Install `spacy` and an English model (e.g. `en_core_web_sm`) to unlock richer
entity extraction, or rely on the built-in heuristics.

## Consolidation Engine

AutoMem runs a background consolidator that keeps memories fresh even when the
API is idle. The scheduler (see `consolidation.py`) cycles through four
processes:

- **Decay** (hourly by default) recalculates relevance scores with finer-grained
  exponential decay so short-lived activity still nudges scores.
- **Creative** (hourly) samples mid/high relevance memories and adds
  `DISCOVERED` edges for surprising pairings.
- **Cluster** (every 6 hours) groups similar embeddings and can emit
  `MetaMemory` summaries for large clusters.
- **Forget** (daily) archives or deletes low-relevance memories while keeping
  Qdrant and FalkorDB in sync.

You can override cadences with environment variables such as
`CONSOLIDATION_DECAY_INTERVAL_SECONDS` or switch attendance filters via
`CONSOLIDATION_DECAY_IMPORTANCE_THRESHOLD` (defaults to `0.3`, set empty to
process everything). The decay pass now uses fractional days when scoring, so
frequent runs produce incremental but meaningful updates.

## Configuration

| Variable | Description | Default |
| --- | --- | --- |
| `PORT` | API port | `8001` |
| `FALKORDB_HOST` | FalkorDB host | `localhost` |
| `FALKORDB_PORT` | FalkorDB port | `6379` |
| `FALKORDB_GRAPH` | Graph name | `memories` |
| `QDRANT_URL` | Qdrant API URL | _unset_ |
| `QDRANT_API_KEY` | Qdrant API key | _optional_ |
| `QDRANT_COLLECTION` | Qdrant collection name | `memories` |
| `VECTOR_SIZE` | Embedding vector size | `768` |
| `AUTOMEM_API_TOKEN` | Required API token | _unset_ |
| `ADMIN_API_TOKEN` | Token for `/admin/reembed` | _unset_ |
| `CONSOLIDATION_DECAY_INTERVAL_SECONDS` | Cadence for decay run (seconds) | `3600` |
| `CONSOLIDATION_DECAY_IMPORTANCE_THRESHOLD` | Minimum importance to include (blank = all) | `0.3` |
| `CONSOLIDATION_CREATIVE_INTERVAL_SECONDS` | Creative association interval | `3600` |
| `CONSOLIDATION_CLUSTER_INTERVAL_SECONDS` | Clustering interval | `21600` |
| `CONSOLIDATION_FORGET_INTERVAL_SECONDS` | Forgetting interval | `86400` |
| `ENRICHMENT_MAX_ATTEMPTS` | Automatic retry limit for failed enrichments | `3` |
| `ENRICHMENT_SIMILARITY_LIMIT` | Number of semantic neighbours to consider | `5` |
| `ENRICHMENT_SIMILARITY_THRESHOLD` | Minimum cosine score for `SIMILAR_TO` links | `0.8` |
| `ENRICHMENT_IDLE_SLEEP_SECONDS` | Worker sleep when queue is empty | `2` |
| `ENRICHMENT_FAILURE_BACKOFF_SECONDS` | Backoff between retry attempts | `5` |
| `ENRICHMENT_ENABLE_SUMMARIES` | Enable automatic summary generation | `true` |
| `ENRICHMENT_SPACY_MODEL` | spaCy model name for entity extraction | `en_core_web_sm` |
| `RECALL_RELATION_LIMIT` | Max related memories returned per result | `5` |
| `SEARCH_WEIGHT_*` | Optional scoring weights (vector, keyword, tag, etc.) | see app defaults |

The application loads environment variables from the process, `.env` in the
project root, and `~/.config/automem/.env`.

## Deployment

### Railway (recommended)

AutoMem is designed to run as two services inside a Railway project: this Flask
API, and a dedicated FalkorDB instance with persistent storage. Qdrant can stay
on Qdrant Cloud (provide `QDRANT_URL`/`QDRANT_API_KEY`) or be omitted if you only
need graph recall.

#### 1. Prerequisites

- Install the Railway CLI: `npm i -g @railway/cli`
- Log in: `railway login`
- (Optional) initialise the project from this repo: `railway init`

#### 2. Provision FalkorDB

1. Create a new Railway service using the `falkordb/falkordb:latest` image.
2. Attach a persistent volume so the graph survives restarts.
3. Note the internal host/port Railway assigns (shown in the service settings).
   If you enable password auth, keep the password handy—you will pass it to the
   AutoMem service.

**FalkorDB environment variable**

| Variable | Description |
| --- | --- |
| `REDIS_PASSWORD` | *(optional)* password for the graph database |

Railway automatically exposes `REDIS_HOST`, `REDIS_PORT`, and `REDIS_PASSWORD`
for other services; you can reference them via `${{service.<name>.internalHost}}`
when configuring AutoMem.

#### 3. Deploy AutoMem

1. From this repo run `railway up` (or connect the repo in the Railway UI with
   auto-deploys). The Dockerfile already installs spaCy and the
   `en_core_web_sm` model.
2. Configure the following variables on the AutoMem service:

| Variable | Description |
| --- | --- |
| `AUTOMEM_API_TOKEN` | Required auth token for all client calls |
| `ADMIN_API_TOKEN` | Required for admin/enrichment endpoints |
| `OPENAI_API_KEY` | Enables real embeddings (otherwise deterministic placeholders) |
| `FALKORDB_HOST` | Internal hostname of the FalkorDB service |
| `FALKORDB_PORT` | Port (usually `6379`) |
| `FALKORDB_PASSWORD` | *(optional)* only if you set one on FalkorDB |
| `QDRANT_URL` | *(optional)* Qdrant Cloud endpoint |
| `QDRANT_API_KEY` | *(optional)* Qdrant API key |
| `CONSOLIDATION_*`, `ENRICHMENT_*` | *(optional)* override defaults listed earlier |

3. Redeploy, then verify health:

```bash
curl https://<your-app>.up.railway.app/health
```

Expect `{ "status": "healthy" }`. A `503` typically means the API cannot reach
FalkorDB—double-check host/port/password values.

#### 4. Seed and reprocess (optional)

- Store a memory to confirm writes work:

  ```bash
  curl -X POST https://<your-app>.up.railway.app/memory \
    -H "Authorization: Bearer $AUTOMEM_API_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"content":"First memory from Railway","importance":0.7}'
  ```

- Re-enqueue existing memories now that spaCy is available:

  ```bash
  curl -X POST https://<your-app>.up.railway.app/enrichment/reprocess \
    -H "Authorization: Bearer $AUTOMEM_API_TOKEN" \
    -H "X-Admin-Token: $ADMIN_API_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"ids":["memory-id-1","memory-id-2"]}'
  ```

  Track progress with `GET /enrichment/status`.

#### 5. Local vs production

Locally you can run `make dev` (Docker Compose) or point the app at the remote
FalkorDB/Qdrant by setting the same environment variables. In production keep
AutoMem, FalkorDB (and optionally Qdrant) as separate services so rolling
deploys and scaling do not interrupt the database.

## Troubleshooting

- `401 Unauthorized`: ensure `AUTOMEM_API_TOKEN` matches the client’s
  token/header.
- `503 FalkorDB is unavailable`: confirm the graph is reachable at `FALKORDB_HOST`.
- `Embedding must contain exactly 768 values`: supply the full vector or omit the
  field to let the placeholder generate.
- Qdrant errors are logged but do not block FalkorDB writes; inspect application
  logs to diagnose failures.

## License

MIT License
