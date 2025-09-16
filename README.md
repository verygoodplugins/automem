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
| `SEARCH_WEIGHT_*` | Optional scoring weights (vector, keyword, tag, etc.) | see app defaults |

The application loads environment variables from the process, `.env` in the
project root, and `~/.config/automem/.env`.

## Deployment

Use the provided Dockerfile or Compose stack. Railway works out of the box—set
the environment variables in the dashboard to point at your FalkorDB and Qdrant
instances, plus `AUTOMEM_API_TOKEN` and `ADMIN_API_TOKEN`.

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
