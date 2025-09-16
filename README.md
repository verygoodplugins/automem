# AutoMem

AutoMem is a small Flask service that gives AI assistants a durable memory. It
persists structured memory records in FalkorDB (graph) and Qdrant (vector
search) so downstream tools can store information, discover related memories and
link them together.

## Features

- REST API with three endpoints: store a memory, recall memories, create
  associations between memories.
- FalkorDB graph storage for canonical memory records and relationships.
- Optional Qdrant integration for semantic recall (skips gracefully when not
  configured).
- Deterministic placeholder embeddings when vectors are not provided, making it
  possible to test the API without an embedding service.
- Containerised development environment with FalkorDB, Qdrant and the API.
- Automated tests for request validation and association handling.

## Quick Start

### Prerequisites

- Python 3.10+ (for local development commands)
- Docker and Docker Compose

### Local development

```bash
# Clone and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

# Run the full stack (FalkorDB + Qdrant + API)
make dev
```

The API starts on `http://localhost:8001`, FalkorDB is available on
`localhost:6379`, and Qdrant on `localhost:6333`.

### Running the API without Docker

```bash
source venv/bin/activate
PORT=8001 python app.py
```

Make sure FalkorDB is reachable at the host defined by `FALKORDB_HOST`. Qdrant
is optional when you just want to exercise the graph API.

### Tests

```bash
make test
```

> **Note**
> Some global pytest plugins conflict with the pinned pytest version. The
> `make test` target sets `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` to ensure the local
> suite runs in isolation. If you invoke `pytest` directly, export the same
> environment variable.

### MCP bridge

To exercise the API through an MCP client (Codex, Cursor, etc.) use the
provided bridge script:

```bash
# Local API
MCP_MEMORY_HTTP_ENDPOINT=http://127.0.0.1:8001 \
  node scripts/http-bridge.js

# Hosted Railway deployment
MCP_MEMORY_HTTP_ENDPOINT=https://automem.up.railway.app \
  node scripts/http-bridge.js
```

The bridge exposes three tools today: `store_memory`, `recall_memory`, and
`check_database_health`. Point an MCP inspector at the command to invoke and
test the endpoints.

## API Overview

### Store a memory

`POST /memory`

```json
{
  "content": "Finished integrating FalkorDB",
  "tags": ["deployment", "success"],
  "importance": 0.9,
  "embedding": [0.12, 0.56, ...]  // optional
}
```

- Returns `201 Created` with the memory ID.
- If Qdrant is configured but no embedding is provided, a deterministic
  placeholder vector is generated so subsequent recall works consistently.
- When Qdrant is not configured the graph write still succeeds and the
  response indicates that the vector store was skipped.

### Recall memories

`GET /recall`

Query parameters:

- `query`: full-text search on FalkorDB memories.
- `embedding`: comma-separated 768-d vector for Qdrant semantic search.
- `limit`: maximum number of results (default 5, max 50).

The response merges vector hits (when available) with graph matches and includes
any related memories discovered via graph traversal.

### Create an association

`POST /associate`

```json
{
  "memory1_id": "uuid-of-source",
  "memory2_id": "uuid-of-target",
  "type": "RELATES_TO",  // one of RELATES_TO, LEADS_TO, OCCURRED_BEFORE
  "strength": 0.8          // 0.0 – 1.0
}
```

Associations are validated to prevent self-links and unexpected relationship
types. A `404` is returned if either memory is missing.

## Configuration

Environment variables recognised by the service:

| Variable | Description | Default |
| --- | --- | --- |
| `PORT` | API port | `8001` |
| `FALKORDB_HOST` | Hostname for FalkorDB | `localhost` |
| `FALKORDB_PORT` | TCP port for FalkorDB | `6379` |
| `FALKORDB_GRAPH` | Graph name | `memories` |
| `QDRANT_URL` | Base URL for Qdrant API | _unset_ (disables Qdrant) |
| `QDRANT_API_KEY` | API key for Qdrant Cloud | _optional_ |
| `QDRANT_COLLECTION` | Qdrant collection name | `memories` |
| `VECTOR_SIZE` | Embedding vector size | `768` |

For local work you can create `~/.config/automem/.env` (or similar); the
application automatically loads that file in addition to a project-level `.env`
if present. A checked-in `.env.example` with placeholders is recommended for
teams.

## Deployment

The included `Dockerfile` builds a slim Python image for the API. Deploy it
alongside managed FalkorDB/Qdrant services, or use the provided Docker Compose
stack for self-hosting.

Railway deployment is supported out of the box: point Railway to this repository
and it will build using the Dockerfile. Configure the environment variables in
Railway to connect to your FalkorDB and Qdrant instances.

## Roadmap / Ideas

- Integrate a real embedding generator (OpenAI, local model, etc.).
- Scheduled consolidation / pruning jobs for the graph.
- Authentication, rate limiting and privacy tooling.
- Richer analytics endpoints (importance trends, recent recalls, etc.).

## Troubleshooting

- `503 FalkorDB is unavailable`: ensure the FalkorDB container is running and
  reachable at the host configured by `FALKORDB_HOST`.
- `Embedding must contain exactly 768 values`: either supply the full embedding
  vector or omit the field so the placeholder generator runs.
- Qdrant errors are logged but do not stop FalkorDB writes; inspect the service
  logs for the exact failure reason.

## License

MIT License
