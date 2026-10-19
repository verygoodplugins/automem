# AutoMem MCP SSE Server

Express service that bridges the AutoMem HTTP API to MCP over SSE.

## Endpoints
- `GET /mcp/sse` and `POST /mcp/messages`: MCP over SSE (tools map to AutoMem HTTP API).
- `GET /health`: Basic health probe.

## Env Vars
- `AUTOMEM_API_URL` (default `http://127.0.0.1:8001`) – AutoMem HTTP base URL.
- `AUTOMEM_API_TOKEN` – Bearer token for AutoMem HTTP calls (required for MCP).
- `PORT` (optional) – Listener port (default 8080).

> **Note**: `AUTOMEM_ENDPOINT` is still supported as a legacy fallback but `AUTOMEM_API_URL` is preferred.
