# AutoMem MCP SSE Server

Express service that bridges the AutoMem HTTP API to MCP over SSE and now exposes a lightweight Alexa skill endpoint.

## Endpoints
- `GET /mcp/sse` and `POST /mcp/messages`: MCP over SSE (tools map to AutoMem HTTP API).
- `POST /alexa`: Alexa Custom skill hook; supports `RememberIntent` (store) and `RecallIntent` (recall).
- `GET /health`: Basic health probe.

## Env Vars
- `AUTOMEM_API_URL` (default `http://127.0.0.1:8001`) – AutoMem HTTP base URL.
- `AUTOMEM_API_TOKEN` – Bearer token for AutoMem HTTP calls (required for Alexa and MCP).
- `PORT` (optional) – Listener port (default 8080).

> **Note**: `AUTOMEM_ENDPOINT` is still supported as a legacy fallback but `AUTOMEM_API_URL` is preferred.

Overrides for testing (Alexa endpoint):
- `?endpoint=` query param or `endpoint` field in the POST body will override `AUTOMEM_API_URL`.
- `api_key` query param or `Authorization: Bearer ...` / `X-API-Key` header will override `AUTOMEM_API_TOKEN`.

## Alexa Notes
- Alexa cannot send custom headers; keep the AutoMem token in `AUTOMEM_API_TOKEN`.
- Sample utterances (Custom model): `remember {note}`, `store {note}`, `recall {query}`, `what do you remember about {query}`.
- Tags applied automatically: `alexa`, plus `user:{userId}` and `device:{deviceId}` when present in the request.
