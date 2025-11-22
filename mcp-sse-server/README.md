# AutoMem MCP SSE Server

Express service that bridges the AutoMem HTTP API to MCP over SSE and now exposes a lightweight Alexa skill endpoint.

## Endpoints
- `GET /mcp/sse` and `POST /mcp/messages`: MCP over SSE (tools map to AutoMem HTTP API).
- `POST /alexa`: Alexa Custom skill hook; supports `RememberIntent` (store) and `RecallIntent` (recall).
- `GET /health`: Basic health probe.

## Env Vars
- `AUTOMEM_ENDPOINT` (default `http://127.0.0.1:8001`) – AutoMem HTTP base URL.
- `AUTOMEM_API_TOKEN` – Bearer token for AutoMem HTTP calls (required for Alexa and MCP).
- `PORT` (optional) – Listener port (default 8080).

## Alexa Notes
- Alexa cannot send custom headers; keep the AutoMem token in `AUTOMEM_API_TOKEN`.
- Sample utterances (Custom model): `remember {note}`, `store {note}`, `recall {query}`, `what do you remember about {query}`.
- Tags applied automatically: `alexa`, plus `user:{userId}` and `device:{deviceId}` when present in the request.
