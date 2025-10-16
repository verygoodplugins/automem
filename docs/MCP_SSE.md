# MCP over SSE Sidecar (Railway)

This sidecar exposes AutoMem as an MCP server over SSE so cloud clients like ChatGPT and ElevenLabs Agents can connect via HTTPS and use your memories.

Service endpoint (on Railway):
- GET `/mcp/sse` — SSE stream (server → client). Include `Authorization: Bearer <AUTOMEM_API_TOKEN>`.
- POST `/mcp/messages?sessionId=<id>` — Client → server JSON-RPC messages.
- GET `/health` — Health probe.

Auth model:
- Use your AutoMem API token as the Bearer token for the SSE connection. The sidecar forwards this token to the AutoMem API for tool calls.

Supported tools:
- `store_memory`, `recall_memory`, `associate_memories`, `update_memory`, `delete_memory`, `check_database_health`

Deploy (one‑click template):
- The template adds a new service `automem-mcp-sse` alongside `automem-api` and `FalkorDB`.
- It preconfigures `AUTOMEM_ENDPOINT` to the internal URL of `automem-api`.

Client setup summary:
- ChatGPT MCP: Configure server URL `https://<your-mcp-domain>/mcp/sse` with the same Bearer token you use for AutoMem.
- ElevenLabs Agents: Point the agent’s MCP to the same SSE URL + token.

Notes:
- Keepalive heartbeats are sent every 20s to prevent idle timeouts.
- Rate limiting and multi-tenant token scoping can be added in front of this service if needed.

