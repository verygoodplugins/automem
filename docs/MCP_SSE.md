# MCP over SSE Sidecar (Railway)

This sidecar exposes AutoMem as an MCP server over SSE so cloud clients like ChatGPT and ElevenLabs Agents can connect via HTTPS and use your memories.

Service endpoint (on Railway):
- GET `/mcp/sse` — SSE stream (server → client). Include `Authorization: Bearer <AUTOMEM_API_TOKEN>`.
- POST `/mcp/messages?sessionId=<id>` — Client → server JSON-RPC messages.
- GET `/health` — Health probe.

Auth model:
- Preferred: `Authorization: Bearer <AUTOMEM_API_TOKEN>` header
- Supported: `X-API-Key: <AUTOMEM_API_TOKEN>` header
- Fallback (if platform can’t set headers): append `?api_key=<AUTOMEM_API_TOKEN>` to the SSE URL
  - Example: `https://<your-mcp-domain>/mcp/sse?api_key=...`
  - Note: URL tokens may appear in logs/proxy metadata; use headers when possible.

Supported tools:
- `store_memory`, `recall_memory`, `associate_memories`, `update_memory`, `delete_memory`, `check_database_health`

Deploy (one‑click template):
- The template adds a new service `automem-mcp-sse` alongside `memory-service` and `FalkorDB`.
- It preconfigures `AUTOMEM_ENDPOINT` to the internal URL of `memory-service`: `http://${memory-service.RAILWAY_PRIVATE_DOMAIN}:8001`.
- **Manual setup**: Use `AUTOMEM_ENDPOINT=http://memory-service.railway.internal:8001` (hardcoded internal DNS is more stable).
- **Important**: The internal DNS must match your memory service's `RAILWAY_PRIVATE_DOMAIN`. If you renamed the service, verify with `railway variables --service memory-service | grep RAILWAY_PRIVATE_DOMAIN`.

Client setup summary:
- ChatGPT MCP: Configure server URL `https://<your-mcp-domain>/mcp/sse` with the same Bearer token you use for AutoMem.
- ElevenLabs Agents: Point the agent's MCP to the same SSE URL + token.

Notes:
- Keepalive heartbeats are sent every 20s to prevent idle timeouts.
- Rate limiting and multi-tenant token scoping can be added in front of this service if needed.

Troubleshooting `fetch failed` errors:
1. **Check memory-service has `PORT=8001`** - Most common cause. Without it, Flask runs on wrong port.
2. **Verify `AUTOMEM_ENDPOINT`** - Should be `http://memory-service.railway.internal:8001` (or your service's actual `RAILWAY_PRIVATE_DOMAIN`).
3. **Check SSE logs** - Enable debug mode and check logs for actual error: `railway logs --service automem-mcp-sse`.
4. **Alternative**: Use public URL as fallback: `AUTOMEM_ENDPOINT=https://<your-memory-service-domain>` (but internal is faster).
