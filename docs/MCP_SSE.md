# MCP over SSE Sidecar (Railway)

This sidecar exposes AutoMem as an MCP server over SSE so cloud AI platforms can connect via HTTPS and use your memories.

**Supported platforms:**
- **ChatGPT** (requires developer mode: Settings >> Connectors >> Advanced)
- **Claude.ai** (web interface)
- **Claude Mobile App** (iOS/Android)
- **ElevenLabs Agents**

Service endpoint (on Railway):
- GET `/mcp/sse` — SSE stream (server → client). Include `Authorization: Bearer <AUTOMEM_API_TOKEN>`.
- POST `/mcp/messages?sessionId=<id>` — Client → server JSON-RPC messages.
- GET `/health` — Health probe.

Auth model:
- **Header-based** (ElevenLabs): `Authorization: Bearer <AUTOMEM_API_TOKEN>` header
- **URL-based** (ChatGPT, Claude): append `?api_token=<AUTOMEM_API_TOKEN>` to the SSE URL
  - Example: `https://<your-mcp-domain>/mcp/sse?api_token=...`
  - Required for platforms that only support OAuth for custom connectors
  - Note: URL tokens may appear in logs/proxy metadata

Supported tools:
- `store_memory`, `recall_memory`, `associate_memories`, `update_memory`, `delete_memory`, `check_database_health`

Deploy (one‑click template):
- The template adds a new service `automem-mcp-sse` alongside `memory-service` and `FalkorDB`.
- It preconfigures `AUTOMEM_ENDPOINT` to the internal URL of `memory-service`: `http://${memory-service.RAILWAY_PRIVATE_DOMAIN}:8001`.
- **Manual setup**: Use `AUTOMEM_ENDPOINT=http://memory-service.railway.internal:8001` (hardcoded internal DNS is more stable).
- **Important**: The internal DNS must match your memory service's `RAILWAY_PRIVATE_DOMAIN`. If you renamed the service, verify with `railway variables --service memory-service | grep RAILWAY_PRIVATE_DOMAIN`.

## Client Setup

### ChatGPT
ChatGPT only supports OAuth for custom connectors, so authentication must be via URL parameter:

1. Enable **Developer Mode**: Settings >> Connectors >> Advanced
2. Configure MCP server:
   - **Server URL**: `https://<your-mcp-domain>/mcp/sse?api_token=<AUTOMEM_API_TOKEN>`
   - Replace `<AUTOMEM_API_TOKEN>` with your actual token

### Claude.ai (Web Interface)
Claude.ai only supports OAuth for custom connectors, so authentication must be via URL parameter:

- **Server URL**: `https://<your-mcp-domain>/mcp/sse?api_token=<AUTOMEM_API_TOKEN>`
- Replace `<AUTOMEM_API_TOKEN>` with your actual token

### Claude Mobile App
Claude mobile only supports OAuth for custom connectors, so authentication must be via URL parameter:

- **Server URL**: `https://<your-mcp-domain>/mcp/sse?api_token=<AUTOMEM_API_TOKEN>`
- Replace `<AUTOMEM_API_TOKEN>` with your actual token

### ElevenLabs Agents
ElevenLabs supports custom headers, so you can use either method:

**Option 1: Custom Header (Recommended)**
- **Server URL**: `https://<your-mcp-domain>/mcp/sse`
- **Custom Header**:
  - Name: `Authorization`
  - Value: `Bearer <AUTOMEM_API_TOKEN>`

**Option 2: URL Parameter**
- **Server URL**: `https://<your-mcp-domain>/mcp/sse?api_token=<AUTOMEM_API_TOKEN>`

> **📚 Comprehensive Setup Guides**: Detailed step-by-step setup instructions for each platform are available in the [MCP-Automem project documentation](https://github.com/verygoodplugins/mcp-automem/blob/main/INSTALLATION.md) (coming soon).

Notes:
- Keepalive heartbeats are sent every 20s to prevent idle timeouts.
- Rate limiting and multi-tenant token scoping can be added in front of this service if needed.

Troubleshooting `fetch failed` errors:
1. **Check memory-service has `PORT=8001`** - Most common cause. Without it, Flask runs on wrong port.
2. **Verify `AUTOMEM_ENDPOINT`** - Should be `http://memory-service.railway.internal:8001` (or your service's actual `RAILWAY_PRIVATE_DOMAIN`).
3. **Check SSE logs** - Enable debug mode and check logs for actual error: `railway logs --service automem-mcp-sse`.
4. **Alternative**: Use public URL as fallback: `AUTOMEM_ENDPOINT=https://<your-memory-service-domain>` (but internal is faster).
