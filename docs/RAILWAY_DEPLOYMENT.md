# Railway Deployment Guide

Complete guide to deploying AutoMem on Railway with persistent storage, backups, and zero data loss.

## Quick Start (One-Click Deploy)

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.com/deploy/automem-ai-memory-service)

This template automatically sets up:

- ✅ **memory-service** — AutoMem Flask API with health checks
- ✅ **falkordb** — Graph database with **persistent volumes** and password protection
- ✅ **mcp-sse-server** — MCP bridge for cloud AI platforms (ChatGPT, Claude.ai, ElevenLabs)
- ✅ Automatic secret generation and service networking

```mermaid
flowchart TB
    subgraph internet [Public Internet]
        Users[AI Clients<br/>Direct API]
        CloudAI[Cloud AI Platforms<br/>ChatGPT, Claude.ai, ElevenLabs]
        Local[Local Tools<br/>Cursor, Claude Desktop, Claude Code]
    end

    subgraph railway [Railway Project]
        subgraph mcpbridge [MCP Bridge Service]
            MCPServer[MCP Bridge<br/>Streamable HTTP<br/>Port 8080]
            MCPDomain[Public Domain<br/>mcp.up.railway.app]
        end

        subgraph api [memory-service Service]
            FlaskAPI[Flask API<br/>Port 8001]
            APIDomain[Public Domain<br/>automem.up.railway.app]

            Enrichment[Enrichment Pipeline]
            Consolidation[Consolidation Engine]
        end

        subgraph db [falkordb Service]
            FalkorDB[(FalkorDB<br/>Port 6379)]
            Volume[Persistent Volume<br/>/var/lib/falkordb/data]
            TCPProxy[TCP Proxy<br/>Optional admin access]
        end

        subgraph qdrantsvc [qdrant Service — Option B]
            Qdrant[(Qdrant<br/>Port 6333)]
            QdrantVolume[Persistent Volume<br/>/qdrant/storage]
        end

        subgraph external [External Services]
            QdrantCloud[(Qdrant Cloud<br/>Vector DB — Option A)]
            OpenAI[OpenAI API<br/>Embeddings]
        end
    end

    Users -->|HTTPS| APIDomain
    CloudAI -->|Streamable HTTP| MCPDomain
    Local -->|HTTPS| APIDomain

    MCPDomain --> MCPServer
    APIDomain --> FlaskAPI

    MCPServer -->|Internal<br/>memory-service.railway.internal:8001| FlaskAPI

    FlaskAPI -->|Internal<br/>falkordb.railway.internal:6379| FalkorDB
    FlaskAPI -.->|Option A| QdrantCloud
    FlaskAPI -.->|Option B · Internal<br/>qdrant.railway.internal:6333| Qdrant
    FlaskAPI --> OpenAI

    Enrichment --> FalkorDB
    Enrichment -.-> QdrantCloud
    Enrichment -.-> Qdrant
    Consolidation --> FalkorDB

    FalkorDB --> Volume
    Qdrant --> QdrantVolume

    TCPProxy -.->|Optional<br/>Admin-only TCP access| FalkorDB
```

---

## Post-Deploy Checklist

After deploying, complete these steps to fully configure AutoMem:

### Required: Add Your API Keys

1. **Go to `memory-service` → Variables**
2. **Set these variables:**

| Variable         | Required    | How to Get                                                           |
| ---------------- | ----------- | -------------------------------------------------------------------- |
| `OPENAI_API_KEY` | Yes\*       | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `QDRANT_URL`     | Option A    | Qdrant Cloud URL — see [Qdrant Setup Guide](QDRANT_SETUP.md)         |
| `QDRANT_API_KEY` | Option A    | Qdrant Cloud API key                                                  |
| `QDRANT_HOST`    | Option B    | `qdrant` (self-hosted on Railway — see [Step 1b](#step-1b-qdrant-vector-database)) |

\*Without `OPENAI_API_KEY`, semantic search won't work (embeddings skipped).

👉 **Qdrant Cloud?** Follow the [Qdrant Setup Guide](QDRANT_SETUP.md) for step-by-step collection setup.
👉 **Self-hosted Qdrant?** See [Step 1b](#step-1b-qdrant-vector-database) — remember to set `QDRANT__SERVICE__HOST=::` on the Qdrant service.

3. **Redeploy** the memory-service after adding variables.

### Verify Deployment

```bash
# Check health (should show falkordb: connected)
curl https://your-automem.up.railway.app/health

# Store a test memory
curl -X POST "https://your-automem.up.railway.app/memory" \
  -H "Authorization: Bearer YOUR_AUTOMEM_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"content": "Test memory from Railway", "tags": ["test"]}'
```

### MCP Bridge (Included)

The template includes **mcp-sse-server**, which exposes AutoMem as an MCP server over HTTPS. This enables cloud AI platforms to access your memories:

| Platform          | How to Connect                                                                                 |
| ----------------- | ---------------------------------------------------------------------------------------------- |
| **ChatGPT**       | Settings → Connectors → Add Server → `https://your-sse.up.railway.app/mcp?api_token=TOKEN` |
| **Claude.ai**     | Settings → MCP → `https://your-sse.up.railway.app/mcp?api_token=TOKEN`                     |
| **Claude Mobile** | Settings → MCP Servers → same URL as above                                                   |
| **Claude Desktop** | Prefer the same remote MCP URL if you want connector state shared with Claude.ai/iOS        |
| **ElevenLabs**    | Agent config → MCP → same URL (or use Authorization header)                                  |

👉 **See [MCP_SSE.md](MCP_SSE.md)** for detailed setup per platform.

**Don't need the MCP bridge?** If you only use Cursor, Claude Code, or direct API access:

1. Go to Railway Dashboard → `mcp-sse-server` service
2. Click the three dots menu → **Delete Service**
3. This saves ~$2-3/month and has no impact on the core memory API

Claude Desktop can still use the local MCP package, but Anthropic now syncs connector state across Desktop, Claude.ai, and iOS. If you want one shared Claude connector, keep the Railway MCP bridge enabled and use the remote URL everywhere.

### Optional: Standalone Graph Viewer Service

Deploy `automem-graph-viewer` as a separate Railway service, then configure these on `memory-service`:

| Variable | Value |
|----------|-------|
| `ENABLE_GRAPH_VIEWER` | `true` |
| `GRAPH_VIEWER_URL` | `https://<your-viewer-domain>` |
| `VIEWER_ALLOWED_ORIGINS` | `https://<your-viewer-domain>` |

AutoMem keeps `/viewer` for compatibility and redirects/bootstraps users to the standalone viewer.
Use `/viewer` on `memory-service` as the public entrypoint; the standalone viewer is the backing service behind that route. Keep FalkorDB internal and avoid using direct public DB access as a viewer transport.

### Get Your API Tokens

Your tokens were auto-generated during deployment. Find them in:

- Railway Dashboard → `memory-service` → Variables
- Look for `AUTOMEM_API_TOKEN` and `ADMIN_API_TOKEN`

---

## Manual Setup (Recommended for Production)

### Step 1: Create FalkorDB Service with Persistence

1. **Create new service in Railway**

   - Click "+ New Service"
   - Select "Docker Image"
   - Image: `falkordb/falkordb:latest`

2. **Add persistent volume** (CRITICAL!)

   - Go to service → Settings → Volumes
   - Click "Add Volume"
   - Mount path: `/var/lib/falkordb/data`
   - This ensures data survives restarts

3. **Configure environment variables**:

   ```bash
   PORT=6379
   FALKOR_PASSWORD=${{shared.FALKOR_PASSWORD}}  # Auto-generated secret
   FALKOR_USERNAME=default
   FALKOR_HOST=${{RAILWAY_PRIVATE_DOMAIN}}
   FALKOR_PORT=${{PORT}}
   FALKOR_PUBLIC_HOST=${{RAILWAY_TCP_PROXY_DOMAIN}}
   FALKOR_PUBLIC_PORT=${{RAILWAY_TCP_PROXY_PORT}}
   REDIS_ARGS=--save 60 1 --appendonly yes --appendfsync everysec
   ```

   **Note**: FalkorDB handles authentication via `FALKOR_PASSWORD` - don't add `--requirepass` to `REDIS_ARGS`.

4. **Health check**: Leave blank/disabled (FalkorDB doesn't have HTTP endpoints). Railway monitors container status automatically.

5. **Note the internal URL**: `${{FalkorDB.RAILWAY_PRIVATE_DOMAIN}}`

### Step 1b: Qdrant Vector Database

Choose **one** of these options:

#### Option A: Qdrant Cloud (Managed)

Use [Qdrant Cloud](https://cloud.qdrant.io) for a fully managed vector database. See the [Qdrant Setup Guide](QDRANT_SETUP.md) for step-by-step instructions. Set `QDRANT_URL` and `QDRANT_API_KEY` on memory-service.

#### Option B: Self-Hosted Qdrant on Railway

Run Qdrant inside your Railway project for lower latency (internal networking, no HTTPS overhead) and simpler architecture.

1. **Create new service in Railway**

   - Click "+ New Service"
   - Select "Docker Image"
   - Image: `qdrant/qdrant:v1.11.3`

2. **Add persistent volume** (CRITICAL!)

   - Go to service → Settings → Volumes
   - Click "Add Volume"
   - Mount path: `/qdrant/storage`
   - This ensures vectors survive restarts

3. **Configure environment variables**:

   ```bash
   PORT=6333
   QDRANT__SERVICE__HOST=::
   ```

   > **⚠️ `QDRANT__SERVICE__HOST=::` is required.** Railway's internal networking uses IPv6. Qdrant defaults to binding `0.0.0.0` (IPv4 only), which silently refuses all internal connections. Setting `::` enables dual-stack (IPv6 + IPv4) binding. Without this, you'll get "Connection refused" errors that are extremely difficult to diagnose — DNS resolves fine, Qdrant logs show it's running, but no connections arrive.

4. **No health check needed**: Qdrant doesn't expose an HTTP health path that Railway can use. Container monitoring handles restarts.

5. **On memory-service**, set:

   ```bash
   QDRANT_HOST=qdrant   # Must match your Qdrant service name in Railway
   ```

   AutoMem constructs `http://qdrant:6333` automatically. No `QDRANT_URL` or `QDRANT_API_KEY` needed for internal networking.

   > **Note:** The hostname must match your Railway service name. If you named the service something other than "qdrant", use that name instead (e.g. `QDRANT_HOST=my-vector-db`). You can also use Railway's reference variable: `QDRANT_HOST=${{Qdrant.RAILWAY_PRIVATE_DOMAIN}}`.

### Step 2: Deploy AutoMem API

1. **Connect GitHub repo** or **Deploy from Docker**

   - If using GitHub: Connect repository and set root directory
   - If using Docker: Use existing Dockerfile

2. **Configure environment variables**:

   **Option A: Variable References (template style)**

   ```bash
   # Database connections
   FALKORDB_HOST=${{FalkorDB.RAILWAY_PRIVATE_DOMAIN}}
   FALKORDB_PORT=6379
   FALKORDB_PASSWORD=${{FalkorDB.FALKOR_PASSWORD}}
   FALKORDB_GRAPH=memories

   # API authentication (Railway auto-generates secrets)
   AUTOMEM_API_TOKEN=${{shared.AUTOMEM_API_TOKEN}}
   ADMIN_API_TOKEN=${{shared.ADMIN_API_TOKEN}}

   # OpenAI for embeddings (required for semantic search)
   OPENAI_API_KEY=<your-openai-key>

   # Vector search — pick ONE:
   # Option A: Qdrant Cloud
   QDRANT_URL=<your-qdrant-cloud-url>
   QDRANT_API_KEY=<your-qdrant-api-key>
   # Option B: Self-hosted Qdrant on Railway (see Step 1b)
   # QDRANT_HOST=qdrant
   QDRANT_COLLECTION=memories

   # Port (REQUIRED - Flask needs explicit port)
   PORT=8001
   ```

   **Option B: Hardcoded Values (recommended for stability)**

   ```bash
   # Database connections - use actual values from FalkorDB service
   FALKORDB_HOST=falkordb.railway.internal
   FALKORDB_PORT=6379
   FALKORDB_PASSWORD=<copy-from-falkordb-service>
   FALKORDB_GRAPH=memories

   # API authentication - generate or copy from shared variables
   AUTOMEM_API_TOKEN=<your-generated-token>
   ADMIN_API_TOKEN=<your-generated-token>

   # OpenAI for embeddings
   OPENAI_API_KEY=<your-openai-key>

   # Vector search — pick ONE:
   # Option A: Qdrant Cloud
   QDRANT_URL=<your-qdrant-cloud-url>
   QDRANT_API_KEY=<your-qdrant-api-key>
   # Option B: Self-hosted Qdrant on Railway (see Step 1b)
   # QDRANT_HOST=qdrant
   QDRANT_COLLECTION=memories

   # Port (REQUIRED - Flask needs explicit port)
   PORT=8001
   ```

   **Note**: Hardcoded values (Option B) are more stable and easier to debug, while variable references (Option A) update automatically but can be harder to troubleshoot.

   **⚠️ Important**: `PORT=8001` is **required** for the memory-service. Without it, Flask defaults to port 5000, causing connection failures from other services.

3. **Set health check**:

   - Path: `/health`
   - Timeout: 100s

4. **Generate public domain**:
   - Settings → Networking → Generate Domain
   - Save your URL: `https://your-automem.up.railway.app`

### Step 3: Verify Deployment

```bash
# Check health
curl https://your-automem.up.railway.app/health

# Expected response:
{
  "status": "healthy",
  "falkordb": "connected",
  "qdrant": "connected",
  "memory_count": 1234,  # Added in recent versions
  "enrichment": {
    "status": "running",
    "queue_depth": 0,
    "pending": 0,
    "inflight": 0,
    "processed": 0,
    "failed": 0
  },
  "graph": "memories",
  "timestamp": "2025-10-20T03:47:39+00:00"
}
```

**Note**: `memory_count` field requires AutoMem commit from Oct 20, 2025 or later. For detailed analytics, use `/analyze` endpoint.

```bash
# Check detailed memory analytics
curl "https://your-automem.up.railway.app/analyze?api_key=YOUR_API_TOKEN"

# Shows:
# - Total memories by type (Context, Decision, Insight, etc.)
# - Entity frequency (projects, tools)
# - Confidence distribution
# - Temporal insights (activity by hour)
```

If you get `503`:

- Check FalkorDB is running and healthy
- Verify `FALKORDB_HOST` is set to private domain (use `falkordb.railway.internal`, not `${{...}}` syntax)
- Confirm `FALKORDB_PASSWORD` matches between services
- Test connection: `railway logs --service memory-service | grep -i falkordb`

### Step 4: Store First Memory

```bash
export AUTOMEM_URL="https://your-automem.up.railway.app"
export AUTOMEM_TOKEN="your-api-token"

curl -X POST "$AUTOMEM_URL/memory" \
  -H "Authorization: Bearer $AUTOMEM_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "First memory from Railway!",
    "tags": ["test", "deployment"],
    "importance": 0.8
  }'
```

---

## Data Persistence & Backups

### Persistent Volumes (Required)

Railway volumes ensure data survives:

- Service restarts
- Deployments
- Platform maintenance

**Volume Configuration**:

- Mount path: `/var/lib/falkordb/data`
- Minimum size: 1GB (adjust based on needs)
- Backed up automatically by Railway

### Automated Backups

Railway provides built-in volume backups for FalkorDB (automatic, one-click restore).

For comprehensive backups covering both FalkorDB and Qdrant with S3 upload:

👉 **See [MONITORING_AND_BACKUPS.md](MONITORING_AND_BACKUPS.md)** for complete backup setup including:

- Railway volume backups (built-in)
- GitHub Actions automated backups (recommended, free)
- Manual backup scripts

### Disaster Recovery

If FalkorDB data is lost but Qdrant is intact:

```bash
# SSH into AutoMem service
railway run

# Run recovery script
python scripts/recover_from_qdrant.py

# This will:
# - Read all 636+ memories from Qdrant
# - Rebuild FalkorDB graph structure
# - Restore all relationships
```

---

## Optional: FalkorDB Browser (Admin / Debug)

For low-level graph debugging by operators (separate from the public standalone graph-viewer):

1. **Create new service**:

   - Image: `falkordb/falkordb-browser:latest`
   - Port: 3000

2. **Configure connection**:

   ```bash
   FALKORDB_URL=redis://default:${{FalkorDB.FALKOR_PASSWORD}}@${{FalkorDB.RAILWAY_PRIVATE_DOMAIN}}:6379
   ```

3. **Access**:
   - Prefer private networking / access-controlled exposure (for example Cloudflare Access)
   - Avoid treating this as the public product viewer path
   - Visual query builder included

---

## Monitoring & Health Checks

### Built-in Health Endpoint

```bash
# Check system health
curl https://your-automem.up.railway.app/health

# Response includes:
# - FalkorDB connection status
# - Qdrant connection status
# - Memory count
# - Timestamp
```

### Automated Health Monitoring

Deploy health monitor as separate service:

```bash
# In new Railway service
docker run automem/health-monitor \
  -e AUTOMEM_API_URL=${{AutoMemAPI.RAILWAY_PUBLIC_DOMAIN}} \
  -e CHECK_INTERVAL=300
```

This will:

- Monitor FalkorDB/Qdrant health every 5min
- Check memory count consistency
- Auto-trigger recovery if >5% drift detected
- Send alerts via webhook (configure ALERT_WEBHOOK_URL)

---

## Cost Optimization

**Recommended Railway Plan**: Pro ($20/mo)

**Service Sizing**:

- **memory-service**: 512MB RAM, 0.5 vCPU (~$5/mo)
- **FalkorDB**: 1GB RAM, 1 vCPU + 2GB volume (~$10/mo)
- **mcp-sse-server**: 256MB RAM, 0.25 vCPU (~$2-3/mo)
- **Qdrant** (choose one):
  - *Cloud*: Free tier (1GB) or $25/mo (10GB)
  - *Self-hosted on Railway*: ~$3-5/mo (256MB RAM + 1GB volume)

**Total**: ~$17-40/month depending on usage

**Cost Saving Tips**:

- Self-host Qdrant on Railway to avoid Qdrant Cloud costs and reduce latency
- Or use Qdrant Cloud free tier initially
- Start with smaller FalkorDB volume (1GB)
- Use Railway's usage-based pricing (scales down when idle)
- **Remove mcp-sse-server** if you only use Cursor/Claude Desktop (saves ~$2-3/mo)

---

## Troubleshooting

### Connection Issues

**Problem**: API can't connect to FalkorDB

**Solution**:

```bash
# Check internal networking
railway logs --service memory-service | grep FalkorDB

# Verify private domain
echo $FALKORDB_HOST  # Should be: falkordb.railway.internal

# Test connection
railway run --service memory-service
> redis-cli -h $FALKORDB_HOST -p 6379 -a $FALKORDB_PASSWORD ping
```

### Service Connection Refused (ECONNREFUSED)

**Problem**: SSE or other services get "fetch failed" or "ECONNREFUSED" when connecting to memory-service

**Symptoms**:

```
Error: connect ECONNREFUSED fd12:ca03:42be:0:1000:50:1079:5b6c:8001
```

**Causes & Solutions**:

1. **Missing PORT variable** (most common):

   - Check memory-service variables: `PORT` must be set to `8001`
   - Without it, Flask defaults to port 5000
   - **Fix**: Add `PORT=8001` to memory-service environment variables and redeploy

2. **IPv6 binding issue** (fixed in latest code):

   - Railway internal networking uses IPv6
   - Older AutoMem versions bound to IPv4 only (`0.0.0.0`)
   - **Fix**: Update to latest code (Flask now binds to `::` for IPv6 dual-stack)
   - Check startup logs should show: `* Running on http://[::1]:8001`

3. **Wrong internal hostname**:
   - Verify `AUTOMEM_API_URL` in SSE service matches memory-service's `RAILWAY_PRIVATE_DOMAIN`
   - Should be: `http://memory-service.railway.internal:8001`

### Qdrant "Connection Refused" on Internal Networking

**Problem**: memory-service can't connect to self-hosted Qdrant via internal networking. Health endpoint shows `qdrant: "disconnected"`, logs show `Connection refused (errno 111)`.

**Root cause**: Railway's internal networking uses **IPv6**, but Qdrant's Docker image binds to `0.0.0.0` (IPv4 only) by default. The internal hostname resolves to an IPv6 address, but Qdrant isn't listening on IPv6 — the kernel refuses the connection before it ever reaches the Qdrant process. This is why Qdrant logs show zero incoming connection attempts even though DNS resolves correctly.

**Fix**: Set this environment variable on the **Qdrant service** (not memory-service):

```bash
QDRANT__SERVICE__HOST=::
```

`::` is the IPv6 equivalent of `0.0.0.0` — it enables dual-stack binding (IPv6 + IPv4). Redeploy Qdrant after setting this.

**Also verify** on memory-service:
- `QDRANT_HOST=qdrant` (Railway resolves short service names internally)
- Or `QDRANT_URL=http://qdrant:6333` (explicit URL form)
- Do NOT use `https://` for internal connections

### Variable Reference Issues

**Problem**: Variables using `${{...}}` syntax not resolving (showing literal `${{...}}` in logs)

**Cause**: Railway variable references only work in templates, not manual service configuration

**Solution**: Use hardcoded values instead

```bash
# ❌ Don't use in manual setup:
FALKORDB_HOST=${{FalkorDB.RAILWAY_PRIVATE_DOMAIN}}

# ✅ Do use in manual setup:
FALKORDB_HOST=falkordb.railway.internal
FALKORDB_PASSWORD=<copy-exact-value-from-falkordb-service>
```

**Benefit**: Hardcoded values are more stable, easier to debug, and work consistently across redeployments.

### Data Loss

**Problem**: FalkorDB restarted and lost data

**Cause**: No persistent volume configured

**Solution**:

1. Add volume to FalkorDB service (Settings → Volumes)
2. Run recovery: `python scripts/recover_from_qdrant.py`
3. Redeploy FalkorDB service

### High Memory Usage

**Problem**: FalkorDB using too much RAM

**Solution**:

```bash
# Optimize Redis memory
REDIS_ARGS=--maxmemory 512mb --maxmemory-policy allkeys-lru
```

---

## Security Best Practices

1. **Always set FALKOR_PASSWORD** (Railway auto-generates)
2. **Use Railway's private networking** for service-to-service
3. **Don't expose FalkorDB publicly** (use private domain only)
4. **Rotate API tokens** periodically via Railway dashboard
5. **Enable Railway's Audit Logs** (Enterprise plan)

**Note on Service Naming**: Railway's internal DNS is based on the service name (e.g., `memory-service.railway.internal`). If you rename a service, its `RAILWAY_PRIVATE_DOMAIN` updates automatically, but you'll need to update any hardcoded hostnames in other services' environment variables.

---

## Next Steps

- [ ] Set up monitoring alerts (see [MONITORING_AND_BACKUPS.md](MONITORING_AND_BACKUPS.md))
- [ ] Configure automated backups (see [MONITORING_AND_BACKUPS.md](MONITORING_AND_BACKUPS.md))
- [x] Add Remote MCP server integration — see docs/MCP_SSE.md
- [ ] Deploy standalone graph-viewer (public UI) if needed
- [ ] Deploy FalkorDB Browser only for admin/debug workflows
- [ ] Set up staging environment

**Questions?** Open an issue: https://github.com/verygoodplugins/automem/issues
