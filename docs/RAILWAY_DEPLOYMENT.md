# Railway Deployment Guide

Complete guide to deploying AutoMem on Railway with persistent storage, backups, and zero data loss.

## Quick Start (One-Click Deploy)

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/automem)

This template automatically sets up:
- ✅ AutoMem Flask API with health checks
- ✅ FalkorDB with **persistent volumes** and password protection
- ✅ Automatic secret generation
- ✅ Service networking configured

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
   - Mount path: `/data`
   - This ensures data survives restarts

3. **Configure environment variables**:
   ```bash
   REDIS_ARGS=--save 60 1 --appendonly yes --appendfsync everysec --dir /data --requirepass ${{shared.REDIS_PASSWORD}}
   REDIS_PASSWORD=${{shared.REDIS_PASSWORD}}  # Auto-generated secret
   ```

4. **Set health check**:
   - Command: `redis-cli -a $REDIS_PASSWORD ping`
   - Interval: 30s
   - Timeout: 3s

5. **Note the internal URL**: `${{FalkorDB.RAILWAY_PRIVATE_DOMAIN}}`

### Step 2: Deploy AutoMem API

1. **Connect GitHub repo** or **Deploy from Docker**
   - If using GitHub: Connect repository and set root directory
   - If using Docker: Use existing Dockerfile

2. **Configure environment variables**:
   ```bash
   # Database connections
   FALKORDB_HOST=${{FalkorDB.RAILWAY_PRIVATE_DOMAIN}}
   FALKORDB_PORT=6379
   FALKORDB_PASSWORD=${{shared.REDIS_PASSWORD}}
   FALKORDB_GRAPH=memories
   
   # API authentication (Railway auto-generates secrets)
   AUTOMEM_API_TOKEN=${{shared.AUTOMEM_API_TOKEN}}
   ADMIN_API_TOKEN=${{shared.ADMIN_API_TOKEN}}
   
   # OpenAI for embeddings (required for semantic search)
   OPENAI_API_KEY=<your-openai-key>
   
   # Optional: Qdrant Cloud for vector search
   QDRANT_URL=<your-qdrant-cloud-url>
   QDRANT_API_KEY=<your-qdrant-api-key>
   QDRANT_COLLECTION=memories
   
   # Port
   PORT=8001
   ```

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
  "graph": "memories",
  "timestamp": "2025-10-05T12:00:00Z"
}
```

If you get `503`:
- Check FalkorDB is running and healthy
- Verify `FALKORDB_HOST` is set to private domain
- Confirm `FALKORDB_PASSWORD` matches between services

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
- Mount path: `/data`
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

## Optional: FalkorDB Browser

For visual graph exploration:

1. **Create new service**:
   - Image: `falkordb/falkordb-browser:latest`
   - Port: 3000

2. **Configure connection**:
   ```bash
   FALKORDB_URL=redis://default:${{shared.REDIS_PASSWORD}}@${{FalkorDB.RAILWAY_PRIVATE_DOMAIN}}:6379
   ```

3. **Access**:
   - Generate public domain
   - Open in browser
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
- **AutoMem API**: 512MB RAM, 0.5 vCPU (~$5/mo)
- **FalkorDB**: 1GB RAM, 1 vCPU + 2GB volume (~$10/mo)
- **Qdrant Cloud**: Free tier (1GB) or $25/mo (10GB)

**Total**: ~$15-35/month depending on usage

**Cost Saving Tips**:
- Use Qdrant Cloud free tier initially
- Start with smaller FalkorDB volume (1GB)
- Use Railway's usage-based pricing (scales down when idle)

---

## Troubleshooting

### Connection Issues

**Problem**: API can't connect to FalkorDB

**Solution**:
```bash
# Check internal networking
railway logs --service automem-api | grep FalkorDB

# Verify private domain
echo $FALKORDB_HOST  # Should be: falkordb.railway.internal

# Test connection
railway run --service automem-api
> redis-cli -h $FALKORDB_HOST -p 6379 -a $FALKORDB_PASSWORD ping
```

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

1. **Always set REDIS_PASSWORD** (Railway auto-generates)
2. **Use Railway's private networking** for service-to-service
3. **Don't expose FalkorDB publicly** (use private domain only)
4. **Rotate API tokens** periodically via Railway dashboard
5. **Enable Railway's Audit Logs** (Enterprise plan)

---

## Next Steps

- [ ] Set up monitoring alerts (see [MONITORING_AND_BACKUPS.md](MONITORING_AND_BACKUPS.md))
- [ ] Configure automated backups (see [MONITORING_AND_BACKUPS.md](MONITORING_AND_BACKUPS.md))
- [ ] Add MCP server integration
- [ ] Deploy FalkorDB Browser
- [ ] Set up staging environment

**Questions?** Open an issue: https://github.com/verygoodplugins/automem/issues
