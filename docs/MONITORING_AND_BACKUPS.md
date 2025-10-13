# AutoMem Monitoring & Backups

Complete guide to setting up automated health monitoring and backups for AutoMem on Railway.

## Overview

AutoMem includes three layers of data protection:

1. **Persistent Volumes** - Railway volumes for FalkorDB data
2. **Dual Storage** - Data stored in both FalkorDB (graph) and Qdrant (vectors)
3. **Automated Backups** - Scheduled exports to compressed JSON + optional S3 upload

---

## Health Monitoring

The `health_monitor.py` script continuously monitors system health and can automatically trigger recovery.

### Quick Start

**Option 1: Deploy as Railway Service (Recommended)**

Create a new Railway service for continuous monitoring:

```bash
# In Railway dashboard
1. Create new service from GitHub repo
2. Set Dockerfile path: scripts/Dockerfile.health-monitor (we'll create this)
3. Configure environment variables (same as main service)
4. Deploy
```

**Option 2: Run as Cron Job**

```bash
# One-time health check (safe)
railway run --service memory-service python scripts/health_monitor.py --once

# Alert-only monitoring (no auto-recovery)
railway run --service memory-service python scripts/health_monitor.py --interval 300

# With Slack webhook alerts
railway run --service memory-service python scripts/health_monitor.py \
  --interval 300 \
  --webhook https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

### Configuration

Set these environment variables on your monitoring service:

```bash
# Required (same as main service)
FALKORDB_HOST=falkordb.railway.internal
FALKORDB_PORT=6379
FALKORDB_PASSWORD=<your-password>
QDRANT_URL=<your-qdrant-url>
QDRANT_API_KEY=<your-qdrant-key>
AUTOMEM_API_URL=https://your-automem-deployment.up.railway.app

# Optional monitoring settings
HEALTH_MONITOR_DRIFT_THRESHOLD=5          # Warning at 5% drift
HEALTH_MONITOR_CRITICAL_THRESHOLD=50      # Critical at 50% drift
HEALTH_MONITOR_WEBHOOK=<slack-webhook>    # Alert webhook
```

### Auto-Recovery (Use with Caution!)

Enable automatic recovery when data loss is detected:

```bash
python scripts/health_monitor.py \
  --auto-recover \
  --interval 300 \
  --critical-threshold 50
```

**⚠️ Warning**: Auto-recovery will automatically run the recovery script when critical drift is detected. Only enable this if you trust the system to self-heal.

---

## Automated Backups

### Railway Volume Backups (Built-in) ✅

**Already configured!** If you're using Railway, your FalkorDB service has automatic volume backups enabled.

**Features:**
- ✅ Automatic snapshots (default: every 24 hours)
- ✅ One-click restore from Railway dashboard
- ✅ Included with Railway Pro (no extra cost)
- ✅ Instant volume snapshots

**Access backups:**
1. Railway Dashboard → `falkordb` service
2. Click "Backups" tab
3. View backup history and schedule
4. Click "Restore" to recover from any snapshot

**Limitations:**
- Only backs up FalkorDB (not Qdrant)
- Platform-locked (can't export/download)
- Use for quick recovery; combine with script backups for full protection

---

### Script-Based Backups

For portable backups that cover both databases, use the `backup_automem.py` script:

#### Local Backups (Development)

The `backup_automem.py` script exports both FalkorDB and Qdrant to compressed JSON files:

```bash
# Basic backup to ./backups/
python scripts/backup_automem.py

# Backup with cleanup (keep last 7)
python scripts/backup_automem.py --cleanup --keep 7

# Custom backup directory
python scripts/backup_automem.py --backup-dir /mnt/backups
```

### Cloud Backups (Production)

Upload backups to S3 for disaster recovery:

```bash
# Install AWS SDK
pip install boto3

# Configure AWS credentials (Railway secrets)
export AWS_ACCESS_KEY_ID=<your-key>
export AWS_SECRET_ACCESS_KEY=<your-secret>
export AWS_DEFAULT_REGION=us-east-1

# Backup with S3 upload
python scripts/backup_automem.py \
  --s3-bucket my-automem-backups \
  --cleanup --keep 7
```

### Automated Script Backups

**Recommended: GitHub Actions (Free)**

GitHub Actions is the simplest way to automate backups - free and doesn't consume Railway resources.

**Setup (5 minutes):**

1. **Workflow file already exists:** `.github/workflows/backup.yml`

2. **Add GitHub secrets:**
   - Go to: GitHub repo → Settings → Secrets and variables → Actions
   - Add these secrets:
     ```
     FALKORDB_HOST         = your-host.proxy.rlwy.net (your Railway TCP proxy)
     FALKORDB_PORT         = 12345 (your Railway TCP proxy port)
     FALKORDB_PASSWORD     = (from Railway)
     QDRANT_URL            = (from Railway)
     QDRANT_API_KEY        = (from Railway)
     ```
   - Optional for S3: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`

3. **Push and test:**
   ```bash
   git push origin main
   ```
   - Go to Actions tab → "AutoMem Backup" → Run workflow

**Runs every 6 hours automatically.** Free tier: 2000 minutes/month.

---

**Advanced: Railway Backup Service**

For Railway Pro users who want backups running on Railway:

⚠️ **Note:** Railway's UI makes Dockerfile configuration complex. This method is for advanced users.

The `scripts/Dockerfile.backup` exists and runs backups every 6 hours in a loop. However, deploying it requires CLI:

```bash
cd /path/to/automem
railway link
railway up --service backup-service
```

Then configure in Railway dashboard:
- Set Builder to Dockerfile
- Dockerfile Path: `scripts/Dockerfile.backup`
- Add environment variables (same as memory-service)

**Cost:** ~$1-2/month

**Recommendation:** Use GitHub Actions instead unless you have specific requirements for Railway-hosted backups.

---

## Backup Restoration

### Restore from Qdrant (Fastest)

If FalkorDB data is lost but Qdrant is intact:

```bash
railway run --service memory-service python scripts/recover_from_qdrant.py
```

This rebuilds the FalkorDB graph from Qdrant vectors and payloads.

### Restore from Backup Files

If both FalkorDB and Qdrant are lost, restore from backup:

```bash
# Download from S3
aws s3 cp s3://my-automem-backups/qdrant/qdrant_20251005_143000.json.gz ./restore/

# Extract
gunzip restore/qdrant_20251005_143000.json.gz

# Restore to Qdrant
python scripts/restore_from_backup.py restore/qdrant_20251005_143000.json

# Then restore FalkorDB from Qdrant
python scripts/recover_from_qdrant.py
```

**Note**: We'll create `restore_from_backup.py` if you need it.

---

## Monitoring Dashboards

### Built-in Health Endpoint

Check system health via API:

```bash
curl https://your-automem-deployment.up.railway.app/health | jq
```

Response:
```json
{
  "status": "healthy",
  "falkordb": "connected",
  "qdrant": "connected",
  "graph": "memories",
  "timestamp": "2025-10-05T14:45:00Z"
}
```

### Railway Dashboard

Monitor your services:
- **Metrics**: CPU, memory, network usage
- **Logs**: Real-time log streaming
- **Deployments**: Build history and status
- **Health Checks**: Automated uptime monitoring

### External Monitoring (Optional)

Set up external monitoring with:

1. **UptimeRobot** - Free HTTP monitoring
   - Monitor: `https://your-automem-deployment.up.railway.app/health`
   - Alert when status != "healthy"

2. **Better Uptime** - Advanced monitoring
   - HTTP checks + keyword monitoring
   - SMS/Slack/Email alerts

3. **Grafana Cloud** - Full observability
   - Custom dashboards
   - Metrics aggregation
   - Log correlation

---

## Backup Schedule Recommendations

### For Personal Use
- **Health checks**: Every 5 minutes (alert-only)
- **Backups**: Every 24 hours, keep 7 days
- **Recovery**: Manual trigger

### For Team Use
- **Health checks**: Every 2 minutes (with auto-recovery)
- **Backups**: Every 6 hours, keep 14 days + S3
- **Recovery**: Automatic on critical drift

### For Production Use
- **Health checks**: Every 30 seconds (with auto-recovery)
- **Backups**: Every 1 hour, keep 30 days + S3 + cross-region replication
- **Recovery**: Automatic with alerts

---

## Alerting Integrations

### Slack Webhook

```bash
# Get webhook URL from Slack App settings
# https://api.slack.com/messaging/webhooks

python scripts/health_monitor.py \
  --webhook https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXX
```

### Discord Webhook

```bash
# Discord webhooks work the same as Slack
python scripts/health_monitor.py \
  --webhook https://discord.com/api/webhooks/123456789/abcdefg
```

### Custom Webhook

The health monitor sends JSON payloads:

```json
{
  "level": "critical",
  "title": "Data Loss Detected",
  "message": "FalkorDB has 52.3% drift from Qdrant",
  "details": {
    "drift_percent": 52.3,
    "falkordb_count": 420,
    "qdrant_count": 884
  },
  "timestamp": "2025-10-05T14:45:00Z",
  "system": "AutoMem Health Monitor"
}
```

---

## Cost Estimates

### Railway (Hobby Plan - $5/month)
- ✅ Main API service
- ✅ FalkorDB service with 1GB volume
- ❌ Not enough resources for monitoring service

### Railway (Pro Plan - $20/month)
- ✅ Main API service (~$5)
- ✅ FalkorDB service (~$10)
- ✅ Health monitoring service (~$2)
- ✅ Backup service (~$1)
- **Total**: ~$18/month

### Railway + External Services (Hybrid)
- Railway Pro for main services (~$15)
- GitHub Actions for backups (free)
- UptimeRobot for monitoring (free)
- **Total**: ~$15/month

### AWS S3 Backup Costs
- **Storage**: ~$0.023/GB/month (Standard)
- **Requests**: ~$0.005/1000 PUTs
- **Example**: 100MB backup every 6 hours = ~$0.30/month

---

## Troubleshooting

### Health Monitor Shows Drift

**Problem**: FalkorDB and Qdrant counts don't match

**Causes**:
- In-flight writes during check (normal, <1% drift)
- Failed writes to one store (>5% drift - warning)
- Data loss event (>50% drift - critical)

**Solution**:
```bash
# Check health details
python scripts/health_monitor.py --once

# If critical, run recovery
python scripts/recover_from_qdrant.py
```

### Backup Failed

**Problem**: Backup script fails with connection error

**Solution**:
```bash
# Test connections
curl https://your-automem-deployment.up.railway.app/health

# Check credentials
echo $FALKORDB_PASSWORD
echo $QDRANT_API_KEY

# Try manual backup
python scripts/backup_automem.py
```

### S3 Upload Failed

**Problem**: Backup created but S3 upload failed

**Solution**:
```bash
# Check AWS credentials
aws s3 ls s3://my-automem-backups/

# Test upload manually
aws s3 cp backups/falkordb/latest.json.gz s3://my-automem-backups/test/

# Check boto3 installation
python -c "import boto3; print(boto3.__version__)"
```

---

## Next Steps

- [ ] Set up health monitoring service on Railway
- [ ] Configure Slack/Discord webhook alerts
- [ ] Schedule automated backups (every 6 hours)
- [ ] Test recovery process in staging environment
- [ ] Set up S3 bucket with versioning enabled
- [ ] Configure cross-region replication (optional)

**Questions?** Check the main Railway deployment guide: [RAILWAY_DEPLOYMENT.md](RAILWAY_DEPLOYMENT.md)
