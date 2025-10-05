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
AUTOMEM_API_URL=https://automem.up.railway.app

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

### Local Backups (Development)

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

### Railway Cron Job Setup

**Method 1: Separate Backup Service**

Create a new Railway service that runs backups on a schedule:

1. **Create `scripts/Dockerfile.backup`**:
   ```dockerfile
   FROM python:3.11-slim
   
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt boto3
   
   COPY scripts/backup_automem.py scripts/
   COPY consolidation.py .
   
   # Run backup every 6 hours (21600 seconds)
   CMD ["sh", "-c", "while true; do python scripts/backup_automem.py --cleanup --keep 7; sleep 21600; done"]
   ```

2. **Deploy to Railway**:
   - Create new service
   - Point to `scripts/Dockerfile.backup`
   - Set environment variables (same as main service + AWS creds)
   - Deploy

**Method 2: Railway Cron (Pro Plan)**

Use Railway's native cron feature:

```bash
# In railway.toml
[[services]]
name = "backup-cron"
cron = "0 */6 * * *"  # Every 6 hours
command = "python scripts/backup_automem.py --s3-bucket my-automem-backups --cleanup"
```

**Method 3: External Cron (GitHub Actions)**

Run backups from GitHub Actions:

```yaml
# .github/workflows/backup.yml
name: AutoMem Backup
on:
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours
  workflow_dispatch:  # Manual trigger

jobs:
  backup:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -r requirements.txt boto3
      
      - name: Run backup
        env:
          FALKORDB_HOST: ${{ secrets.FALKORDB_HOST }}
          FALKORDB_PASSWORD: ${{ secrets.FALKORDB_PASSWORD }}
          QDRANT_URL: ${{ secrets.QDRANT_URL }}
          QDRANT_API_KEY: ${{ secrets.QDRANT_API_KEY }}
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          python scripts/backup_automem.py \
            --s3-bucket my-automem-backups \
            --cleanup --keep 7
```

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
curl https://automem.up.railway.app/health | jq
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
   - Monitor: `https://automem.up.railway.app/health`
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
curl https://automem.up.railway.app/health

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
