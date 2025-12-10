# Health Monitoring Guide

AutoMem includes a built-in health monitoring system that watches for data inconsistencies and optionally triggers automatic recovery.

## Quick Start

### Alert-Only Mode (Recommended)

```bash
# Run health checks every 5 minutes (alert only, no auto-recovery)
python scripts/health_monitor.py --interval 300
```

This will:
- ✅ Monitor FalkorDB, Qdrant, and API health
- ✅ Check memory count consistency
- ✅ Log warnings if drift detected
- ✅ Send alerts via webhook (if configured)
- ❌ **NOT** automatically trigger recovery (safe!)

### With Webhook Alerts

```bash
# Send alerts to Slack/Discord/etc
python scripts/health_monitor.py \
  --interval 300 \
  --webhook https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

---

## Safety Features

### Default: Alert Only

**By design, auto-recovery is DISABLED by default.** This prevents unexpected system changes without human oversight.

When drift is detected, the monitor will:
1. Log a warning with drift percentage
2. Send webhook alert (if configured)
3. Provide recovery command to run manually
4. **NOT** automatically trigger recovery

### Opt-In Auto-Recovery

To enable auto-recovery (use with caution):

```bash
python scripts/health_monitor.py \
  --auto-recover \
  --interval 300 \
  --webhook https://your-webhook-url
```

**10-second safety delay**: When starting with `--auto-recover`, you have 10 seconds to cancel (Ctrl+C) before it activates.

---

## Thresholds

### Warning Threshold (5% default)

Minor drift - sends warning alert but **does not trigger recovery**.

**Example**: FalkorDB has 610 memories, Qdrant has 636 (4.1% drift)
- Status: Warning
- Action: Alert sent
- Recovery: No

### Critical Threshold (50% default)

Major data loss - triggers recovery process (if enabled).

**Example**: FalkorDB has 200 memories, Qdrant has 636 (68.6% drift)
- Status: Critical
- Action: Alert sent + recovery triggered (if `--auto-recover`)
- Recovery: Yes (if enabled)

### Customize Thresholds

```bash
python scripts/health_monitor.py \
  --drift-threshold 10 \
  --critical-threshold 30 \
  --interval 300
```

Or via environment:

```bash
export HEALTH_MONITOR_DRIFT_THRESHOLD=10
export HEALTH_MONITOR_CRITICAL_THRESHOLD=30
python scripts/health_monitor.py --interval 300
```

---

## Alert Channels

### Webhook (Slack, Discord, etc.)

```bash
# Slack
python scripts/health_monitor.py \
  --webhook https://hooks.slack.com/services/T00/B00/XXXX

# Discord
python scripts/health_monitor.py \
  --webhook https://discord.com/api/webhooks/XXXX/YYYY
```

**Webhook Payload**:
```json
{
  "level": "critical",
  "title": "Data Loss Detected - Manual Recovery Required",
  "message": "Major data loss detected. Drift: 68.6%",
  "details": {
    "drift_percent": 68.6,
    "auto_recover_enabled": false,
    "recovery_command": "python scripts/recover_from_qdrant.py"
  },
  "timestamp": "2025-10-05T12:00:00Z",
  "system": "AutoMem Health Monitor"
}
```

### Email (Coming Soon)

Email alerts are planned but not yet implemented. Use webhooks for now.

---

## Usage Examples

### One-Time Health Check

```bash
# Quick check without continuous monitoring
python scripts/health_monitor.py --once
```

**Output**:
```json
{
  "timestamp": "2025-10-05T12:00:00Z",
  "falkordb": {
    "status": "healthy",
    "memory_count": 636
  },
  "qdrant": {
    "status": "healthy",
    "points_count": 636
  },
  "api": {
    "status": "healthy"
  },
  "consistency": {
    "status": "consistent",
    "drift_percent": 0.0
  }
}
```

### Continuous Monitoring (Production)

```bash
# Run as background service with systemd
sudo tee /etc/systemd/system/automem-health.service << EOF
[Unit]
Description=AutoMem Health Monitor
After=network.target

[Service]
Type=simple
User=automem
WorkingDirectory=/opt/automem
Environment="PATH=/opt/automem/venv/bin:/usr/bin"
ExecStart=/opt/automem/venv/bin/python scripts/health_monitor.py --interval 300 --webhook https://your-webhook
Restart=always
RestartSec=60

[Install]
WantedBy=multi-tier.target
EOF

sudo systemctl enable automem-health
sudo systemctl start automem-health
```

### Docker Compose

Add to `docker-compose.yml`:

```yaml
services:
  health-monitor:
    build: .
    command: python scripts/health_monitor.py --interval 300
    environment:
      - FALKORDB_HOST=falkordb
      - QDRANT_URL=http://qdrant:6333
      - AUTOMEM_API_URL=http://flask-api:8001
      - HEALTH_MONITOR_WEBHOOK=${WEBHOOK_URL}
    depends_on:
      - falkordb
      - qdrant
      - flask-api
    restart: unless-stopped
```

### Railway Deployment

Deploy as separate service:

1. Create new service: "Health Monitor"
2. Use same repo, different start command
3. Set environment variables:
   ```bash
   FALKORDB_HOST=${{FalkorDB.RAILWAY_PRIVATE_DOMAIN}}
   QDRANT_URL=${{Qdrant.QDRANT_URL}}
   AUTOMEM_API_URL=${{AutoMemAPI.RAILWAY_PUBLIC_DOMAIN}}
   HEALTH_MONITOR_WEBHOOK=https://your-webhook
   ```
4. Start command: `python scripts/health_monitor.py --interval 300`

---

## What Gets Monitored

### FalkorDB Health

- ✅ Connection status
- ✅ Memory count (via `MATCH (m:Memory) RETURN count(m)`)
- ✅ Response time
- ❌ Graph integrity (coming soon)

### Qdrant Health

- ✅ Connection status
- ✅ Points count
- ✅ Collection status
- ❌ Vector quality (coming soon)

### API Health

- ✅ HTTP status (via `/health` endpoint)
- ✅ Response time
- ✅ FalkorDB/Qdrant connection status from API

### Consistency Check

- ✅ Memory count drift between FalkorDB and Qdrant
- ✅ Drift percentage calculation
- ✅ Severity classification (ok/warning/critical)
- ❌ Content checksum validation (coming soon)

---

## Recovery Behavior

### Alert-Only Mode (Default)

When critical drift detected:

1. **Log warning**:
   ```
   ⚠️  CRITICAL: FalkorDB has 68.6% drift from Qdrant
   🚨 AUTO-RECOVERY DISABLED - Please run recovery manually:
      python scripts/recover_from_qdrant.py
   ```

2. **Send webhook alert**:
   - Level: `critical`
   - Title: "Data Loss Detected - Manual Recovery Required"
   - Includes recovery command

3. **No automatic action** - human decides when to recover

### Auto-Recovery Mode (Opt-In)

When critical drift detected:

1. **Send "recovery starting" alert**
2. **Execute**: `python scripts/recover_from_qdrant.py`
3. **Monitor recovery progress**
4. **Send completion/failure alert**

**Example Alert Flow**:
```
1. 🚨 CRITICAL: Data Loss Detected
   → Webhook: "Data Loss Detected"

2. 🔧 AUTO-RECOVERY ENABLED: Starting recovery
   → Webhook: "Auto-Recovery Triggered"

3. ✅ Recovery completed successfully
   → Webhook: "Auto-Recovery Completed - 636 memories restored"
```

---

## Troubleshooting

### Monitor Won't Start

**Error**: `Cannot connect to FalkorDB`

**Fix**: Check environment variables:
```bash
echo $FALKORDB_HOST
echo $FALKORDB_PORT
echo $FALKORDB_PASSWORD
```

### No Alerts Received

**Check webhook URL**:
```bash
curl -X POST https://your-webhook-url \
  -H "Content-Type: application/json" \
  -d '{"text":"Test alert from AutoMem"}'
```

### False Positive Alerts

Drift can occur normally due to:
- In-flight writes (memory being stored)
- Consolidation in progress
- Network delays

**Solution**: Increase drift threshold:
```bash
python scripts/health_monitor.py --drift-threshold 10  # More lenient
```

### Recovery Not Triggering

Auto-recovery only triggers when:
1. `--auto-recover` flag is set
2. Drift exceeds critical threshold (default: 50%)
3. Both stores are healthy (can connect)

**Check**: Run one-time check to see current drift:
```bash
python scripts/health_monitor.py --once | grep drift_percent
```

---

## Best Practices

### Production Recommendations

1. **Start with alert-only mode** - monitor for a week before enabling auto-recovery
2. **Set up webhook alerts** - know immediately when issues occur
3. **Run as systemd service** - restart automatically if it crashes
4. **Monitor the monitor** - use systemd status checks
5. **Test recovery manually** - verify it works before enabling auto-recovery

### When to Enable Auto-Recovery

✅ **Good use cases**:
- Stable production environment
- Tested recovery process multiple times
- 24/7 webhook monitoring
- Clear runbooks for failures

❌ **Bad use cases**:
- Development/staging environments
- Untested recovery process
- No alerting configured
- Unclear root cause of drift

### Alert Fatigue Prevention

- Set appropriate thresholds (too sensitive = noise)
- Use different channels for warnings vs critical
- Implement rate limiting (built-in: won't spam)
- Review and adjust thresholds based on experience

---

## See Also

- [Recovery Script Documentation](../scripts/recover_from_qdrant.py)
- [Environment Variables](./ENVIRONMENT_VARIABLES.md)
- [Railway Deployment](./RAILWAY_DEPLOYMENT.md)
- [Deployment Checklist](./DEPLOYMENT_CHECKLIST.md)
