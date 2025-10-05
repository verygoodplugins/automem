#!/bin/bash
# Automated FalkorDB backup script
# Run via cron or Railway scheduled task

set -e

BACKUP_DIR="${BACKUP_DIR:-/data/backups}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"

echo "🔄 Starting FalkorDB backup at $TIMESTAMP"

# Trigger Redis SAVE
redis-cli SAVE

# Copy RDB file
if [ -f /data/dump.rdb ]; then
    cp /data/dump.rdb "$BACKUP_DIR/dump_${TIMESTAMP}.rdb"
    echo "✅ Backup created: dump_${TIMESTAMP}.rdb"
    
    # Compress old backups
    find "$BACKUP_DIR" -name "dump_*.rdb" -mtime +1 -exec gzip {} \;
    
    # Clean old backups
    find "$BACKUP_DIR" -name "dump_*.rdb.gz" -mtime +${RETENTION_DAYS} -delete
    echo "🧹 Cleaned backups older than ${RETENTION_DAYS} days"
else
    echo "⚠️  No dump.rdb found"
    exit 1
fi

# Optional: Upload to S3 if credentials available
if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$S3_BACKUP_BUCKET" ]; then
    aws s3 cp "$BACKUP_DIR/dump_${TIMESTAMP}.rdb" \
        "s3://${S3_BACKUP_BUCKET}/automem/falkordb/dump_${TIMESTAMP}.rdb"
    echo "☁️  Uploaded to S3"
fi

echo "✅ Backup complete"
