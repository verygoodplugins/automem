# Migration Guide

This document provides step-by-step instructions for migrating between different AutoMem configurations.

**Heads up for existing deployments:** The default embedding dimension is now **3072d** for new installs. If your Qdrant collection is still 768d, set `VECTOR_SIZE=768` (and keep `text-embedding-3-small`) until you complete the upgrade steps below. AutoMem will fail fast if the configured dimension does not match your collection to prevent accidental corruption.

## Table of Contents
- [Upgrading to 3072d Embeddings](#upgrading-to-3072d-embeddings)
- [Downgrading to 768d Embeddings](#downgrading-to-768d-embeddings)
- [Troubleshooting](#troubleshooting)

---

## Upgrading to 3072d Embeddings

**When to upgrade:** If you need better semantic precision and have the storage budget for 4x larger embeddings.

### Pros ✅
- **Better semantic precision**: ~5-10% improvement on benchmarks
- **Improved multi-hop reasoning**: Better at connecting related concepts
- **Recommended for production**: If accuracy is critical and storage is not a constraint

### Cons ❌
- **4x storage cost**: 768 → 3072 dimensions (4x more disk space)
- **4x embedding cost**: OpenAI charges per dimension
- **~20% slower search**: More dimensions = more computation
- **Migration required**: Cannot reuse existing embeddings

### Cost Comparison

| Metric | 768d (small) | 3072d (large) | Multiplier |
|--------|--------------|---------------|------------|
| Storage per 1M memories | ~3GB | ~12GB | 4x |
| OpenAI cost per 1M tokens | $0.02 | $0.13 | 6.5x |
| Search latency | ~50ms | ~60ms | 1.2x |
| Benchmark accuracy | 88.2% | 90.5% | +2.3pp |

### Migration Steps

#### 1. Backup Your Data
```bash
python scripts/backup_automem.py
```

This creates timestamped backups in `backups/`:
- `backups/falkordb/memories_YYYYMMDD_HHMMSS.rdb`
- `backups/qdrant/qdrant_snapshot_YYYYMMDD_HHMMSS.tar.gz`

#### 2. Update Configuration
```bash
# Add to your .env file
echo "VECTOR_SIZE=3072" >> .env
echo "EMBEDDING_MODEL=text-embedding-3-large" >> .env
```

Or export temporarily:
```bash
export VECTOR_SIZE=3072
export EMBEDDING_MODEL=text-embedding-3-large
```

#### 3. Re-embed All Memories
```bash
python scripts/reembed_embeddings.py
```

This will:
- Fetch all memories from FalkorDB (source of truth)
- Generate new 3072d embeddings using OpenAI API
- Recreate Qdrant collection with new dimensions
- Upsert all embeddings in batches

**Expected time:** ~5-10 minutes per 10k memories

#### 4. Verify Migration
Check Qdrant collection info:
```bash
curl http://localhost:6333/collections/memories | jq '.result.config.params.vectors'
```

Should show:
```json
{
  "size": 3072,
  "distance": "Cosine"
}
```

#### 5. Test Recall
```bash
curl -X POST http://localhost:8001/recall \
  -H "Authorization: Bearer $AUTOMEM_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "test recall", "limit": 5}'
```

Verify results are returned and scores look reasonable.

#### 6. Restart Application
```bash
# If using Docker
make restart

# If using systemd
sudo systemctl restart automem

# If using Railway
railway up
```

### Rollback Procedure
If migration fails or results are poor:

```bash
# 1. Stop application
docker-compose down  # or however you run AutoMem

# 2. Restore from backup
python scripts/restore_from_backup.py backups/qdrant/qdrant_snapshot_YYYYMMDD_HHMMSS.tar.gz
python scripts/restore_from_backup.py backups/falkordb/memories_YYYYMMDD_HHMMSS.rdb

# 3. Revert configuration
export VECTOR_SIZE=768
export EMBEDDING_MODEL=text-embedding-3-small

# 4. Restart
docker-compose up -d
```

---

## Downgrading to 768d Embeddings

**When to downgrade:** If storage costs are too high or 3072d isn't providing enough value.

### Steps

Follow the same migration steps above, but use:
```bash
export VECTOR_SIZE=768
export EMBEDDING_MODEL=text-embedding-3-small
```

Then run `reembed_embeddings.py` to recreate the collection with 768d vectors.

---

## Troubleshooting

### Error: "Vector dimension mismatch"

**Symptom:**
```
ValueError: VECTOR DIMENSION MISMATCH
Qdrant collection 'memories': 768d
Config VECTOR_SIZE: 3072d
```

**Solution:**
Either:
- Keep existing: `export VECTOR_SIZE=768`
- Migrate: Follow "Upgrading to 3072d Embeddings" above

### Error: "OpenAI API rate limit"

**Symptom:**
```
Rate limit exceeded during re-embedding
```

**Solution:**
The `reembed_embeddings.py` script has exponential backoff, but for large datasets:
1. Run during off-peak hours
2. Increase OpenAI rate limits (pay-as-you-go tier)
3. Split migration into batches using `--batch-size` flag

### Error: "Qdrant collection already exists"

**Symptom:**
```
Collection 'memories' already exists with different dimension
```

**Solution:**
Delete and recreate:
```bash
curl -X DELETE http://localhost:6333/collections/memories
python scripts/reembed_embeddings.py
```

**⚠️ Warning:** This deletes all embeddings. Make sure FalkorDB still has the memories (embeddings will be regenerated from there).

### Migration is slow

**Symptoms:**
- Taking hours for thousands of memories
- High OpenAI API costs

**Solutions:**
1. **Check batch size**: Script uses batches of 100 by default
2. **Parallel processing**: Use `--workers` flag (if implemented)
3. **Spot check first**: Test on a subset before full migration
4. **Use cheaper model for testing**: 
   ```bash
   export EMBEDDING_MODEL=text-embedding-3-small
   python scripts/reembed_embeddings.py --dry-run
   ```

### Backup failed

**Symptoms:**
- Backup script errors
- Empty backup files

**Solutions:**
1. **Check disk space**: `df -h`
2. **Check permissions**: `ls -la backups/`
3. **Manual backup**:
   ```bash
   # FalkorDB
   docker exec automem-falkordb-1 redis-cli --rdb /data/dump.rdb
   
   # Qdrant
   curl -X POST http://localhost:6333/collections/memories/snapshots
   ```

---

## Best Practices

### Before Any Migration
1. ✅ **Always backup first** - Don't skip this step
2. ✅ **Test in staging** - If you have a staging environment
3. ✅ **Monitor costs** - Check OpenAI usage dashboard during migration
4. ✅ **Document current state** - Note current VECTOR_SIZE and EMBEDDING_MODEL

### After Migration
1. ✅ **Run benchmark tests** - Verify accuracy hasn't degraded
2. ✅ **Monitor performance** - Check search latency and throughput
3. ✅ **Update documentation** - Note when migration occurred and why
4. ✅ **Store migration record**:
   ```bash
   curl -X POST http://localhost:8001/memory \
     -H "Authorization: Bearer $AUTOMEM_API_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "content": "Migrated to 3072d embeddings for better semantic precision",
       "tags": ["migration", "config", "embeddings"],
       "importance": 0.8
     }'
   ```

### Choosing Between 768d and 3072d

**Use 768d (text-embedding-3-small) if:**
- Cost-conscious deployment
- Storage is limited
- Speed > slight accuracy gains
- Personal/development use
- Small dataset (<100k memories)

**Use 3072d (text-embedding-3-large) if:**
- Production deployment
- Accuracy is critical
- Complex multi-hop reasoning needed
- Large dataset benefits from precision
- Storage/compute costs are acceptable

---

## Related Documentation

- [Environment Variables](ENVIRONMENT_VARIABLES.md) - Configuration reference
- [Testing Guide](TESTING.md) - Benchmark testing
- [Monitoring & Backups](MONITORING_AND_BACKUPS.md) - Backup strategies
- [Railway Deployment](RAILWAY_DEPLOYMENT.md) - Cloud deployment guide

