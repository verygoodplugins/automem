# Migration Guide

This document provides step-by-step instructions for migrating between different AutoMem configurations.

**Heads up for existing deployments:** The default embedding dimension is now **1024d** (voyage-4) for new installs. If your Qdrant collection uses a different dimension (e.g. 3072d from `text-embedding-3-large` or 768d from `text-embedding-3-small`), **no action is needed** — `VECTOR_SIZE_AUTODETECT=true` (the default) automatically adopts your existing collection dimension on startup. To explicitly pin your dimension, set `VECTOR_SIZE=<your-dimension>` in your `.env`. To enforce strict matching (fail on mismatch), set `VECTOR_SIZE_AUTODETECT=false`.

## Table of Contents

**Embedding dimension migrations**
- [Migrating to 1024d (voyage-4 default)](#migrating-to-1024d-voyage-4-default)
- [Upgrading to 3072d Embeddings](#upgrading-to-3072d-embeddings)
- [Downgrading to 768d Embeddings](#downgrading-to-768d-embeddings)

**Data & schema migrations**
- [Upgrading to 0.16.0](#upgrading-to-0160)
- [Importing from the legacy MCP SQLite store](#importing-from-the-legacy-mcp-sqlite-store)

**Reference**
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

> Every script below is cataloged in [scripts/README.md](../scripts/README.md).
> The embedding-dimension sections are about re-vectorizing; the data & schema
> sections are one-time, idempotent FalkorDB/Qdrant migrations.

---

## Migrating to 1024d (voyage-4 default)

**When to migrate:** If you're switching from OpenAI embeddings to Voyage AI (the new recommended default).

### Steps

1. **Backup your data**: `python scripts/backup_automem.py`
2. **Set environment variables**:
   ```bash
   EMBEDDING_PROVIDER=voyage    # or auto (will prefer Voyage if VOYAGE_API_KEY is set)
   VOYAGE_API_KEY=pa-...
   VECTOR_SIZE=1024
   ```
3. **Delete and recreate the Qdrant collection**:
   ```bash
   curl -X DELETE http://localhost:6333/collections/memories
   ```
4. **Re-embed all memories**:
   ```bash
   python scripts/reembed_embeddings.py
   ```
5. **Verify**: Check that `/health` shows `vector_size: 1024` and recall returns results.

> **Alternatively**, set `VECTOR_SIZE_AUTODETECT=true` (the default) and AutoMem will adopt your existing collection dimension without migration. Only migrate when you want to switch embedding providers.

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
docker compose up -d

# Or rerun the foreground dev stack
make dev

# If using systemd
sudo systemctl restart automem

# If using Railway
railway up
```

### Rollback Procedure
If migration fails or results are poor:

```bash
# 1. Stop application
docker compose down  # or however you run AutoMem

# 2. Restore from backup
python scripts/restore_from_backup.py backups/qdrant/qdrant_snapshot_YYYYMMDD_HHMMSS.tar.gz
python scripts/restore_from_backup.py backups/falkordb/memories_YYYYMMDD_HHMMSS.rdb

# 3. Revert configuration
export VECTOR_SIZE=768
export EMBEDDING_MODEL=text-embedding-3-small

# 4. Restart
docker compose up -d
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

## Upgrading to 0.16.0

0.16.0 adds several **one-time, idempotent** data migrations. None are required
to keep an existing instance running — run them to adopt the new behavior. Each
is safe to re-run and most support `--dry-run`. **Back up first**
(`python scripts/backup_automem.py`).

### Promote entity tags to first-class Entity nodes

Earlier versions recorded entities only as `entity:{category}:{slug}` tags on
Memory nodes. This migration creates real `Entity` nodes and links them with
`REFERENCED_IN` relationships, which enables entity-centric recall and graph
queries.

```bash
python scripts/migrate_entity_nodes.py --dry-run   # preview
python scripts/migrate_entity_nodes.py             # apply (idempotent)
```

> First-class entity-node **synthesis** (`IDENTITY_SYNTHESIS_ENABLED`) is
> experimental and gated **off** by default while people-entity word-pair noise
> is addressed ([#181](https://github.com/verygoodplugins/automem/issues/181)).
> Promote nodes when you opt into that feature. If your clone has noisy generated
> entity tags, clean them first with
> [`scripts/lab/repair_entity_tags.py`](../scripts/lab/repair_entity_tags.py)
> (`--mode audit` → review the plan → `--mode execute`, with `--mode rollback`
> available).

### Backfill tag prefixes

Recomputes the `tag_prefixes` sidecar on every memory in FalkorDB and Qdrant so
prefix-match recall stays consistent with the current tag set. Run once after
upgrading.

```bash
python scripts/backfill_tag_prefixes.py
```

### Rescore relevance

Recomputes every `relevance_score` with the corrected consolidation decay
(`base_decay_rate=0.01` + importance floor), undoing the damage from the old
over-aggressive `0.1` rate. Targets a local clone by default; pass `--target` to
point at production.

```bash
python scripts/rescore_relevance.py --dry-run     # preview against local
python scripts/rescore_relevance.py               # apply against local
```

### Clean up invalid memory types

Reclassifies legacy/invalid type values (e.g. `session_start`, `interaction`)
back to the valid set (`Decision`, `Pattern`, `Preference`, `Style`, `Habit`,
`Insight`, `Context`). Reads `.env`; takes no flags.

```bash
python scripts/cleanup_memory_types.py
```

---

## Importing from the legacy MCP SQLite store

If you're moving off the old MCP memory service, replay its `sqlite_vec.db` into
AutoMem via the API. Original timestamps, tags, and importance are preserved, and
the legacy payload is kept under `metadata['legacy']`. Always `--dry-run` first.

```bash
# Preview what will be imported
python scripts/migrate_mcp_sqlite.py --dry-run

# Run the import against a deployed instance
python scripts/migrate_mcp_sqlite.py \
  --db /path/to/sqlite_vec.db \
  --automem-url https://automem.example.com \
  --api-token $AUTOMEM_API_TOKEN

# Refresh embeddings afterward
python scripts/reembed_embeddings.py --limit 200
```

---

## Troubleshooting

### Error: "Vector dimension mismatch"

**Symptom** (only when `VECTOR_SIZE_AUTODETECT=false`):
```
FATAL: Vector dimension mismatch detected!
  Existing Qdrant collection: 3072d
  Configured VECTOR_SIZE:     1024d
```

**Solution** (pick one):
1. Set `VECTOR_SIZE_AUTODETECT=true` (default) to automatically adopt the existing collection dimension
2. Set `VECTOR_SIZE=<existing-dimension>` in your `.env` to match your data
3. Migrate to the new dimension: follow the [1024d](#migrating-to-1024d-voyage-4-default), [3072d](#upgrading-to-3072d-embeddings), or [768d](#downgrading-to-768d-embeddings) migration steps above

### Error: "OpenAI API rate limit"

**Symptom:**
```
Rate limit exceeded during re-embedding
```

**Solution:**
The `reembed_embeddings.py` script uses whatever embedding provider is configured (Voyage, OpenAI, local, etc.). For large datasets:
1. Run during off-peak hours
2. Increase your provider's rate limits if applicable
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
