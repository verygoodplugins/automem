# Migration Guide

This document provides step-by-step instructions for migrating between different AutoMem configurations.

**Heads up for existing deployments:** New installs default to **1024d** with Voyage (`voyage-4`). If you are only updating AutoMem and keeping the same embedding provider and model, `VECTOR_SIZE_AUTODETECT=true` (the default) can adopt an existing collection dimension to avoid a startup mismatch. It does **not** migrate embeddings. Any provider or model change requires backing up, recreating the Qdrant collection, and fully re-embedding every memory—even when both models output 1024 dimensions. To explicitly pin a dimension, set `VECTOR_SIZE=<your-dimension>`; set `VECTOR_SIZE_AUTODETECT=false` to fail on mismatch.

## Table of Contents

**Embedding provider and model migrations**
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

> **Re-embed prerequisite:** `scripts/reembed_embeddings.py` requires a full `QDRANT_URL` (for example, `http://localhost:6333`); unlike the API service, the script does not construct one from `QDRANT_HOST` and `QDRANT_PORT`.

---

## Migrating to 1024d (voyage-4 default)

**When to migrate:** If you're switching from OpenAI embeddings to Voyage AI (the new recommended default).

### Steps

1. **Backup your data**: `python scripts/backup_automem.py`
2. **Set environment variables**:
   ```bash
   export EMBEDDING_PROVIDER=voyage    # or auto (will prefer Voyage if VOYAGE_API_KEY is set)
   export VOYAGE_API_KEY=pa-...
   export VOYAGE_MODEL=voyage-4
   export VECTOR_SIZE=1024
   # The re-embed script needs a URL; derive one if your deployment uses host/port.
   export QDRANT_URL="${QDRANT_URL:-http://${QDRANT_HOST:-localhost}:${QDRANT_PORT:-6333}}"
   # Set QDRANT_API_KEY when the selected Qdrant instance requires authentication.
   export QDRANT_API_KEY="${QDRANT_API_KEY:-}"
   ```
3. **Pause writes, then delete and recreate the Qdrant collection**:
   ```bash
   curl -X DELETE "$QDRANT_URL/collections/memories" \
     -H "api-key: $QDRANT_API_KEY"
   curl -X PUT "$QDRANT_URL/collections/memories" \
     -H 'Content-Type: application/json' \
     -H "api-key: $QDRANT_API_KEY" \
     -d '{"vectors": {"size": 1024, "distance": "Cosine"}}'
   ```
4. **Re-embed all memories**:
   ```bash
   python scripts/reembed_embeddings.py --batch-size 32
   ```
5. **Verify**: Check that `/health` shows `vector_size: 1024` and recall returns results.

> `VECTOR_SIZE_AUTODETECT=true` can preserve the old collection dimension only when you keep its provider and model. It never makes an existing OpenAI, Voyage, Ollama, or FastEmbed vector compatible with another model space.

---

## Upgrading to 3072d Embeddings

**When to upgrade:** If an evaluation shows that you need the explicit OpenAI `text-embedding-3-large` option and can accept its extra storage and API usage. This is not the recommended cloud default; new deployments should use Voyage `voyage-4` at 1024d.

### Pros ✅
- **Higher-dimensional OpenAI option**: Native 3072d output
- **Explicit control**: Useful when an OpenAI-specific evaluation requires it

### Cons ❌
- **4x storage cost**: 768 → 3072 dimensions (4x more disk space)
- **More API usage cost**: Review current provider pricing before migration
- **More search work**: Larger vectors increase storage and computation
- **Migration required**: Cannot reuse existing embeddings

### Storage Impact

| Metric | 768d (small) | 3072d (large) | Multiplier |
|--------|--------------|---------------|------------|
| Vector dimensions | 768 | 3072 | 4x |
| Vector storage (all else equal) | Baseline | Approximately 4x | 4x |

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
echo "EMBEDDING_PROVIDER=openai" >> .env
echo "VECTOR_SIZE=3072" >> .env
echo "EMBEDDING_MODEL=text-embedding-3-large" >> .env
echo "QDRANT_URL=http://localhost:6333" >> .env
```

Or export temporarily:
```bash
export EMBEDDING_PROVIDER=openai
export VECTOR_SIZE=3072
export EMBEDDING_MODEL=text-embedding-3-large
export QDRANT_URL=http://localhost:6333
```

#### 3. Pause Writes and Recreate the Qdrant Collection

`reembed_embeddings.py` upserts into an existing collection; it does not recreate one. After the backup, stop or pause writes and recreate the collection for the new model space:

```bash
curl -X DELETE http://localhost:6333/collections/memories
curl -X PUT http://localhost:6333/collections/memories \
  -H 'Content-Type: application/json' \
  -d '{"vectors": {"size": 3072, "distance": "Cosine"}}'
```

#### 4. Re-embed All Memories
```bash
python scripts/reembed_embeddings.py --batch-size 32
```

This will:
- Fetch all memories from FalkorDB (source of truth)
- Generate new 3072d embeddings using OpenAI API
- Upsert all embeddings in batches

**Expected time:** ~5-10 minutes per 10k memories

#### 5. Verify Migration
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

#### 6. Test Recall
```bash
curl -X POST http://localhost:8001/recall \
  -H "Authorization: Bearer $AUTOMEM_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "test recall", "limit": 5}'
```

Verify results are returned and scores look reasonable.

#### 7. Restart Application
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
export EMBEDDING_PROVIDER=openai
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
export EMBEDDING_PROVIDER=openai
export VECTOR_SIZE=768
export EMBEDDING_MODEL=text-embedding-3-small
```

After backing up and pausing writes, recreate the collection with a 768d vector schema, then run:

```bash
python scripts/reembed_embeddings.py --batch-size 32
```

The script upserts the replacement vectors; it does not recreate the collection. A full re-embed is required even if a future source model happens to share the same dimension.

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
> (`--mode canonicalize-safe` → review the generated `plan.jsonl` →
> `--execute --plan <plan.jsonl>`, with `--rollback <rollback.jsonl>` available).

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
1. If you are keeping the same provider and model, set `VECTOR_SIZE_AUTODETECT=true` (default) to adopt the existing collection dimension
2. If you are keeping the same provider and model, set `VECTOR_SIZE=<existing-dimension>` in `.env`
3. If you are changing the provider or model, follow the [1024d](#migrating-to-1024d-voyage-4-default), [3072d](#upgrading-to-3072d-embeddings), or [768d](#downgrading-to-768d-embeddings) procedure: back up, recreate the collection, and fully re-embed

### Error: "OpenAI API rate limit"

**Symptom:**
```
Rate limit exceeded during re-embedding
```

**Solution:**
The `reembed_embeddings.py` script uses the configured provider (Voyage, OpenAI, Ollama, FastEmbed, and so on). For large datasets:
1. Run during off-peak hours
2. Increase your provider's rate limits if applicable
3. Split work into smaller embedding requests with `--batch-size`

### Error: "Qdrant collection already exists"

**Symptom:**
```
Collection 'memories' already exists with different dimension
```

**Solution:**
Back up, pause writes, then delete **and recreate** the collection with the selected model's dimension before re-embedding. The script only upserts vectors:
```bash
curl -X DELETE http://localhost:6333/collections/memories
curl -X PUT http://localhost:6333/collections/memories \
  -H 'Content-Type: application/json' \
  -d '{"vectors": {"size": 1024, "distance": "Cosine"}}'
python scripts/reembed_embeddings.py --batch-size 32
```

**⚠️ Warning:** Replace `1024` with the selected model's output size. This deletes all embeddings; make sure FalkorDB still has the memories so they can be regenerated.

### Migration is slow

**Symptoms:**
- Taking hours for thousands of memories
- High embedding-provider costs

**Solutions:**
1. **Check batch size**: Script defaults to batches of 32; lower `--batch-size` if the provider or host is constrained
2. **Spot check first**: On a newly recreated collection, test a subset before the full run:
   ```bash
   python scripts/reembed_embeddings.py --limit 100 --batch-size 32
   ```
   Then rerun without `--limit` to migrate every memory.
3. **Use a non-production provider configuration for experiments** and keep Voyage (`voyage-4`) as the cloud default for the final migration. See [Voyage pricing](https://docs.voyageai.com/docs/pricing) for current limits and rates.

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
3. ✅ **Pause writes and recreate the collection** - Required for every provider/model change, including same-dimension swaps
4. ✅ **Monitor costs** - Check the selected provider's usage dashboard during migration
5. ✅ **Document current state** - Note the current provider, model, and `VECTOR_SIZE`

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
