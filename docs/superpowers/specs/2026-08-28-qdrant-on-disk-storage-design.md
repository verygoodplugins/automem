# Qdrant On-Disk Storage Design

## Goal

Reduce AutoMem's Qdrant RAM residency by creating collections with on-disk
vectors, HNSW index, and payload storage, and provide a safe operator-run
rebuild path for the existing production collection.

## Scope

- New Qdrant collections use `on_disk=True` for vectors, `HnswConfigDiff(on_disk=True)`,
  and `on_disk_payload=True`.
- The existing restore utility gains an optional HNSW on-disk setting so an
  operator can rebuild a collection using the same profile.
- Documentation provides a production runbook for backup, Qdrant-only restore,
  verification, rollback, and a subsequent manual Railway volume resize.

## Non-goals

- Do not mutate an existing collection at application startup; Qdrant creation
  settings only apply to a newly created collection.
- Do not run a production migration, change Railway volume size, or alter
  replicas as part of this change.
- Do not change embedding dimensions, distance metric, payload indexes, or
  recall semantics.

## Architecture

`ensure_qdrant_collection()` remains responsible only for collection creation.
When it creates a collection, it passes the on-disk profile to Qdrant. Existing
collections keep their settings unchanged.

`scripts/restore_from_backup.py` remains the only destructive recovery path.
It will accept `QDRANT_RESTORE_HNSW_ON_DISK` alongside its existing optional
vector and payload storage settings and applies all selected settings only when
it recreates a collection during a forced Qdrant-only restore.

The runbook uses the existing authenticated `/backup` export and restore tool:
take a portable backup, rebuild Qdrant with explicitly set on-disk options,
validate collection point count and representative recall, and retain the
backup for rollback. Railway volume reduction is a manual post-verification
operation.

## Data Flow

1. A fresh AutoMem deployment starts with no Qdrant collection.
2. Runtime creates the collection with on-disk vector, HNSW, and payload
   settings, then creates the existing payload indexes.
3. For an existing collection, an operator exports a backup and runs a
   Qdrant-only forced restore with the three on-disk options enabled.
4. Restore deletes and recreates only Qdrant, upserts the backup's points, and
   waits for the expected point count.
5. The operator verifies count and recall before manually resizing Railway
   storage. A retained backup supports restoring the prior data if validation
   fails.

## Error Handling and Rollback

- Collection creation errors keep the existing fail-soft behavior: Qdrant is
  disabled and the service continues without vector storage.
- Restore remains opt-in and destructive only with its existing confirmation or
  `--force`; no startup path invokes it.
- The runbook requires retaining the pre-migration backup until validation is
  complete. Re-running restore from that backup recreates the collection if
  rollback is needed.
- Railway volume resize is deliberately outside automation because it is an
  infrastructure and capacity decision requiring observed post-migration usage.

## Testing

- Unit-test creation of a missing collection and assert all three on-disk
  settings are passed to `create_collection`.
- Unit-test restore configuration with HNSW on-disk enabled and disabled.
- Run focused Qdrant/restore tests, full unit tests, formatting, import sorting,
  lint, and compilation.

## Documentation

Update the environment-variable reference for the restore setting and the
backup/monitoring guide with the exact production migration, verification, and
rollback commands.
