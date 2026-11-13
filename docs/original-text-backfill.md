# Original-text search backfill

`python scripts/backfill_original_search.py` prepares the graph-side search source for existing `Memory` nodes. It sets the `original_search_text` property and `MemoryOriginalText` label together in a single guarded Cypher update. It does **not** create the database full-text index, modify Qdrant, change memory content or metadata, call an embedding provider, or remove memories. Ensure the service has created the corresponding full-text index and check its database status separately; this command's verification covers properties and labels, not full-text index readiness or query ranking.

The command is read-only unless `--apply` is supplied. Supply an explicit loopback host (or `FALKORDB_HOST`) and port. A non-loopback host requires `--allow-remote`; that switch does not imply that a remote migration is safe. `FALKORDB_PASSWORD` and `FALKORDB_USERNAME` are read from the environment and are not printed.

```sh
# Read-only plan. Reports counts and the last scanned ID; does not write.
python scripts/backfill_original_search.py --host 127.0.0.1 --port 6379 --dry-run

# Only after inspecting the plan: apply the safe default.
python scripts/backfill_original_search.py --host 127.0.0.1 --port 6379 --apply

# Independent read-only coverage scan of the property and label.
python scripts/backfill_original_search.py --host 127.0.0.1 --port 6379 --verify
```

## Legacy originals are not provably current

Earlier single-memory ingest saved oversized originals in `metadata.original_content`, but older content PATCH operations could replace the current content without clearing that metadata. Their `updated_at` is not reliable evidence of freshness. The backfill cannot distinguish a still-valid original from a stale one from these records alone. By default it indexes the **current `content`**, not that ambiguous original, and reports `unverified_legacy`. Existing nonempty `original_search_text` values are preserved as authoritative; `--verify` reports originals already indexed as `index_uses_unverified_original` without asserting that they are stale.

If an operator has independently established that a legacy corpus has no stale original metadata, the explicitly risky option below indexes preserved originals on legacy rows lacking an index, and upgrades a previous fallback only when the existing index equals the current content:

```sh
python scripts/backfill_original_search.py --host 127.0.0.1 --port 6379 \
  --assume-legacy-original-current --dry-run
python scripts/backfill_original_search.py --host 127.0.0.1 --port 6379 \
  --assume-legacy-original-current --apply
```

The opt-in is an **assumption**, not a validation. It can expose stale facts to search. Content and metadata should be independently audited before applying it. Historical batch ingestion sometimes summarized without preserving any original or a summary marker; such originals cannot be reconstructed and the affected records cannot reliably be counted. `known_original_missing` only counts records that explicitly say `was_summarized: true` but have no usable original.

## Progress, concurrency, and verification

IDs are scanned in ascending order with keyset batches (`--batch-size`, default 250, maximum 1000). `--limit N` bounds a run; `--after-id ID` resumes strictly after the last ID. Each write matches an existing memory, requires exactly one node for that ID, and compares its current content, raw metadata and prior index property to the values observed during scanning. It never uses `MERGE`, so a concurrent deletion cannot recreate a record. Immediately after each successful update the script reads the node back and verifies those values and the label.

The JSON result distinguishes the pre-write coverage counts (`indexed`, `missing_index`, `missing_label`), planned and applied writes, pre-existing indexes preserved, unverifiable originals, known missing originals, malformed rows, conflicts and verification mismatches. `conflict_ids` and `verification_mismatch_ids` identify retries. The script exits nonzero for malformed rows, missing IDs, duplicate ID groups, compare-and-set conflicts or failed verification; `--verify` also exits nonzero on missing index properties or labels. Duplicate IDs abort the scan without writing. An indexable row with malformed metadata is skipped, not silently rewritten.

Rerun from the beginning after conflicts or concurrent writes: records can be inserted at or before a completed keyset cursor, and a failed ID can be behind `last_id`. Repeated runs are idempotent and preserve already indexed content. Run a final `--verify` over the full scope after writes have settled; its coverage counts do not prove that any legacy original is current. Do not treat a limited or resumed verification as full-corpus verification.

## Tests

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_backfill_original_search.py -q
# Opt-in real FalkorDB test: creates a uniquely named disposable Docker container,
# publishes only a random loopback port, uses tmpfs, and removes that container.
RUN_ORIGINAL_INDEX_DOCKER_TESTS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest tests/test_backfill_original_search.py -q
```

The disposable integration test exercises actual FalkorDB keyset traversal, null-aware compare-and-set, label creation, native full-text indexing and updates, idempotent reruns, explicit legacy opt-in, and duplicate-ID rejection. It never connects to an existing AutoMem stack.

## Current-state verification

For current-only recall, an error while querying replacement relationships now
returns HTTP 503 rather than assuming that no replacement exists. History-mode
recall does not perform this replacement check. This prevents a graph query
timeout from silently returning an unverified superseded memory as current.
The existing vector-only fallback when the graph itself is unavailable remains
unchanged: it can filter lifecycle timestamps in vector payloads, but cannot
verify supersession relationships. Consumers that require verified current
state should ensure the graph is available.
