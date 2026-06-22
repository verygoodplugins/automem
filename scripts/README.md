# AutoMem scripts

Operational, migration, recovery, and evaluation tooling for an AutoMem
instance. This file is the catalog — the single place to find "what scripts
exist and when do I run each one." For the narrative runbooks (why and when),
see the [docs portal](https://automem.ai/docs/); for exact flags, run a script
with `--help` or read its docstring header.

## How to read this

Most Python scripts are self-documenting: run `python scripts/<name>.py --help`,
or read the docstring at the top of the file. Most connect to FalkorDB (and
Qdrant) using credentials from `.env` in the repo root or
`~/.config/automem/.env`. A handful of harness scripts are fronted by `make`
targets (noted inline).

Each script is tagged by **lifecycle** — the single most important thing to know
before running it:

| Tag | Meaning |
|---|---|
| `routine` | Safe to run repeatedly as normal operations. |
| `one-time` | Run once per instance or per upgrade. Idempotent, but not part of day-to-day ops. See [docs/MIGRATIONS.md](../docs/MIGRATIONS.md). |
| `recovery` | Break-glass. Only after data loss or corruption. See [docs/MONITORING_AND_BACKUPS.md](../docs/MONITORING_AND_BACKUPS.md). |
| `dev` | Local development / deployment helpers. |
| `bench` · `lab` | Contributor evaluation and recall-tuning harnesses. Not needed to run AutoMem. See [docs/TESTING.md](../docs/TESTING.md) and [docs/RECALL_QUALITY_LAB.md](../docs/RECALL_QUALITY_LAB.md). |

---

## Routine operations

Run these as part of normal upkeep.

| Script | Lifecycle | What it does |
|---|---|---|
| [`backup_automem.py`](backup_automem.py) | `routine` | Timestamped FalkorDB + Qdrant backup; optional S3 upload and old-backup cleanup. Cron-friendly. `--s3-bucket`, `--cleanup --keep N`. |
| [`restore_from_backup.py`](restore_from_backup.py) | `routine` · `recovery` | Restore FalkorDB + Qdrant from a backup (local snapshot or downloaded API tarball). `--backup-timestamp`, `--backup-dir snapshot.tar.gz`, `--dry-run`. |
| [`health_monitor.py`](health_monitor.py) | `routine` | Background service: watches FalkorDB/Qdrant health, checks graph↔vector consistency, triggers recovery, alerts. `--interval 300`. Containerized via [`Dockerfile.health-monitor`](Dockerfile.health-monitor). See [docs/HEALTH_MONITORING.md](../docs/HEALTH_MONITORING.md). |
| [`automem_watch.py`](automem_watch.py) | `routine` | Real-time terminal UI over the SSE event stream; flags garbage-write patterns and consolidation timing. `--url`, `--token`. |
| [`audit_relevance.py`](audit_relevance.py) | `routine` | Audit the `relevance_score` distribution from a backup file (default) or a live instance (`--live`). |
| [`reembed_embeddings.py`](reembed_embeddings.py) | `routine` | Re-embed all memories and upsert vectors into Qdrant using the configured provider. `--batch-size`, `--limit`. Used after provider/dimension changes — see [docs/MIGRATIONS.md](../docs/MIGRATIONS.md). |
| [`reclassify_with_llm.py`](reclassify_with_llm.py) | `routine` | Reclassify fallback `type='Memory'` records via the configured classification LLM. `--provider`; env `CLASSIFICATION_MODEL` / `CLASSIFICATION_BASE_URL` / `CLASSIFICATION_API_KEY`. |
| [`reenrich_batch.py`](reenrich_batch.py) | `routine` | Re-run enrichment over a batch of memories with current classification logic. |

### `browse_memories.py` — read-only database browser `dev`

Interactive CLI over the production FalkorDB graph + Qdrant vectors. Connects
with `.env` credentials and **never modifies data**. Four subcommands:

```bash
# search — by text, date range, type, tag, importance
python scripts/browse_memories.py search --text "Eva" --from 2025-10
python scripts/browse_memories.py search --type Decision --min-importance 0.8 --sort relevance -n 50
python scripts/browse_memories.py search --text "old project" --include-archived

# inspect — full record for one memory (4+ char id prefix works)
python scripts/browse_memories.py inspect 2751e70e

# stats — overview; --full adds a FalkorDB↔Qdrant consistency check
python scripts/browse_memories.py stats --full

# diagnose — why a memory isn't surfacing in recall (decay, access,
# relationships, importance floor, embedding quality, current weights)
python scripts/browse_memories.py diagnose 2751e70e
```

`inspect` shows full content, all FalkorDB properties, Qdrant presence/payload,
and every graph relationship. `diagnose` reports issues at `[CRITICAL]` /
`[WARNING]` / `[INFO]` severity.

---

## One-time migrations

Run once per instance or when upgrading. Idempotent and safe to re-run, but not
part of routine ops. Full runbook: [docs/MIGRATIONS.md](../docs/MIGRATIONS.md).

| Script | Lifecycle | What it does |
|---|---|---|
| [`migrate_mcp_sqlite.py`](migrate_mcp_sqlite.py) | `one-time` | Import the legacy MCP `sqlite_vec.db` memory store into AutoMem via the API, preserving timestamps/tags/importance. `--db`, `--automem-url`, `--api-token`, `--dry-run`. |
| [`migrate_entity_nodes.py`](migrate_entity_nodes.py) | `one-time` | Promote `entity:{category}:{slug}` tags on Memory nodes into first-class `Entity` nodes linked by `REFERENCED_IN`. `--dry-run`. (0.16.0) |
| [`backfill_tag_prefixes.py`](backfill_tag_prefixes.py) | `one-time` | Compute and backfill `tag_prefixes` in FalkorDB + Qdrant from existing tags (keeps prefix-match recall consistent). |
| [`rescore_relevance.py`](rescore_relevance.py) | `one-time` | Recompute every `relevance_score` with the corrected decay formula (undoes the old over-aggressive 0.1 rate). `--dry-run`, `--target`. |
| [`cleanup_memory_types.py`](cleanup_memory_types.py) | `one-time` | Reclassify invalid memory types (e.g. `session_start`, `interaction`) back to valid types. No flags; reads `.env`. |

> `scripts/lab/repair_entity_tags.py` (`lab`, below) is the companion repair tool
> for entity-tag noise on a local clone before promoting entity nodes.

---

## Break-glass recovery

Only reach for these after data loss or corruption. See
[docs/MONITORING_AND_BACKUPS.md](../docs/MONITORING_AND_BACKUPS.md).

| Script | Lifecycle | What it does |
|---|---|---|
| [`recover_from_qdrant.py`](recover_from_qdrant.py) | `recovery` | Rebuild the FalkorDB graph from Qdrant: reads every vector's payload and re-inserts via the API, which regenerates relationships. No flags; reads `.env`. |
| [`deduplicate_qdrant.py`](deduplicate_qdrant.py) | `recovery` | Remove duplicate Qdrant points (e.g. after a recovery run double-inserted). `--dry-run`, `--auto-confirm`. |
| [`restore_from_backup.py`](restore_from_backup.py) | `recovery` · `routine` | See Routine operations above. |

---

## Developer & deployment

| Script | Lifecycle | What it does | Make target |
|---|---|---|---|
| [`bootstrap_dev.sh`](bootstrap_dev.sh) | `dev` | Create `.venv` (Python 3.12), refresh `venv -> .venv`, install dev deps + pre-commit hooks. | `make install` |
| [`deploy_check.sh`](deploy_check.sh) | `dev` | Compare the live Railway deployment commit against `origin/main` to catch a silently disconnected GitHub integration. `DEPLOY_CHECK_QUIET=1` for CI. | `make deploy-check` |

---

## Benchmark harness — `scripts/bench/`

Snapshot-based LoCoMo / LongMemEval evaluation. See [docs/TESTING.md](../docs/TESTING.md).

| Script | What it does | Make target |
|---|---|---|
| [`bench/ingest_and_snapshot.sh`](bench/ingest_and_snapshot.sh) | Ingest a benchmark dataset into Docker AutoMem and snapshot the volumes (run once). | `make bench-ingest BENCH=locomo` |
| [`bench/restore_and_eval.sh`](bench/restore_and_eval.sh) | Restore a snapshot and evaluate a config (no re-ingest). | `make bench-eval BENCH=locomo CONFIG=baseline` |
| [`bench/compare_configs.sh`](bench/compare_configs.sh) | A/B two scoring configs against the same snapshot. | `make bench-compare` |
| [`bench/compare_branch.sh`](bench/compare_branch.sh) | Compare a git branch against `main` on a snapshot. | `make bench-compare-branch BRANCH=…` |
| [`bench/compare_results.py`](bench/compare_results.py) | Side-by-side table of two result JSON files. `--baseline`, `--test`, `--output`. | — |
| [`bench/analyze_locomo_results.py`](bench/analyze_locomo_results.py) | Markdown failure report from a LoCoMo results JSON. `--output`. (0.16.0) | — |
| [`bench/health_check.py`](bench/health_check.py) | Post-restore diagnostics: score distribution, entity quality, latency, precision on curated queries. | `make bench-health` |
| [`run_longmemeval_watch.sh`](run_longmemeval_watch.sh) | LongMemEval with persistent logging + desktop completion/crash notifications. | `make test-longmemeval-watch` |

---

## Recall Quality Lab — `scripts/lab/`

Data-driven recall scoring experiments against a clone of production. Full
workflow: [docs/RECALL_QUALITY_LAB.md](../docs/RECALL_QUALITY_LAB.md).

| Script | What it does | Make target |
|---|---|---|
| [`lab/clone_production.sh`](lab/clone_production.sh) | Clone production data into an isolated local Docker stack (direct DB backup, or `--restore-only` from a saved API tarball; supports custom ports for parallel sweeps). | `make lab-clone` |
| [`lab/create_test_queries.py`](lab/create_test_queries.py) | Generate a natural-question test set from local memories (via GPT-4o-mini). `--count`, `--output`, `--api-url`. | `make lab-queries` |
| [`lab/run_recall_test.py`](lab/run_recall_test.py) | Run a test set under a config, compute Recall@K / MRR / NDCG, A/B compare, and sweep a parameter. `--config`, `--compare`, `--sweep`. | `make lab-test` · `make lab-compare` · `make lab-sweep` |
| [`lab/repair_entity_tags.py`](lab/repair_entity_tags.py) | Audit → plan → execute/rollback repair of noisy generated entity tags on a clone. `--mode audit\|execute\|rollback`. (0.16.0) | — |
| [`lab/lab_metrics.py`](lab/lab_metrics.py) | **Library module** (not a CLI): pure, deterministic IR scoring functions (Recall@K, MRR, NDCG, distractor rate). Imported by `run_recall_test.py`. (0.16.0) | — |
| [`lab/lab_corpus.py`](lab/lab_corpus.py) | **Library module** (not a CLI): recall/corpus HTTP helpers behind injectable clients for unit-testable lab logic. (0.16.0) | — |
| `lab/configs/` | JSON scoring-weight overrides for A/B testing (`baseline.json`, `issue78_*.json`). | — |

---

## Shared

| File | What it is |
|---|---|
| [`lib/common.sh`](lib/common.sh) | Shell helpers (color codes, `wait_for_api`) sourced by the `bench/` scripts. |
| [`archive/`](archive/) | Retired one-off scripts kept for reference (e.g. dated release-sweep summarizers). Not maintained. |

---

## See also

- [docs/MIGRATIONS.md](../docs/MIGRATIONS.md) — embedding-dimension and one-time data migrations
- [docs/MONITORING_AND_BACKUPS.md](../docs/MONITORING_AND_BACKUPS.md) — backup/restore/recovery runbook
- [docs/HEALTH_MONITORING.md](../docs/HEALTH_MONITORING.md) — `health_monitor.py` deployment
- [docs/RECALL_QUALITY_LAB.md](../docs/RECALL_QUALITY_LAB.md) — the `lab/` harness end to end
- [docs/TESTING.md](../docs/TESTING.md) — unit, integration, and benchmark testing
