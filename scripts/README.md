# AutoMem scripts

Operational, migration, recovery, and evaluation tooling for an AutoMem
instance. This is the canonical inventory: every active executable in
`scripts/`, when to use it, and the safest way to start.

Run commands from the repository root with the project environment available:

```bash
source .venv/bin/activate
```

Most CLI scripts support `--help`; `cleanup_memory_types.py` and
`recover_from_qdrant.py` intentionally have no flags, so use their documented
commands only. For narrative runbooks, follow the linked documents rather than
inventing a new sequence.

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
| `read-only` | Inspects data or streams events without writing. |
| `maintenance` | Operational tool that may write or queue work; review the scope and back up first when it changes stored data. |
| `one-time` | Run once per instance or per upgrade. Idempotent where noted, but not part of day-to-day ops. See [docs/MIGRATIONS.md](../docs/MIGRATIONS.md). |
| `recovery` | Break-glass and potentially destructive. Only after data loss or corruption. See [docs/MONITORING_AND_BACKUPS.md](../docs/MONITORING_AND_BACKUPS.md). |
| `dev` | Local development / deployment helpers. |
| `bench` · `lab` | Contributor evaluation and recall-tuning harnesses. Not needed to run AutoMem. See [docs/TESTING.md](../docs/TESTING.md) and [docs/RECALL_QUALITY_LAB.md](../docs/RECALL_QUALITY_LAB.md). |

Before running a script that can change FalkorDB or Qdrant, confirm the target
from `.env`, take a backup, and start with `--dry-run` or a small `--limit` when
the script offers one. Do not run recovery, re-embedding, or migration scripts
against production while writers are active.

---

## Operations and maintenance

Use these for normal operations or deliberate small maintenance tasks.

| Script | Lifecycle | When to use it | Start here |
|---|---|---|---|
| [`backup_automem.py`](backup_automem.py) | `maintenance` | Create a portable FalkorDB + Qdrant backup before a migration or on a schedule. | `python scripts/backup_automem.py`; add `--s3-bucket … --cleanup --keep 7` for off-site retention. |
| [`health_monitor.py`](health_monitor.py) | `maintenance` | Continuously check API, FalkorDB, Qdrant, and drift; use alert-only mode by default. | `python scripts/health_monitor.py --once` or `--interval 300`. `--auto-recover` is an explicit, high-risk opt-in. See [docs/HEALTH_MONITORING.md](../docs/HEALTH_MONITORING.md). |
| [`automem_watch.py`](automem_watch.py) | `read-only` | Observe live store/recall/update/delete, enrichment, and consolidation events. | `python scripts/automem_watch.py --url "$AUTOMEM_API_URL" --token "$AUTOMEM_API_TOKEN"` |
| [`audit_relevance.py`](audit_relevance.py) | `read-only` | Inspect `relevance_score` distribution before or after scoring changes. | `python scripts/audit_relevance.py` reads the newest backup; use `--live` only for the configured instance. |
| [`reembed_embeddings.py`](reembed_embeddings.py) | `maintenance` | Replace vectors after an embedding provider, model, or dimension migration. | After backing up and recreating the collection, run `python scripts/reembed_embeddings.py --batch-size 32`. Requires `QDRANT_URL`; use `--limit 100` only for a smoke check. See [docs/MIGRATIONS.md](../docs/MIGRATIONS.md). |
| [`reclassify_with_llm.py`](reclassify_with_llm.py) | `maintenance` | Reclassify fallback `type='Memory'` records after changing classification logic. | Start with `python scripts/reclassify_with_llm.py --dry-run --limit 25`; apply only after review (the script asks for confirmation unless `--yes` is supplied). |
| [`reenrich_batch.py`](reenrich_batch.py) | `maintenance` | Queue a small batch for re-enrichment after an enrichment logic change. | `python scripts/reenrich_batch.py --limit 10`; this calls the configured API and queues work. |

### `browse_memories.py` — read-only database browser

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

Run when adopting a specific upgrade or repair. Most are idempotent, but they
are not routine operations; back up first and follow the linked runbook.

| Script | When to use it | Start here |
|---|---|---|
| [`migrate_mcp_sqlite.py`](migrate_mcp_sqlite.py) | Moving from the legacy MCP `sqlite_vec.db` store into AutoMem. | Preview first: `python scripts/migrate_mcp_sqlite.py --dry-run`; then pass `--db`, `--automem-url`, and `--api-token`. |
| [`migrate_entity_nodes.py`](migrate_entity_nodes.py) | Adopting first-class `Entity` nodes for legacy `entity:{category}:{slug}` tags. | `python scripts/migrate_entity_nodes.py --dry-run`, then rerun without the flag. |
| [`backfill_tag_prefixes.py`](backfill_tag_prefixes.py) | Restoring or introducing `tag_prefixes` so prefix recall remains consistent. | `python scripts/backfill_tag_prefixes.py --dry-run`; apply after review. Use `--no-qdrant` only when vector payload sync is intentionally deferred. |
| [`rescore_relevance.py`](rescore_relevance.py) | Repairing relevance scores produced with the previous over-aggressive decay formula. | `python scripts/rescore_relevance.py --dry-run`, then choose the intended `--target` before applying. |
| [`cleanup_memory_types.py`](cleanup_memory_types.py) | Repairing legacy invalid types such as `session_start` or `interaction`. | Back up, verify `.env` targets the intended instance, then run `python scripts/cleanup_memory_types.py`. It has no preview mode. |

> `scripts/lab/repair_entity_tags.py` (`lab`, below) is the companion repair tool
> for entity-tag noise on a local clone before promoting entity nodes.

---

## Break-glass recovery

Only reach for these after data loss or corruption. See
[docs/MONITORING_AND_BACKUPS.md](../docs/MONITORING_AND_BACKUPS.md).

| Script | When to use it | Start here |
|---|---|---|
| [`restore_from_backup.py`](restore_from_backup.py) | Restoring one or both stores from a tested local backup or API-exported tarball. | Always preview first: `python scripts/restore_from_backup.py --dry-run`. Then select `--backup-timestamp` or `--backup-dir`, optionally `--falkordb-only` / `--qdrant-only`, and use `--force` only after review. |
| [`recover_from_qdrant.py`](recover_from_qdrant.py) | FalkorDB is lost or corrupt while Qdrant is known-good and complete. | **Destructive:** it clears the configured FalkorDB graph, then rebuilds it from Qdrant. Back up and verify Qdrant before `python scripts/recover_from_qdrant.py`. |
| [`deduplicate_qdrant.py`](deduplicate_qdrant.py) | Qdrant contains duplicates, commonly after a failed recovery or manual import. | `python scripts/deduplicate_qdrant.py --dry-run`; review, then rerun with `--yes` to delete. |

---

## Developer and deployment

| Script | When to use it | Start here | Make target |
|---|---|---|---|
| [`bootstrap_dev.sh`](bootstrap_dev.sh) | Setting up or repairing a local contributor environment. | `make install` (or `./scripts/bootstrap_dev.sh`) creates `.venv`, refreshes `venv -> .venv`, installs development dependencies, and installs pre-commit hooks. | `make install` |
| [`deploy_check.sh`](deploy_check.sh) | Checking that Railway is deploying the expected Git commit. | `./scripts/deploy_check.sh automem`; use `DEPLOY_CHECK_QUIET=1` only for CI-style exit-code checks. Requires linked Railway and GitHub CLIs. | `make deploy-check` |

---

## Benchmark harness — `scripts/bench/`

Snapshot-based LoCoMo / LongMemEval evaluation. See [docs/TESTING.md](../docs/TESTING.md).

| Script | When to use it | Start here | Make target |
|---|---|---|---|
| [`bench/ingest_and_snapshot.sh`](bench/ingest_and_snapshot.sh) | Creating a fresh benchmark snapshot after a corpus or embedding change. | `make bench-ingest BENCH=locomo`; this starts Docker and writes reusable snapshots. | `make bench-ingest BENCH=locomo` |
| [`bench/restore_and_eval.sh`](bench/restore_and_eval.sh) | Evaluating one scoring configuration against an existing snapshot. | `make bench-eval BENCH=locomo CONFIG=baseline` | `make bench-eval` |
| [`bench/compare_configs.sh`](bench/compare_configs.sh) | A/B comparing two scoring configurations on the same snapshot. | `make bench-compare BENCH=locomo BASELINE=baseline CONFIG=<candidate>` | `make bench-compare` |
| [`bench/compare_branch.sh`](bench/compare_branch.sh) | Comparing a branch against `main` using a snapshot. | `make bench-compare-branch BRANCH=<branch>`; the script temporarily checks out refs, so begin from a clean worktree. | `make bench-compare-branch` |
| [`bench/compare_results.py`](bench/compare_results.py) | Reading two existing result JSON files without rerunning a benchmark. | `python scripts/bench/compare_results.py --baseline <base.json> --test <candidate.json>` | — |
| [`bench/analyze_locomo_results.py`](bench/analyze_locomo_results.py) | Producing a Markdown failure report from a LoCoMo result JSON. | `python scripts/bench/analyze_locomo_results.py <results.json> --output report.md` | — |
| [`bench/health_check.py`](bench/health_check.py) | Checking post-restore score distributions, entity quality, latency, and curated-query precision. | `make bench-health` | `make bench-health` |
| [`run_longmemeval_watch.sh`](run_longmemeval_watch.sh) | Running LongMemEval with persistent logging and local completion/crash notifications. | `make test-longmemeval-watch`; for a smaller run use `./scripts/run_longmemeval_watch.sh --max-questions 50`. | `make test-longmemeval-watch` |

---

## Recall Quality Lab — `scripts/lab/`

Data-driven recall scoring experiments against a clone of production. Full
workflow: [docs/RECALL_QUALITY_LAB.md](../docs/RECALL_QUALITY_LAB.md).

| Script | When to use it | Start here | Make target |
|---|---|---|---|
| [`lab/clone_production.sh`](lab/clone_production.sh) | Creating an isolated local copy of production data for recall experiments. | `make lab-clone`; use `--restore-only <snapshot>` for repeat experiments so production is not contacted again. | `make lab-clone` |
| [`lab/create_test_queries.py`](lab/create_test_queries.py) | Generating a natural-language evaluation set from a local clone. | `make lab-queries` or `python scripts/lab/create_test_queries.py --count 100`. | `make lab-queries` |
| [`lab/run_recall_test.py`](lab/run_recall_test.py) | Measuring one configuration, an A/B comparison, or a parameter sweep. | `make lab-test CONFIG=baseline`; use `make lab-compare` or `make lab-sweep` for the other modes. | `make lab-test` · `make lab-compare` · `make lab-sweep` |
| [`lab/repair_entity_tags.py`](lab/repair_entity_tags.py) | Repairing noisy generated entity tags on a **local clone**. | Plan first: `python scripts/lab/repair_entity_tags.py --mode canonicalize-safe`; review `<report-dir>/plan.jsonl`, then apply with `--execute --plan <report-dir>/plan.jsonl` or undo with `--rollback <report-dir>/rollback.jsonl`. | — |
| [`lab/lab_metrics.py`](lab/lab_metrics.py) | **Library module**, not a CLI; implements deterministic Recall@K, MRR, NDCG, and distractor-rate metrics. | Import from `run_recall_test.py` or tests. | — |
| [`lab/lab_corpus.py`](lab/lab_corpus.py) | **Library module**, not a CLI; centralizes injectable recall/corpus HTTP helpers. | Import from `run_recall_test.py` or tests. | — |
| [`lab/configs/`](lab/configs/) | Creating named JSON scoring-weight overrides for A/B tests. | Copy `baseline.json`, edit weights, then pass the filename without `.json` to `CONFIG`. | — |

---

## Shared

| File | What it is |
|---|---|
| [`lib/common.sh`](lib/common.sh) | Support module, not a CLI. Provides color helpers and `wait_for_api` to benchmark shell scripts. |
| [`Dockerfile.health-monitor`](Dockerfile.health-monitor) | Container recipe for running `health_monitor.py` in alert-only mode. Use only when you intentionally operate monitoring as a separate service. |

## See also

- [docs/MIGRATIONS.md](../docs/MIGRATIONS.md) — embedding-provider, model, and one-time data migrations
- [docs/MONITORING_AND_BACKUPS.md](../docs/MONITORING_AND_BACKUPS.md) — backup/restore/recovery runbook
- [docs/HEALTH_MONITORING.md](../docs/HEALTH_MONITORING.md) — `health_monitor.py` deployment
- [docs/RECALL_QUALITY_LAB.md](../docs/RECALL_QUALITY_LAB.md) — the `lab/` harness end to end
- [docs/TESTING.md](../docs/TESTING.md) — unit, integration, and benchmark testing
