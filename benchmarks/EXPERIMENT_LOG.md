# AutoMem Experiment Log

Tracks recall quality experiments with before/after benchmark results.

**Baselines** are created with Voyage 4 embeddings (`EMBEDDING_PROVIDER=voyage`, `VECTOR_SIZE=1024`)
on the snapshot-based bench infrastructure (PR #97, merged 2026-03-02).

## Tiered Benchmarking

| Tier | Benchmark | Runtime | Cost | When to use |
|------|-----------|---------|------|-------------|
| 0 | `make test` (unit) | 30s | free | Every change |
| 1 | `locomo-mini` (2 convos, 198 Qs) | 2-3 min | free | Rapid iteration |
| 2 | `locomo` (10 convos, 1986 Qs) | 5-10 min | free | Before merge |
| 3 | `longmemeval-mini` (20 Qs) | 15 min | ~$1 | Scoring/entity changes |
| 4 | `longmemeval` (500 Qs) | 1-2 hr | ~$10 | Milestones only |

## Results

| Date | Issue/PR | Branch | LoCoMo-mini | LoCoMo-full | LME-mini | Notes |
|------|----------|--------|-------------|-------------|----------|-------|
| 2026-03-02 | baseline | main | 76.97% (234/304) | 80.06% (1590/1986) | -- | Voyage 4, 1024d. Health: DEGRADED (low score variance) |

## How to add an entry

1. Run the benchmark: `make bench-eval BENCH=locomo-mini CONFIG=baseline`
2. Record the overall accuracy from the output JSON
3. Add a row to the table above with the date, issue/PR, branch, and scores
4. For deltas, show as `XX.X% (+Y.Y)` relative to the baseline row

## Snapshot metadata

| Snapshot | Created | Git SHA | Embedding | Memories |
|----------|---------|---------|-----------|----------|
| locomo-mini | 2026-03-02 | main @ 80a6f93 | voyage:voyage-4 1024d | 788 (2 convos) |
| locomo | 2026-03-02 | main @ 80a6f93 | voyage:voyage-4 1024d | 5828 (10 convos) |
