# AutoMem Experiment Log

Tracks recall quality experiments with before/after benchmark results.

**Baselines** are created with Voyage 4 embeddings (`EMBEDDING_PROVIDER=voyage`, `VECTOR_SIZE=1024`)
on the snapshot-based bench infrastructure (PR #97, merged 2026-03-02).

## Tiered Benchmarking

| Tier | Benchmark | Runtime | Cost | When to use |
|------|-----------|---------|------|-------------|
| 0 | `make test` (unit) | 30s | free | Every change |
| 1 | `locomo-mini` (2 convos, 304 Qs) | 2-3 min | free | Rapid iteration |
| 2 | `locomo` (10 convos, 1986 Qs) | 5-10 min | free | Before merge |
| 3 | `longmemeval-mini` (20 Qs) | 15 min | ~$1 | Scoring/entity changes |
| 4 | `longmemeval` (500 Qs) | 1-2 hr | ~$10 | Milestones only |

## Results

| Date | Issue/PR | Branch | LoCoMo-mini | LoCoMo-full | LME-mini | Notes |
|------|----------|--------|-------------|-------------|----------|-------|
| 2026-03-02 | baseline | main | 76.97% (234/304) | 80.06% (1590/1986) | -- | Voyage 4, 1024d. Health: DEGRADED (low score variance) |
| 2026-03-02 | #73 | exp/73-min-score-threshold | 76.97% (+0.0) | -- | -- | min_score + adaptive floor. No regression. Needs #78 for impact |
| 2026-03-02 | PR #80 | jescalan/feat/enhanced-recall | BLOCKED | -- | -- | Merge conflicts with main (recall.py), needs rebase before eval |
| 2026-03-02 | PR #87 | jescalan/feat/write-time-dedup | 76.97% (+0.0) | -- | -- | Write-time dedup gate. Neutral on recall (expected) |
| 2026-03-02 | #78 | exp/78-decay-fix | 76.97% (+0.0) | 79.51% (-0.55) | -- | Decay rate 0.1→0.01, importance floor, archive filter. Within variance. Impact is on production (rehabilitated via rescore) |
| 2026-03-10 | pre-refactor | main (@ 795368a) | 76.97% (+0.0) | -- | -- | Baseline re-confirmed after #73, #78, #115, #116 merged. Stable. Pre-relation-tier-refactor checkpoint. |
| 2026-03-10 | eval-fix | docs/benchmark-agent-guidelines | **89.27% (208/233)** | -- | -- | Fix temporal matching (answer vs memory dates) + skip cat5 (no ground truth). Honest score, beats CORE by 1.03pp. |

### Category Breakdown (LoCoMo-mini)

Categories 1-4 scored by word-overlap/date matching. Category 5 requires LLM judge (not yet implemented).

| Date | Issue/PR | Single-hop | Temporal | Multi-hop | Open Domain | Complex |
|------|----------|------------|----------|-----------|-------------|---------|
| 2026-03-02 | baseline | 76.7% (33/43) | 22.2%\* (14/63) | 46.2% (6/13) | 96.5% (110/114) | 100%\*\* (71/71) |
| 2026-03-10 | pre-refactor | 76.7% (33/43) | 22.2%\* (14/63) | 46.2% (6/13) | 96.5% (110/114) | 100%\*\* (71/71) |
| 2026-03-10 | eval-fix | **79.1% (34/43)** | **92.1% (58/63)** | 46.2% (6/13) | 96.5% (110/114) | N/A (71 skipped) |

\* Temporal was artificially low: evaluator compared question dates (empty) vs memory dates instead of answer dates.
\*\* Complex was artificially 100%: dataset has no `answer` field for cat5 → empty string → `"" in content` always True.

## How to add an entry

1. Run the benchmark: `make bench-eval BENCH=locomo-mini CONFIG=baseline`
2. Record the overall accuracy from the output JSON
3. Add a row to the Results table with the date, issue/PR, branch, and scores
4. Add a row to the Category Breakdown table with per-category scores
5. For deltas, show as `XX.X% (+Y.Y)` relative to the baseline row

## Snapshot metadata

| Snapshot | Created | Git SHA | Embedding | Memories |
|----------|---------|---------|-----------|----------|
| locomo-mini | 2026-03-02 | main @ 80a6f93 | voyage:voyage-4 1024d | 788 (2 convos) |
| locomo | 2026-03-02 | main @ 80a6f93 | voyage:voyage-4 1024d | 5828 (10 convos) |
