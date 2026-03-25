# AutoMem Experiment Log

Tracks recall quality experiments with before/after benchmark results.

**Baselines** are created with Voyage 4 embeddings (`EMBEDDING_PROVIDER=voyage`, `VECTOR_SIZE=1024`)
on the snapshot-based bench infrastructure (PR #97, merged 2026-03-02).

## Tiered Benchmarking

| Tier | Benchmark | Runtime | Cost | When to use |
|------|-----------|---------|------|-------------|
| 0 | `make test` (unit) | 30s | free | Every change |
| 1 | `locomo-mini` (2 convos, 304 Qs) | 2-3 min | free / ~$0.20 with judge | Rapid iteration |
| 2 | `locomo` (10 convos, 1986 Qs) | 5-10 min | free / ~$1-3 with judge | Before merge |
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
| 2026-03-10 | cat5-judge | feat/bench-cat5-judge | **89.80% (273/304)** | **87.56% (1739/1986)** | -- | Opt-in GPT-4o judge for cat5. Full run scored cat5 at 95.74% (427/446) with 0 judge skips/errors; added 90s OpenAI request timeout to prevent stuck full runs. |
| 2026-03-10 | main-refresh (no judge) | main | **89.36% (210/235)** | -- | -- | Fresh current-main rerun before PR #80 experiment. Comparison anchor for judge-off. |
| 2026-03-10 | main-refresh (judge) | main | **90.13% (274/304)** | -- | -- | Fresh current-main rerun with `BENCH_JUDGE_MODEL=gpt-4o`. Comparison anchor for judge-on. |
| 2026-03-11 | PR #80 port (no judge) | exp/pr80-enhanced-recall-v2 | 85.53% (201/235) | -- | -- | BM25 + query expansion + rerank port. Regressed **-3.83pp** vs fresh main. Open-domain -11.4pp. Runtime 7.4x. → [postmortem](postmortems/2026-03-11_pr80_enhanced_recall.md) |
| 2026-03-11 | PR #80 port (judge) | exp/pr80-enhanced-recall-v2 | 88.16% (268/304) | -- | -- | Same branch with GPT-4o cat5 judge. Regressed **-1.97pp** vs fresh main. Runtime 10.2x. → [postmortem](postmortems/2026-03-11_pr80_enhanced_recall.md) |
| 2026-03-11 | PR #80 BM25-only f10 | exp/pr80-bm25-only-f10 | 88.09% (-1.28) | -- | -- | Best config variant, still regressed. → [postmortem](postmortems/2026-03-11_pr80_enhanced_recall.md) |
| 2026-03-11 | PR #80 BM25-only f20 | exp/pr80-bm25-only-f20 | 87.66% (-1.70) | -- | -- | More BM25 results = more dilution. → [postmortem](postmortems/2026-03-11_pr80_enhanced_recall.md) |
| 2026-03-11 | PR #80 BM25+rerank top5 | exp/pr80-bm25-rerank-top5 | 87.23% (-2.13) | -- | -- | Reranking didn't recover regression. → [postmortem](postmortems/2026-03-11_pr80_enhanced_recall.md) |
| 2026-03-11 | PR #80 BM25+rerank top10 | exp/pr80-bm25-rerank-top10 | 86.81% (-2.55) | -- | -- | Wider rerank window = worse. → [postmortem](postmortems/2026-03-11_pr80_enhanced_recall.md) |
| 2026-03-11 | #74 entity expansion | exp/74-entity-expansion-precision-v1 | 89.36% (+0.0) | -- | -- | Hub-node detection. Zero delta — benchmark doesn't exercise graph expansion. → [postmortem](postmortems/2026-03-11_issue74_entity_expansion_precision.md) |
| 2026-03-12 | #79 (PR #125) | exp/79-priority-ids-fetch-v1 | 89.36% (+0.0) | -- | -- | Bug fix: priority_ids now fetches by ID. Merged. → [postmortem](postmortems/2026-03-12_issue79_priority_ids_fetch.md) |

### Category Breakdown (LoCoMo-mini)

Categories 1-4 are scored by word-overlap/date matching. Category 5 uses an opt-in LLM judge when `BENCH_JUDGE_MODEL` or `--judge` is enabled; otherwise it remains `N/A`.

| Date | Issue/PR | Single-hop | Temporal | Multi-hop | Open Domain | Complex |
|------|----------|------------|----------|-----------|-------------|---------|
| 2026-03-02 | baseline | 76.7% (33/43) | 22.2%\* (14/63) | 46.2% (6/13) | 96.5% (110/114) | 100%\*\* (71/71) |
| 2026-03-10 | pre-refactor | 76.7% (33/43) | 22.2%\* (14/63) | 46.2% (6/13) | 96.5% (110/114) | 100%\*\* (71/71) |
| 2026-03-10 | eval-fix | **79.1% (34/43)** | **92.1% (58/63)** | 46.2% (6/13) | 96.5% (110/114) | N/A (71 skipped) |
| 2026-03-10 | cat5-judge | **79.1% (34/43)** | **92.1% (58/63)** | 46.2% (6/13) | 96.5% (110/114) | **91.5% (65/71)** |
| 2026-03-10 | main-refresh (no judge) | **79.1% (34/43)** | **92.1% (58/63)** | 46.2% (6/13) | **96.5% (110/114)** | 100.0% (2/2, 69 skipped) |
| 2026-03-10 | main-refresh (judge) | **79.1% (34/43)** | **92.1% (58/63)** | 46.2% (6/13) | **96.5% (110/114)** | **93.0% (66/71)** |
| 2026-03-11 | PR #80 port (no judge) | **86.0% (37/43)** | **93.7% (59/63)** | 46.2% (6/13) | 85.1% (97/114) | 100.0% (2/2, 69 skipped) |
| 2026-03-11 | PR #80 port (judge) | **88.4% (38/43)** | **92.1% (58/63)** | 46.2% (6/13) | 86.0% (98/114) | **95.8% (68/71)** |
| 2026-03-11 | PR #80 BM25-only f10 | 81.4% (35/43) | 92.1% (58/63) | 46.2% (6/13) | 93.0% (106/114) | N/A |
| 2026-03-11 | #74 entity expansion | 79.1% (34/43) | 92.1% (58/63) | 46.2% (6/13) | 96.5% (110/114) | N/A |
| 2026-03-12 | #79 (PR #125) | 79.1% (34/43) | 92.1% (58/63) | 46.2% (6/13) | 96.5% (110/114) | N/A |

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
