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
| 3 | `longmemeval-mini` (stratified 30 Qs) | 15 min | ~$1 | Scoring/entity changes |
| 4 | `longmemeval` (500 Qs) | 1-2 hr | ~$10 | Milestones only |

## Results

Current headline results:

| Benchmark | Scope | Score | Retrieval | Notes |
|-----------|-------|-------|-----------|-------|
| LongMemEval full | 500 questions | **86.20% (431/500)** | recall@5 **97.20% (486/500)** | Canonical `gpt-5-mini` answerer + `gpt-5.4-mini-2026-03-17` judge; `judge_errors=0`, `memory_ingest_failures=0`. |
| LongMemEval mini | 30 questions, stratified 5 per type | **60.0% (18/30)** | recall@5 **96.67% (29/30)** | Representative canary; do not compare to legacy prefix slices. |
| LoCoMo full | 10 conversations, 1986 questions | **83.99% (1668/1986)** | -- | Latest recorded full judge-on run from #128; cat5 scored 92.83% with 0 skips. |

Detailed experiment history:

| Date | Issue/PR | Branch | LoCoMo-mini | LoCoMo-full | LME-mini | LME-full | Notes |
|------|----------|--------|-------------|-------------|----------|----------|-------|
| 2026-03-02 | baseline | main | 76.97% (234/304) | 80.06% (1590/1986) | -- | -- | Voyage 4, 1024d. Health: DEGRADED (low score variance) |
| 2026-03-02 | #73 | exp/73-min-score-threshold | 76.97% (+0.0) | -- | -- | -- | min_score + adaptive floor. No regression. Needs #78 for impact |
| 2026-03-02 | PR #80 | jescalan/feat/enhanced-recall | BLOCKED | -- | -- | -- | Merge conflicts with main (recall.py), needs rebase before eval |
| 2026-03-02 | PR #87 | jescalan/feat/write-time-dedup | 76.97% (+0.0) | -- | -- | -- | Write-time dedup gate. Neutral on recall (expected) |
| 2026-03-02 | #78 | exp/78-decay-fix | 76.97% (+0.0) | 79.51% (-0.55) | -- | -- | Decay rate 0.1→0.01, importance floor, archive filter. Within variance. Impact is on production (rehabilitated via rescore) |
| 2026-03-10 | pre-refactor | main (@ 795368a) | 76.97% (+0.0) | -- | -- | -- | Baseline re-confirmed after #73, #78, #115, #116 merged. Stable. Pre-relation-tier-refactor checkpoint. |
| 2026-03-10 | eval-fix | docs/benchmark-agent-guidelines | **89.27% (208/233)** | -- | -- | -- | Fix temporal matching (answer vs memory dates) + skip cat5 (no ground truth). Honest corrected baseline after evaluator fixes. |
| 2026-03-10 | cat5-judge | feat/bench-cat5-judge | **89.80% (273/304)** | **87.56% (1739/1986)** | -- | -- | Opt-in GPT-4o judge for cat5. Full run scored cat5 at 95.74% (427/446) with 0 judge skips/errors; added 90s OpenAI request timeout to prevent stuck full runs. |
| 2026-03-10 | main-refresh (no judge) | main | **89.36% (210/235)** | -- | -- | -- | Fresh current-main rerun before PR #80 experiment. Comparison anchor for judge-off. |
| 2026-03-10 | main-refresh (judge) | main | **90.13% (274/304)** | -- | -- | -- | Fresh current-main rerun with `BENCH_JUDGE_MODEL=gpt-4o`. Comparison anchor for judge-on. |
| 2026-03-11 | PR #80 port (no judge) | exp/pr80-enhanced-recall-v2 | 85.53% (201/235) | -- | -- | -- | BM25 + query expansion + rerank port. Regressed **-3.83pp** vs fresh main. Open-domain -11.4pp. Runtime 7.4x. → [postmortem](postmortems/2026-03-11_pr80_enhanced_recall.md) |
| 2026-03-11 | PR #80 port (judge) | exp/pr80-enhanced-recall-v2 | 88.16% (268/304) | -- | -- | -- | Same branch with GPT-4o cat5 judge. Regressed **-1.97pp** vs fresh main. Runtime 10.2x. → [postmortem](postmortems/2026-03-11_pr80_enhanced_recall.md) |
| 2026-03-11 | PR #80 BM25-only f10 | exp/pr80-bm25-only-f10 | 88.09% (-1.28) | -- | -- | -- | Best config variant, still regressed. → [postmortem](postmortems/2026-03-11_pr80_enhanced_recall.md) |
| 2026-03-11 | PR #80 BM25-only f20 | exp/pr80-bm25-only-f20 | 87.66% (-1.70) | -- | -- | -- | More BM25 results = more dilution. → [postmortem](postmortems/2026-03-11_pr80_enhanced_recall.md) |
| 2026-03-11 | PR #80 BM25+rerank top5 | exp/pr80-bm25-rerank-top5 | 87.23% (-2.13) | -- | -- | -- | Reranking didn't recover regression. → [postmortem](postmortems/2026-03-11_pr80_enhanced_recall.md) |
| 2026-03-11 | PR #80 BM25+rerank top10 | exp/pr80-bm25-rerank-top10 | 86.81% (-2.55) | -- | -- | -- | Wider rerank window = worse. → [postmortem](postmortems/2026-03-11_pr80_enhanced_recall.md) |
| 2026-03-11 | #74 entity expansion | exp/74-entity-expansion-precision-v1 | 89.36% (+0.0) | -- | -- | -- | Hub-node detection. Zero delta — benchmark doesn't exercise graph expansion. → [postmortem](postmortems/2026-03-11_issue74_entity_expansion_precision.md) |
| 2026-03-12 | #79 (PR #125) | exp/79-priority-ids-fetch-v1 | 89.36% (+0.0) | -- | -- | -- | Bug fix: priority_ids now fetches by ID. Merged. → [postmortem](postmortems/2026-03-12_issue79_priority_ids_fetch.md) |
| 2026-04-23 | #128 | fix/128-recall-keyword-scoring-dead-for-vector-results-adaptive-floor-too-aggressive | **85.53% (201/235)** | -- | -- | -- | Content keyword fallback + gentler adaptive floor. Improved **+3.40pp** vs the same-day baseline (`82.13%`, `193/235`) with no sampled question-level regressions across conv-26/conv-30. |
| 2026-04-23 | #128 full judge | fix/128-recall-keyword-scoring-dead-for-vector-results-adaptive-floor-too-aggressive | -- | **83.99% (1668/1986)** | -- | -- | Full judge-on rerun after harness fixes. Judge preflight passed and cat-5 scored **92.83% (414/446)** with **0 skips**. Improves **+3.93pp** vs full baseline (`80.06%`, `1590/1986`), so #128 is strong enough to move forward to broader validation. |
| 2026-04-23 | #142 | fix/142-expansion-tag-filter | -- | 77.30% (-0.07) | -- | -- | Expansion tag-filter bypass. Effectively flat vs pre-fix `77.37%` — canonical configs don't exercise `expand_relations`. Validated via scoped repro + helper/API tests. |
| 2026-04-26 | LongMemEval harness | fix/longmemeval-harness-resume-and-stratified-mini | -- | -- | **60.0% (18/30)** | **86.20% (431/500)** | Representative stratified mini and full canonical run. Full recall@5 **97.20% (486/500)**; `judge_errors=0`, `memory_ingest_failures=0`. |

### Category Breakdown (LoCoMo-mini)

Categories 1-4 are scored by word-overlap/date matching. Category 5 uses an opt-in LLM judge when `BENCH_JUDGE_MODEL` or `--judge` is enabled; otherwise it remains `N/A`.

| Date | Issue/PR | Single-hop | Temporal | Multi-hop | Open Domain | Complex | Overall |
|------|----------|------------|----------|-----------|-------------|---------|---------|
| 2026-03-02 | baseline | 76.7% (33/43) | 22.2%\* (14/63) | 46.2% (6/13) | 96.5% (110/114) | 100%\*\* (71/71) | 76.97% (234/304) |
| 2026-03-10 | pre-refactor | 76.7% (33/43) | 22.2%\* (14/63) | 46.2% (6/13) | 96.5% (110/114) | 100%\*\* (71/71) | 76.97% |
| 2026-03-10 | eval-fix | **79.1% (34/43)** | **92.1% (58/63)** | 46.2% (6/13) | 96.5% (110/114) | N/A (71 skipped) | **89.27% (208/233)** |
| 2026-03-10 | cat5-judge | **79.1% (34/43)** | **92.1% (58/63)** | 46.2% (6/13) | 96.5% (110/114) | **91.5% (65/71)** | **89.80% (273/304)** |
| 2026-03-10 | main-refresh (no judge) | **79.1% (34/43)** | **92.1% (58/63)** | 46.2% (6/13) | **96.5% (110/114)** | 100.0% (2/2, 69 skipped) | **89.36% (210/235)** |
| 2026-03-10 | main-refresh (judge) | **79.1% (34/43)** | **92.1% (58/63)** | 46.2% (6/13) | **96.5% (110/114)** | **93.0% (66/71)** | **90.13% (274/304)** |
| 2026-03-11 | PR #80 port (no judge) | **86.0% (37/43)** | **93.7% (59/63)** | 46.2% (6/13) | 85.1% (97/114) | 100.0% (2/2, 69 skipped) | 85.53% (201/235) |
| 2026-03-11 | PR #80 port (judge) | **88.4% (38/43)** | **92.1% (58/63)** | 46.2% (6/13) | 86.0% (98/114) | **95.8% (68/71)** | 88.16% (268/304) |
| 2026-03-11 | PR #80 BM25-only f10 | 81.4% (35/43) | 92.1% (58/63) | 46.2% (6/13) | 93.0% (106/114) | N/A | 88.09% |
| 2026-03-11 | #74 entity expansion | 79.1% (34/43) | 92.1% (58/63) | 46.2% (6/13) | 96.5% (110/114) | N/A | 89.36% |
| 2026-03-12 | #79 (PR #125) | 79.1% (34/43) | 92.1% (58/63) | 46.2% (6/13) | 96.5% (110/114) | N/A | 89.36% |
| 2026-04-23 | #128 | **65.1% (28/43)** | 92.1% (58/63) | **38.5% (5/13)** | **94.7% (108/114)** | 100.0% (2/2, 69 skipped) | **85.53% (201/235)** |

\* Temporal was artificially low: evaluator compared question dates (empty) vs memory dates instead of answer dates.
\*\* Complex was artificially 100%: dataset has no `answer` field for cat5 → empty string → `"" in content` always True.

### Category Breakdown (LongMemEval full)

Canonical run: `benchmarks/results/longmemeval_full_gpt5mini_20260425_231308.json`.
Answerer `gpt-5-mini`; judge `gpt-5.4-mini-2026-03-17`.

| Question type | Accuracy | Recall@5 |
|---------------|----------|----------|
| knowledge-update | 88.46% (69/78) | 100.00% (78/78) |
| multi-session | 81.20% (108/133) | 98.50% (131/133) |
| single-session-assistant | 98.21% (55/56) | 100.00% (56/56) |
| single-session-preference | 60.00% (18/30) | 90.00% (27/30) |
| single-session-user | 91.43% (64/70) | 92.86% (65/70) |
| temporal-reasoning | 87.97% (117/133) | 96.99% (129/133) |

Failure split from the result-analysis helper: 58 wrong answers had the answer
session retrieved at recall@5; 11 were retrieval misses. This is the basis for
follow-up issues #158 and #159.

## Exploratory and Historical Benchmarks

These results are useful validation signals, but they are not current headline
claims. BEAM entries came from `automem-evals` exploratory runners. LongMemEval
prefix entries are legacy/provisional because dataset prefixes are biased.

| Date | Benchmark | Scope | Score | Retrieval | Notes |
|------|-----------|-------|-------|-----------|-------|
| 2026-04-22 | BEAM 100K V1 raw-dialogue shim | 20 conversations, 400 questions | **76.25% (305/400)**, avg 0.677 | top-k 200 | `gpt-5-mini` answerer/judge, zero errors. Diagnostic result only: BEAM 100K is easier than mem0's published 1M/10M settings, and the V1 shim stores raw dialogue rather than mem0-style extracted facts. |
| 2026-04-22 | BEAM 100K V2 fact-extraction shim | 20 conversations, 400 questions | **73.75% (295/400)**, avg 0.653 | top-k 200 | -2.50pp vs V1 overall, within the estimated noise floor. Category signal was useful: abstention +15pp and knowledge_update +7.5pp; event_ordering -20pp and information_extraction -12.5pp. |
| 2026-04-24 | LongMemEval partial legacy prefix (50q) | 50 questions, single-session-user type | **82.0% (41/50)** | recall@5 **92.0% (46/50)** | Provisional prefix run with legacy `gpt-4o` answerer; recall_limit=10, no entity/relation expansion. Not reproduced by the current stratified `bench-mini-longmemeval` target and not directly comparable to the older 35.6% / 500-question setup or to a full LongMemEval claim. |

## How to add an entry

1. Run the benchmark: `make bench-eval BENCH=locomo-mini CONFIG=baseline`
2. Record the overall accuracy from the output JSON
3. Add a row to the detailed experiment history with the date, issue/PR, branch, and scores
4. Add a row to the Category Breakdown table with per-category scores
5. For deltas, show as `XX.X% (+Y.Y)` relative to the baseline row

## Snapshot metadata

| Snapshot | Created | Git SHA | Embedding | Memories |
|----------|---------|---------|-----------|----------|
| locomo-mini | 2026-03-02 | main @ 80a6f93 | voyage:voyage-4 1024d | 788 (2 convos) |
| locomo | 2026-03-02 | main @ 80a6f93 | voyage:voyage-4 1024d | 5828 (10 convos) |
