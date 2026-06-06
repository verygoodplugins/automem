# PR #80: Enhanced Recall — BM25, LLM Reranking, Query Expansion

**Status**: REJECTED — regression across all tested configurations
**Date**: 2026-03-11
**PR**: #80 (by jescalan)
**Branch**: `exp/pr80-enhanced-recall-v2` (commit `a122ba2`)
**Author**: jescalan

## Hypothesis

Adding BM25 full-text search, LLM reranking, and query expansion to the recall pipeline will improve accuracy by catching keyword matches that vector search misses and filtering false positives.

**Pipeline**: query expansion → vector + graph + BM25 search → RRF fusion → metadata scoring → LLM reranking

## Benchmark Results

### Full Port (all features enabled)

| Metric | Baseline (main) | PR #80 | Delta |
|--------|-----------------|--------|-------|
| LoCoMo-mini (judge-off) | 89.36% | 85.53% | **-3.83pp** |
| LoCoMo-mini (judge-on) | 90.13% | 88.16% | **-1.97pp** |

### Category Breakdown (judge-off)

| Category | Baseline | PR #80 | Delta |
|----------|----------|--------|-------|
| Single-hop | 79.1% | 86.0% | **+7.0pp** |
| Temporal | 92.1% | 93.7% | **+1.6pp** |
| Multi-hop | 46.2% | 46.2% | **0.0** |
| Open Domain | 96.5% | 85.1% | **-11.4pp** |
| Complex | N/A | N/A | -- |

### Config Sweep (Round 1, judge-off)

| Config | Accuracy | Delta vs Main | Notes |
|--------|----------|---------------|-------|
| BM25-only f10 | 88.09% | **-1.28pp** | Best variant, still regressed |
| BM25-only f20 | 87.66% | **-1.70pp** | More BM25 results = more dilution |
| BM25 + rerank top-5 | 87.23% | **-2.13pp** | Reranking didn't help |
| BM25 + rerank top-10 | 86.81% | **-2.55pp** | Wider rerank window = worse |

### Runtime Impact

| Config | Runtime | vs Baseline |
|--------|---------|-------------|
| Main (baseline) | 209s | -- |
| Full port (judge-off) | 1546s | **7.4x slower** |
| Full port (judge-on) | 2136s | **10.2x slower** |

## Root Cause Analysis

**Open-domain regression is the primary problem.** Open-domain questions (e.g., "What does Alex think about remote work?") rely on semantic similarity — the answer is conceptually related but doesn't share keywords with the question. BM25 keyword matches dilute the vector results via Reciprocal Rank Fusion (RRF), pushing semantically-relevant results down.

The single-hop improvement (+7.0pp) shows BM25 *does* help for factual lookups where exact keywords matter. But the open-domain loss (-11.4pp) is 1.6x larger and affects more questions (114 vs 43).

**Runtime cost** is driven by LLM reranking (one API call per candidate per question) and query expansion (one API call per question). Even BM25-only configs add SQLite FTS5 overhead.

## Commands Run

```bash
# Full port evaluation
make bench-eval BENCH=locomo-mini CONFIG=baseline  # on exp/pr80-enhanced-recall-v2
make bench-compare BENCH=locomo-mini CONFIG=baseline BASELINE=baseline

# Config sweep (4 variants)
make bench-eval BENCH=locomo-mini CONFIG=bm25_only_f10
make bench-eval BENCH=locomo-mini CONFIG=bm25_only_f20
make bench-eval BENCH=locomo-mini CONFIG=bm25_rerank_top5
make bench-eval BENCH=locomo-mini CONFIG=bm25_rerank_top10
```

## Outcome

**REJECTED.** No configuration recovered the open-domain regression. The best variant (BM25-only f10) still regressed -1.28pp overall.

## Decision

PR #80's always-on BM25 fusion approach hurts more than it helps on current benchmarks. The single-hop gains don't compensate for open-domain losses. The 7-10x runtime increase is also prohibitive for production use.

## Promoted Artifacts

- `tests/benchmarks/results/compare_pr80_judge_off_20260311.json`
- `tests/benchmarks/results/compare_pr80_judge_on_20260311.json`
- `tests/benchmarks/results/compare_pr80_bm25_only_f10_judge_off.json`

## Follow-up Recommendations

1. **Targeted BM25 fallback**: Only invoke BM25 when vector search returns low-confidence results (e.g., all scores < 0.5), rather than always-on fusion
2. **Category-aware fusion**: Weight BM25 differently for factual vs open-domain queries (requires query classification)
3. **RRF tuning**: The RRF constant (k=60 default) may be too aggressive for BM25 results — could try lower k to reduce BM25 influence
4. **Reranking without BM25**: Test LLM reranking on vanilla vector results (without BM25 dilution) as a separate experiment
5. **Contributor summary**: If closing PR #80, provide benchmark evidence (this postmortem) rather than preference-based feedback
