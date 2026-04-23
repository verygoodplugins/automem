# Issue #79: priority_ids parameter only boosts relevance

**Status**: ACCEPTED / MERGED
**Date**: 2026-03-12
**PR**: #125 (commit `5d3708c`)
**Branch**: `exp/79-priority-ids-fetch-v1`

## Hypothesis

The `priority_ids` parameter should fetch memories directly by ID and guarantee their inclusion in results, not merely boost their score if they happen to appear in normal search results.

## Benchmark

| Metric | Baseline | Test | Delta |
|--------|----------|------|-------|
| LoCoMo-mini | 89.36% | 89.36% | **0.0** |

**Category deltas**: All zero across Single-hop, Temporal, Multi-hop, Open Domain, Complex.

Delta of zero is expected — this is a fetch-behavior bug fix, not a recall scoring change. The benchmark exercises natural-language queries, not ID-based lookups.

## Commands Run

```bash
make bench-eval BENCH=locomo-mini CONFIG=baseline  # on exp/79 branch
make bench-compare BENCH=locomo-mini CONFIG=baseline BASELINE=baseline  # vs main
```

## Outcome

Bug fix merged without benchmark regression. MCP clients can now reliably fetch specific memories by ID via `priority_ids`.

## Promoted Artifact

`tests/benchmarks/results/compare_issue79_priority_ids_20260311.json`

## Follow-up

None needed. Related MCP-side fix: `verygoodplugins/mcp-automem#67`.
