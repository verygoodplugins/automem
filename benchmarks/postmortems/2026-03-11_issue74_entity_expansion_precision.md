# Issue #74: Graph expansion follows too many hops through hub nodes

**Status**: NON-PROMOTED / REJECTED (this direction)
**Date**: 2026-03-11
**Issue**: #74 (remains open)
**Branch**: `exp/74-entity-expansion-precision-v1` (commit `236b075`)

## Hypothesis

Hub-node pollution in graph expansion degrades recall precision. Querying "Alex Panagis" with `expand_entities=true` incorrectly pulls in unrelated memories through shared generic nodes (e.g., the "AutoMem" tool node connects all users). Hub-node detection and deprioritization should improve precision without harming recall.

## Benchmark

| Metric | Baseline | Test | Delta |
|--------|----------|------|-------|
| LoCoMo-mini | 89.36% | 89.36% | **0.0** |

**Category deltas**: All zero across all five categories.

## Why Zero Delta

The LoCoMo benchmark doesn't exercise graph expansion (`expand_entities`) in its query path. All LoCoMo questions are answered via vector + keyword search. The hub-node problem is real in production (where users query with `expand_entities=true`), but this benchmark can't surface the improvement.

## Commands Run

```bash
make bench-eval BENCH=locomo-mini CONFIG=baseline  # on exp/74 branch
make bench-compare BENCH=locomo-mini CONFIG=baseline BASELINE=baseline  # vs main
```

## Outcome

Hub-node detection alone doesn't move the needle on LoCoMo. The experiment was correctly executed but the benchmark isn't the right instrument to measure this improvement.

## Promoted Artifact

`tests/benchmarks/results/compare_issue74_entity_precision_20260311.json`

## Follow-up

- Issue #74 remains open — the problem is real, just not benchmark-measurable yet
- Consider Personalized PageRank (#100) or configurable `max_hops` as alternative approaches
- Need a graph-expansion-specific test suite (queries with `expand_entities=true`) to measure future attempts
- Should not be the default direction for recall improvement work
