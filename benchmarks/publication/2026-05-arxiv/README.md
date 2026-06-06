# AutoMem arXiv Publication Bundle

This bundle collects the repository-side material for the May 2026 AutoMem
arXiv preprint effort. It is intentionally conservative: canonical claims come
from this repository's official benchmark log, while exploratory and external
numbers are labeled separately.

## Claim Posture

AutoMem should be described as an open-source, inspectable, MCP-first
graph-vector memory service for AI agents with transparent benchmark harnesses
and strong canonical LoCoMo / LongMemEval results.

Do not claim "best memory system", "SOTA", or "beats Mem0" from this bundle.
Those claims require apples-to-apples reruns against the current external
systems, judge policies, dataset versions, and scale settings.

## Canonical Results

| Status | Benchmark | Scope | Score | Retrieval | Source |
|---|---:|---:|---:|---:|---|
| canonical | LongMemEval full | 500 questions | 87.00% (435/500) | recall@5 97.00% (485/500) | Fresh publication verification run; see `fresh-verification.md` |
| representative canary | LongMemEval mini | 30 stratified questions | 70.00% (21/30) | recall@5 96.67% (29/30) | Fresh publication verification run; see `fresh-verification.md` |
| canonical | LoCoMo full | 10 conversations, 1,986 questions | 84.74% (1683/1986) | not reported | Fresh publication verification run with pinned `gpt-5.4-mini-2026-03-17` judge |

Canonical LongMemEval model policy:

- Answerer: `gpt-5-mini`
- Judge: `gpt-5.4-mini-2026-03-17`
- Judge errors: `0`
- Memory ingest failures: `0`
- Harness publishable flag: `true`

## Supplemental Signals

| Status | Benchmark / Evidence | Scope | Result | Caveat |
|---|---|---:|---:|---|
| exploratory | BEAM 100K V1 raw-dialogue shim | 20 conversations, 400 questions | 76.25% (305/400), avg 0.677 | Not comparable to published BEAM 1M/10M claims. |
| exploratory | BEAM 100K V2 fact-extraction shim | 20 conversations, 400 questions | 73.75% (295/400), avg 0.653 | Diagnostic failure-mode signal only. |
| exploratory | Writ drift integration | 5 drift scenarios | 100% recall accuracy, 20% update fidelity, 0% drift rate | Lives in `automem-evals`; must remain labeled supplemental until promoted. |
| exploratory | Claude Code hook replay | fixture suite | metrics harness only | Lives in `automem-evals`; workflow-continuity signal, not a memory benchmark. |
| external reported | Mem0 managed platform | LoCoMo / LongMemEval / BEAM | see cited Mem0 docs | Proprietary managed-platform optimizations; not directly comparable. |
| not yet run | BEAM official 1M/10M | official BEAM scale | -- | Required before any BEAM-competitive claim. |
| not yet run | LongMemEval-V2 | web-agent memory | -- | Required before "experienced colleague" claims. |
| not yet run | Memora / FAMA | invalidated-memory reuse | -- | Natural fit for `INVALIDATED_BY`/`CONTRADICTS`, but not run yet. |

## Bundle Files

- `benchmark-summary.md` - paper-ready benchmark and limitation summary.
- `artifact-manifest.json` - machine-readable manifest for claims, generated-artifact paths, and commands.
- `commands.md` - verification and reproduction command inventory.
- `fresh-verification.md` - latest local verification notes and generated artifact hashes.

## Promotion Rule

Results from `../automem-evals` may inform the paper only as supplemental
evidence until a result is reproduced or explicitly summarized in this
repository. Official benchmark claims remain owned by
`benchmarks/EXPERIMENT_LOG.md`.
