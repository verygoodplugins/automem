# Benchmark Summary For Paper Draft

## Recommended Headline

AutoMem is an open-source graph-vector memory service for AI agents that
publishes transparent benchmark harnesses and current canonical results of
87.00% on LongMemEval full and 84.74% on LoCoMo full.

This is a reproducibility and systems claim, not a state-of-the-art claim.

## Canonical Results

The official source of truth is `benchmarks/EXPERIMENT_LOG.md`.

| Benchmark | Scope | Status | Score | Retrieval | Models / Judge |
|---|---:|---|---:|---:|---|
| LongMemEval full | 500 questions | canonical | 87.00% (435/500) | recall@5 97.00% (485/500) | `gpt-5-mini` answerer, `gpt-5.4-mini-2026-03-17` judge |
| LongMemEval mini | 30 stratified questions | representative canary | 70.00% (21/30) | recall@5 96.67% (29/30) | `gpt-5-mini` answerer, script LLM eval |
| LoCoMo full | 10 conversations, 1,986 questions | canonical | 84.74% (1683/1986) | not reported | Pinned `gpt-5.4-mini-2026-03-17` judge, 444 judge calls, 0 skips/errors |

LongMemEval failure split: 54 wrong answers had the answer session retrieved at
recall@5, while 11 wrong answers were retrieval misses. This supports a paper discussion that
future improvements are likely in answer synthesis, memory representation, and
preference handling, not only first-stage retrieval.

## Historical / Exploratory Context

Older LoCoMo mini/full values, including the 89.27% judge-off mini and 87.56%
March full judge-on run, remain useful trend anchors but should not be used as
current headline claims.

BEAM 100K shim results, Writ drift runs, and hook replay metrics came from
`automem-evals` and are explicitly diagnostic. They are not comparable to
published BEAM 1M/10M numbers or production memory benchmarks because the
scale, adapter, extraction policy, and judge setup differ.

## External Comparisons

The paper may cite external reported numbers from Mem0, Zep/Graphiti, Letta,
A-MEM, BEAM, LongMemEval-V2, and Memora/FAMA, but those rows must be labeled
`external reported` unless rerun through an AutoMem-controlled harness.

For any comparison table, include:

- system and version/date
- open-source vs managed
- benchmark and dataset version/hash
- scope and question count
- ingest/extraction protocol
- retrieval method
- answer model and judge/evaluator
- token/context budget
- latency/cost if available
- score and recall@k
- artifact URL or repro command
- claim status

## Limitations To State

- No current SOTA claim.
- External systems use different extraction policies, judges, hosted services,
  and token budgets.
- BEAM 1M/10M, LongMemEval-V2, and Memora/FAMA have not yet been run as
  canonical AutoMem benchmarks.
- LoCoMo and LongMemEval primarily test recall and answer synthesis; they do
  not fully measure write precision, forgetting, privacy boundaries, or
  long-running coding-agent workflows.
- Some detailed JSON result artifacts are local/generated and gitignored; the
  committed experiment log is the current durable source.
