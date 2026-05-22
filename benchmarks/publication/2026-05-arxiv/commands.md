# Publication Verification Commands

Run these from the repository root unless noted.

## Repository Checks

```bash
make test
.venv/bin/black --check .
.venv/bin/isort --check-only .
make lint
make test-integration
make bench-health
```

## Canonical Benchmarks

```bash
BENCH_JUDGE_MODEL=gpt-5.4-mini-2026-03-17 make bench-eval BENCH=locomo-mini CONFIG=baseline
BENCH_JUDGE_MODEL=gpt-5.4-mini-2026-03-17 make bench-eval BENCH=locomo CONFIG=baseline
./test-longmemeval-benchmark.sh --llm-eval --llm-model gpt-5-mini --per-type 5 --output benchmarks/results/longmemeval-mini-publication
./test-longmemeval-benchmark.sh --llm-eval --llm-model gpt-5-mini --output benchmarks/results/longmemeval-full-publication
```

The current 84.74% LoCoMo full result in `benchmarks/EXPERIMENT_LOG.md` is the
fresh publication verification artifact produced by the pinned
`gpt-5.4-mini-2026-03-17` command above.

The LongMemEval full run is expensive and long-running. Use `--resume` with the
same `--output` base if interrupted.

## Supplemental Evals

From a clone of the `automem-evals` repository:

```bash
python3 -m unittest discover -s runners -p 'test_*.py'
python3 -m unittest discover -s scripts -p 'test_*.py'
python3 scripts/seed_from_snapshot.py
python3 scripts/seed_associations.py
python3 runners/compare_rulesets.py --rulesets baseline_v1 bare_tag_1m_v2
python3 scripts/beam_shim_smoke.py --self-spawn
python3 runners/run_writ.py --compare automem baseline --scenarios drift
```

Supplemental outputs are not canonical publication claims until they are
summarized here or reproduced by a canonical harness.
