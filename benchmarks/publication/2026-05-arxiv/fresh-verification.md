# Fresh Verification Notes

Local verification run date: 2026-05-17 / 2026-05-18 UTC.

## Repository Checks

| Check | Result | Notes |
|---|---|---|
| `make test` | pass | 238 passed, 1 skipped, 25 deselected |
| `.venv/bin/black --check .` | pass | Added `pyproject.toml` with the repo's documented 100-column Black configuration and generated-directory excludes. |
| `.venv/bin/isort --check-only .` | pass | Added `pyproject.toml` isort config; excludes virtualenvs, snapshots, worktrees, showcase output, and vendored LoCoMo source. |
| `make lint` | pass | Required `.flake8` exclude update for generated/env/worktree directories. |
| `make test-integration` | pass | 11 passed, 253 deselected after rebuilding the Docker API image. |
| `make bench-health` | pass | Required Makefile fix to use the venv Python; health reported `HEALTHY`. |

## Benchmark Reruns

| Benchmark | Command | Result | Generated artifact |
|---|---|---|---|
| LoCoMo mini | `BENCH_JUDGE_MODEL=gpt-5.4-mini-2026-03-17 make bench-eval BENCH=locomo-mini CONFIG=baseline` | 85.20% (259/304) | `benchmarks/results/locomo-mini_baseline_20260517_182318.json`, sha256 `ba2b98b0055f92ca17de9bc36207d7f39cf90b6270c2c3d903d69b8044aa7015` |
| LoCoMo full | `BENCH_JUDGE_MODEL=gpt-5.4-mini-2026-03-17 make bench-eval BENCH=locomo CONFIG=baseline` | 84.74% (1683/1986) | `benchmarks/results/locomo_baseline_20260517_193934.json`, sha256 `a75816e9a6d3302c22b34852b75ac19a9d9f5cb27d1a109e0af7e49359330716` |
| LongMemEval mini | `./test-longmemeval-benchmark.sh --llm-eval --llm-model gpt-5-mini --per-type 5 --output benchmarks/results/longmemeval-mini-publication-20260517` | 70.00% (21/30), recall@5 96.67% (29/30) | `benchmarks/results/longmemeval-mini-publication-20260517.json`, sha256 `7ea922b77e312a17c313bbf8c0e81f0268b48d1082080cae1db3c38e906577b8` |
| LongMemEval full | `./test-longmemeval-benchmark.sh --llm-eval --llm-model gpt-5-mini --eval-llm-model gpt-5.4-mini-2026-03-17 --output benchmarks/results/longmemeval-full-publication-20260518` | 87.00% (435/500), recall@5 97.00% (485/500) | `benchmarks/results/longmemeval-full-publication-20260518.json`, sha256 `ed6f7cf69b7be6fa0050536ec2b0f947f5510afd8c2a374b3fafb9cde009da75` |

The official LongMemEval full claim is now the fresh publication verification
artifact:

- `benchmarks/results/longmemeval-full-publication-20260518.json`
- sha256 `ed6f7cf69b7be6fa0050536ec2b0f947f5510afd8c2a374b3fafb9cde009da75`
- 87.00% (435/500), recall@5 97.00% (485/500)
- `memory_ingest_failures=0`, `judge_errors=0`, `publishable=true`

Console caveat: the full run logged transient `gpt-5-mini` empty-answer
warnings and one local recall read timeout, but the harness completed and marked
the aggregate artifact publishable. Treat those warnings as answerer/service
stability notes rather than hidden failures.

The official LoCoMo full claim is now the fresh pinned-judge publication
verification artifact:

- `benchmarks/results/locomo_baseline_20260517_193934.json`
- sha256 `a75816e9a6d3302c22b34852b75ac19a9d9f5cb27d1a109e0af7e49359330716`
- 84.74% (1683/1986)
- Judge: `gpt-5.4-mini-2026-03-17`, 444 calls, 0 skips/errors

## Supplemental Eval Repo Checks

From a clone of the `automem-evals` repository. The local verification used a
sibling checkout referenced as `../automem-evals` below.

| Check | Result |
|---|---|
| `python3 -m unittest discover -s runners -p 'test_*.py'` | pass, 95 tests |
| `python3 -m unittest discover -s scripts -p 'test_*.py'` | pass, 10 tests |
| `npm test` in `../automem-evals/third_party/writ` | pass, 72 tests |
| `npm run build` in `../automem-evals/third_party/writ` | pass |

Writ drift evidence remains exploratory and lives in
`../automem-evals/docs/writ_integration.md`: AutoMem recall accuracy 100.0%,
update fidelity 20.0%, drift rate 0.0% across 5 drift scenarios.

## Paper Checks

The separate AutoMem paper source checkout passed static checks for input-file
existence and BibTeX cite-key resolution. No local LaTeX compiler (`pdflatex`,
`latexmk`, `tectonic`, or `pandoc`) was available, so no PDF compilation is
claimed.
