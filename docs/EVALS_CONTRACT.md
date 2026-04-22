# Eval Contract for External Benchmark Repos

`automem` is the source of truth for official benchmark claims and release-gating benchmark workflows.
External eval repos such as `automem-evals` should treat this repository as the system under test, not as a shared scratch space.

## Ownership Boundary

- Keep official LoCoMo and LongMemEval harnesses in `automem`.
- Keep published baselines and methodology notes in `benchmarks/EXPERIMENT_LOG.md`.
- Keep user-facing benchmark commands and release-gating flows in this repo.
- Put exploratory ruleset experiments, seeded corpora, scenario authoring, cross-agent comparisons, and bulky per-run artifacts in `automem-evals`.

## Supported Local Surface

When another repo needs to evaluate AutoMem locally, it should rely on the running service rather than repo internals.

- Start the stack from this repo with `docker compose up -d`.
- Wait for `http://localhost:8001/health` to report healthy FalkorDB and Qdrant services.
- Local Docker defaults expose:
  - API base URL: `http://localhost:8001`
  - API token: `test-token`
  - Admin token: `test-admin-token`
- Eval repos should pass endpoint and token explicitly when possible instead of depending on implicit shell state.

## Stable Endpoints for Eval Work

- `GET /health` for readiness and memory counts
- `POST /memory` for seeding synthetic corpora
- `GET /recall` for retrieval evaluation
- `GET /memory/<id>` for debugging a returned memory
- `POST /associate` only when the experiment explicitly needs typed graph edges

If an eval needs behavior outside those endpoints, document the dependency clearly before relying on it.

## Rules for External Evals

- Treat AutoMem as a black-box server under test.
- Do not make published benchmark claims from experimental harnesses unless the result is also reproduced through the official `automem` benchmark flow.
- If an experiment needs LoCoMo or LongMemEval, call the official harness here or label any adapter as experimental rather than canonical.
- Do not make `automem` depend on uncommitted files from an external eval repo.
