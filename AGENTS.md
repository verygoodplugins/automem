# Repository Guidelines

## Project Structure & Modules

- `automem/`: Core package. Notable dirs: `api/` (Flask blueprints), `utils/`, `stores/`, `config.py`.
- `app.py`: Flask API entry point used in local/dev and tests.
- `tests/`: Pytest suite (`test_*.py`), plus legacy benchmark harnesses under `tests/benchmarks/`.
- `benchmarks/`: Snapshot-based benchmark system. See `EXPERIMENT_LOG.md` for current baselines and results.
- `scripts/bench/`: Benchmark tooling (ingest, eval, compare, health check).
- `docs/`: API, testing, deployment, monitoring, and env var references.
- `scripts/`: Maintenance and ops helpers (backup, reembed, health monitor).
- `mcp-sse-server/`: Optional MCP bridge used in some deployments.

## Build, Test, and Development

- `make install`: Create `.venv` (and symlink `venv -> .venv`) and install dev deps. Prefers Python 3.12 and fails fast on incompatible `python3`.
- `source .venv/bin/activate`: Activate the virtualenv.
- `make dev`: Start local stack via Docker (FalkorDB, Qdrant, API).
- `make test`: Run unit tests (fast, no services).
- `make test-integration`: Start Docker and run full integration tests.
- `make fmt` / `make lint`: Format with Black/Isort and lint with Flake8.
- `make bench-eval BENCH=locomo-mini`: Run snapshot-based benchmark (~2 min). See Benchmarking section below.
- `make deploy` / `make status`: Deploy/check Railway. Quick health: `curl :8001/health`.

## Coding Style & Naming

- Python with type hints. Indent 4 spaces; line length 100 (Black).
- Tools: Black, Isort (profile=black), Flake8; pre-commit hooks available.
- Run `pre-commit install` and `make fmt && make lint` before committing.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.

## Testing Guidelines

- Framework: Pytest. Place tests in `tests/` named `test_*.py`.
- Unit tests: `make test`.
- Integration: `make test-integration` (requires Docker). See `docs/TESTING.md` for env flags and live testing options.
- Add/adjust tests for new endpoints, stores, or utils; prefer fixtures over globals.

## Benchmarking

The benchmark system uses **snapshot-based evaluation**: ingest once, eval many times from the same snapshot. This keeps runs deterministic and fast.

**Source of truth**: `benchmarks/EXPERIMENT_LOG.md` — contains current baselines, all experiment results, and the tiered benchmark table.

### Tiered System

| Tier | Benchmark | Command | Runtime | Cost | When to use |
|------|-----------|---------|---------|------|-------------|
| 0 | Unit tests | `make test` | 30s | free | Every change |
| 1 | LoCoMo-mini (2 convos, 304 Qs) | `make bench-eval BENCH=locomo-mini` | 2-3 min | free / ~$0.20 with judge | Rapid iteration |
| 2 | LoCoMo-full (10 convos, 1986 Qs) | `make bench-eval BENCH=locomo` | 5-10 min | free / ~$1-3 with judge | Before merge |
| 3 | LongMemEval-mini (20 Qs) | `make bench-mini-longmemeval` | 15 min | ~$1 | Scoring/entity changes |
| 4 | LongMemEval-full (500 Qs) | `make test-longmemeval` | 1-2 hr | ~$10 | Milestones only |

### Key Commands

- `make bench-eval BENCH=locomo-mini CONFIG=baseline` — eval from snapshot (~2 min).
- `make bench-compare BENCH=locomo CONFIG=<name> BASELINE=baseline` — A/B compare two configs.
- `make bench-compare-branch BRANCH=<branch>` — compare a branch against baseline.
- `make bench-ingest BENCH=locomo` — ingest + snapshot (run once per embedding change).
- `make bench-health` — recall health check (score distribution, entity quality, latency).

### Workflow for Recall/Retrieval Changes

1. Run `make bench-eval BENCH=locomo-mini` on `main` to confirm the current baseline.
2. Create a feature branch and implement changes.
3. Run the same eval on the branch.
4. Record both results as a new row in `benchmarks/EXPERIMENT_LOG.md`.
5. Promote to `make bench-eval BENCH=locomo` (full) before merge.

### Directory Layout

- `benchmarks/EXPERIMENT_LOG.md` — results table and experiment metadata (committed).
- `benchmarks/baselines/` — baseline result JSONs (small files committed, large ones gitignored).
- `benchmarks/snapshots/` — Qdrant/FalkorDB snapshot data (gitignored, regenerate with `make bench-ingest`).
- `benchmarks/results/` — per-run result JSONs (gitignored).
- `scripts/bench/` — shell and Python scripts driving ingest, eval, compare, and health checks.
- `tests/benchmarks/` — legacy benchmark harnesses (LoCoMo, LongMemEval) and historical result markdown files.

## Commit & Pull Requests

- Use Conventional Commits style: `feat`, `fix`, `docs`, `refactor`, `test`, `chore` (e.g., `feat(api): add /analyze endpoint`).
- PRs must include: clear description and scope, linked issues, test plan/output, and notes on API or config changes. Update relevant docs under `docs/`.
- CI must pass; formatting/lint clean.

## Security & Configuration

- Never commit secrets. Configure via env vars: `AUTOMEM_API_TOKEN`, `ADMIN_API_TOKEN`, `OPENAI_API_KEY`, `FALKORDB_PASSWORD`, `QDRANT_API_KEY`.
- Local dev uses Docker defaults; see `docs/ENVIRONMENT_VARIABLES.md` and `docker-compose.yml` for ports and credentials.

## Agent Memory Protocol

Follow rules in `.cursor/rules/automem.mdc` for memory operations.
