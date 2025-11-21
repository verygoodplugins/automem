# Repository Guidelines

## Project Structure & Modules
- `automem/`: Core package. Notable dirs: `api/` (Flask blueprints), `utils/`, `stores/`, `config.py`.
- `app.py`: Flask API entry point used in local/dev and tests.
- `tests/`: Pytest suite (`test_*.py`), plus benchmarks under `tests/benchmarks/`.
- `docs/`: API, testing, deployment, monitoring, and env var references.
- `scripts/`: Maintenance and ops helpers (backup, reembed, health monitor).
- `mcp-sse-server/`: Optional Node SSE bridge used in some deployments.

## Build, Test, and Development
- `make install`: Create `venv` and install dev deps.
- `source venv/bin/activate`: Activate the virtualenv.
- `make dev`: Start local stack via Docker (FalkorDB, Qdrant, API).
- `make test`: Run unit tests (fast, no services).
- `make test-integration`: Start Docker and run full integration tests.
- `make fmt` / `make lint`: Format with Black/Isort and lint with Flake8.
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

## Commit & Pull Requests
- Use Conventional Commits style: `feat`, `fix`, `docs`, `refactor`, `test`, `chore` (e.g., `feat(api): add /analyze endpoint`).
- PRs must include: clear description and scope, linked issues, test plan/output, and notes on API or config changes. Update relevant docs under `docs/`.
- CI must pass; formatting/lint clean.

## Security & Configuration
- Never commit secrets. Configure via env vars: `AUTOMEM_API_TOKEN`, `ADMIN_API_TOKEN`, `OPENAI_API_KEY`, `FALKORDB_PASSWORD`, `QDRANT_API_KEY`.
- Local dev uses Docker defaults; see `docs/ENVIRONMENT_VARIABLES.md` and `docker-compose.yml` for ports and credentials.

## Agent Memory Protocol
- Start of task: fetch context with recall. Example:
  - `curl -H "Authorization: Bearer $AUTOMEM_API_TOKEN" "http://localhost:8001/recall?query=$TASK_SUMMARY&tags=repo:automem&tags=task:$TASK_ID&limit=20"`
  - Optionally read `GET /startup-recall` for rules and lessons.
- Persist key steps: POST `/memory` for starts, decisions, assumptions, and outputs. Always tag and set importance.
  - `curl -X POST -H 'Content-Type: application/json' -H "Authorization: Bearer $AUTOMEM_API_TOKEN" \`
    `-d '{"content":"Started: write AGENTS.md","tags":["agent","repo:automem","task:AGENTS","docs"],"importance":0.6}' http://localhost:8001/memory`
- Finish or checkpoints: summarize results and store another memory; PATCH to refine if needed.
  - `curl -X POST ... -d '{"content":"Completed: AGENTS.md contributor guide","tags":["agent","task:AGENTS","result"],"importance":0.7}' http://localhost:8001/memory`
- Associate related memories: link start/end, decisions → results, artifacts → tasks.
  - `curl -X POST -H 'Content-Type: application/json' -H "Authorization: Bearer $AUTOMEM_API_TOKEN" \`
    `-d '{"memory1_id":"$START_ID","memory2_id":"$END_ID","type":"RELATES_TO","strength":0.9}' http://localhost:8001/associate`
- Tagging conventions: `agent`, `repo:<name>`, `task:<id>`, `file:<path>`, `pr:<num>`, `issue:<num>`, `result`, `decision`, `docs`.
- Temporal fields: when appropriate, include `t_valid`/`t_invalid` to bound applicability.
