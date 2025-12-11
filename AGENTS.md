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

## Task Completion Checklist
**CRITICAL**: Before declaring any coding task complete, ALWAYS:
1. **Run the build**: `make build` (Python) or `npm run build` (graph-viewer)
2. **Run lints**: `make lint` (Python) or `npm run tsc` (TypeScript)
3. **Run tests** (if applicable): `make test` or `npm test`
4. **If any fail**: Iterate and fix until all pass
5. **Never commit or deploy** code that doesn't build

For graph-viewer specifically:
```bash
cd packages/graph-viewer && npm run build
```

## Agent Memory Protocol
Follow rules in `.cursor/rules/automem.mdc` for memory operations.
