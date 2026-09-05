# Repository Guidelines

## Astra 协作约定

- 以用户当前目标和本轮明确约束为准。任务要求实施时，完成实际修改与必要验证，不停在计划、建议或“是否继续”。普通实现选择自行决定；只有缺失信息会实质改变结果或操作超出授权时才询问，并先完成不依赖答案的工作。
- 用户指令优先于本地 skills 的工作流建议。只读取与当前任务直接相关的文件和技能；不因关键词命中就串联整套技能、生成流程工件或增加审批。
- 保留既有业务规则、数据所有权、用户改动和明确的工具限制。只改当前目标需要的内容，不顺手重构、升级依赖、搬目录或扩展产品范围。

## 拒绝过度防御性编程

- 直接使用已有输入、文件、依赖和运行环境，不重复做环境、权限、目录或文件存在性预检查。
- 不为假想故障添加重复参数验证、大量极端输入分支、宽泛 `try/catch`、默认值兜底、静默失败或伪造成功。契约不满足时暴露具体错误。
- 不主动新增重试、退避、熔断、降级、备用实现、兼容层、自动备份、回滚、迁移或恢复机制。
- 不主动添加 SHA、MD5、签名、文件哈希、完整性校验、CI/CD、发布门禁、安全扫描、许可证审计、复杂日志、监控、遥测或诊断框架。
- 不为未来需求预建插件系统、通用框架或抽象层，不为小改动铺设大量单元测试、回归测试、故障注入或性能基准。
- 只在缺少检查会立即阻止核心功能、造成明显数据损坏或掩盖真实错误时保留最小必要检查。现有鉴权、真实业务校验和数据保护功能继续遵守其契约；本规则不授权删除这些功能。
- 例外必须来自用户明确要求，或与本次改动直接相关的既有产品契约。旧文档中泛化的“每次全量检查”“必须先审批”“自动完善”不构成额外任务。

## 验证与交付

- 选择能证明本次行为的最小验证：文档或提示词改动检查内容和 diff；代码改动运行相关构建、现有定向测试或核心流程冒烟。低影响、可逆改动不新增仅复述实现的测试。
- 必要检查通过即交付；只有新改动、失败或具体未解决疑点才扩大或重复验证。不要为了收尾重跑无关全量测试、打包、实机流程或基准。
- 错误如实报告。区分实际运行通过、静态检查、未运行与真实环境验证；历史测试数量不能当作本次证据。
- 仅在任务需要时使用子代理；不强制委派、切换模型或修改推理档位，遵守当前会话设置与工具权限。
- 按当前授权和项目约定执行 Git 操作，只提交本任务文件；不要为清空工作区而夹带其他改动，不强推或丢弃用户内容。没有远端时报告，不擅自创建远端。
- 用简明中文交代实际修改、验证结果和已知问题。只有需求、接口或已验证事实改变时同步相关文档，不追加与交付无关的报告。

## Project Structure & Modules

- `automem/`: Core package. Notable dirs: `api/` (Flask blueprints), `utils/`, `stores/`, `config.py`.
- `app.py`: Flask API entry point used in local/dev and tests.
- `tests/`: Pytest suite (`test_*.py`), plus legacy benchmark harnesses under `tests/benchmarks/`.
- `benchmarks/`: Snapshot-based benchmark system. See `EXPERIMENT_LOG.md` for current baselines and results.
- `scripts/bench/`: Benchmark tooling (ingest, eval, compare, health check).
- `docs/`: API, testing, deployment, monitoring, and env var references.
- `scripts/`: Maintenance and ops helpers (backup, reembed, health monitor). See the canonical [scripts catalog](scripts/README.md) for lifecycle and usage.
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
- Reuse existing formatting/lint tools for affected Python changes; do not install hooks as part of unrelated work.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.

## Testing Guidelines

- Framework: Pytest. Place tests in `tests/` named `test_*.py`.
- Unit tests: `make test`.
- Integration: `make test-integration` (requires Docker). See `docs/TESTING.md` for env flags and live testing options.
- Run the directly affected tests. Add one only when needed to prove a new behavior or reproduce the defect; prefer fixtures over globals.

## Benchmarking

Run benchmarks only for requested retrieval/scoring/performance work. Prefer the existing snapshot-based `make bench-eval BENCH=locomo-mini` for a bounded before/after comparison; use a full benchmark only when required to answer the task. Do not attach paid judge runs or full benchmark tiers to ordinary commits.

`benchmarks/EXPERIMENT_LOG.md` stores official results. `automem-evals` holds exploratory corpora and bulky results; external evaluations follow `docs/EVALS_CONTRACT.md`. Keep snapshots and generated results in existing ignored paths. Record only measurements actually run.

## Commit & Pull Requests

- Feature PRs target `develop` (repo default). Promote `develop` to `main` with a validated release merge; release-please and GHCR `:stable` then run on `main`. Do not open feature work onto `main`.
- PR titles must use Conventional Commit format because squash merges use the PR title as the release commit title. Do not prefix titles with `[codex]`, `[claude]`, `[copilot]`, `[wip]`, or similar labels; put agent/status context in the PR body.
- Use Conventional Commit types: `feat`, `fix`, `docs`, `refactor`, `test`, `ci`, `build`, `chore`, `perf`, `revert` (e.g., `feat(api): add /analyze endpoint`).
- For public API changes, use `feat(api): ...` unless the change is strictly a bug fix with no new public surface. For docs-only changes, use `docs: ...`; for release automation, use `ci(release): ...` or `chore(release): ...`.
- PRs must include: clear description and scope, linked issues, test plan/output, and notes on API or config changes. Update relevant docs under `docs/`.
- Observe existing checks when a PR is requested; do not add CI or make unrelated workflows a gate for a local documentation change.

## Security & Configuration

- Never commit secrets. Configure via env vars: `AUTOMEM_API_TOKEN`, `ADMIN_API_TOKEN`, `OPENAI_API_KEY`, `FALKORDB_PASSWORD`, `QDRANT_API_KEY`.
- Local dev uses Docker defaults; see `docs/ENVIRONMENT_VARIABLES.md` and `docker-compose.yml` for ports and credentials.

## Agent Memory Protocol

Consult `.cursor/rules/automem.mdc` only when the user requests AutoMem memory operations. Ordinary repository work does not authorize writing agent memories.
