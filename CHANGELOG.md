# Changelog

All notable changes to AutoMem will be documented in this file.

## [0.10.1](https://github.com/verygoodplugins/automem/compare/v0.10.0...v0.10.1) (2026-02-07)


### Bug Fixes

* bound temporal query to 7-day window with timeout ([#62](https://github.com/verygoodplugins/automem/issues/62)) ([9c720de](https://github.com/verygoodplugins/automem/commit/9c720dea686e579d2f85881624db8c3c48ce395d))
* skip temperature param for o-series and gpt-5 models ([#65](https://github.com/verygoodplugins/automem/issues/65)) ([f1d5b1d](https://github.com/verygoodplugins/automem/commit/f1d5b1d283e3b0e929697ceed97aa99755c9ca27))
* use max_completion_tokens for gpt-5 model family ([#61](https://github.com/verygoodplugins/automem/issues/61)) ([1ec25db](https://github.com/verygoodplugins/automem/commit/1ec25dbc7616f1a00008df59f34abd589feec770))

## [0.10.0](https://github.com/verygoodplugins/automem/compare/v0.9.3...v0.10.0) (2026-02-06)


### Features

* Add `exclude_tags` parameter to recall API ([#58](https://github.com/verygoodplugins/automem/issues/58)) ([8aeca2e](https://github.com/verygoodplugins/automem/commit/8aeca2ebba472ce065ed97b46ef495519a836067))
* add memory content size governance with auto-summarization ([e24a86b](https://github.com/verygoodplugins/automem/commit/e24a86be0a0dde80804aef6b6d88953d0e4dd9a5))
* add memory content size governance with auto-summarization ([876aabf](https://github.com/verygoodplugins/automem/commit/876aabf320cc5937c44d6678e02f0aed33b051b8))
* Add real-time observability with SSE streaming and TUI monitor ([54d2701](https://github.com/verygoodplugins/automem/commit/54d2701f42992de6172aa862d794f6b09ff0c95b))
* Add real-time observability with SSE streaming and TUI monitor ([c5f6370](https://github.com/verygoodplugins/automem/commit/c5f63708f0bcb3ab63c56ae0ae63da0dd4b3229b))
* Add real-time observability with SSE streaming and TUI monitor ([944168e](https://github.com/verygoodplugins/automem/commit/944168e8e00bcdc42af09944577e32475ce25119))
* **api:** add graph visualization API endpoints ([#26](https://github.com/verygoodplugins/automem/issues/26)) ([a0d0dd3](https://github.com/verygoodplugins/automem/commit/a0d0dd3719ac9dad7c4e24e78f290c9364d2b2e1))
* **api:** add time-based sorting to recall API ([c063845](https://github.com/verygoodplugins/automem/commit/c063845bb72af7aaaff519083b850da84f20003e))
* **api:** add time-based sorting to recall API ([a511f0c](https://github.com/verygoodplugins/automem/commit/a511f0c6b6ab74e1c98d29eaae19c7ca69f6542b))
* create coderabbit.yml ([e05811f](https://github.com/verygoodplugins/automem/commit/e05811f32904a0355b34957be0d5a15629f73f2b))
* **embedding:** add Ollama embedding provider and expand docs ([#56](https://github.com/verygoodplugins/automem/issues/56)) ([b506a90](https://github.com/verygoodplugins/automem/commit/b506a90661ee8f329bfdc40fff6dda7b2c6a0504))
* **mcp-sse:** agent-friendly recall output + tests ([71201d3](https://github.com/verygoodplugins/automem/commit/71201d33b5b8a9044f23f3a911dbf193de67b669))
* **mcp-sse:** improve recall tool output and options ([97c4f72](https://github.com/verygoodplugins/automem/commit/97c4f7240b5319a95412e7d1e65ae65e67150a46))


### Bug Fixes

* Add TTL cleanup to InMemoryEventStore ([0631af6](https://github.com/verygoodplugins/automem/commit/0631af6cee60f70505aa0f392e5073f8245b7923))
* Address CodeRabbit security and lint issues ([a701c57](https://github.com/verygoodplugins/automem/commit/a701c578340c90bb08456a066cdf1e304b11ddda))
* **backup:** Fix GitHub Actions backup failing with connection error ([19ed1e8](https://github.com/verygoodplugins/automem/commit/19ed1e817f2faab4b570a3d7cdc829c3d9a664ff))
* **backup:** Fix GitHub Actions backup failing with connection error ([10b37d0](https://github.com/verygoodplugins/automem/commit/10b37d0f92273dbaf29bad4cb32d95ff80df5e81))
* **consolidation:** prevent accidental forgetting ([4330c20](https://github.com/verygoodplugins/automem/commit/4330c207d01f68fda96347f36cac41dc466ce93a))
* **consolidation:** prevent accidental forgetting ([39dee3e](https://github.com/verygoodplugins/automem/commit/39dee3ec44baf73bf319d16b3ca11667ad063fd2))
* Correct token parameter for gpt-5 models (max_output_tokens) ([0786705](https://github.com/verygoodplugins/automem/commit/0786705597493779ee16d64b6d111428336da7a7))
* **enrichment:** sync tags/tag_prefixes to Qdrant ([3ccf874](https://github.com/verygoodplugins/automem/commit/3ccf8748901c6f297a7cb0487d5ac9ddf63bd076))
* **enrichment:** sync tags/tag_prefixes to Qdrant ([7ba9d76](https://github.com/verygoodplugins/automem/commit/7ba9d76d2be0575fe688ada42f9a29542b890edf))
* Handle streamable HTTP client disconnects ([f15a7ee](https://github.com/verygoodplugins/automem/commit/f15a7ee81ef690581bbc54deca38c168401a4cac))
* **mcp-sse:** update test for SDK 1.20+ Accept header behavior ([597909e](https://github.com/verygoodplugins/automem/commit/597909e40a2aecc28420c3a9bbb0a41deda0bfa7))
* **mcp:** close transport/server on sweep and update Accept header test ([ce7f6fa](https://github.com/verygoodplugins/automem/commit/ce7f6fad6720bd091883805ddbf8d4e702a91e60))
* **mcp:** replace broken res.on('close') with TTL sweep for Streamable HTTP sessions ([3d53263](https://github.com/verygoodplugins/automem/commit/3d53263389c72c2c2fd624ea3cb18b8d5d1e1227))
* **mcp:** replace broken res.on('close') with TTL sweep for Streamable HTTP sessions ([#59](https://github.com/verygoodplugins/automem/issues/59)) ([d4d3259](https://github.com/verygoodplugins/automem/commit/d4d325916f167c0f69a97038d8d335b0dc15a95d))
* **mcp:** use Promise.resolve().catch() for async close() in session sweep ([a8cc589](https://github.com/verygoodplugins/automem/commit/a8cc5895dd826883e3c8e3fbeda66c9a8f7b5790))
* Production bugs - datetime tz, OpenAI tokens, Qdrant race ([14491c4](https://github.com/verygoodplugins/automem/commit/14491c47ea78729d03cd36390cda67f3cc70670a))
* Production bugs - datetime tz, OpenAI tokens, Qdrant race ([0771360](https://github.com/verygoodplugins/automem/commit/07713600e3b5c13c9c3662b89a2baa6c1d1bb90b))
* Refactor tests to use double quotes for consistency ([21da453](https://github.com/verygoodplugins/automem/commit/21da45310652ee700ddb97da11db65b6ddeb6e90))
* Refine summarization token limit and improve tests ([6cab145](https://github.com/verygoodplugins/automem/commit/6cab145ea6e4e65ad172ccf242b2a57c8f585018))
* Update test for POST /mcp Accept header error ([67a5a10](https://github.com/verygoodplugins/automem/commit/67a5a10a8ae416eeb6992c1ffaafe33aae881ff1))
* YAML indentation in backup workflow and update docs ([a606b3e](https://github.com/verygoodplugins/automem/commit/a606b3e4ae160d0f3b7e390579ab7b0adac70947))


### Documentation

* add mermaid diagrams for architecture visualization ([0d70a2a](https://github.com/verygoodplugins/automem/commit/0d70a2a9037743059dbb3bc1ca1902cdc9ad5b5c))
* Add Qdrant setup guide and update docs for AUTOMEM_API_URL ([5c440aa](https://github.com/verygoodplugins/automem/commit/5c440aa7fa38c3169655601953859b5ffd4a009e))
* Add Qdrant setup guide and update docs for AUTOMEM_API_URL ([123ff28](https://github.com/verygoodplugins/automem/commit/123ff282508e648a97fe0a0d40c522cf1ceded62))
* Enhance README.md with formatted NPM bridge link ([09ac4ac](https://github.com/verygoodplugins/automem/commit/09ac4acc6e9f7738f25b06cbe12dff0975ad2e0f))
* merge mermaid diagrams from docs/add-mermaid-diagrams ([4e9ae4f](https://github.com/verygoodplugins/automem/commit/4e9ae4fcd18b58a356cb87e34ca43a29e81a5bf7))
* Update API documentation and configuration settings ([9281149](https://github.com/verygoodplugins/automem/commit/928114926cb7795b3ee70783d38a2ba747209355))
* Update API documentation and configuration settings ([#51](https://github.com/verygoodplugins/automem/issues/51)) ([ad9500f](https://github.com/verygoodplugins/automem/commit/ad9500f698c9d78bcae29682bad43164cba8028b))
* Update API endpoint count and project description ([9e71299](https://github.com/verygoodplugins/automem/commit/9e71299be2bb6744d78e0e596b2d3afcc760458a))
* update diagrams for Streamable HTTP transport ([1a2ce22](https://github.com/verygoodplugins/automem/commit/1a2ce22caea6d46b27b59dd235cc8076be2a5e1b))
* Update docs to clarify MCP bridge and improve formatting ([af69a2f](https://github.com/verygoodplugins/automem/commit/af69a2fd6c7f91e82e96c0b48b24d21796782651))
* Update MCP documentation and server implementation for Streamable HTTP support ([8fdef09](https://github.com/verygoodplugins/automem/commit/8fdef09863989bdde0a514b152d570cc93d69d37))
* Update MCP documentation and server implementation for Streamable HTTP support ([335a78d](https://github.com/verygoodplugins/automem/commit/335a78dc60cf2821eb095cd276c74e8d81418469))
* update MCP_SSE.md to reflect MCP bridge terminology ([940a776](https://github.com/verygoodplugins/automem/commit/940a776e0cb43ad74c4dd1c7c2820f9317b3a4c9))
* Update Railway deployment docs and improve README ([5480eb5](https://github.com/verygoodplugins/automem/commit/5480eb52136162893236e0b5278cb241dd57481f))
* Update Railway deployment docs and improve README ([82c33b0](https://github.com/verygoodplugins/automem/commit/82c33b05b7b24d66922888ba4ba3b09ba55a10d6))

## [Unreleased]

### Fixed

- **GitHub Actions Backup**: Fixed backup workflow failing with "Connection reset by peer" error
  - Root cause: Backup script tried to connect to Railway internal hostname which isn't accessible from GitHub Actions
  - Solution: Use Railway TCP Proxy for external access to FalkorDB
  - Added pre-flight connectivity check with clear error messages and troubleshooting guidance
  - Updated documentation with TCP Proxy setup instructions

### Documentation

- Expanded `docs/MONITORING_AND_BACKUPS.md` with:
  - TCP Proxy setup instructions for GitHub Actions backups
  - Network architecture diagram explaining internal vs external access
  - Troubleshooting section for common backup failures
  - Debug checklist for verifying backup configuration

## [0.9.3] - 2025-12-10

### Fixed

- **Critical**: Fixed missing `RECALL_EXPANSION_LIMIT` import that caused crashes when using expansion features

### CI/CD

- Added GitHub Actions CI workflow with tests on Python 3.10, 3.11, 3.12
- Added pre-commit hooks for code quality enforcement
  - Black formatting (100 char line length)
  - isort import sorting
  - Flake8 syntax checking
  - Conventional commit validation
  - Secret detection
- Added release-please for automated versioning and releases

### Style

- Applied consistent formatting across entire codebase with Black and isort

## [0.9.2] - 2025-12-10

### Added

- **Expansion Filtering**: Added `expand_min_strength` and `expand_min_importance` parameters to `recall` API
  - Filters expanded relation results by connection strength and target memory importance
  - Reduces noise in multi-hop reasoning and graph expansion
  - Seed results are never filtered, only expanded memories
  - Recommended values: 0.3-0.5 for broad context, 0.6+ for focused results
- **Vector Dimension Autodetect**: Added `VECTOR_SIZE_AUTODETECT` env var
  - Automatically detects and adopts existing Qdrant collection dimension
  - Prevents service crashes when upgrading defaults (e.g. 768d -> 3072d)

### Documentation

- Expanded `docs/API.md` with detailed expansion filtering documentation and examples
- Added graph expansion examples to `README.md` API examples section

## [0.9.1] - 2025-12-02

### 🏆 Benchmark Improvement

- Achieved **90.53%** on LoCoMo-10 (SOTA, +2.29pp over CORE at 88.24%)
- Multi-hop reasoning improved from 37.5% to **50%** (+12.5pp)
- New benchmark report: `tests/benchmarks/BENCHMARK_2025-12-02.md`

### 🔗 Entity-to-Entity Expansion

- **Feature**: New `expand_entities` parameter for multi-hop reasoning via entity tags
- **How it works**: Extracts entities from seed results → searches for `entity:people:{name}` tags
- **Use case**: Query about "Amanda's sister's career" finds "Amanda's sister is Rachel" → entity expansion finds "Rachel works as counselor"
- Respects original tag filters for proper context scoping

### 🧪 Test Infrastructure

- Fixed test cleanup: was missing 99% of tagged memories (used wrong endpoint)
- Added `test_multihop_quick.py` for rapid multi-hop iteration
- Improved Q+A embedding similarity checking in test harness

### 📝 Documentation

- Updated API.md with `expand_entities` and `expand_relations` params
- Updated ENVIRONMENT_VARIABLES.md with `RECALL_EXPANSION_LIMIT`
- Updated TESTING.md with latest benchmark results
- Updated README with 90.53% score

### 🔧 Configuration

- New param: `expand_entities` (default: false) - enables entity-to-entity expansion
- New param: `entity_expansion` (alias for expand_entities)
- Existing: `RECALL_EXPANSION_LIMIT=25` - total expansion results limit

## [0.9.0] - 2025-11-20

### 🏆 Benchmark Breakthrough

- Achieved **90.38%** on LoCoMo-10 (SOTA, +2.14pp over CORE at 88.24%)
- New benchmark report: `tests/benchmarks/BENCHMARK_2025-11-20.md`

### 🧠 Retrieval Improvements

- Added **multi-hop bridge discovery** to connect disparate seed memories
- Introduced **temporal alignment scoring** for time-aware recall
- Added **content token overlap** to boost lexical precision
- Hybrid scoring now combines 9 weighted components (vector, keyword, relation, content, temporal, tag, importance, confidence, recency, exact)

### 🔧 Configuration

- New env vars: `RECALL_BRIDGE_LIMIT`, `SEARCH_WEIGHT_CONTENT`, `SEARCH_WEIGHT_TEMPORAL`
- Tuned defaults for relation/path expansion (`expand_relations`, `expand_paths`, bridge limits)

### 📝 Documentation

- Documented LoCoMo SOTA run and configuration in `docs/TESTING.md`
- Updated changelog and benchmark references

## [0.8.0] - 2025-11-08

### 🏗️ Major Refactoring

#### API Modularization

- **Breaking**: Refactored monolithic `app.py` into modular API blueprints
- New structure: `automem/api/` with separate modules:
  - `memory.py` - Core memory CRUD operations
  - `recall.py` - Memory search and retrieval
  - `admin.py` - Administrative endpoints (reembedding, analysis)
  - `health.py` - Health checks and monitoring
  - `enrichment.py` - Enrichment pipeline management
  - `consolidation.py` - Memory consolidation
- Improved code organization and maintainability
- Reduced `app.py` from 1,251 lines to ~200 lines

### 🔒 Security Fixes

#### Memory ID Security

- **Critical**: Removed support for client-supplied memory IDs
- Now always generates server-side UUIDs to prevent collision/overwrite attacks
- Eliminates potential for malicious clients to overwrite existing memories

#### Batch Processing Reliability

- Fixed batch failure counter in reembedding endpoint
- Now correctly tracks only successfully upserted items
- Prevents incorrect success counts when Qdrant upserts fail

### 🚀 Railway Deployment Improvements

#### Template Fixes

- **Critical**: Fixed secret generation syntax in `railway-template.json`
- Changed from `{"generator": "secret"}` to `${{secret()}}` (Railway's correct syntax)
- Fixes blank `FALKOR_PASSWORD`, `ADMIN_API_TOKEN`, and `AUTOMEM_API_TOKEN` on deploy
- Removed `--requirepass` from `REDIS_ARGS` to prevent Redis config parse errors
- FalkorDB now handles authentication via `FALKOR_PASSWORD` env var only

#### Documentation Enhancements

- Expanded `docs/RAILWAY_DEPLOYMENT.md` with detailed troubleshooting
- Added step-by-step deployment instructions
- Included cost optimization tips and backup strategies
- Documented persistent storage configuration
- Added environment variable reference and best practices

### 🧪 Testing Improvements

#### Live Integration Tests

- Added safety confirmation prompt to `test-live-server.sh`
- Prevents accidental test runs against production
- New `--non-interactive` flag for CI/CD pipelines
- Fixed argument parsing bug (shift mutation in for-loop)
- `test-live-server-auto.sh` now wrapper for backward compatibility

#### LoCoMo Benchmark Support

- Added LoCoMo benchmark integration for memory system evaluation
- Documents AutoMem's 76.08% accuracy on LoCoMo-10 (vs CORE at 88.24%) as of November 8, 2025
- Comprehensive benchmark results in `tests/benchmarks/BENCHMARK_2025-11-08.md` and summarized in `docs/TESTING.md`
- Test coverage for multi-hop reasoning, temporal understanding, complex reasoning

#### Test Infrastructure

- Added OpenAI chat completions mock in `conftest.py`
- Improved test reliability with proper cache invalidation
- Removed unused test dependencies (`requests`)

### 🛠️ Code Quality Improvements

#### Utility Consolidation

- Removed duplicate `_parse_iso_datetime` from `consolidation.py`
- Now imports from shared `automem.utils.time` module
- Improved code reusability and consistency

#### Code Cleanup

- Removed erroneous `or True` from hasattr checks
- Added proper error logging for import failures
- Removed trailing whitespace across codebase
- Added syntax highlighting to documentation code blocks
- Cleaned up unused dependencies in `requirements-dev.txt`

### 📝 Documentation Updates

#### New Documentation

- Added `docs/API.md` - Comprehensive API reference
- Added `docs/TESTING.md` - Testing guide with benchmark instructions
- Enhanced `README.md` with clearer setup instructions

#### Removed Obsolete Documentation

- Archived outdated optimization notes
- Removed deprecated benchmark files
- Consolidated deployment documentation

### 🔧 Configuration Changes

#### Pre-commit Hooks

- Added `.pre-commit-config.yaml` for automated code quality checks
- Includes trailing whitespace removal, end-of-file fixing

#### Build System

- Updated `Makefile` with new testing targets
- Improved development workflow commands

### 🐛 Bug Fixes

- Fixed Qdrant delete selector to properly check for HTTP models
- Added null-safety guard for enrichment queue in health endpoint
- Fixed test consolidation engine cache invalidation
- Corrected JSON syntax highlighting in documentation

## [0.7.1] - 2025-10-20

### 🐛 Railway Deployment Fixes

#### Fixed - IPv6 Networking Support

- **Critical**: Flask now binds to `::` (IPv6 dual-stack) instead of `0.0.0.0` (IPv4 only)
- Railway's internal networking uses IPv6 addresses (e.g., `fd12:ca03:42be:...`)
- Resolves `ECONNREFUSED` errors when services try to connect via internal DNS
- Memory-service now accepts connections from SSE sidecar and other services

#### Fixed - Port Configuration

- **Critical**: Added `PORT=8001` to Railway template for memory-service
- Without explicit PORT, Flask defaults to 5000, causing connection failures
- Template now includes PORT for all services that require it
- Documentation emphasizes PORT is **required** for Railway deployments

#### Fixed - FalkorDB Data Persistence

- Corrected volume mount path: `/var/lib/falkordb/data` (was `/data`)
- FalkorDB now properly persists data across restarts
- Updated `REDIS_ARGS` in template to match official FalkorDB Railway template
- Removed health check configuration (Railway uses container monitoring for databases)

#### Fixed - Service Naming Consistency

- Standardized service name to `memory-service` across all configs and docs
- Updated `AUTOMEM_ENDPOINT` in SSE service to use correct internal DNS
- Eliminated confusion from mixed naming (`flask`, `automem-api`, `memory-service`)
- `RAILWAY_PRIVATE_DOMAIN` now consistently resolves to `memory-service.railway.internal`

### 📝 Documentation Improvements

#### Enhanced - Railway Deployment Guide

- New troubleshooting section: "Service Connection Refused (ECONNREFUSED)"
- Added detailed explanation of Railway's IPv6 networking
- Documented PORT requirement and common pitfalls
- Clarified when to use hardcoded values vs `${{...}}` variable references
- Updated FalkorDB setup instructions with correct volume paths
- Added memory_count to health check examples for easier debugging

#### Enhanced - Environment Variables Reference

- Marked `PORT` as **required** for Railway deployments
- Added Railway networking notes explaining IPv6 dual-stack binding
- Clarified auto-populated Railway variables and their usage
- Updated FalkorDB connection variable documentation

#### Enhanced - MCP SSE Documentation

- Added comprehensive troubleshooting steps for `fetch failed` errors
- Updated service references to use `memory-service` consistently
- Added note about internal DNS matching and verification commands

### 🔧 Improvements

#### Added - Debug Logging in SSE Server

- SSE server now logs actual URLs being requested (`[AutoMem] GET http://...`)
- Detailed error messages for fetch failures with endpoint information
- Helps diagnose connection issues between services
- Logged errors include cause and network details

#### Updated - Railway Template

- All critical fixes incorporated into `railway-template.json`
- Template now follows official FalkorDB patterns for environment variables
- Removed incorrect health check configuration for FalkorDB service
- Added `PORT` environment variables for all services
- Uses `FALKOR_PASSWORD` consistently (matches official template)

### 🧹 Cleanup

#### Removed

- Backup files: `app.py.original`, `app.py.withchanges`
- Empty/unused files: `claude-desktop-mcp.js`
- Test/benchmark logs: `*.log`, `locomo_results.json`
- Empty directories: `backups/`
- Unused Railway configs: `railway-backup.json`, `railway-health-monitor.json`

### 📁 Files Modified

- `app.py` (Flask IPv6 binding)
- `mcp-sse-server/server.js` (debug logging)
- `railway-template.json` (PORT, volume paths, variable names)
- `docs/RAILWAY_DEPLOYMENT.md` (comprehensive troubleshooting)
- `docs/ENVIRONMENT_VARIABLES.md` (PORT requirements, IPv6 notes)
- `docs/MCP_SSE.md` (troubleshooting guide)

### 🎯 Migration Notes

**For existing Railway deployments:**

1. **Update memory-service variables**: Add `PORT=8001` (required)
2. **Redeploy memory-service**: Pull latest code for IPv6 binding fix
3. **Verify FalkorDB volume**: Should be mounted at `/var/lib/falkordb/data`
4. **Check SSE endpoint**: Should use `http://memory-service.railway.internal:8001`
5. **Test health**: `/health` should show `memory_count` field

**For new deployments:**

- Use the updated Railway template (all fixes included)
- Follow updated `docs/RAILWAY_DEPLOYMENT.md` for manual setup

## [0.7.0] - 2025-10-17

### 🌉 MCP over SSE Sidecar

#### Added - Hosted MCP Server via SSE

- Exposes AutoMem as an MCP server over SSE for cloud clients (ChatGPT, ElevenLabs Agents)
- Endpoints:
  - `GET /mcp/sse` — SSE stream (server → client); auth via `Authorization: Bearer`, `X-API-Key`, or `?api_key=`
  - `POST /mcp/messages?sessionId=<id>` — Client → server JSON-RPC messages
  - `GET /health` — Sidecar health probe
- Implementation: Node 18 + `@modelcontextprotocol/sdk` with JSON-RPC tool handlers mapping to AutoMem API
- Keepalive: Heartbeats every 20s to avoid idle proxy timeouts

#### Security & Auth

- Preferred auth: `Authorization: Bearer <AUTOMEM_API_TOKEN>`
- Also supports: `X-API-Key` and `?api_key=` (URL tokens may appear in logs; prefer headers)

#### Documentation

- New: `docs/MCP_SSE.md` with deployment, auth, and troubleshooting
- README updated to link SSE sidecar alongside the NPM MCP bridge

### 🚀 Deployment

#### Railway Template Enhancements

- Template now provisions 3 services: `automem-api`, `automem-mcp-sse`, and `FalkorDB`
- `automem-mcp-sse` preconfigured with `AUTOMEM_ENDPOINT` → `http://${{automem-api.RAILWAY_PRIVATE_DOMAIN}}`
- Health checks added for the sidecar service
- Fixed: Corrected Dockerfile path in `mcp-sse-server/railway.json`
- Added: `package-lock.json` for reproducible SSE builds

### 📊 Benchmarks

#### Added - LoCoMo Benchmark Integration

- End-to-end evaluation flow, docs (`docs/LOCOMO_BENCHMARK.md`), and test harness
- Make targets for local and live (Railway) runs
- Initial optimization phases (1–2.5) implemented and documented

### 📝 Documentation

- INSTALLATION: One‑click Railway section updated to reflect new SSE service
- README: Added SSE sidecar docs link under MCP and Documentation sections

### 📁 Files Modified

- `mcp-sse-server/`: `server.js`, `Dockerfile`, `package.json`, `package-lock.json`, `railway.json`
- `docs/`: `MCP_SSE.md`, `LOCOMO_BENCHMARK.md`
- `railway-template.json`
- `INSTALLATION.md`, `README.md`

### Commits Since 0.6.0

- Add MCP SSE sidecar and LoCoMo quick wins
- Add MCP SSE server deployment config for Railway
- Add multi-auth support to SSE server (Bearer, X-API-Key, query param)
- Add package-lock.json for mcp-sse-server
- Fix Dockerfile path in mcp-sse-server railway.json
- Implement LoCoMo benchmark optimizations (Phases 1–2.5)
- Add LoCoMo benchmark integration and documentation
- Update INSTALLATION.md

## [0.6.0] - 2025-10-14

### 🚀 Performance Optimizations

#### Added - Embedding Batching (40-50% Cost Reduction)

- **Feature**: Batch processing for OpenAI embedding generation
- **Implementation**:
  - New config: `EMBEDDING_BATCH_SIZE` (default: 20) and `EMBEDDING_BATCH_TIMEOUT_SECONDS` (default: 2.0)
  - Created `_generate_real_embeddings_batch()` for bulk embedding generation
  - Rewrote `embedding_worker()` to accumulate and batch-process memories
  - Added `_process_embedding_batch()` helper function
  - Extracted `_store_embedding_in_qdrant()` for reusability
- **Impact**:
  - 40-50% reduction in API calls and overhead
  - Better throughput during high-memory periods
  - $8-15/year savings at 1000 memories/day

#### Added - Relationship Count Caching (80% Consolidation Speedup)

- **Feature**: LRU cache for relationship counts with hourly invalidation
- **Implementation**:
  - Added `@lru_cache(maxsize=10000)` to `_get_relationship_count_cached_impl()`
  - Cache key includes hour timestamp for automatic hourly refresh
  - Updated `calculate_relevance_score()` to use cached method
- **Impact**:
  - 80% reduction in graph queries during consolidation
  - 5x faster decay consolidation runs
  - Minimal memory overhead with automatic LRU eviction

#### Added - Async Embedding Generation (60% Faster /memory POST)

- **Feature**: Background embedding generation with queue-based processing
- **Implementation**:
  - Embeddings generated asynchronously in worker thread
  - `/memory` POST returns immediately after FalkorDB write
  - Queue-based system prevents blocking API responses
- **Impact**:
  - `/memory` endpoint 60% faster (was 250-400ms, now 100-150ms)
  - Better user experience for high-frequency memory storage
  - No loss in reliability - queued embeddings processed in order

#### Added - Enrichment Stats in /health Endpoint

- **Feature**: Public enrichment metrics in health check (no auth required)
- **Implementation**: Enhanced `/health` endpoint with enrichment section
- **Response Format**:
  ```json
  {
    "enrichment": {
      "status": "running",
      "queue_depth": 12,
      "pending": 15,
      "inflight": 3,
      "processed": 1234,
      "failed": 5
    }
  }
  ```
- **Impact**: Better monitoring and early detection of enrichment backlogs

#### Added - Structured Logging

- **Feature**: Performance metrics in logs for analysis and debugging
- **Implementation**:
  - Added structured logging to `/recall` endpoint with query, latency, results, filters
  - Added structured logging to `/memory` (store) endpoint with type, size, latency, status
  - Logs include `extra={}` dict with machine-parseable fields
- **Impact**:
  - Easy performance analysis via log aggregation
  - Better debugging for production issues
  - Foundation for metrics dashboards

#### Added - Query Time Metrics

- **Feature**: All API responses now include `query_time_ms` field
- **Implementation**: Added `time.perf_counter()` tracking to all endpoints
- **Impact**: Easy performance monitoring without parsing logs

### 🏗️ Code Quality & Infrastructure

#### Refactored - Utility Module Structure

- **Change**: Extracted reusable helpers from `app.py` into `automem/` package
- **Created Modules**:
  - `automem/utils/graph.py` - Graph query helpers
  - `automem/utils/tags.py` - Tag normalization and processing
  - `automem/utils/text.py` - Text processing utilities
  - `automem/utils/time.py` - Timestamp parsing and formatting
  - `automem/utils/scoring.py` - Relevance scoring functions
- **Impact**: Better code organization, easier testing, reduced duplication

#### Improved - Docker Build

- **Added**: `.dockerignore` file for faster builds
- **Change**: Updated Dockerfile to use new utility module structure
- **Impact**: Smaller Docker images, faster builds

#### Improved - Backup Infrastructure

- **Added**: GitHub Actions workflow for automated backups to S3
- **Added**: Comprehensive backup documentation in `docs/MONITORING_AND_BACKUPS.md`
- **Added**: Backup service deployment guides for Railway
- **Impact**: Better disaster recovery, automated backup schedules

### 🐛 Bug Fixes

#### Fixed - Project Name Extraction

- **Issue**: Feature names like "file-ops-automation" were incorrectly extracted as projects
- **Fix**: Added filtering to prevent feature/module names from entity extraction
- **Impact**: Cleaner entity metadata, more accurate project tracking

#### Fixed - /health Endpoint Attribute Error

- **Issue**: `/health` endpoint crashed with `AttributeError: 'EnrichmentStats' object has no attribute 'success_count'`
- **Fix**: Corrected attribute names to `successes` and `failures` (actual class fields)
- **Impact**: `/health` endpoint now works correctly with enrichment stats

### 📁 Files Modified

- `app.py` - Embedding batching, async generation, health endpoint, structured logging, query timing
- `consolidation.py` - Relationship count caching
- `automem/utils/` - New utility module structure
- `Dockerfile` - Updated for new module structure
- `.dockerignore` - Created for faster builds
- `.github/workflows/backup.yml` - Created automated backup workflow
- `docs/MONITORING_AND_BACKUPS.md` - Updated with comprehensive backup strategies
- `OPTIMIZATIONS.md` - Created comprehensive optimization documentation
- `test-optimizations.sh` - Created test script for validating optimizations

### 📊 Performance Comparison

| Metric                   | Before    | After        | Improvement |
| ------------------------ | --------- | ------------ | ----------- |
| OpenAI API calls         | 1000/day  | ~50-100/day  | 40-50% ↓    |
| Annual embedding cost    | $20-30    | $12-18       | $8-15 saved |
| `/memory` latency        | 250-400ms | 100-150ms    | 60% faster  |
| Consolidation time (10k) | ~5 min    | ~1 min       | 80% faster  |
| Production visibility    | Limited   | Full metrics | ∞ better    |

### 🎓 Based On

- Audit by Steve (October 11, 2025)
- Implemented highest-ROI recommendations from audit
- Total implementation time: ~3 hours for optimizations
- Expected ROI: 200-300% in year 1

---

## [0.5.0] - 2025-10-13

### 🎯 Major Data Quality Overhaul

#### Fixed - Memory Type Pollution (Critical)

- **Issue**: 490/773 memories (64%) had invalid types from recovery script bug
- **Root Cause**: Recovery script flattened `metadata.type` as top-level property, overwriting actual memory type
- **Fix**:
  - Added `RESERVED_FIELDS` filter in `scripts/recover_from_qdrant.py` to exclude type, confidence, content, etc.
  - Created `scripts/cleanup_memory_types.py` to reclassify all 490 polluted memories
  - 100% success rate - all memories reclassified to 7 valid types
- **Impact**: Analytics now accurate with only valid memory types (Decision, Pattern, Preference, Style, Habit, Insight, Context)

#### Fixed - Tag-Based Search Not Working

- **Issue**: Tag searches returned 0 results despite memories having tags
- **Root Cause**: Qdrant missing keyword indexes on `tags` and `tag_prefixes` payload fields
- **Fix**: Added `PayloadSchemaType.KEYWORD` indexes in `_ensure_qdrant_collection()`
- **Impact**: Tag filtering now works for both exact and prefix matching

#### Improved - Entity Extraction Quality

**Phase 1 - Error Code Filtering:**

- Added `ENTITY_BLOCKLIST` for HTTP errors (Bad Request, Not Found, etc.)
- Added `ENTITY_BLOCKLIST` for network errors (ECONNRESET, ETIMEDOUT, etc.)

**Phase 2 - Code Artifact Filtering:**

- Block markdown formatting (`- **File:**`, `# Header`, etc.)
- Block class name suffixes (Adapter, Handler, Manager, Service, etc.)
- Block JSON field names (starting with underscore like `_embedded`)
- Block boolean literals (`'false'`, `true`, `null`)
- Block environment variables (all-caps with underscores)
- Block text fragments (strings ending with colons)

**Phase 3 - Project Name Extraction:**

- **Issue**: Project extraction regex required `Project ProjectName` but memories use `project: project-name`
- **Fix**: Added pattern `r"(?:in |on )?project:\s+([a-z][a-z0-9\-]+)"`
- **Impact**: Now correctly extracts claude-automation-hub, mcp-automem, autochat from session starts

#### Added - LLM-Based Memory Classification

- **Feature**: Hybrid classification system (regex first, LLM fallback)
- **Implementation**:
  - Enhanced `MemoryClassifier` with `_classify_with_llm()` method
  - Uses GPT-4o-mini for accurate, cost-effective classification
  - Falls back to LLM only when regex patterns don't match (~30% of cases)
- **Reclassification Completed**: 413 memories reclassified with 100% success rate
- **Cost**: $0.04 total for one-time reclassification, ~$0.24/year ongoing (10 memories/day)
- **Accuracy**: 85-95% confidence on technical content, work logs, session starts
- **Script**: Created `scripts/reclassify_with_llm.py` for batch reclassification

#### Added - Automated Backups

- **GitHub Actions**: Workflow runs every 6 hours, backs up to S3
- **Railway Volumes**: Built-in volume backups for redundancy
- **Dual Storage**: FalkorDB (primary) + Qdrant (backup) provides data resilience
- **Documentation**: Consolidated all backup docs into `docs/MONITORING_AND_BACKUPS.md`

#### Fixed - FalkorDB Data Loss Recovery

- **Issue**: FalkorDB restarted and lost all memories (persistence not working)
- **Recovery**: Used `scripts/recover_from_qdrant.py` to restore 778/780 memories from Qdrant
- **Root Cause Analysis**: `appendonly` was set to `no` instead of `yes`
- **Prevention**: Verified `REDIS_ARGS` in Railway and documented proper persistence configuration

### 📁 Files Modified

- `app.py` - Enhanced entity extraction, LLM classification, Qdrant indexes
- `scripts/recover_from_qdrant.py` - Fixed metadata field handling
- `scripts/cleanup_memory_types.py` - Created tool for type reclassification
- `scripts/reclassify_with_llm.py` - Created LLM-based reclassification tool
- `docs/MONITORING_AND_BACKUPS.md` - Updated with comprehensive backup strategies
- `docs/RAILWAY_DEPLOYMENT.md` - Updated references to backup documentation
- `README.md` - Updated architecture and documentation links
- `.github/workflows/backup.yml` - Created automated backup workflow

### 📊 Impact Summary

- **Memory Types**: 79 invalid types → 7 valid types (100% cleanup)
- **Reclassified**: 903 total memories
  - 490 via cleanup script (invalid types → valid types)
  - 413 via LLM (fallback "Memory" → specific types)
  - 100% success rate on both
- **Final Distribution**: Context (47%), Decision (16%), Insight (13%), Habit (12%), Preference (9%), Pattern (3%), Style (<1%)
- **Entity Quality**: Clean extraction, no error codes or code artifacts
- **Tag Search**: Fully functional with proper indexing
- **Backups**: Automated every 6 hours to S3
- **Data Recovery**: 99.7% recovery rate (778/780) from Qdrant
- **Classification**: Hybrid system with 85-95% LLM confidence

### 🚀 Deployment

All changes deployed to Railway at `automem.up.railway.app`

### 💰 Cost Analysis

- **LLM Classification**: $0.04 one-time + $0.24/year ongoing (~$0.02/month)
- **S3 Backups**: Minimal (~$0.05/month for storage + lifecycle rules)
- **Qdrant**: Existing free tier (1GB)
- **Railway**: Existing deployment ($5/month hobby plan)

---

## [Prior Releases]

See git history for previous changes.
