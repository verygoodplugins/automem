# Changelog

All notable changes to AutoMem will be documented in this file.

## [0.10.0](https://github.com/verygoodplugins/automem/compare/v0.9.3...v0.10.0) (2025-12-14)


### Features

* **api:** add graph visualization API endpoints ([#26](https://github.com/verygoodplugins/automem/issues/26)) ([a0d0dd3](https://github.com/verygoodplugins/automem/commit/a0d0dd3719ac9dad7c4e24e78f290c9364d2b2e1))
* **mcp-sse:** agent-friendly recall output + tests ([71201d3](https://github.com/verygoodplugins/automem/commit/71201d33b5b8a9044f23f3a911dbf193de67b669))
* **mcp-sse:** improve recall tool output and options ([97c4f72](https://github.com/verygoodplugins/automem/commit/97c4f7240b5319a95412e7d1e65ae65e67150a46))

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

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| OpenAI API calls | 1000/day | ~50-100/day | 40-50% ↓ |
| Annual embedding cost | $20-30 | $12-18 | $8-15 saved |
| `/memory` latency | 250-400ms | 100-150ms | 60% faster |
| Consolidation time (10k) | ~5 min | ~1 min | 80% faster |
| Production visibility | Limited | Full metrics | ∞ better |

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
