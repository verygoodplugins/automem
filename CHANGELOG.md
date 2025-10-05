# Changelog

## [0.4.0] - 2025-10-05

### Added
- **Railway Deployment Support**
  - One-click Railway deployment with persistent volumes and automatic backups
  - Comprehensive Railway deployment guide (`docs/RAILWAY_DEPLOYMENT.md`) and checklist (`docs/DEPLOYMENT_CHECKLIST.md`)
  - Custom FalkorDB Dockerfile with backup scripts for Railway
  - Railway template configuration (`railway.json`, `railway-template.json`)
  
- **Data Recovery & Management Tools**
  - `scripts/recover_from_qdrant.py` - Rebuild FalkorDB from Qdrant after data loss (direct graph writes, preserves types)
  - `scripts/deduplicate_qdrant.py` - Remove duplicate memories with dry-run mode and batch processing
  - `scripts/reenrich_batch.py` - Re-classify memories with updated classification logic
  
- **Health Monitoring System**
  - `scripts/health_monitor.py` - Monitor service health and data consistency
  - Opt-in auto-recovery with configurable thresholds
  - Alert integrations (webhooks, email) for drift detection
  - Comprehensive monitoring guide (`docs/HEALTH_MONITORING.md`)

- **Memory Classification Enhancements**
  - Explicit `type` and `confidence` parameters in `POST /memory` endpoint
  - Type validation against allowed memory types (`Decision`, `Pattern`, `Preference`, `Style`, `Habit`, `Insight`, `Context`)
  - Hybrid approach: explicit types preferred, auto-classification as fallback

### Changed
- **Docker Infrastructure**
  - Added persistent volumes for FalkorDB and Qdrant with local backup paths
  - Implemented aggressive persistence configuration (`REDIS_ARGS`) for FalkorDB
  - Added FalkorDB password support with automatic authentication
  - Set `restart: unless-stopped` for all services
  - Added optional FalkorDB Browser service on port 3001

- **Memory Classification**
  - Removed "Memory" as explicit type (now internal fallback only)
  - Changed default classification from "Memory" to "Context"
  - Recovery scripts now preserve original `type` and `confidence` from Qdrant

- **Documentation**
  - Complete API parameter documentation in `INSTALLATION.md` (all 13 parameters for `/memory`)
  - Clarified admin endpoint authentication (requires both `AUTOMEM_API_TOKEN` and `ADMIN_API_TOKEN`)
  - Added decision framework for Railway vs local deployment
  - Updated `README.md` with explicit type usage examples and research paper links
  - Created comprehensive environment variables guide (`docs/ENVIRONMENT_VARIABLES.md`)
  - Split installation from marketing content for better UX

- **Environment Variables**
  - Cleaned up and organized variables by category
  - Removed deprecated variables (`MCP_MEMORY_AUTO_DISCOVER`, `DEVELOPMENT`)
  - Renamed `MCP_MEMORY_HTTP_ENDPOINT` to `AUTOMEM_API_URL` (backward compatible)
  - Added health monitor configuration variables
  - Created `.env.example` for quick setup

### Fixed
- FalkorDB TCP proxy connection support for Railway deployment
- Recovery script authentication (now uses both API and admin tokens)
- Memory type preservation during recovery operations
- Entity extraction picking up markdown formatting artifacts

### Notes
- **Data Migration**: Use `scripts/reenrich_batch.py` to update memory classifications
- **Railway Cost**: ~$0.50/month typical usage with $5 free trial credits

## [0.3.0] - 2025-09-30

- Added
  - Admin endpoint to re-embed existing memories with OpenAI: `POST /admin/reembed` (batching, limits, and force options).
  - Enrichment pipeline with entity extraction, pattern linking, and a status endpoint `GET /enrichment/status`.
  - Tests covering re-embed and enrichment functionality.

- Changed
  - Improved entity extraction quality and relation fetching for recall responses.
  - Refined consolidation decay logic for smoother, incremental relevance updates.
  - Restructured docs and scripts; updated migration references and paths.
  - Updated README with clearer setup and feature documentation.

- Notes
  - API token auth and foundational recall/scoring improvements shipped earlier remain compatible with these changes.

## [0.2.0] - 2025-09-17

- Added
  - Update (`PATCH /memory/<id>`), delete (`DELETE /memory/<id>`), and tag filter endpoints (`GET /memory/by-tag`).
  - API token authentication via `Authorization: Bearer`, `X-API-Key`, or `?api_key`.

- Changed
  - Enhanced recall scoring and result ordering.

## [0.1.0] - 2025-09-16

- Added
  - Initial API for storing and recalling memories backed by FalkorDB + Qdrant.
  - Consolidation engine (initial version), OpenAI embeddings integration, and HTTP bridge draft.
  - `reembed_embeddings.py` helper.
