# Changelog

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
