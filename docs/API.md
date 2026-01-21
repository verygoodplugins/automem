# AutoMem API Reference

This document lists the primary API endpoints and examples. All JSON responses include `status` and primary payload fields for LLM-friendliness.

Authentication
- All endpoints except `/health` require authentication via one of:
  - `Authorization: Bearer <AUTOMEM_API_TOKEN>` header (recommended)
  - `X-API-Key: <AUTOMEM_API_TOKEN>` header
  - `?api_key=<AUTOMEM_API_TOKEN>` query parameter
- Admin endpoints additionally require `X-Admin-Token: <ADMIN_API_TOKEN>` header.

Health
- GET `/health`
  - Returns overall status and service connectivity.

Memory
- POST `/memory`
  - Body: `{ "content": "...", "tags": ["tag"], "importance": 0.7, "metadata": {} }`
  - Response: `{ "status": "success", "memory_id": "...", ... }`

- PATCH `/memory/{id}`
  - Body: any subset of fields (`content`, `tags`, `importance`, `type`, `confidence`, `timestamp`, `metadata`)
  - Response: `{ "status": "success", "memory_id": "..." }`

- DELETE `/memory/{id}`
  - Response: `{ "status": "success", "memory_id": "..." }`

- GET `/memory/by-tag?tags=foo&tags=bar&limit=20`
  - Response: `{ "status": "success", "tags": ["foo","bar"], "count": N, "memories": [...] }`

- POST `/associate`
  - Body: `{ "memory1_id": "...", "memory2_id": "...", "type": "RELATES_TO", "strength": 0.9 }`
  - Response: `{ "status": "success", ... }`

Recall
- GET `/recall`
  - **Basic parameters**: `query`, `limit`, `tags`, `tag_mode` (any|all), `tag_match` (prefix|exact), `time_query` (e.g. "last week"), `start`, `end`, `embedding`
  - **Ordering**: `sort` (or `order_by`) supports:
    - `score` (default) - hybrid relevance/importance ranking
    - `time_desc` / `time_asc` - chronological ordering by `updated_at`/`timestamp` within the filter window (use for "what happened since X")
    - `updated_desc` / `updated_asc` - explicit alias (same ordering key as time_*)
  - **Context hints**: `context`, `language`, `active_path`, `context_tags`, `context_types`, `priority_ids`
  - **Graph expansion**:
    - `expand_relations` - Follow graph edges from seed results to related memories
    - `expand_entities` - Multi-hop reasoning via entity tags (finds "Amanda → Rachel" then "Rachel's job")
    - `relation_limit` - Max relations per seed (default: 5)
    - `expansion_limit` - Total max expanded memories (default: 25)
  - **Expansion filtering** (reduce noise in expanded results):
    - `expand_min_importance` - Minimum importance (0-1) for expanded memories. Seed results are never filtered, only expanded ones. Recommended: 0.3-0.5 for broad context, 0.6+ for focused results.
    - `expand_min_strength` - Minimum relation strength (0-1) to traverse during graph expansion. Only edges above this threshold are followed. Recommended: 0.3 for exploratory, 0.6+ for high-confidence connections.
  - Response: `{ "status": "success", "results": [...], "count": M, "context_priority": {...} }`
  - When `expand_entities=true`: includes `entity_expansion: { enabled, expanded_count, entities_found }`
  - When `expand_relations=true`: includes `expansion: { enabled, seed_count, expanded_count, relation_limit }`

**Expansion Filtering Example:**
```bash
# Get architecture decisions with related context, but filter out low-importance noise
GET /recall?query=database%20architecture&expand_relations=true&expand_min_importance=0.5&expand_min_strength=0.3
```

- GET `/memories/{id}/related`
  - Query: `relationship_types`, `max_depth` (1..3), `limit` (<=200)
  - Response: `{ "status": "success", "related_memories": [...] }`

- GET `/startup-recall`
  - Returns critical lessons and system rules.

- GET `/analyze`
  - Returns analytics (type counts, preferences, temporal insights, entities, confidence distribution).

Enrichment
- GET `/enrichment/status`
  - Response: queue size, inflight/pending, stats.

- POST `/enrichment/reprocess`
  - Body: `{ "ids": ["..."] }` or query `?ids=a,b,c`
  - Response: `{ "status": "queued", "count": N }`

Admin
- POST `/admin/reembed`
  - Headers: requires both API and Admin tokens.
  - Body: `{ "batch_size": 32, "limit": 100, "force": false }`
  - Response: `{ "status": "complete", "processed": N, "failed": K }`

Consolidation
- POST `/consolidate`
  - Body: `{ "mode": "full"|"decay"|"creative"|"cluster"|"forget", "dry_run": true }`
  - Response: `{ "status": "success", "consolidation": {...} }`

- GET `/consolidate/status`
  - Response: `{ "status": "success", "next_runs": {...}, "history": [...] }`

Notes
- Tag matching supports exact and prefix semantics; vector searches are filtered by tag conditions when provided.
- Time filtering accepts ISO timestamps (`start`, `end`) or a natural expression via `time_query`.
- Context hints boost matching preferences (e.g., Python coding style) and guarantee at least one anchor memory when applicable; responses echo what was applied via `context_priority`.
