# AutoMem API Reference

This document lists the primary API endpoints and examples. All JSON responses include `status` and primary payload fields for LLM-friendliness.

Authentication
- Pass `Authorization: Bearer <AUTOMEM_API_TOKEN>` for all endpoints.
- Admin endpoints also require `X-Admin-Token: <ADMIN_API_TOKEN>`.

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
  - Query: `query`, `limit`, `tags`, `tag_mode` (any|all), `tag_match` (prefix|exact), `time_query` (e.g. "last week"), `start`, `end`, `embedding`
  - Optional context hints: `context`, `language`, `active_path`, `context_tags`, `context_types`, `priority_ids`
  - Graph expansion: `expand_relations` (follow graph edges), `expand_entities` (multi-hop via entity tags)
  - Expansion limits: `relation_limit` (per-seed, default 5), `expansion_limit` (total, default 25)
  - Response: `{ "status": "success", "results": [...], "count": M, "context_priority": {...} }`
  - When `expand_entities=true`: includes `entity_expansion: { enabled, expanded_count, entities_found }`

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

