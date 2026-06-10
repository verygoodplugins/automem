# AutoMem API Reference

This document lists the primary API endpoints and examples. JSON responses include `status` and primary payload fields for LLM-friendliness unless an endpoint explicitly returns a binary payload.

Authentication

- All endpoints except `/health` require authentication via one of:
  - `Authorization: Bearer <AUTOMEM_API_TOKEN>` header (recommended)
  - `X-API-Key: <AUTOMEM_API_TOKEN>` header
  - `?api_key=<AUTOMEM_API_TOKEN>` query parameter
- Admin endpoints additionally require `X-Admin-Token: <ADMIN_API_TOKEN>` header.
- `GET /backup` is admin-only and does not accept the regular API token by itself.

Health

- GET `/health`
  - Returns overall status and service connectivity.

Memory

- POST `/memory`
  - Body: `{ "content": "...", "tags": ["tag"], "importance": 0.7, "metadata": {} }`
  - Response: `{ "status": "success", "memory_id": "...", ... }`

- GET `/memory/{id}`
  - Response: `{ "status": "success", "memory": { ... } }`
  - Errors: `404` if memory is missing, `500` on query failure, `503` if graph database is unavailable.

- PATCH `/memory/{id}`
  - Body: any subset of fields (`content`, `tags`, `importance`, `type`, `confidence`, `timestamp`, `metadata`)
  - Response: `{ "status": "success", "memory_id": "..." }`

- DELETE `/memory/{id}`
  - Response: `{ "status": "success", "memory_id": "..." }`

- GET `/memory/by-tag?tags=foo&tags=bar&limit=20&offset=0`
  - Response: `{ "status": "success", "tags": ["foo","bar"], "count": N, "limit": 20, "offset": 0, "has_more": false, "memories": [...] }`

- DELETE `/memory/by-tag?tags=foo&tags=bar`
  - Response: `{ "status": "success", "tags": ["foo","bar"], "deleted_count": N }`

- POST `/associate`
  - Body: `{ "memory1_id": "...", "memory2_id": "...", "type": "RELATES_TO", "strength": 0.9 }`
  - Response: `{ "status": "success", ... }`
  - `type` accepts only the 11 authorable semantic relationship types. System-generated labels such as `SIMILAR_TO`, `PRECEDED_BY`, and `DISCOVERED` are readable/filterable but cannot be created via this endpoint.

Recall

- GET `/recall`
  - **Basic parameters**: `query`, `limit`, `tags`, `exclude_tags`, `tag_mode` (any|all), `tag_match` (prefix|exact, default prefix), `time_query` (e.g. "last week"), `start`, `end`, `embedding`
    - `exclude_tags` - Comma-separated list or multiple params to exclude memories containing ANY of these tags. Supports both exact and prefix matching (independent of `tag_match`). Example: `exclude_tags=conversation_5` or `exclude_tags=temp,draft`
  - **Tag semantics** (per retrieval path):
    - `tags` is a **hard scope filter** on the vector, keyword, metadata-sidecar, and tag-only fallback paths: memories without a matching tag are excluded *before* scoring.
    - Graph/entity **expansion bypasses the tag scope by default**; pass `expand_respect_tags=true` to keep expanded results inside the scope.
    - `context_tags` is the **soft-boost** channel: it raises matching results' scores without excluding anything.
    - Within the tag-scoped pool, query-independent score components (importance, confidence, recency, tag overlap) can be ramped down for results with little topical evidence via the `RECALL_RELEVANCE_GATE` env var (default `0.0` = off; see ENVIRONMENT_VARIABLES.md).
    - `scope_fallback` (boolean, default `false`) - When tag-scoped results come up short of `limit` and a text/semantic query is present, fill the remaining slots from an *unscoped* vector search. Fills are appended after all scoped results (never interleaved with or displacing them) and each carries `"outside_tag_scope": true`. Fills get filter parity with the scoped path: `exclude_tags`, time filters, `min_score`, and current-state filtering (payload-level reasons plus graph-edge supersession) all still apply — only the tags scope is lifted. Candidates whose tags match the request's `tags` are never returned as fills: they are in-scope by definition (already returned, or dropped by a score filter) and are not resurrected.
  - **Ordering**: `sort` (or `order_by`) supports:
    - `score` (default) - hybrid relevance/importance ranking; exact score ties order newest-first
    - `time_desc` / `time_asc` - chronological ordering by `updated_at`/`timestamp` within the filter window (use for "what happened since X")
    - `updated_desc` / `updated_asc` - explicit alias (same ordering key as time\_\*)
    - `recency_bias` (`off`|`on`|`auto`, default from `RECALL_RECENCY_BIAS` env, ships `off`) - relative-recency re-rank for score ordering: candidate timestamps are min-max normalized across the result set and `SEARCH_WEIGHT_TEMPORAL × relative_recency` is added to each final score, so the newest version of a conflicting fact can outrank an older, heavier one. `auto` activates only when the query expresses temporal intent ("latest", "current", "what changed", ...). The response echoes `"recency_bias": "on"` and results carry a `temporal` score component when the re-rank ran.
  - **State filtering**:
    - `state_mode=current|history` controls whether recall returns only currently valid state or full state history. Default: `current`.
    - `current` is equivalent to `current_only=true`: suppresses memories that are expired, not yet valid, archived, or invalidated/evolved by active replacements. Supersession chains (`INVALIDATED_BY`/`EVOLVED_INTO`) are resolved to their *head* — A→B→C surfaces C — bounded at 5 hops and cycle-safe; the surfaced head's `state_replaces`/`relations[].from` provenance still points at the originally suppressed memory.
    - `history` is equivalent to `current_only=false`: returns stale and future state when it matches the query/filter.
    - `current_only` remains supported for backward compatibility and wins over `state_mode` when both resolve to a value. A malformed `state_mode` is still rejected with `400` even when `current_only` is supplied — the value is validated before precedence is applied.
    - `state_debug=true` includes suppression/replacement details in `state_filter`.
  - **Context hints**: `context`, `language`, `active_path`, `context_tags`, `context_types`, `priority_ids`
  - **Metadata sidecar search**: Text queries can admit bounded metadata candidates when the query strongly matches whitelisted metadata values. This is enabled by default and controlled by `RECALL_METADATA_SEARCH_ENABLED`; no extra request parameter is required.
  - **Graph expansion**:
    - `expand_relations` - Follow graph edges from seed results to related memories
    - `expand_entities` - Multi-hop reasoning via entity tags (finds "Amanda → Rachel" then "Rachel's job")
    - `relation_limit` - Max relations per seed (default: 5)
    - `expansion_limit` - Total max expanded memories (default: 25)
  - **Expansion filtering** (reduce noise in expanded results):
    - `expand_min_importance` - Minimum importance (0-1) for expanded memories. Seed results are never filtered, only expanded ones. Recommended: 0.3-0.5 for broad context, 0.6+ for focused results.
    - `expand_min_strength` - Minimum relation strength (0-1) to traverse during graph expansion. Only edges above this threshold are followed. Recommended: 0.3 for exploratory, 0.6+ for high-confidence connections.
  - Response: `{ "status": "success", "results": [...], "count": M, "state_mode": "current", "context_priority": {...} }`
  - Echoed filters (for debugging): `tags`, `exclude_tags`, `tag_mode`, `tag_match`
  - When `tags` were passed, the response includes scope diagnostics: `tag_scope: { "filtered": true, "pool_size_hint": <int|null>, "gated_low_evidence": <int> }` — `pool_size_hint` is the post-tag-filter, pre-limit vector candidate count (null when no semantic query ran, e.g. tag-only recall), and `gated_low_evidence` counts returned results whose score components were ramped down by `RECALL_RELEVANCE_GATE`. Note the hint is capped by the vector fetch limit and sums per-query counts when the request decomposes into multiple queries (`queries[]`/`auto_decompose`), so the same memory can be counted more than once — treat it as a rough pool-size signal, not an exact count. `scope_fallback: true` is echoed when the fallback ran.
  - Echoed state: `state_mode` always reflects the resolved mode after `current_only` precedence. When current-state filtering runs, `state_filter` may include suppressed and replacement IDs, including `INVALIDATED_BY`, `EVOLVED_INTO`, and `CONTRADICTS` handling details when `state_debug=true`.
  - When `expand_entities=true`: includes `entity_expansion: { enabled, expanded_count, entities_found }`
  - When `expand_relations=true`: includes `expansion: { enabled, seed_count, expanded_count, relation_limit }`

**Expansion Filtering Example:**

```bash
# Get architecture decisions with related context, but filter out low-importance noise
GET /recall?query=database%20architecture&expand_relations=true&expand_min_importance=0.5&expand_min_strength=0.3
```

**Exclude Tags Example:**

```bash
# Get long-term memories from all conversations EXCEPT the current one
GET /recall?tags=user_1&exclude_tags=conversation_5&limit=3

# Exclude multiple conversations
GET /recall?tags=user_1&exclude_tags=conversation_5,conversation_6&limit=5

# Exclude temporary/draft memories
GET /recall?query=project%20plan&exclude_tags=temp,draft&limit=10
```

**Current-State Recall Examples:**

```bash
# Default: return active/current state only
GET /recall?query=current%20project%20plan

# Historical audit: include expired, future, invalidated, and evolved facts
GET /recall?query=project%20plan&state_mode=history
```

- GET `/memories/{id}/related`
  - Query: `relationship_types`, `max_depth` (1..3), `limit` (<=200)
  - Default traversal uses the 11 authorable semantic relationship types.
  - Explicit opt-ins may include `SIMILAR_TO`, `PRECEDED_BY`, and `DISCOVERED`.
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

- GET `/backup`
  - Headers: requires `X-Admin-Token: <ADMIN_API_TOKEN>` or `X-Admin-Api-Key: <ADMIN_API_TOKEN>`.
  - Query: optional `include=falkordb,qdrant`; defaults to both. Use `include=falkordb` or `include=qdrant` for a partial export.
  - Response: binary `application/gzip` attachment named `automem-backup-<timestamp>.tar.gz`.
  - Archive contents are restore-compatible: `falkordb/falkordb_<timestamp>.json.gz` and/or `qdrant/qdrant_<timestamp>.json.gz`.
  - Example: `curl -H "X-Admin-Token: $ADMIN_API_TOKEN" "$AUTOMEM_API_URL/backup" -o snapshot.tar.gz`

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
- Exclusion filtering (`exclude_tags`) removes any memory containing ANY of the excluded tags, supporting both exact and prefix matching.
- Time filtering accepts ISO timestamps (`start`, `end`) or a natural expression via `time_query`.
- Context hints boost matching preferences (e.g., Python coding style) and guarantee at least one anchor memory when applicable; responses echo what was applied via `context_priority`.
