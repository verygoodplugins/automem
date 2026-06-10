# Metadata Behavior

This document describes the current end-to-end behavior of the `metadata` object
on memories. It is intentionally a product/runtime spec, not an experiment note.

## Storage

- `POST /memory` accepts `metadata` only when it is a JSON object. Missing
  metadata is stored as `{}`; non-object metadata is rejected with `400`.
- FalkorDB stores metadata as a JSON string on `m.metadata`.
- Qdrant stores metadata as a parsed object in the point payload.
- If server-side content summarization runs, the original content audit fields
  are added to metadata: `original_content`, `was_summarized`, and
  `original_length`.
- Metadata is not included in the primary content embedding. The stored vector
  remains an embedding of the memory content.

## Recall Response Shape

- Recall results include parsed `memory.metadata` when the graph or Qdrant
  payload provides it, along with `updated_at` and `last_accessed` timestamps;
  malformed graph metadata is treated as an empty or raw parsed value depending
  on the caller path.
- The MCP server's `json` recall format passes the raw response through, so it
  exposes the full metadata object. The MCP `detailed` format renders a
  size-capped `Metadata:` line (single-line JSON truncated to 300 characters
  with a trailing ellipsis) plus an `Updated:` line when present, and omits the
  metadata line entirely for empty or missing metadata. The `text` and `items`
  formats do not include metadata.
- Final scoring can use metadata terms as weak evidence for candidates that are
  already present from another channel.

## Search

- Metadata search is an additive `/recall` candidate channel. It does not change
  the HTTP API and does not require a user-visible metadata mode.
- The sidecar channel is enabled by default with
  `RECALL_METADATA_SEARCH_ENABLED=true`. Set it to `false`, `0`, or `no` to
  restore baseline behavior without metadata-sidecar candidates.
- Metadata search runs only for text queries and is bounded to a small candidate
  budget. Results are deduplicated against vector and keyword candidates before
  final ranking.
- Candidate admission requires value evidence against whitelisted metadata
  fields. Field words such as `repo` or `source agent` are scoring context, not a
  hard trigger.
- Searchable metadata fields are `source`, `source_agent`, `source_agents`,
  `repo`, `project`, `tool`, `surface`, `applies_to`, `trigger`, `provider`,
  `model`, and structured `entities`.
- Structured `entities` are searched only when the query has entity-field
  context. Enrichment-generated entities are often derived from content, so they
  are not used as a general hidden-metadata sidecar signal.
- Entity people are excluded from metadata sidecar search by default to avoid
  noisy personal-name matching even when entity-field context is present.
- The sidecar skips `original_content`, `enrichment`, `semantic_neighbors`,
  `patterns_detected`, dict-valued non-entity fields, long strings, large arrays,
  and unstructured entity lists.
- Tag filters, exclude-tag filters, time windows, archive state, and current
  state suppression still apply to metadata candidates.
- Metadata candidates expose `match_type: "metadata"` and populate
  `score_components.metadata`.

## Update

- `PATCH /memory/{id}` preserves existing metadata when the request omits the
  `metadata` field.
- When a patch includes `metadata`, the provided object replaces the previous
  metadata object. There is no deep merge for arbitrary user metadata today.
- If content changes, the vector is regenerated from the new content only. If
  content is unchanged, the existing vector is preserved while Qdrant payload
  metadata is refreshed.

## Enrichment

- Async and just-in-time enrichment parse existing metadata and merge generated
  entity data into `metadata.entities`.
- Enrichment also writes operational details under `metadata.enrichment`.
- Enrichment can add entity tags and tag prefixes, but metadata sidecar search
  does not depend on those generated tags.

## Consolidation

- Decay updates `relevance_score`; forget may archive or delete memories and
  sync those state changes to Qdrant.
- Creative consolidation adds graph relationships. Identity consolidation
  deduplicates Entity nodes and can synthesize entity identity summaries.
- Cluster consolidation may create `MetaMemory` nodes and `SUMMARIZES`
  relationships.
- Current consolidation does not merge two ordinary memory records into one
  replacement record, so there is no general-purpose memory-metadata merge policy
  yet. If future consolidation introduces memory merging, provenance fields such
  as `source_agents` should be merged intentionally rather than overwritten.
