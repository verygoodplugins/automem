# MCP Transport Parity

AutoMem ships its MCP surface over two transports:

| Transport  | Implementation                              | Package / path                                |
| ---------- | ------------------------------------------- | --------------------------------------------- |
| **stdio**  | `@verygoodplugins/mcp-automem` (TypeScript) | [verygoodplugins/mcp-automem](https://github.com/verygoodplugins/mcp-automem) |
| **remote** | Streamable HTTP + SSE bridge (ESM JS)       | `mcp-sse-server/server.js` (this repo)        |

`mcp-automem`'s `server.json` publishes stdio, streamable-HTTP, and SSE as **one
server with one shared 6-tool array**. That is the contract: a client that picks
"AutoMem" out of the registry must get the same tools, the same request mapping,
and the same rendered output regardless of which transport it connected over.

This document records the audit that established where the two had drifted, and
the short list of differences that remain intentional.

## Why this exists

[PR #224](https://github.com/verygoodplugins/automem/pull/224) fixed one symptom:
the remote bridge's compact recall block dropped the memory's stored date. An
agent replaying that text read a two-week-old itinerary as today's plan, because
nothing in the output said how old the memory was. AutoHub's parser already
looked for a `Created:` line — the remote transport simply never emitted one,
while the stdio package always had.

That single missing line turned out to be one instance of a much wider split. A
manual schema sync had already been attempted once
(`d99b86d fix(mcp-sse): sync tool schemas for SSE/MCP parity (#104)`) and had
since drifted again, which is the case against maintaining two hand-synced
copies.

## Audit: where the transports had diverged

Snapshot taken against `mcp-automem` 0.15.0 and `mcp-sse-server` at
`4b5eaaf`. Both transports registered the same six tool names in the same order —
`store_memory`, `recall_memory`, `associate_memories`, `update_memory`,
`delete_memory`, `check_database_health`. Everything below that differed.

### Tool definitions

| Axis                              | stdio                                                                | remote                                                      |
| --------------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------- |
| `description`                     | Multi-paragraph: modes, "When to use", "Examples"                     | One line each                                                |
| `title`                           | Present (and duplicated in `annotations.title`)                       | Absent                                                       |
| `outputSchema`                    | On all 6                                                              | On none                                                      |
| `annotations`                     | `readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`  | `readOnlyHint`, `destructiveHint` only                       |
| `_meta['anthropic/alwaysLoad']`   | On `store_memory`, `recall_memory`, `associate_memories`              | Absent                                                       |
| server `instructions`             | Set (~450 chars)                                                      | Not set                                                      |
| `serverInfo`                      | `mcp-automem` / package version                                       | `automem-mcp-sse` / `0.1.0`, disagreeing with its own `0.2.0` |

### Capability missing on the remote transport

- **`recall_memory`** — 38 params vs 27. Absent remotely: `memory_id` (ID-fetch
  mode), `exhaustive` + `offset` (tag-enumeration mode), `exclude_tags`,
  `current_only`, `state_mode`, `state_debug`, `recency_bias`, `min_score`,
  `adaptive_floor`, `expand_respect_tags`. `limit` capped at 50 remotely vs 200
  on stdio. stdio declared defaults (`expansion_limit: 25`, `relation_limit: 5`,
  `current_only: true`); remote declared none.
- **`store_memory`** — no `memories[]` batch mode, and no supersede mode
  (`supersedes_memory_id` / `supersede_relation` / `supersede_reason`, a
  four-call sequence with a compensating delete on partial failure).
- **`delete_memory`** — no bulk-delete-by-tag.
- **`associate_memories`** — none of the nine relation-specific properties
  (`context`, `reason`, `pattern_type`, `confidence`, `resolution`,
  `observations`, `timestamp`, `transformation`, `role`). `additionalProperties:
  true` let them through unvalidated, but an agent reading the schema never
  learned to send them.
- **`update_memory`** — remote advertised `embedding` and forwarded it;
  `PATCH /memory/<id>` (`automem/api/memory.py:789`) never reads it. Dead field.

### Response rendering

| Call                    | stdio                                                                                                                                   | remote                                                          |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| recall `text`           | `1. <preview>[tags] (importance: raw) score=0.123 [match] relations=N`, `   ID:`, `   Created: … Updated: …`; summary-first, 400-char preview, 18k-token budget, trailer | `1. <full content> [tags] score=0.123`, `   ID:` — no date, no budget, no summary |
| recall `detailed`       | 2-space indent, no numbering, `Created:`                                                                                                  | no indent, numbered, `Timestamp:`                                |
| recall `items`          | `[<id>] <text>` blocks, no header                                                                                                         | `Found N memories:` header block plus compact blocks             |
| recall `json`           | own structured envelope                                                                                                                   | raw upstream response                                            |
| recall empty            | `No memories found matching your query.`                                                                                                  | `No memories found.`                                             |
| `store_memory`          | `Memory stored successfully!\n\nMemory ID: <id>`                                                                                           | `Memory stored: <id>`                                            |
| `update_memory`         | `Memory <id> updated successfully!`                                                                                                       | `Updated <id>`                                                   |
| `delete_memory`         | `Memory <id> deleted successfully!`                                                                                                       | `Deleted <id>`                                                   |
| `check_database_health` | formatted block with a status emoji                                                                                                       | `JSON.stringify(r)`                                              |
| error                   | `Error: <msg>`                                                                                                                            | `AutoMem error: <msg> (request_id: …)`                           |
| `structuredContent`     | on all six successes                                                                                                                      | never                                                            |

### Not a parity gap: `store_memory.id`

Both transports advertised an `id` parameter, and
`automem/api/memory.py:475` mints a server-side UUID unconditionally ("Always
generate server-side UUID to prevent collision/overwrite attacks"). Neither
transport could honor it. Rather than copy the lie into parity, `id` was dropped
from the shared schema.

## Status: `recall_memory` is at parity

Closed after the audit. On 2026-10-06 AutoHub's agents called the hosted
bridge with `recall_memory({ memory_id: "<uuid>" })` and got five unrelated
memories back with `isError: false`: the bridge dropped `memory_id` and ran an
unfiltered ranked search. Agents asking for one memory by ID got plausible
wrong answers, and in one trial a model invented a cause to fill the gap.

The bridge's `recall_memory` is now a port of `mcp-automem` 0.16.0:

- **Definition:** `title`, `description`, `annotations`, `_meta`,
  `inputSchema` and `outputSchema` are copied from `src/mcp-surface.ts`.
- **Routing and request mapping:** from `src/automem-client.ts`.
  - `memory_id` goes to `GET /memory/{id}` and ignores every other argument.
  - `exhaustive` goes to `GET /memory/by-tag`, using stdio's validation.
  - Ranked recall forwards `exclude_tags`, `offset`, `current_only`,
    `state_mode`, `state_debug`, `recency_bias`, `min_score`,
    `adaptive_floor` and `expand_respect_tags` with the same type guards.
- **Rendering:** from `src/recall-memory.ts`. This covers previews, the
  response budget and `structuredContent`.

Two differences remain, both deliberate:

| Difference                  | remote                                                                 | stdio                                                                                                        |
| --------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Error framing               | `isError` result, `AutoMem error: <msg> (request_id: …)`               | `Error: <msg>`. Its recall handler returns the builder's promise unawaited, so recall errors currently reach the client as JSON-RPC errors |
| Non-UUID `memory_id`        | Rejected before calling AutoMem, with the API's own message            | Sent to the API, which rejects it with the same message                                                       |

The pre-check exists because ids that are not UUIDs can resolve to other
routes. `by-tag` hits `GET /memory/by-tag`, and `..` normalizes to the viewer
at `/`. The message inside the framing must match on both sides.

`mcp-sse-server/test/recall-parity.test.js` enforces this without a live stack,
so CI's `node-test` job runs it on every PR. It drives the published package's
own MCP server and client next to the bridge, against a canned AutoMem API
(`mcp-sse-server/parity/fake-automem.js`), and requires the same tool
definition, the same upstream requests, and the same client-visible result.
Bumping the `@verygoodplugins/mcp-automem` devDependency re-runs it against the
new release. If it fails, port the change into `server.js` rather than loosening
the test.

Known gaps in the shared contract itself. Fix them in `mcp-automem` first, then
port them here:

- `offset` in ranked mode is forwarded to `/recall`, which does not read it.
- The `tag_match` description says `exact` is the default; `/recall` defaults
  to `prefix`.
- The `limit` description says the default is 5. Enumeration actually uses
  `/memory/by-tag`'s default of 20.

## Accepted transport-level differences

These are intentional and are **not** parity violations. The differential harness
allowlists exactly these and nothing else.

| Difference             | remote                                                                                | stdio                        | Why                                                                                                                  |
| ---------------------- | ------------------------------------------------------------------------------------- | ---------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `serverInfo.name`      | `automem-mcp-sse`                                                                       | `mcp-automem`                | Clients must be able to tell the transports apart.                                                                    |
| `serverInfo.version`   | the bridge's `package.json`                                                             | the published package's      | Two packages version independently. Each side must still report its **own** package version — the harness pins that self-consistency, which is how the remote's stale `0.1.0` is caught. |
| Auth mechanisms        | `Authorization: Bearer`, `X-API-Key` / `X-API-Token`, `?api_key=` / `?apiKey=` / `?api_token=` | `Authorization: Bearer` only | Browser and EventSource clients cannot set headers.                                                                    |
| Error text suffix      | ` (request_id: <uuid>)`                                                                 | none                         | Observability on a hosted service. Same code path — the bridge supplies a `requestIdProvider`, stdio does not.         |
| Timeout / retry policy | `UPSTREAM_TIMEOUT_MS` (15 s), `UPSTREAM_MAX_RETRIES` (2)                                | 25 s, 3                      | Tuned per deployment; injected via `AutoMemConfig`.                                                                    |
| Transport surface      | `/mcp`, `/mcp/sse`, `/mcp/messages`, `/health`, `/ready`, `/alexa`                       | stdio only                   | Not part of the MCP tool surface.                                                                                      |

Everything else — tool names, order, `title`, `description`, `inputSchema`,
`outputSchema`, `annotations`, `_meta`, server `instructions`, and every byte of
rendered `tools/call` text — must be identical.

## How parity is enforced

A differential harness drives both transports against a single
docker-compose AutoMem and diffs them:

```bash
make test-parity
```

It lives in `mcp-sse-server/parity/` with its entry point at
`mcp-sse-server/test/parity.test.js`, and it asserts these things:

1. `tools/list` is deep-equal across transports after key-order normalization.
2. Server capabilities and `instructions` match; `serverInfo.name` is allowlisted
   to differ.
3. A 27-scenario `tools/call` matrix renders identical text on both, after
   redacting values that legitimately vary per run (UUIDs, timestamps, scores,
   `query_time_ms`, and the per-transport tag namespace).
4. The `recall_memory` entry in `tools/list` is identical, and every
   `recall_memory` call in that matrix gets the same outcome on both
   transports. An outcome is the error message, or the mode, the paging, and
   which of the scenario's fixtures came back in what order. This covers:
   - ID fetch: valid, unknown and non-UUID ids, and `memory_id` overriding
     every other argument;
   - enumeration paging and rejections;
   - `exclude_tags`;
   - the state and score params.

   These checks hold even while checks 1 to 3 are red for the other five tools.
   Rendering is compared byte for byte in `test/recall-parity.test.js`
   instead. Against a live service the rendered text also picks up background
   work that reaches one transport's fixtures before the other's: enrichment
   summaries, JIT enrichment, temporal edges, and the clock-driven recency
   score.

The harness is gated on `AUTOMEM_RUN_PARITY_TESTS=1` because it needs a live
service. Without the gate it skips cleanly, which is what CI's `node-test` job
sees. `.github/workflows/mcp-parity.yml` runs it with the stack up on PRs that
touch `mcp-sse-server/**`, `automem/api/**`, or `app.py`.

It brings its own stack up on port **8011**, not the usual 8001. The harness
writes fixtures and bulk-deletes by tag, and a developer running a local
AutoMem install (`~/.automem/server`) already holds 8001 — pointing the harness
at that instance would seed test data into a real memory store.

**There is no scheduled drift run yet, deliberately.** While the harness is
intentionally red against the known gaps, `continue-on-error` leaves the
workflow successful either way, so a cron job could not distinguish "same known
gaps" from "a newly published `mcp-automem` added drift". That would be an alarm
that cannot alarm. The weekly check lands with the change that turns the harness
green, where a failure is a real signal.
