# AML adapter local smoke report

**Run date:** 2026-09-10 15:06 CEST (13:06 UTC)

**Verdict:** PASS — local contract pre-check passed. This is not the official
AML compatibility smoke; that requires an AML-issued key and public endpoint.

## Public input used

AML's public repository intentionally excludes benchmark corpora and held-out
questions. The pre-check therefore uses the public Add/Search API Guide example
payload, preserved in [`public_sample.json`](public_sample.json). It includes
the documented `request_id`, ordered user/assistant messages, `user_id`,
`session_id`, choice options, and formal `top_k: 100`.

The public guide was rechecked immediately before this run. It requires a
synchronous HTTP 200 Add response with exact identifier echoes, exact
`user_id` retrieval isolation, and an ordered Search `data` array containing
no more than `top_k` records.

## Command

```bash
.venv/bin/python scripts/benchmarks/aml_adapter/run_smoke.py
```

## Output

```text
PASS: AML public API-guide Add/Search sample completed through the real MCP bridge.
PASS: Add echoed identifiers after two synchronous store_memory calls.
PASS: Search returned ranked AML data and enforced exact user scope tag.
PASS: Bearer authentication was enforced and an invalid X-Api-Key was rejected.
PASS: Each stored message preserved the exact user_id metadata.
PASS: Each AML request negotiated, refreshed, and schema-validated the live MCP tool surface.
PASS: top_k=100 was safely bounded to AutoMem MCP's per-call limit of 50.
```

## Assertions exercised

| Surface | Evidence |
| --- | --- |
| AML Add | Two messages produced two synchronous MCP `store_memory` calls before a 200 response with exact identifier echoes. |
| AML Search | A `top_k: 100` query produced a contract-shaped, relevance-ordered `data` array with non-empty `id` and `content`. |
| Isolation | The MCP `recall_memory` call carried one SHA-256-derived exact user scope tag, `tag_match: exact`, and `scope_fallback: false`. |
| Authentication | Valid Bearer authentication succeeded; an invalid `X-Api-Key` was rejected with 401. |
| Stored scope | Each MCP `store_memory` call carries the exact `user_id` in metadata as well as the derived, exact scope tag. |
| MCP transport | The smoke started the repository's actual Streamable HTTP MCP bridge (`mcp-sse-server`), which translated JSON-RPC `tools/call` into the local AutoMem HTTP seam; the adapter decoded `recall_memory`'s `format: json` content response. |
| MCP discovery | Before each authenticated adapter request, the adapter performs stateless `initialize` capability negotiation, issues `tools/list`, confirms the three required tools, and fails closed if a live input schema no longer supports its fixed arguments. |

## Supporting MCP verification

```bash
npm --prefix mcp-sse-server test
```

Result: **22 passed, 0 failed, 3 skipped**. The skipped parity tests require a
live AutoMem stack (`AUTOMEM_RUN_PARITY_TESTS=1`).

## Additional local checks

```text
.venv/bin/black --check scripts/benchmarks/aml_adapter
4 files would be left unchanged.

.venv/bin/flake8 scripts/benchmarks/aml_adapter
(passed)

docker build -t automem-aml-adapter:local scripts/benchmarks/aml_adapter
(passed)
```

## Limitation and submission recommendation

The published AutoMem `recall_memory` MCP schema limits an individual call to
50 results. The adapter therefore returns at most 50 results for AML's formal
`top_k: 100`; AML permits a response with *at most* the requested count, so the
response schema is compliant, but retrieval breadth may reduce score.

**NO-GO for a formal submission today.** Before Cycle 2 opens on 2026-09-20,
deploy a stable HTTPS hosted API with a pinned AutoMem MCP upstream, obtain and
bind the AML key, pass AML's official smoke, pin and record the submitted
AutoMem configuration (including any Add/Search model, for which AML currently
requires `gpt-4o-mini`), complete authenticated capacity testing, assign
30-day operations ownership, and decide whether the 50-result MCP ceiling is
acceptable. The adapter image alone is not sufficient for AML's
platform-managed Docker route because it requires the separately deployed MCP
upstream. **GO** only after those gates pass; the local adapter contract itself
is ready for that preflight.
