# AML adapter local smoke report

**Run date:** 2026-09-10

**Verdict:** PASS — local contract pre-check passed. This is not the official
AML compatibility smoke; that requires an AML-issued key and public endpoint.

## Public input used

AML's public repository intentionally excludes benchmark corpora and held-out
questions. The pre-check therefore uses the public Add/Search API Guide example
payload, preserved in [`public_sample.json`](public_sample.json). It includes
the documented `request_id`, ordered user/assistant messages, `user_id`,
`session_id`, choice options, and formal `top_k: 100`.

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
PASS: Each AML request refreshed and schema-validated the live MCP tool surface.
PASS: top_k=100 was safely bounded to AutoMem MCP's per-call limit of 50.
```

## Assertions exercised

| Surface | Evidence |
| --- | --- |
| AML Add | Two messages produced two synchronous MCP `store_memory` calls before a 200 response with exact identifier echoes. |
| AML Search | A `top_k: 100` query produced a contract-shaped, relevance-ordered `data` array with non-empty `id` and `content`. |
| Isolation | The MCP `recall_memory` call carried one SHA-256-derived exact user scope tag, `tag_match: exact`, and `scope_fallback: false`. |
| Authentication | Valid Bearer authentication succeeded; an invalid `X-Api-Key` was rejected with 401. |
| MCP transport | The smoke started the repository's actual Streamable HTTP MCP bridge (`mcp-sse-server`), which translated JSON-RPC `tools/call` into the local AutoMem HTTP seam; the adapter decoded `recall_memory`'s `format: json` content response. |
| MCP discovery | Before each authenticated adapter request, the adapter issues `tools/list`, confirms the three required tools, and fails closed if a live input schema no longer supports its fixed arguments. |

## Supporting MCP verification

```bash
npm --prefix mcp-sse-server test
```

Result: **22 passed, 0 failed, 3 skipped**. The skipped parity tests require a
live AutoMem stack (`AUTOMEM_RUN_PARITY_TESTS=1`).

## Limitation and submission recommendation

The published AutoMem `recall_memory` MCP schema limits an individual call to
50 results. The adapter therefore returns at most 50 results for AML's formal
`top_k: 100`; AML permits a response with *at most* the requested count, so the
response schema is compliant, but retrieval breadth may reduce score.

**NO-GO for a formal submission today.** Before Cycle 2 opens on 2026-09-20,
deploy a stable HTTPS endpoint, obtain/bind the AML key, pass AML's official
smoke, verify AML's required `gpt-4o-mini` model lock for all model-backed
memory operations, complete authenticated capacity testing, assign 30-day
operations ownership, and decide whether the 50-result MCP ceiling is
acceptable. **GO** only after those gates pass; the local adapter contract
itself is ready for that preflight.
