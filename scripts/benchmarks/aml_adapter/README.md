# AML Add/Search adapter

This service exposes the synchronous [Agent Memory Leaderboard (AML) Add/Search
contract](https://agentmemoryleaderboard.ai/) while using AutoMem only through
its Streamable HTTP MCP tools: `store_memory` and `recall_memory`.

## Run

Start AutoMem and its MCP bridge first, then run:

```bash
export AML_ADAPTER_MCP_URL=https://your-mcp-host/mcp
export AML_ADAPTER_MCP_TOKEN=your_automem_api_token
export AML_ADAPTER_API_KEY=the_key_registered_with_aml
python -m flask --app scripts.benchmarks.aml_adapter.app run --host 0.0.0.0 --port 8090
```

Register `https://your-host/add`, `https://your-host/search`, and
`https://your-host/health` with AML. `AML_ADAPTER_API_KEY` is optional only for
local smoke tests; production must bind one of `Authorization: Bearer`,
`Authorization: Token`, or `X-Api-Key`.

`POST /add` writes every message synchronously before returning 200. It stores
the original content, role/session metadata, and a SHA-256-derived exact tag
for the supplied `user_id`. `POST /search` uses that exact tag with no scope
fallback, so a caller can only retrieve its own AML scope. It returns AML's
ordered `{data:[{id,content,score?,created_at?}]}` response and never answers
the benchmark question.

Before every authenticated Add, Search, or health request, the adapter calls
MCP `tools/list` and validates the live input schemas of `store_memory`,
`recall_memory`, and `check_database_health`. This detects upstream tool drift
without holding an MCP session or logging request content.

## Container deployment

The included [`Dockerfile`](Dockerfile) runs the adapter under Gunicorn:

```bash
docker build -t automem-aml-adapter scripts/benchmarks/aml_adapter
docker run --rm -p 8090:8090 \
  -e AML_ADAPTER_MCP_URL=https://your-mcp-host/mcp \
  -e AML_ADAPTER_MCP_TOKEN=your_automem_api_token \
  -e AML_ADAPTER_API_KEY=the_key_registered_with_aml \
  automem-aml-adapter
```

Set `PORT`, `AML_ADAPTER_WORKERS`, `AML_ADAPTER_THREADS`, and
`AML_ADAPTER_GUNICORN_TIMEOUT_SECONDS` to match the capacity plan in
[`OPS.md`](OPS.md). The Docker build context must be this adapter directory.

## Local contract smoke

```bash
python scripts/benchmarks/aml_adapter/run_smoke.py
```

The smoke uses the public request examples reproduced in
[`public_sample.json`](public_sample.json), since AML's public repository
deliberately excludes held-out benchmark corpora. It starts a local fake
AutoMem HTTP upstream and the repository's actual `mcp-sse-server` Streamable
HTTP bridge, then calls real JSON-RPC `tools/call` requests through that MCP
surface. It validates Add, Search, result ordering, response echoes, and MCP
scope arguments without using any private AML evaluation data.

## Current retrieval bound

AML formal jobs request `top_k=100`; AutoMem's currently published
`recall_memory` MCP schema caps a single call at 50. The adapter therefore
returns up to 50 results, which remains compliant with AML's “at most top_k”
contract but is a quality limitation to resolve before a competitive run.

See [OPS.md](OPS.md) for the production runbook and
[SMOKE_REPORT.md](SMOKE_REPORT.md) for checked-in evidence.
