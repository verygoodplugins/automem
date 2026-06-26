<p align="center">
  <img src="https://automem.ai/img/G_Vd7EYWYAM_3SF.jpeg" alt="AutoMem" width="600" />
</p>

<p align="center">
  <a href="https://www.npmjs.com/package/@verygoodplugins/mcp-automem"><img src="https://img.shields.io/npm/v/@verygoodplugins/mcp-automem?label=mcp-automem" alt="npm version" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/verygoodplugins/automem" alt="License" /></a>
  <a href="https://automem.ai/discord"><img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white" alt="Discord" /></a>
  <a href="https://x.com/automem_ai"><img src="https://img.shields.io/badge/X-@automem__ai-000000?logo=x&logoColor=white" alt="X" /></a>
  <a href="benchmarks/EXPERIMENT_LOG.md"><img src="https://img.shields.io/badge/LongMemEval-87.00%25-success" alt="LongMemEval benchmark" /></a>
  <a href="https://automem.ai/benchmarks"><img src="https://img.shields.io/badge/BEAM%2010M-57.4%25-success" alt="BEAM 10M on the Agent Memory Benchmark" /></a>
  <a href="https://railway.com/deploy/automem-ai-memory-service?referralCode=VuFE6g&utm_medium=integration&utm_source=github&utm_campaign=generic"><img src="https://img.shields.io/badge/Deploy%20on-Railway-0B0D0E?logo=railway&logoColor=white" alt="Deploy on Railway" /></a>
</p>

<p align="center">
  <strong>Long-term memory for AI assistants. Graph + vector. Runs on your hardware.</strong>
</p>

<!-- VIDEO PLACEHOLDER — replace this comment with a centered <p> containing a thumbnail image linking to the vidrush demo URL once recorded. Suggested markup:
<p align="center">
  <a href="https://vidrush.app/v/REPLACE-ME"><img src="docs/img/video-thumb.jpg" alt="Watch the demo" width="480" /></a>
</p>
-->

# AutoMem

Your AI forgets between sessions. RAG dumps documents that look similar. Vector databases match keywords but miss meaning. None of them learn.

AutoMem stores typed relationships *and* embeddings. When you ask "why did we choose PostgreSQL?", recall returns not just the matching memory — but the alternatives you considered, the principle behind the choice, and the related decisions that came after.

Current canonical benchmark results are **87.00%** on LongMemEval full with **97.00% recall@5**, and **84.74%** on LoCoMo full. See [`benchmarks/EXPERIMENT_LOG.md`](benchmarks/EXPERIMENT_LOG.md) for methodology, judge policy, category breakdowns, and historical runs.

### On the neutral Agent Memory Benchmark

AutoMem **0.16.0** was run through the neutral [Agent Memory Benchmark](https://automem.ai/benchmarks) (AMB, by vectorize-io) on a self-spinning FalkorDB + Qdrant stack with **FastEmbed-local `bge-base-en-v1.5` (768d)** — no embedding API keys. The honest summary: AutoMem's strength is **large-context scaling and efficiency**, not verbatim conversational recall.

- **BEAM is the apples-to-apples axis** (same benchmark, same Gemini answerer + judge). AutoMem scores above Honcho at every BEAM tier, and the gap widens with scale: **+4.5pp at 100k, +0.7pp at 500k, +0.7pp at 1M, +16.8pp at 10M**. AutoMem degrades gracefully — **67.5% → 57.4%** (−10pp) across a 100× haystack increase — while Honcho holds roughly flat through 1M, then drops to 40.6% at 10M. That places AutoMem **#2 on BEAM**, behind vectorize's own Hindsight (~73→64% across the curve).
- **At 10M tokens, AutoMem holds 57.4% ±5.5%** while Honcho falls to ~41%. At that scale, context-stuffing is physically impossible, so the score reflects retrieval architecture, not context window.
- **Efficiency is architectural:** AutoMem feeds the answerer **~2.6–4.8k context tokens** at every scale (mean), versus 17–27k for the board leader on BEAM.
- **The honest other half:** on conversational Core-3, AutoMem **trails** the AMB leader Hindsight — locomo 85.1% vs 92%, longmemeval 74.4% vs 94.6%, personamem 76.1% vs 86.6%. Pick AutoMem for large-context scaling and efficiency, not for top-of-board verbatim recall.

Outputs are committed and public, and [`AUTOMEM_REPRODUCE.md`](https://automem.ai/benchmarks) gives one command per split so you can **run it yourself**. AutoMem is **submitted to the neutral board (provider PR under review)** — not yet live on the public leaderboard. Full head-to-head numbers live at [automem.ai/benchmarks](https://automem.ai/benchmarks).

## Should you use AutoMem?

| Use AutoMem if... | Look elsewhere if... |
|---|---|
| You want one memory across Claude / Cursor / ChatGPT / Codex | You need SOC2 / HIPAA audit logs and row-level ACLs |
| You're comfortable self-hosting (Docker or Railway) | You want a managed SaaS with a polished dashboard |
| You're a solo dev, prosumer, or small team | You're running a multi-agent swarm needing per-agent memory isolation |
| You want to own your memory data | You need an enterprise SLA and dedicated support |

If your row is on the right, AutoMem isn't it — yet. Try [Mem0](https://mem0.ai), [Letta](https://letta.com), or [Zep](https://www.getzep.com) instead.

## How it works

AutoMem combines two storage layers behind a single API:

- **[FalkorDB](https://www.falkordb.com/)** stores memories as nodes with **11 typed relationships** between them. The graph is the canonical record.
- **[Qdrant](https://qdrant.tech/)** stores an embedding for every memory. Recall is a hybrid query — semantic similarity, graph traversal, temporal alignment, tag overlap, and importance — ranked by a 9-component score.

```mermaid
flowchart TB
    subgraph service [AutoMem Service Flask]
        API[REST API<br/>Memory Lifecycle]
        Enrichment[Background Enrichment<br/>Pipeline]
        Consolidation[Consolidation<br/>Engine]
        Backups[Automated Backups<br/>Optional]
    end

    subgraph storage [Dual Storage Layer]
        FalkorDB[(FalkorDB<br/>Graph Database)]
        Qdrant[(Qdrant<br/>Vector Database)]
    end

    Client[AI Client] -->|Store/Recall/Associate| API
    API --> FalkorDB
    API --> Qdrant
    Enrichment -->|11 edge types<br/>Pattern nodes| FalkorDB
    Enrichment -->|Semantic search<br/>1024-d vectors| Qdrant
    Consolidation --> FalkorDB
    Consolidation --> Qdrant
    Backups -.->|Optional| FalkorDB
    Backups -.->|Optional| Qdrant
```

If Qdrant is unavailable, the graph still serves recall in a degraded mode. If FalkorDB is down, the API returns 503 — the graph is the source of truth.

### Multi-hop bridge discovery

Ask "why boring tech for Kafka?" and AutoMem doesn't just match the word "Kafka". It traverses the graph from the seed memories to find the bridge that connects them:

- **Seed 1**: "Migrated to PostgreSQL for operational simplicity"
- **Seed 2**: "Evaluating Kafka vs RabbitMQ for message queue"
- **Bridge**: "Team prefers boring technology — proven, debuggable systems"

Both seeds carry an `EXEMPLIFIES` edge to the bridge memory. AutoMem ranks the bridge above the seeds and surfaces it in the recall response, so the assistant answers with your reasoning, not isolated facts. Tune via `expand_relations`, `relation_limit`, and `expansion_limit` on `GET /recall`.

### 11 authorable relationship types

| Type | Use case | Example |
|---|---|---|
| `RELATES_TO` | General connection | Bug report → Related issue |
| `LEADS_TO` | Causal relationship | Problem → Solution |
| `OCCURRED_BEFORE` | Temporal sequence | Planning → Execution |
| `PREFERS_OVER` | User preferences | PostgreSQL → MongoDB |
| `EXEMPLIFIES` | Pattern examples | Code review → Best practice |
| `CONTRADICTS` | Conflicting info | Old approach → New approach |
| `REINFORCES` | Supporting evidence | Decision → Validation |
| `INVALIDATED_BY` | Outdated info | Legacy docs → Current docs |
| `EVOLVED_INTO` | Knowledge evolution | Initial design → Final design |
| `DERIVED_FROM` | Source tracking | Implementation → Spec |
| `PART_OF` | Hierarchical structure | Feature → Epic |

Three more edge types are added automatically by the [enrichment pipeline](docs/API.md) and consolidation engine: `SIMILAR_TO`, `PRECEDED_BY`, and `DISCOVERED`.

### Memory consolidation, neuroscience-inspired

AutoMem implements [biological memory consolidation cycles](https://pmc.ncbi.nlm.nih.gov/articles/PMC4648295/). Wrong rabbit holes fade naturally. Important memories with strong connections strengthen over time.

| Cycle | Frequency | Purpose |
|---|---|---|
| **Decay** | Daily | Exponential relevance scoring (age, access, connections, importance) |
| **Creative** | Weekly | REM-like processing that discovers non-obvious connections |
| **Cluster** | Monthly | Groups similar memories, generates meta-patterns |
| **Forget** | Off by default | Archives low-relevance memories (<0.2), deletes very old (<0.05) |

Tune intervals via `CONSOLIDATION_*_INTERVAL_SECONDS`. See [`docs/ENVIRONMENT_VARIABLES.md`](docs/ENVIRONMENT_VARIABLES.md).

For more on the recall scoring formula, enrichment internals, and how AutoMem differs from RAG and pure vector databases, see [`docs/COMPARISON.md`](docs/COMPARISON.md).

## Research foundation

AutoMem implements techniques from peer-reviewed memory research:

- **[HippoRAG 2](https://arxiv.org/abs/2502.14802)** (Ohio State, 2025) — graph + vector hybrid for associative memory
- **[A-MEM](https://arxiv.org/abs/2502.12110)** (2025) — Zettelkasten-inspired dynamic memory organization
- **[MELODI](https://arxiv.org/html/2410.03156v1)** (DeepMind, 2024) — gist-based memory compression
- **[ReadAgent](https://arxiv.org/abs/2402.09727)** (DeepMind, 2024) — episodic memory for context extension

Full writeups, findings, and how AutoMem implements each → [`docs/RESEARCH.md`](docs/RESEARCH.md).

## Run it

### Railway (60 seconds)

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/deploy/automem-ai-memory-service?referralCode=VuFE6g&utm_medium=integration&utm_source=github&utm_campaign=generic)

Recommended Railway projects run AutoMem as a small service group: `automem` (the API), `automem-graph-viewer` (the standalone UI), `falkordb` (graph), `qdrant` (vectors), and `mcp-automem` (the MCP bridge for ChatGPT, Claude.ai, and ElevenLabs). Services use pre-built Docker images and auto-redeploy on `:stable`, so Railway does not spend compute rebuilding source.

→ Full setup: [INSTALLATION.md](INSTALLATION.md#deployment)

### Docker Compose (local)

```bash
git clone https://github.com/verygoodplugins/automem.git
cd automem
make dev
```

| Service | URL | Purpose |
|---|---|---|
| AutoMem API | `http://localhost:8001` | Memory REST API |
| FalkorDB | `localhost:6379` | Graph database |
| Qdrant | `localhost:6333` | Vector database |
| FalkorDB Browser | `http://localhost:3000` | Local graph inspection UI |

→ Full setup: [INSTALLATION.md](INSTALLATION.md#docker-compose-local)

### Python (development)

```bash
make install
source .venv/bin/activate
PORT=8001 python app.py
```

Requires Python 3.10+ (3.12 recommended). → [INSTALLATION.md](INSTALLATION.md#bare-api-development)

**Contributing:** feature PRs target the `develop` branch. `main` only moves via validated release merges, so users deploying from `main` (e.g. Railway auto-deploys) see one deploy per release instead of one per PR.

## Connect your AI

| Client | Mode | Setup |
|---|---|---|
| Claude Desktop, Cursor, Claude Code, Codex, Copilot, Antigravity | Local MCP bridge | `npx @verygoodplugins/mcp-automem setup` |
| ChatGPT (developer mode), Claude.ai web/mobile, ElevenLabs Agents | Remote MCP (HTTPS) | [`docs/MCP_SSE.md`](docs/MCP_SSE.md) |
| Anything else | Direct REST API | [`docs/API.md`](docs/API.md) |

The MCP bridge is published as [`@verygoodplugins/mcp-automem`](https://www.npmjs.com/package/@verygoodplugins/mcp-automem). It handles client-specific config (rules files, hooks, templates) and proxies to your AutoMem service — local or Railway.

Direct API call:

```python
import requests

token = "your-automem-api-token"

requests.post(
    "https://your-automem.railway.app/memory",
    headers={"Authorization": f"Bearer {token}"},
    json={
        "content": "Chose PostgreSQL over MongoDB for ACID compliance",
        "type": "Decision",
        "tags": ["database", "architecture"],
        "importance": 0.9,
    },
)
```

## Screenshots

<!-- CAPTURE CHECKLIST — replace each placeholder image once captured:
1. docs/img/graph-recall.png — FalkorDB browser at http://localhost:3000 showing a memory node with multiple typed edges visible (PREFERS_OVER, EXEMPLIFIES, RELATES_TO). Proves the graph is real.
2. docs/img/claude-recall.png — Claude Desktop conversation where the assistant cites a recalled memory inline, with the AutoMem MCP tool call visible in the trace.
3. docs/img/railway-services.png — Railway dashboard showing the production service group (automem, automem-graph-viewer, mcp-automem, qdrant, falkordb). A version exists as a GitHub user-attachment in INSTALLATION.md; capture a clean version and host it in-repo.

Until each PNG exists, the table cells will render as broken-image icons on GitHub. -->

_Screenshots will be added once the referenced in-repo image assets are available._

## Known limitations

AutoMem is pre-1.0 and honest about its rough edges. The active ones for recall quality:

- **Tags are a hard gate, not a soft boost.** Tags filter *before* scoring, so a memory missing the queried tag won't surface even on a perfect semantic match. Within a tag scope, high-importance off-topic memories can still over-rank — mitigated by the opt-in `RECALL_RELEVANCE_GATE`, not yet on by default ([#130](https://github.com/verygoodplugins/automem/issues/130)).
- **Temporal and preference updates.** Recall doesn't yet reliably prefer the newest version of a conflicting fact, or fully resolve multi-session preference updates. The `RECALL_RECENCY_BIAS=auto` re-rank helps temporal-intent queries but stays opt-in pending broader validation ([#158](https://github.com/verygoodplugins/automem/issues/158), [#159](https://github.com/verygoodplugins/automem/issues/159)).
- **The MCP SSE bridge doesn't forward `state_mode`.** The HTTP recall API supports it; the SSE proxy doesn't pass it through yet ([#172](https://github.com/verygoodplugins/automem/issues/172)).
- **Entity-node synthesis is experimental and off by default.** First-class `Entity` nodes (`IDENTITY_SYNTHESIS_ENABLED`) are gated off while people-entity word-pair noise is addressed ([#181](https://github.com/verygoodplugins/automem/issues/181)).

## Docs, community, and license

**Setup**
- [Installation guide](INSTALLATION.md) — Railway, Docker, development
- [Qdrant setup](docs/QDRANT_SETUP.md) — vector database configuration
- [Environment variables](docs/ENVIRONMENT_VARIABLES.md) — full reference

**API and integration**
- [API reference](docs/API.md) — endpoints, scoring, enrichment
- [Remote MCP](docs/MCP_SSE.md) — ChatGPT, Claude.ai, ElevenLabs
- [Migrations](docs/MIGRATIONS.md) — embedding dimensions, 0.16.0 data migrations, MCP SQLite import

**Research and comparison**
- [Research foundation](docs/RESEARCH.md) — papers and how AutoMem implements them
- [Comparison](docs/COMPARISON.md) — vs. RAG, vector DBs, building your own
- [Benchmark history](benchmarks/EXPERIMENT_LOG.md) — internal LoCoMo / LongMemEval harness runs + the neutral AMB (BEAM + Core-3) summary
- [AMB head-to-head + reproducibility](https://automem.ai/benchmarks) — neutral Agent Memory Benchmark results and the `AUTOMEM_REPRODUCE.md` "run it yourself" recipe

**Operations**
- [Scripts](scripts/README.md) — maintenance, migration, recovery, and eval tooling, by lifecycle
- [Health monitoring & backups](docs/MONITORING_AND_BACKUPS.md)
- [Testing guide](docs/TESTING.md) — unit, integration, benchmarks

**Community**
- [automem.ai](https://automem.ai) — official site
- [Discord](https://automem.ai/discord) — community chat
- [X / @automem_ai](https://x.com/automem_ai) — updates
- [YouTube / @AutoJackBot](https://www.youtube.com/@AutoJackBot) — tutorials
- [GitHub issues](https://github.com/verygoodplugins/automem/issues) — bugs and feature requests

**Sibling repos**
- [`mcp-automem`](https://github.com/verygoodplugins/mcp-automem) — universal MCP bridge / install funnel
- [`automem-evals`](https://github.com/verygoodplugins/automem-evals) — exploratory recall-quality lab
- [`automem-graph-viewer`](https://github.com/verygoodplugins/automem-graph-viewer) — standalone graph visualization

MIT licensed. Deploy anywhere. No vendor lock-in.
