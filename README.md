# AutoMem: Research-Validated AI Memory 🧠

> **📅 November 20, 2025 Update** – Enhanced graph-vector hybrid architecture with improved enrichment pipeline, pattern detection, and deployment workflows. [See what's new](#whats-new-in-november-2025) ⚡

**Graph + Vector architecture proven to match human long-term memory performance.**

```bash
# Deploy in 60 seconds
railway up
# Or run locally
make dev
```

Give your AI persistent memory that actually learns and remembers.

**Quick Navigation:** [What's New](#whats-new-in-november-2025) • [Architecture](#architecture) • [Quick Start](#quick-start) • [Features](#features) • [API Examples](#api-examples) • [Connect AI Platforms](#connect-to-ai-platforms) • [Documentation](#documentation)

---

## The Problem We Solve

AI assistants forget everything between sessions. RAG systems retrieve context but can't learn patterns. Vector databases find similar text but miss relationships.

**You need AI that actually remembers.**

## What AutoMem Does

AutoMem is a **graph-vector memory service** that gives AI assistants durable, relational memory:
- 🧠 **Stores memories** with rich metadata, importance scores, and temporal context
- 🔍 **Recalls with hybrid search** - vector similarity + keyword + tags + time
- 🔗 **Builds knowledge graphs** - 11 relationship types between memories
- 🎯 **Learns patterns** - automatic entity extraction, clustering, and consolidation
- ⚡ **Sub-second recall** - even with millions of memories

### Research-Validated Architecture

AutoMem implements principles from:
- **HippoRAG 2** (2025): Graph-vector hybrid for human-like associative memory
- **A-MEM** (2025): Dynamic memory organization with Zettelkasten principles
- **MELODI** (DeepMind, 2025): 8x memory compression without quality loss
- **ReadAgent** (DeepMind, 2024): 20x context extension through gist memories

## What's New in November 2025

🎉 **Major enhancements to the graph-vector memory system:**

- **🔗 Enhanced Knowledge Graphs** – 11 relationship types now include `PREFERS_OVER`, `EXEMPLIFIES`, and `CONTRADICTS` for richer context modeling
- **🤖 Smarter Enrichment Pipeline** – Automatic entity extraction (people, tools, projects, concepts) with improved pattern detection
- **🔄 Background Consolidation** – Memory decay, creative association discovery, clustering, and intelligent forgetting cycles
- **⚡ Improved Hybrid Search** – Vector similarity + keyword + tags + temporal scoring for better recall accuracy
- **🚀 One-Command Deployment** – Railway deployment simplified with `railway up` - production-ready in 60 seconds
- **📊 Better Observability** – Enhanced health monitoring and enrichment status endpoints

[Jump to Quick Start](#quick-start) | [See Full Changelog](CHANGELOG.md)

## Architecture

**Dual-Engine Memory System for Human-Like Recall**

```
┌─────────────────────────────────────────────┐
│           AutoMem Service (Flask)           │
│   • REST API for memory lifecycle           │
│   • Background enrichment pipeline          │
│   • Consolidation engine                    │
│   • Automated backups (optional)            │
└──────────────┬──────────────┬───────────────┘
               │              │
        ┌──────▼──────┐  ┌───▼────────┐
        │  FalkorDB   │  │   Qdrant   │
        │   (Graph)   │  │ (Vectors)  │
        │             │  │            │
        │ • 11 edge   │  │ • Semantic │
        │   types     │  │   search   │
        │ • Pattern   │  │ • 768-d    │
        │   nodes     │  │   vectors  │
        └─────────────┘  └────────────┘
```

**Why Two Databases?**
- **FalkorDB (Graph)** – Canonical storage, relationships, and consolidation logic
- **Qdrant (Vectors)** – Semantic search and similarity-based recall
- **Dual Storage Benefits** – Built-in redundancy, disaster recovery, and graceful degradation

The graph provides structure and relationships while vectors enable fuzzy semantic matching. Together, they create memory that's both precise and contextual.

## Why Graph + Vector?

### Traditional RAG (Vector Only)
```
Memory: "Chose PostgreSQL for reliability"
Query: "What database should I use?"
Result: ✅ Finds the memory
         ❌ Doesn't know WHY you chose it
         ❌ Can't connect to related decisions
```

### AutoMem (Graph + Vector)
```
Memory: "Chose PostgreSQL for reliability"
Graph: PREFERS_OVER MongoDB
       RELATES_TO "team expertise" memory
       DERIVED_FROM "boring technology" principle

Query: "What database should I use?"
Result: ✅ Finds the memory
        ✅ Knows your decision factors
        ✅ Shows related preferences
        ✅ Explains your reasoning pattern
```

## Real-World Performance

### Knowledge Graphs That Learn
```python
# After storing: "Migrated to PostgreSQL for operational simplicity"

AutoMem automatically creates:
├── Entity: PostgreSQL (tagged: entity:tool:postgresql)
├── Entity: operational simplicity (tagged: entity:concept:ops-simplicity)
├── Pattern: "prefers boring technology" (reinforced)
├── Temporal: PRECEDED_BY migration planning memory
└── Similarity: SIMILAR_TO "Redis deployment" (both value simplicity)

# Next query: "Should we use Kafka?"
AI recalls:
- Your PostgreSQL decision
- Your "boring tech" pattern
- Related simplicity preferences
→ Suggests: "Based on your operational simplicity pattern, 
   consider RabbitMQ instead"
```

### Hybrid Search That Works
```bash
# Semantic + keyword + tags + time + importance scoring
GET /recall?query=database&tags=decision&time_query=last%20month

Returns memories ranked by:
- Vector similarity (0.64)
- Tag match (0.50)
- Recency (0.90)
- Exact keyword match (1.00)
Final score: 0.82 (weighted combination)
```

## Features

### Core Memory Operations

Everything your AI needs to build lasting knowledge:

- **📝 Store** – Rich memories with metadata, importance scores, timestamps, and embeddings
- **🔍 Recall** – Hybrid search combining vector similarity, keywords, tags, and time windows
- **✏️ Update** – Modify existing memories with automatic embedding regeneration
- **🗑️ Delete** – Clean removal from both graph and vector stores
- **🔗 Associate** – Create typed relationships between memories (11 relationship types)
- **🏷️ Filter** – Tag-based queries with prefix and exact matching

### Memory Consolidation

AutoMem uses [neuroscience-inspired](https://pmc.ncbi.nlm.nih.gov/articles/PMC4648295/) consolidation cycles to keep memories relevant and organized:

- **⏰ Decay (Hourly)** – Exponential relevance scoring based on age, access patterns, relationships, and importance
- **💡 Creative (Hourly)** – Discovers surprising connections between memories during "REM-like" processing
- **🧩 Cluster (Every 6 Hours)** – Groups similar memories and generates meta-patterns
- **🗂️ Forget (Daily)** – Archives low-relevance memories, permanently deletes extremely old unused ones

**Smart Forgetting:** Memories aren't immediately deleted. They're archived first (relevance 0.05-0.2), only removed if they drop below 0.05. This means wrong paths naturally fade (~30-45 days without use), while important connections survive longer.

### Background Intelligence

#### Enrichment Pipeline
Automatically enhances every memory:
- **Entity extraction** - People, projects, tools, concepts (with spaCy)
- **Auto-tagging** - `entity:<type>:<slug>` for structured queries
- **Summaries** - Lightweight snippets for quick scanning
- **Temporal links** - `PRECEDED_BY` to recent memories
- **Semantic neighbors** - `SIMILAR_TO` via cosine similarity
- **Pattern detection** - Reinforces emerging themes

#### Consolidation Engine
Keeps memory fresh over time:
- **Decay** (hourly) - Exponential relevance scoring
- **Creative** (hourly) - Discovers surprising associations
- **Cluster** (6-hourly) - Groups similar embeddings, generates meta-memories
- **Forget** (daily) - Archives/deletes low-relevance memories

### 11 Relationship Types

Build rich knowledge graphs:

| Type | Use Case | Example |
|------|----------|---------|
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

## Quick Start

Choose your deployment path based on your needs:

### Option 1: Railway (Recommended for Production)

**Best for:** Production deployments, 24/7 availability, multi-device access

Deploy AutoMem with managed FalkorDB and Qdrant in under 60 seconds:

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

After deployment:
1. Set `AUTOMEM_API_TOKEN` in Railway dashboard
2. Copy your Railway URL (e.g., `https://automem-production.up.railway.app`)
3. Test with: `curl https://your-url/health`

👉 **[Complete Deployment Guide](INSTALLATION.md#deployment)** – Railway setup, environment variables, and configuration

### Option 2: Docker Compose (Recommended for Local Development)

**Best for:** Local testing, development, privacy-focused work

Run the full stack (AutoMem + FalkorDB + Qdrant) locally:

```bash
# Clone and start all services
git clone https://github.com/verygoodplugins/automem.git
cd automem
make dev

# Services will be available at:
# • API: http://localhost:8001
# • FalkorDB: localhost:6379
# • Qdrant: localhost:6333
```

### Option 3: Development Mode (API Only)

**Best for:** Quick testing without Docker, or when databases are remote

Run just the Flask API:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

# Start API (requires existing FalkorDB/Qdrant)
PORT=8001 python app.py
```

**Next Steps:** [API Examples](#api-examples) | [Configuration Guide](INSTALLATION.md#configuration)

## API Examples

### Store a Memory
```bash
curl -X POST http://localhost:8001/memory \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{\n    "content": "Chose PostgreSQL over MongoDB for ACID compliance",
    "type": "Decision",
    "confidence": 0.95,
    "tags": ["database", "architecture"],
    "importance": 0.9,
    "metadata": {\n      "source": "architecture-meeting",
      "alternatives": ["MongoDB", "MySQL"],
      "deciding_factors": ["ACID", "team_expertise"]
    }
  }'
```

**Available memory types**: `Decision`, `Pattern`, `Preference`, `Style`, `Habit`, `Insight`, `Context` (default)
- **Explicit `type` recommended** when you know the classification
- **Omit `type`** to let enrichment auto-classify from content

### Recall Memories
```bash
# Hybrid search with tags and time
GET /recall?query=database&tags=decision&time_query=last%20month

# Semantic search with vector
GET /recall?embedding=0.12,0.56,...&limit=10

# Tag prefix matching (finds slack:U123:*, slack:channel-ops, etc.)
GET /recall?tags=slack&tag_match=prefix
```

### Create Relationship
```bash
curl -X POST http://localhost:8001/associate \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{\n    "memory1_id": "uuid-postgres-decision",
    "memory2_id": "uuid-mongodb-evaluation",
    "type": "PREFERS_OVER",
    "strength": 0.9
  }'
```

## Connect to AI Platforms

AutoMem integrates seamlessly with any AI assistant or application:

### 🔌 MCP (Model Context Protocol)

**For Local AI Tools** (Claude Desktop, Cursor, Claude Code):

Install the official MCP bridge to give your desktop AI tools persistent memory:

```bash
# One-command setup
npm install -g @verygoodplugins/mcp-automem
npx @verygoodplugins/mcp-automem setup
```

Your AI assistant can now store and recall memories automatically.

**For Cloud AI Platforms** (ChatGPT, Claude.ai, ElevenLabs):

Connect AutoMem to cloud services via the SSE sidecar:

- ✅ **ChatGPT** (with developer mode)
- ✅ **Claude.ai** web interface
- ✅ **Claude mobile app**
- ✅ **ElevenLabs Agents**

See the [MCP over SSE Setup Guide](docs/MCP_SSE.md) for detailed instructions.

**Resources:**
- 📦 [NPM Bridge Package](https://www.npmjs.com/package/@verygoodplugins/mcp-automem)
- 📚 [SSE Configuration Guide](docs/MCP_SSE.md)

### 🌐 Direct API Integration

Use AutoMem from any language or framework:

```python
import requests

# Store a memory
response = requests.post(
    "https://your-automem.railway.app/memory",
    headers={"Authorization": f"Bearer {token}"},
    json={"content": "Memory content", "importance": 0.8}
)
```

Perfect for custom integrations, backend services, or building your own AI assistant.

## What Makes AutoMem Different

### vs. Traditional RAG Systems
- ✅ **Rich Relationships** – Not just "similar" matches, but explicit relationships like "causes", "prefers", "invalidates"
- ✅ **Temporal Intelligence** – Knows what came before, what evolved from what, and how knowledge changes over time
- ✅ **Pattern Discovery** – Automatically discovers and reinforces recurring themes across memories
- ✅ **Active Consolidation** – Memories improve and organize over time, not just pile up

### vs. Vector-Only Databases
- ✅ **11 Relationship Types** – Structured edges vs. cosine similarity alone
- ✅ **Background Intelligence** – Automatic enrichment, clustering, and relevance decay
- ✅ **Hybrid Scoring** – Vector similarity + keyword matching + tag overlap + temporal context + importance weighting
- ✅ **Graph Traversal** – Navigate relationship chains, not just retrieve similar vectors

### vs. Building Your Own Memory System
- ✅ **Research-Validated** – Implements proven principles from HippoRAG 2, A-MEM, MELODI, and ReadAgent
- ✅ **Production-Ready** – Built-in authentication, admin tools, health monitoring, and backup systems
- ✅ **Battle-Tested** – Robust enrichment pipeline, consolidation logic, and automatic retry mechanisms
- ✅ **Open Source** – MIT license, deploy anywhere, extend freely

## Performance & Reliability

**Built for Real-World Production Use**

- ⚡ **Sub-second recall** – Even with 100k+ memories
- 🔄 **Concurrent writes** – Background enrichment doesn't block API requests
- 🛡️ **Graceful degradation** – Works in graph-only mode if Qdrant is unavailable
- 🔁 **Automatic retries** – Failed enrichments queue for reprocessing
- 📊 **Health monitoring** – `/health` and `/enrichment/status` endpoints
- 💾 **Automated backups** – Optional backup service for disaster recovery
- 🔐 **Dual storage redundancy** – Data persisted in both FalkorDB and Qdrant
- ✅ **Benchmark validated** – Tested against LoCoMo (ACL 2024) with `make test-locomo`

## Configuration

### Essential Settings

**Required:**
- `AUTOMEM_API_TOKEN` – Authentication for all endpoints (except `/health`)
- `FALKORDB_HOST` / `FALKORDB_PORT` – Graph database connection

**Optional but Recommended:**
- `QDRANT_URL` / `QDRANT_API_KEY` – Enable semantic vector search
- `OPENAI_API_KEY` – Generate real embeddings (otherwise uses deterministic placeholders)
- `ADMIN_API_TOKEN` – Required for admin endpoints like `/admin/reembed`

**Advanced Tuning:**
- `CONSOLIDATION_*_INTERVAL_SECONDS` – Adjust decay, creative, cluster, and forget cycles
- `ENRICHMENT_*` – Configure similarity thresholds, retry limits, and worker behavior

👉 **[Complete Configuration Guide](INSTALLATION.md#configuration)** – All environment variables with examples

## Documentation

**Get Started:**
- 📦 **[Installation Guide](INSTALLATION.md)** – Railway, Docker, and development setup
- 🚀 **[Quick Start](#quick-start)** – Deploy in 60 seconds

**Integration:**
- 🌉 **[MCP over SSE](docs/MCP_SSE.md)** – Connect to ChatGPT, Claude, and ElevenLabs
- 🔧 **[API Reference](docs/API.md)** – Complete endpoint documentation with examples

**Operations:**
- 💾 **[Monitoring & Backups](docs/MONITORING_AND_BACKUPS.md)** – Health checks and disaster recovery
- 🧪 **[Testing Guide](docs/TESTING.md)** – Unit, integration, and benchmark tests
- 📊 **[LoCoMo Benchmark](docs/TESTING.md#locomo-benchmark)** – ACL 2024 validation suite

**Migration:**
- 🔄 **[Migration Guide](INSTALLATION.md#migration)** – Move from MCP SQLite or other systems

**Learn More:**
- 🌐 **[automem.ai](https://automem.ai)** – Official website and tutorials

## Community & Support

**Connect with Us:**
- 🌐 **[automem.ai](https://automem.ai)** – Official website
- 🐙 **[GitHub Repository](https://github.com/verygoodplugins/automem)** – Source code and discussions
- 📦 **[NPM MCP Bridge](https://www.npmjs.com/package/@verygoodplugins/mcp-automem)** – Official MCP integration
- 🐛 **[Issue Tracker](https://github.com/verygoodplugins/automem/issues)** – Bug reports and feature requests

## The Science Behind AutoMem

AutoMem isn't just inspired by research—it implements peer-reviewed principles from leading institutions:

**[HippoRAG 2](https://arxiv.org/abs/2502.14802)** (Ohio State, June 2025)  
Graph-vector hybrid architecture achieves **7% better associative memory** than pure vector RAG, approaching human long-term memory performance.

**[A-MEM](https://arxiv.org/abs/2502.12110)** (July 2025)  
Validates dynamic memory organization with Zettelkasten-inspired principles—exactly what AutoMem's pattern detection and clustering implement.

**[MELODI](https://arxiv.org/html/2410.03156v1)** (DeepMind, 2024)  
Demonstrates **8x memory compression** without quality loss through gist representations—AutoMem's summary generation follows these principles.

**[ReadAgent](https://arxiv.org/abs/2402.09727)** (DeepMind, 2024)  
Shows **20x context extension** via episodic memory—AutoMem's consolidation engine implements similar temporal organization.

**We didn't just read the papers. We built the system they describe.**

## Contributing

We welcome contributions from the community! Here's how to get involved:

1. **Fork** the repository on GitHub
2. **Create** a feature branch for your changes
3. **Add tests** for new functionality
4. **Submit** a pull request with a clear description

See our [Testing Guide](TESTING.md) for running the test suite locally.

## License

**MIT License** – Because AI memory should be free and accessible to everyone.

---

## Ready to Transform Your AI?

Give your AI assistant the gift of human-like memory:

```bash
# Deploy to production in 60 seconds
railway up

# Or start locally
make dev
```

**AutoMem turns AI from a tool into a thinking partner.**

*Built with obsession. Validated by neuroscience. Powered by graph theory.*