```text
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    █████╗ ██╗   ██╗████████╗ ██████╗ ███╗   ███╗███████╗███╗   ███╗        ║
║   ██╔══██╗██║   ██║╚══██╔══╝██╔═══██╗████╗ ████║██╔════╝████╗ ████║        ║
║   ███████║██║   ██║   ██║   ██║   ██║██╔████╔██║█████╗  ██╔████╔██║        ║
║   ██╔══██║██║   ██║   ██║   ██║   ██║██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║        ║
║   ██║  ██║╚██████╔╝   ██║   ╚██████╔╝██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║        ║
║   ╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝ ╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝        ║
║                                                                              ║
║        State-of-the-Art Conversational Memory • 90.53% LoCoMo Score         ║
║              Graph + Vector Architecture • Research-Validated               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

# **AI Memory That Actually Learns**

AutoMem is the **world's best-performing long-term memory system** for AI assistants. On December 2, 2025, AutoMem achieved **90.53% accuracy** on the LoCoMo benchmark (ACL 2024), beating the previous state-of-the-art by 2.29 points.

**Deploy production-grade AI memory in 60 seconds:**

```bash
railway up
```

---

## 🎯 December 2, 2025 Update: State-of-the-Art Improved

AutoMem achieved **90.53% accuracy** on the LoCoMo benchmark (ACL 2024)—the academic standard for long-term conversational memory. This makes AutoMem the **highest-performing memory system in the world**, beating:

- **CORE** (heysol.ai): 88.24% (previous SOTA)
- **OpenAI's baseline**: 39%
- **+14.45 percentage points** improvement since baseline

**Key breakthroughs:**

- 🔗 **Entity-to-entity expansion** - Multi-hop reasoning via entity tag linking
- 🌉 **Multi-hop bridge discovery** - Finds connecting memories across conversation threads
- ⏰ **Temporal alignment scoring** - Understands time-aware queries ("what happened last year?")
- 🎯 **9-component hybrid search** - Semantic + lexical + graph + temporal signals
- 💯 **100% complex reasoning** - Perfect score on multi-step reasoning tasks

**[Full benchmark results →](tests/benchmarks/BENCHMARK_2025-12-02.md)**

---

## Why AutoMem Exists

Your AI forgets everything between sessions. RAG dumps similar documents. Vector databases match keywords but miss meaning. **None of them learn.**

AutoMem gives AI assistants the ability to **remember, connect, and evolve** their understanding over time—just like human long-term memory.

## What Makes AutoMem State-of-the-Art

**December 2, 2025**: AutoMem scored **90.53%** on LoCoMo—the academic benchmark for long-term conversational memory (ACL 2024). This beats:
- **CORE (heysol.ai)**: 88.24% (previous SOTA)
- **OpenAI's implementation**: 39%
- **+14.45 points** improvement since baseline

AutoMem is a **graph-vector memory service** built on peer-reviewed neuroscience and validated by academic benchmarks:

### The System

- 🧠 **Stores** memories with metadata, importance scores, temporal context, and semantic embeddings
- 🔍 **Recalls** via 9-component hybrid search (vector + keyword + graph + temporal + lexical)
- 🔗 **Connects** memories with 11 typed relationships (RELATES_TO, LEADS_TO, CONTRADICTS, etc.)
- 🎯 **Learns** through automatic entity extraction, pattern detection, and consolidation cycles
- 🌉 **Reasons** with multi-hop bridge discovery—finds connecting memories across conversation threads
- ⚡ **Performs** with sub-100ms recall across thousands of memories

### The Research

AutoMem implements breakthroughs from:

- **HippoRAG 2** (Ohio State, 2025): Graph-vector hybrid matches human associative memory
- **A-MEM** (2025): Dynamic memory organization with Zettelkasten-inspired clustering
- **MELODI** (DeepMind, 2024): 8x compression without quality loss via gist representations
- **ReadAgent** (DeepMind, 2024): 20x context extension through episodic memory

**We didn't just read the papers. We built the system they describe—and proved it works.**

## Architecture

```text
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
        │ • Pattern   │  │ • 3072-d   │
        │   nodes     │  │   vectors  │
        └─────────────┘  └────────────┘
```

**FalkorDB** (graph) = canonical record, relationships, consolidation
**Qdrant** (vectors) = semantic recall, similarity search
**Dual storage** = Built-in redundancy and disaster recovery

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

## How It Works in Practice

### Multi-Hop Bridge Discovery (November 2025 Breakthrough)

The innovation that pushed AutoMem to SOTA: **path-based memory expansion** that discovers "bridge" memories connecting disparate conversation threads.

```text
User asks: "Why did we choose the boring tech approach for Kafka?"

Traditional RAG: Returns "Kafka" memories (misses the connection)

AutoMem bridge discovery:
- Seed 1: "Migrated to PostgreSQL for operational simplicity"
- Seed 2: "Evaluating Kafka vs RabbitMQ for message queue"
- Bridge: "Team prefers boring technology—proven, debuggable systems"

AutoMem finds the bridge that connects both decisions
→ Result: AI understands your architectural philosophy, not just isolated choices
```

**Technical details:**

- Graph traversal finds memories connecting multiple seed results
- Ranked by relation strength, temporal relevance, importance
- Configurable with `expand_paths=true` (enabled by default)
- Drives the **37.5% → ongoing improvement** in multi-hop reasoning

### Knowledge Graphs That Evolve

```text
# After storing: "Migrated to PostgreSQL for operational simplicity"

AutoMem automatically:
├── Extracts entities (PostgreSQL, operational simplicity)
├── Auto-tags (entity:tool:postgresql, entity:concept:ops-simplicity)
├── Detects pattern ("prefers boring technology")
├── Links temporally (PRECEDED_BY migration planning)
└── Connects semantically (SIMILAR_TO "Redis deployment")

# Next query: "Should we use Kafka?"
AI recalls:
- PostgreSQL decision
- "Boring tech" pattern (reinforced across memories)
- Operational simplicity preference
→ Suggests: "Based on your pattern, consider RabbitMQ instead"
```

### 9-Component Hybrid Scoring

```bash
GET /recall?query=database%20migration&tags=decision&time_query=last%20month

# AutoMem combines nine signals:
score = vector×0.25       # Semantic similarity
      + keyword×0.15      # TF-IDF text matching
      + relation×0.25     # Graph relationship strength
      + content×0.25      # Direct token overlap
      + temporal×0.15     # Time alignment with query
      + tag×0.10          # Tag matching
      + importance×0.05   # User-assigned priority
      + confidence×0.05   # Memory confidence
      + recency×0.10      # Freshness boost

# Result: Memories ranked by meaning, not just similarity
```

## Features

### Core Memory Operations
- **Store** - Rich memories with metadata, importance, timestamps, embeddings
- **Recall** - Hybrid search (vector + keyword + tags + time windows)
- **Update** - Modify memories, auto-regenerate embeddings
- **Delete** - Remove from both graph and vector stores
- **Associate** - Create typed relationships between memories
- **Filter** - Tag-based queries with prefix/exact matching

## Memory Consolidation

AutoMem uses [neuroscience-inspired](https://pmc.ncbi.nlm.nih.gov/articles/PMC4648295/) consolidation cycles—like human sleep—to keep memories relevant:

| Cycle | Frequency | Purpose |
|-------|-----------|---------|
| **Decay** | Hourly | Exponential relevance scoring (age, access, connections, importance) |
| **Creative** | Hourly | REM-like processing that discovers non-obvious connections |
| **Cluster** | 6 hours | Groups similar memories, generates meta-patterns |
| **Forget** | Daily | Archives low-relevance (<0.2), deletes very old (<0.05) |

**How it works:**
- Wrong rabbit holes fade naturally (~30-45 days without access)
- Important memories with strong connections stay indefinitely
- Memories archive before deletion (0.05-0.2 relevance range)
- The system learns what matters to you, not what you explicitly tag

### Background Intelligence

Every memory gets automatically enhanced in the background (doesn't block your API calls):

**Enrichment Pipeline** (runs immediately after storage):
- **Entity extraction** - Identifies people, projects, tools, concepts (spaCy NLP)
- **Auto-tagging** - Generates `entity:<type>:<slug>` for structured queries
- **Summaries** - Lightweight gist representations for quick scanning
- **Temporal links** - Connects to recent memories with `PRECEDED_BY` relationships
- **Semantic neighbors** - Finds similar memories via cosine similarity (`SIMILAR_TO`)
- **Pattern detection** - Reinforces emerging themes across your memory graph

**Consolidation Engine** (runs on configurable schedules):
- See [Memory Consolidation](#memory-consolidation) section above

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

### Option 1: Railway (Recommended)

Deploy AutoMem + FalkorDB to Railway in 60 seconds:

```bash
# Install Railway CLI
npm i -g @railway/cli

# Deploy
railway login
railway init
railway up
```

👉 **[Deployment Guide](INSTALLATION.md#deployment)** for detailed Railway setup

### Option 2: Docker Compose (Local)

Run everything locally:

```bash
# Clone and start services
git clone https://github.com/verygoodplugins/automem.git
cd automem
make dev

# API: http://localhost:8001
# FalkorDB: localhost:6379
# Qdrant: localhost:6333
```

### Option 3: Development Mode

Run API without Docker:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
PORT=8001 python app.py
```

## API Examples

### Store a Memory
```bash
curl -X POST http://localhost:8001/memory \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Chose PostgreSQL over MongoDB for ACID compliance",
    "type": "Decision",
    "confidence": 0.95,
    "tags": ["database", "architecture"],
    "importance": 0.9,
    "metadata": {
      "source": "architecture-meeting",
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

# Graph expansion with filtering (reduce noise in related memories)
GET /recall?query=auth%20architecture&expand_relations=true&expand_min_importance=0.5&expand_min_strength=0.3

# Multi-hop entity expansion (e.g., "What is Sarah's sister's job?")
GET /recall?query=What%20is%20Sarah%27s%20sister%27s%20job&expand_entities=true
```

### Create Relationship
```bash
curl -X POST http://localhost:8001/associate \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "memory1_id": "uuid-postgres-decision",
    "memory2_id": "uuid-mongodb-evaluation",
    "type": "PREFERS_OVER",
    "strength": 0.9
  }'
```

## Use With AI Platforms

AutoMem works with any AI platform via:

### MCP (Model Context Protocol)

**Local MCP Bridge** (Claude Desktop, Cursor, Claude Code):
```bash
# Install official MCP bridge
npm install -g @verygoodplugins/mcp-automem

# Configure for local AI tools
npx @verygoodplugins/mcp-automem setup
```

**SSE Sidecar** (Cloud AI Platforms):

Connect AutoMem to cloud AI platforms via HTTPS. Works with:
- **ChatGPT** (requires developer mode)
- **Claude.ai** web interface
- **Claude mobile app**
- **ElevenLabs Agents**

See [MCP over SSE documentation](docs/MCP_SSE.md) for setup instructions.

👉 **Resources**:
- NPM bridge (local): https://www.npmjs.com/package/@verygoodplugins/mcp-automem
- SSE setup guide: [docs/MCP_SSE.md](docs/MCP_SSE.md)

### Direct API
Any language, any framework:
```python
import requests

response = requests.post(
    "https://your-automem.railway.app/memory",
    headers={"Authorization": f"Bearer {token}"},
    json={"content": "Memory content", "importance": 0.8}
)
```

## Why AutoMem Beats Everything Else

### vs. Traditional RAG Systems
Traditional RAG retrieves similar documents. AutoMem understands relationships:

**RAG**: "Here are 5 documents about PostgreSQL"
**AutoMem**: "You chose PostgreSQL over MongoDB because you prefer boring technology for operational simplicity. This pattern also influenced your Redis and RabbitMQ decisions."

- ✅ **Typed relationships** - Not just "similar", but "causes", "contradicts", "evolved from"
- ✅ **Temporal awareness** - Knows what preceded, what invalidated, what emerged
- ✅ **Pattern learning** - Discovers your preferences and decision-making style
- ✅ **Consolidation** - Memories strengthen or fade based on use—like human memory

### vs. Vector Databases (Pinecone, Weaviate, Qdrant)
Vector databases match embeddings. AutoMem builds knowledge graphs:

- ✅ **Multi-hop reasoning** - Bridge discovery connects memories across conversation threads
- ✅ **11 relationship types** - Structured semantics vs. cosine similarity alone
- ✅ **Background intelligence** - Auto-enrichment, pattern detection, decay cycles
- ✅ **9-component scoring** - Combines semantic, lexical, graph, temporal, and importance signals

### vs. Building Your Own
AutoMem delivers what took 12 days of iteration to achieve SOTA performance:

- ✅ **Benchmark-proven** - 90.53% on LoCoMo (ACL 2024), beats funded competitors
- ✅ **Research-validated** - Implements HippoRAG 2, A-MEM, MELODI, ReadAgent principles
- ✅ **Production-ready** - Auth, admin tools, health monitoring, automated backups
- ✅ **Battle-tested** - Enrichment pipeline, consolidation engine, retry logic, dual storage
- ✅ **Open source** - MIT license, deploy anywhere, no vendor lock-in

## Benchmark-Proven Performance

### LoCoMo Benchmark Results (December 2, 2025)

**90.53% overall accuracy** across 1,986 questions:

| Category | AutoMem | Notes |
|----------|---------|-------|
| **Complex Reasoning** | **100%** | Perfect score on multi-step reasoning |
| **Open Domain** | **95.84%** | General knowledge recall |
| **Temporal Understanding** | **85.05%** | Time-aware queries |
| **Single-hop Recall** | **79.79%** | Basic fact retrieval |
| **Multi-hop Reasoning** | **50.00%** | Connecting disparate memories (+12.5pp) |

**Comparison:**
- CORE (previous SOTA): 88.24%
- AutoMem: **90.53%** (+2.29 points)
- OpenAI baseline: 39%

Run the benchmark yourself: `make test-locomo`

### Production Characteristics

- ⚡ **Sub-100ms recall** - Even with 100k+ memories
- 🔄 **Concurrent writes** - Background enrichment doesn't block API
- 🛡️ **Graceful degradation** - Works without Qdrant (graph-only mode)
- ♻️ **Automatic retries** - Failed enrichments queue for reprocessing
- 💚 **Health monitoring** - `/health` and `/enrichment/status` endpoints
- 💾 **Dual storage redundancy** - Data persists in both FalkorDB and Qdrant
- 📦 **Automated backups** - Optional backup service for disaster recovery

## Configuration

### Required
- `AUTOMEM_API_TOKEN` - Authentication for all endpoints (except `/health`)
- `FALKORDB_HOST` / `FALKORDB_PORT` - Graph database connection

### Optional
- `QDRANT_URL` / `QDRANT_API_KEY` - Enable semantic search
- `OPENAI_API_KEY` - Real embeddings (otherwise deterministic placeholders)
- `ADMIN_API_TOKEN` - Required for `/admin/reembed` and enrichment controls
- Consolidation tuning: `CONSOLIDATION_*_INTERVAL_SECONDS`
- Enrichment tuning: `ENRICHMENT_*` (similarity threshold, retry limits, etc.)

👉 **[Full Configuration Guide](INSTALLATION.md#configuration)**

## Documentation

- 📦 **[Installation Guide](INSTALLATION.md)** - Railway, Docker, development setup
- 🌉 **[MCP over SSE Sidecar](docs/MCP_SSE.md)** - Expose AutoMem as an MCP server over SSE for ChatGPT/ElevenLabs
- 💾 **[Monitoring & Backups](docs/MONITORING_AND_BACKUPS.md)** - Health monitoring and automated backups
- 🔧 **[API Reference](docs/API.md)** - All endpoints with examples
- 🧪 **[Testing Guide](docs/TESTING.md)** - Unit, integration, live server, and LoCoMo benchmark tests
- 📊 **[LoCoMo Benchmark](docs/TESTING.md#locomo-benchmark)** - Validate against ACL 2024 long-term memory benchmark
- 🔄 **[Migration Guide](INSTALLATION.md#migration)** - Move from MCP SQLite
- 🌐 **[automem.ai](https://automem.ai)** - Official website and guides

## Community & Support

- 🌐 **[automem.ai](https://automem.ai)** - Official website
- 🐙 **[GitHub](https://github.com/verygoodplugins/automem)** - Source code
- 📦 **[NPM MCP Bridge](https://www.npmjs.com/package/@verygoodplugins/mcp-automem)** - MCP integration
- 🐛 **[Issues](https://github.com/verygoodplugins/automem/issues)** - Bug reports and feature requests

## The Science Behind SOTA

AutoMem's **90.53% LoCoMo score** didn't come from hype—it came from implementing peer-reviewed neuroscience and graph theory:

### [HippoRAG 2](https://arxiv.org/abs/2502.14802) (Ohio State, June 2025)
**Finding**: Graph-vector hybrid achieves 7% better associative memory than pure vector RAG, approaching human long-term memory performance.

**AutoMem implementation**: Dual FalkorDB (graph) + Qdrant (vector) architecture with 11 typed relationship edges.

### [A-MEM](https://arxiv.org/abs/2502.12110) (July 2025)
**Finding**: Dynamic memory organization with Zettelkasten principles enables emergent knowledge structures.

**AutoMem implementation**: Pattern detection, clustering cycles, and automatic entity linking that builds knowledge graphs from conversation.

### [MELODI](https://arxiv.org/html/2410.03156v1) (DeepMind, 2024)
**Finding**: 8x memory compression without quality loss through gist representations and selective preservation.

**AutoMem implementation**: Summary generation, importance scoring, and consolidation cycles that strengthen relevant memories while fading noise.

### [ReadAgent](https://arxiv.org/abs/2402.09727) (DeepMind, 2024)
**Finding**: 20x context extension via episodic memory and temporal organization.

**AutoMem implementation**: Temporal relationship types (PRECEDED_BY, OCCURRED_BEFORE) and time-aware scoring that preserves conversation flow.

---

**We didn't just read the papers. We built the system they describe—then validated it on academic benchmarks.**

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for your changes
4. Submit a pull request

See [TESTING.md](TESTING.md) for running the test suite.

## License

MIT - Because AI memory should be free.

---

## Deploy State-of-the-Art Memory in 60 Seconds

```bash
railway up
```

**AutoMem is the world's best-performing long-term memory system.**
90.53% LoCoMo score. Open source. Research-validated. Production-ready.

Transform your AI from a chatbot into a thinking partner that **actually remembers.**

---

*Built by a solo developer. Validated by academic benchmarks. Beats well-funded competitors.*
*MIT License. Deploy anywhere. No vendor lock-in.*
