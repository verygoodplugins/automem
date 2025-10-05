# AutoMem: Research-Validated AI Memory 🧠

**Graph + Vector architecture proven to match human long-term memory performance.**

```bash
# Deploy in 60 seconds
railway up
# Or run locally
make dev
```

Persistent memory for your AI.

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

## Architecture

```
┌─────────────────────────────────────────────┐
│           AutoMem Service (Flask)           │
│   • REST API for memory lifecycle           │
│   • Background enrichment pipeline          │
│   • Consolidation engine                    │
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

**FalkorDB** (graph) = canonical record, relationships, consolidation  
**Qdrant** (vectors) = semantic recall, similarity search

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
- **Store** - Rich memories with metadata, importance, timestamps, embeddings
- **Recall** - Hybrid search (vector + keyword + tags + time windows)
- **Update** - Modify memories, auto-regenerate embeddings
- **Delete** - Remove from both graph and vector stores
- **Associate** - Create typed relationships between memories
- **Filter** - Tag-based queries with prefix/exact matching

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
    "content": "Chose PostgreSQL over MongoDB for team expertise",
    "tags": ["decision", "database"],
    "importance": 0.9,
    "metadata": {
      "source": "architecture-meeting",
      "entities": {"tools": ["PostgreSQL", "MongoDB"]}
    }
  }'
```

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
```bash
# Install official MCP bridge
npm install -g @verygoodplugins/mcp-automem

# Configure for Claude Desktop, Cursor, or Claude Code
npx @verygoodplugins/mcp-automem setup
```

👉 **[MCP Integration Guide](https://www.npmjs.com/package/@verygoodplugins/mcp-automem)**

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

## What Makes AutoMem Different

### vs. Traditional RAG
- ✅ **Relationships** - Not just "similar", but "causes", "prefers", "invalidates"
- ✅ **Temporal awareness** - Knows what came before, what evolved from what
- ✅ **Pattern learning** - Discovers themes across memories
- ✅ **Consolidation** - Memories improve over time, not just accumulate

### vs. Vector-Only Databases
- ✅ **Structured relationships** - 11 edge types vs cosine similarity only
- ✅ **Background intelligence** - Auto-enrichment, clustering, decay
- ✅ **Hybrid scoring** - Vector + keyword + tags + time + importance
- ✅ **Knowledge graphs** - Traverse relationships, not just retrieve vectors

### vs. Building Your Own
- ✅ **Research-validated** - Implements HippoRAG 2, A-MEM, MELODI principles
- ✅ **Production-ready** - Authentication, admin tools, health monitoring
- ✅ **Battle-tested** - Enrichment pipeline, consolidation, retry logic
- ✅ **Open source** - MIT license, deploy anywhere

## Performance

- **Sub-second recall** - Even with 100k+ memories
- **Concurrent writes** - Background enrichment doesn't block API
- **Graceful degradation** - Works without Qdrant (graph-only mode)
- **Automatic retries** - Failed enrichments queue for reprocessing
- **Health monitoring** - `/health` and `/enrichment/status` endpoints

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
- 🔧 **[API Reference](INSTALLATION.md#api-reference)** - All endpoints with examples
- 🧪 **[Testing Guide](TESTING.md)** - Unit, integration, and live server tests
- 🔄 **[Migration Guide](INSTALLATION.md#migration)** - Move from MCP SQLite
- 🌐 **[automem.ai](https://automem.ai)** - Official website and guides

## Community & Support

- 🌐 **[automem.ai](https://automem.ai)** - Official website
- 🐙 **[GitHub](https://github.com/verygoodplugins/automem)** - Source code
- 📦 **[NPM MCP Bridge](https://www.npmjs.com/package/@verygoodplugins/mcp-automem)** - MCP integration
- 🐛 **[Issues](https://github.com/verygoodplugins/automem/issues)** - Bug reports and feature requests

## The Science

AutoMem's architecture is validated by peer-reviewed research:

**HippoRAG 2** (Ohio State, June 2025)  
Proves graph-vector hybrid achieves 7% better associative memory than pure vector RAG, approaching human long-term memory performance.

**A-MEM** (July 2025)  
Validates dynamic memory organization with Zettelkasten-inspired principles - exactly what AutoMem's pattern detection and clustering implement.

**MELODI** (DeepMind, 2025)  
Shows 8x memory compression without quality loss through gist representations - AutoMem's summary generation follows these principles.

**ReadAgent** (DeepMind, 2024)  
Demonstrates 20x context extension via episodic memory - AutoMem's consolidation engine implements similar temporal organization.

We didn't just read the papers - we built the system they describe.

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

**Ready to give your AI human-like memory?**

```bash
railway up
```

*Built with obsession. Validated by neuroscience. Powered by graph theory.*

**Transform AI from a tool into a thinking partner. Deploy AutoMem now.**

