# 🧠 AutoMem - Personal AI Memory Service

A cloud-native memory system that gives AI assistants persistent, searchable memory with relationship tracking. Built to integrate with the Claude Automation Hub and provide a foundation for truly personalized AI interactions.

## 🎯 Project Vision

Create a memory service that enables AI assistants to:
- **Remember** conversations, decisions, and patterns across sessions
- **Learn** your communication style, preferences, and decision-making patterns
- **Associate** related memories through graph relationships with varying strengths
- **Consolidate** memories using neurobiologically-inspired algorithms
- **Scale** from personal use to millions of memories without degradation

The ultimate goal: AI that works exactly like you would, maintaining context across all interactions.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│     Client Applications (Claude, Automation Hub) │
└────────────┬───────────────────┬────────────────┘
             │    REST API        │
             ▼                    ▼
┌─────────────────────────────────────────────────┐
│           Memory Service API (Flask)             │
│                  Port: 8000                      │
└────────────┬───────────────────┬────────────────┘
             │                    │
      ┌──────▼──────┐      ┌─────▼──────┐
      │  FalkorDB   │      │   Qdrant   │
      │   (Graph)   │      │  (Vectors) │
      ├─────────────┤      ├────────────┤
      │ • Relations │      │ • Semantic │
      │ • Patterns  │      │ • 768-dim  │
      │ • Decisions │      │ • Similarity│
      │ Port: 6379  │      │ Cloud API  │
      └─────────────┘      └────────────┘
```

## ✅ Completed

### Infrastructure
- [x] FalkorDB deployed on Railway (graph database for relationships)
- [x] Redis 7.4 with FalkorDB module v4.12.5
- [x] 48 thread pool for high-performance graph operations
- [x] Qdrant Cloud cluster configured (vector search)
- [x] Railway project initialized and connected
- [x] Docker containerization working

### Core Decisions
- [x] Chose FalkorDB over ArangoDB (200x faster, simpler deployment, $5/mo vs $150/mo)
- [x] Hybrid architecture design (local + cloud sync)
- [x] Separated memory service from automation hub for modularity

## 🚧 In Progress

### Memory API Development
- [ ] Complete Flask API with endpoints:
  - [ ] `/memory` - Store new memories
  - [ ] `/recall` - Retrieve memories (semantic + graph)
  - [ ] `/associate` - Create relationships between memories
  - [ ] `/consolidate` - Trigger memory consolidation
- [ ] Integration with Qdrant for vector storage
- [ ] Graph traversal algorithms for pattern recognition

### Consolidation Engine
- [ ] Micro-consolidation every 5 minutes (importance > 0.7)
- [ ] Nightly batch consolidation (dream phase)
- [ ] Pattern recognition across memory domains
- [ ] Strength adjustment for associations

## 📋 TODO

### Phase 1: Core Functionality (Week 1)
- [ ] Implement embedding generation (OpenAI/local model)
- [ ] Create memory scoring algorithm (recency, frequency, importance)
- [ ] Build graph traversal queries for related memories
- [ ] Add authentication/API keys
- [ ] Set up health monitoring

### Phase 2: Intelligence Layer (Week 2-3)
- [ ] Decision pattern recognition
- [ ] Communication style extraction
- [ ] Temporal pattern analysis
- [ ] Cross-domain correlation engine
- [ ] Memory pruning for efficiency

### Phase 3: Integration (Week 4)
- [ ] MCP bridge for Claude Desktop
- [ ] Automation Hub webhooks
- [ ] WhatsApp notification system
- [ ] Backup/restore functionality
- [ ] Memory export tools

### Phase 4: Advanced Features (Month 2)
- [ ] Predictive memory loading
- [ ] Speculative execution
- [ ] Memory compression
- [ ] Multi-user support
- [ ] Privacy controls

## 🚀 Quick Start

### Prerequisites
- Railway account with CLI installed
- Qdrant Cloud account (free tier works)
- Python 3.8+
- Docker (for local testing)

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/automem.git
cd automem

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export QDRANT_URL="https://your-cluster.qdrant.io"
export QDRANT_API_KEY="your-api-key"
export RAILWAY_STATIC_URL="localhost"  # for local testing

# Run locally with Docker
docker run -p 6379:6379 -p 3000:3000 falkordb/falkordb:latest

# Start the API
python app.py
```

### Deployment

```bash
# Deploy to Railway
railway up

# Set production variables
railway variables set QDRANT_URL=your-qdrant-url
railway variables set QDRANT_API_KEY=your-api-key

# Get your public URL
railway domain
```

## 📊 Memory Schema

### FalkorDB (Graph Structure)
```cypher
(:Memory {
  id: String,           // UUID
  content: String,      // Memory text
  timestamp: DateTime,  // When created
  importance: Float,    // 0.0-1.0 score
  tags: [String]        // Categories
})

-[:RELATES_TO {strength: Float}]->  // Association strength
-[:LEADS_TO {confidence: Float}]->  // Decision chains
-[:OCCURRED_BEFORE]->                // Temporal relationships
```

### Qdrant (Vector Storage)
```json
{
  "id": "uuid",
  "vector": [768 dimensions],
  "payload": {
    "content": "memory text",
    "tags": ["tag1", "tag2"],
    "importance": 0.8,
    "timestamp": "2025-09-16T00:00:00Z"
  }
}
```

## 🔌 API Endpoints

### Store Memory
```bash
POST /memory
{
  "content": "Important decision about...",
  "embedding": [768 floats],
  "tags": ["decision", "architecture"],
  "importance": 0.9
}
```

### Recall Memories
```bash
GET /recall?query=architecture&limit=10
# Returns semantically similar + graph-related memories
```

### Create Association
```bash
POST /associate
{
  "memory1_id": "uuid-1",
  "memory2_id": "uuid-2",
  "strength": 0.8,
  "type": "RELATES_TO"
}
```

## 📈 Performance Targets

- **Storage**: 1M+ memories
- **Retrieval**: <50ms latency (cloud), <5ms (local)
- **Consolidation**: 5-minute micro-cycles, nightly batch
- **Graph traversal**: 3-depth in <100ms
- **Vector search**: Top-10 in <20ms

## 💰 Cost Analysis

### Current (FalkorDB + Qdrant Cloud)
- FalkorDB on Railway: ~$5/month
- Qdrant Cloud (free tier): $0/month (up to 1GB)
- Total: ~$5/month for personal use

### Scaling Costs
- 100K memories: ~$5/month
- 1M memories: ~$20/month
- 10M memories: ~$50/month

## 🧪 Testing

### Test FalkorDB Connection
```python
from falkordb import FalkorDB
db = FalkorDB(host='your-railway-url', port=6379)
graph = db.select_graph('memories')
print("Connected!")
```

### Test Memory Storage
```python
# Store a memory
result = graph.query("""
    CREATE (m:Memory {
        id: 'test-001',
        content: 'Test memory',
        timestamp: datetime(),
        importance: 0.5
    }) RETURN m
""")
```

### Test Qdrant Integration
```python
from qdrant_client import QdrantClient
client = QdrantClient(
    url="https://your-cluster.qdrant.io",
    api_key="your-api-key"
)
print(client.get_collections())
```

## 🔧 Configuration

### Environment Variables
```env
# Railway
RAILWAY_STATIC_URL=your-app.up.railway.app
RAILWAY_PRIVATE_DOMAIN=your-app.railway.internal
PORT=8001

# Qdrant
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key-here
QDRANT_COLLECTION=memories

# Optional
OPENAI_API_KEY=sk-...  # For embeddings
CONSOLIDATION_INTERVAL=300  # 5 minutes
IMPORTANCE_THRESHOLD=0.7
```

## 📝 Memory Types

The system recognizes several memory patterns:

### Decision Memories
```python
{
  "type": "decision",
  "content": "Chose FalkorDB over ArangoDB",
  "factors": ["cost", "performance", "simplicity"],
  "outcome": "successful",
  "importance": 0.9
}
```

### Knowledge Memories
```python
{
  "type": "knowledge",
  "content": "FalkorDB uses sparse matrices",
  "domain": "database",
  "confidence": 0.95,
  "source": "documentation"
}
```

### Pattern Memories
```python
{
  "type": "pattern",
  "content": "User prefers morning coding sessions",
  "frequency": "daily",
  "confidence": 0.8,
  "evidence_count": 15
}
```

## 🤝 Integration Points

### Claude Desktop (MCP)
```javascript
// MCP configuration
{
  "memory": {
    "command": "node",
    "args": ["bridge.js"],
    "env": {
      "MEMORY_API": "https://automem.up.railway.app"
    }
  }
}
```

### Automation Hub
```python
# In automation hub
from memory_client import MemoryClient
memory = MemoryClient(api_url="https://automem.up.railway.app")
memory.store("Automation completed successfully")
```

### WhatsApp Bridge
```python
# Auto-capture WhatsApp decisions
@on_message_sent
def capture_decision(message):
    if "decided" in message.lower():
        memory.store(message, tags=["whatsapp", "decision"])
```

## 🔐 Security

- [ ] API authentication via Bearer tokens
- [ ] End-to-end encryption for sensitive memories
- [ ] Rate limiting (100 req/min per client)
- [ ] GDPR compliance (right to forget)
- [ ] Automatic PII detection and masking

## 📚 References

- [FalkorDB Documentation](https://docs.falkordb.com)
- [Qdrant Documentation](https://qdrant.tech/documentation)
- [Railway Deployment Guide](https://docs.railway.app)
- [MCP Protocol Spec](https://modelcontextprotocol.io)

## 🐛 Troubleshooting

### FalkorDB won't start
```bash
# Check logs
railway logs

# Verify Redis is running
redis-cli ping

# Check module is loaded
redis-cli MODULE LIST
```

### Qdrant connection failed
```bash
# Test with curl
curl https://your-cluster.qdrant.io/collections \
  -H "api-key: your-api-key"
```

### Memory API errors
```bash
# Check health endpoint
curl https://your-app.up.railway.app:8001/health
```

## 👥 Contributors

- Jack Arturo - Initial architecture and implementation

## 📄 License

MIT License - Use freely for personal AI memory systems

---

**Remember**: The more memories you store, the smarter your AI becomes! 🚀