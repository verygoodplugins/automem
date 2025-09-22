# AutoMem Onboarding Guide for Developers

Welcome to AutoMem! This guide will get you up to speed with the project architecture, development workflow, and immediate priorities.

## 🎯 Project Mission

Build a personal AI memory system that learns from all digital interactions to enable AI assistants that work exactly like the user would. The system should capture decisions, patterns, preferences, and styles to eventually simulate the user's decision-making with high fidelity.

**Long-term Vision**: AutoMem is the first component in a distributed automation hub architecture. We're establishing patterns for deploying single-purpose microservices on Railway that communicate via standardized protocols. The memory service patterns and protocols developed here will be reused throughout the broader automation ecosystem.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────┐
│     Claude Desktop / Claude Code / Cursor    │
└────────────┬──────────────┬─────────────────┘
             │ MCP Protocol  │ 
             ▼               │
┌─────────────────────────┐  │
│   MCP AutoMem Server    │  │ Direct REST API
│  (@verygoodplugins/     │  │
│    mcp-automem)         │  │
└────────────┬────────────┘  │
             │ HTTP          │
             ▼               ▼
┌─────────────────────────────────────────────┐
│          AutoMem Service (Flask)             │
│    https://automem.up.railway.app            │
├─────────────────────────────────────────────┤
│ • 11 REST endpoints                          │
│ • Authentication via Bearer tokens           │
│ • Consolidation scheduler                    │
│ • PKG enrichment pipeline                    │
└────────┬──────────────────┬─────────────────┘
         │                  │
    ┌────▼─────┐      ┌────▼─────────────────┐
    │ FalkorDB │      │  Qdrant Cloud        │
    │  (Graph) │      │  (Vector Search)     │
    │  Railway │      │  cloud.qdrant.io     │
    └──────────┘      └──────────────────────┘
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone and enter project
cd /Users/jgarturo/Projects/OpenAI/automem

# Create virtual environment
make install
source venv/bin/activate

# Copy environment template
cp .env.example .env

# Edit .env with required values:
# - OPENAI_API_KEY (for real embeddings)
# - AUTOMEM_API_TOKEN (for API auth)
# - ADMIN_API_TOKEN (for admin endpoints)
```

### 2. Local Development

```bash
# Start full stack with Docker
make dev

# This runs:
# - FalkorDB on localhost:6379 (UI on :3000)
# - Qdrant on localhost:6333
# - Flask API on localhost:8001

# In another terminal, run tests
make test

# View logs
make logs
```

### 3. Production Access

```bash
# AutoMem Service (Railway - Live and Working!)
https://automem.up.railway.app

# Test health endpoint (no auth required):
curl https://automem.up.railway.app/health

# API endpoints (require authentication):
curl -H "Authorization: Bearer YOUR_TOKEN" \
     https://automem.up.railway.app/recall?query=test
```

#### Service Dashboards (You've Been Invited!)

1. **Railway Dashboard** (AutoMem Service):
   - URL: https://railway.app
   - Status: ✅ Live and running
   - Access: Sorin has been invited
   - Environment variables managed here

2. **Qdrant Cloud Dashboard**:
   - URL: https://cloud.qdrant.io/accounts/fda24ffc-3475-42da-93d6-4e0205e008b1/cloud-access/roles/all-users
   - Access: Sorin has been invited
   - Collection: `memories`
   - API key in Railway environment

3. **MCP Server** (for local AI clients):
   - GitHub: https://github.com/verygoodplugins/mcp-automem
   - NPM Package: `@verygoodplugins/mcp-automem`
   - Local repo: `/Users/jgarturo/Projects/OpenAI/mcp-servers/mcp-automem/`

## 📚 Key Concepts

### Memory Structure

Each memory contains:
```python
{
    "id": "uuid",                    # Unique identifier
    "content": "text",                # The actual memory
    "timestamp": "ISO-8601",          # When created
    "importance": 0.0-1.0,            # User-defined priority
    "tags": ["tag1", "tag2"],        # Categories
    "metadata": {                    # Flexible properties
        "source": "automation-hub",
        "entities": {...}
    },
    "embedding": [768 floats],       # Vector for similarity
    "memory_type": "decision",       # Classification
    "type_confidence": 0.85          # Classification confidence
}
```

### Memory Types

- **Decision**: Explicit choices with rationale
- **Pattern**: Recurring behaviors
- **Preference**: Likes and favorites
- **Style**: Communication/coding patterns
- **Habit**: Regular practices
- **Insight**: Discoveries and learnings
- **Context**: Environmental information
- **Memory**: Default/unclassified

### Relationship Types

Memories connect via typed edges:
- `RELATES_TO`: General connection
- `LEADS_TO`: Causal relationship
- `PREFERS_OVER`: Preference ranking
- `CONTRADICTS`: Conflicting information
- `REINFORCES`: Strengthens patterns
- `EVOLVED_INTO`: Knowledge evolution
- [See app.py for full list]

## 🛠️ Development Workflow

### Adding a New Feature

1. **Write tests first** (TDD approach):
```python
# tests/test_new_feature.py
def test_my_feature():
    # Test implementation
```

2. **Implement in app.py**:
```python
@app.route('/new-endpoint', methods=['POST'])
@require_auth
def new_feature():
    # Implementation
```

3. **Update documentation**:
- Add to README.md API section
- Update CLAUDE.md for AI assistance
- Add example to this onboarding doc

4. **Test locally**:
```bash
make dev
pytest tests/test_new_feature.py -v
```

5. **Deploy**:
```bash
git add -A
git commit -m "feat: add new feature"
git push origin main
# Railway auto-deploys from main
```

## 🔌 MCP Integration (AI Assistant Access)

The MCP server enables Claude Desktop, Claude Code, and Cursor to interact with AutoMem memories directly.

### Setting Up MCP Server

#### For Claude Desktop
```json
// ~/Library/Application Support/Claude/claude_desktop_config.json
{
  "mcpServers": {
    "automem": {
      "command": "npx",
      "args": ["@verygoodplugins/mcp-automem"],
      "env": {
        "AUTOMEM_ENDPOINT": "https://automem.up.railway.app",
        "AUTOMEM_API_KEY": "get-from-railway-dashboard"
      }
    }
  }
}
```

#### For Claude Code
```bash
# Install via command line
claude mcp add automem "npx @verygoodplugins/mcp-automem"

# Configure endpoint
export AUTOMEM_ENDPOINT="https://automem.up.railway.app"
export AUTOMEM_API_KEY="your-api-key"
```

### MCP Tools Available
- `store_memory` - Save memories with tags and importance
- `recall_memory` - Search memories by text or semantic similarity
- `associate_memories` - Create relationships between memories
- `check_database_health` - Monitor service status

### Local MCP Development
```bash
cd /Users/jgarturo/Projects/OpenAI/mcp-servers/mcp-automem
npm install
npm run dev  # Watch mode with auto-reload

# Test with local AutoMem
export AUTOMEM_ENDPOINT="http://127.0.0.1:8001"
npm test
```

## 🏛️ Architectural Patterns (Foundation for Automation Hub)

AutoMem establishes key patterns for the broader automation ecosystem:

### Microservice Design Principles
1. **Single Responsibility** - Each service does one thing well (AutoMem = memory management)
2. **Stateless API** - All state in databases, not in service
3. **Graceful Degradation** - Service continues if optional dependencies fail
4. **Standardized Auth** - Bearer token pattern reusable across services
5. **Health Monitoring** - `/health` endpoint standard for all services

### Inter-Service Communication (Future)
```python
# Proposed standard protocol for service discovery
SERVICE_REGISTRY = {
    "memory": "https://automem.up.railway.app",
    "notifications": "https://notify.up.railway.app",  # Future
    "scheduler": "https://scheduler.up.railway.app",   # Future
    "nlp": "https://nlp.up.railway.app"               # Future
}

# Standard message format
{
    "service": "memory",
    "action": "store",
    "payload": {...},
    "metadata": {
        "source": "automation-hub",
        "timestamp": "ISO-8601",
        "correlation_id": "uuid"
    }
}
```

### Deployment Pattern
- Each service in its own Railway project
- Environment variables for service discovery
- Internal Railway networking for zero-latency communication
- Shared databases where appropriate (FalkorDB for all graph needs)

## 🎯 Immediate Priorities

### Week 1: Testing & Stability
- [ ] Add integration tests for graph traversal
- [ ] Test consolidation engine thoroughly
- [ ] Add performance benchmarks
- [ ] Document test coverage goals

### Week 2: Security & Robustness
- [ ] Implement rate limiting
- [ ] Add request validation middleware
- [ ] Set up logging/monitoring
- [ ] Create backup/restore scripts

### Week 3: Integration
- [ ] Complete MCP bridge for Claude Desktop
- [ ] Connect WhatsApp notification system
- [ ] Add automation hub webhooks
- [ ] Test end-to-end flows

### Week 4: Intelligence Layer
- [ ] Implement decision pattern extraction
- [ ] Add communication style learning
- [ ] Build cross-domain correlations
- [ ] Create pattern visualization

## 📊 Testing Strategy

### Unit Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_app.py -v

# Run with coverage
pytest --cov=. --cov-report=html
```

### Integration Tests (TODO)
```python
# Test graph operations
def test_graph_traversal():
    # Store connected memories
    # Query relationships
    # Verify traversal

# Test consolidation
def test_consolidation_cycle():
    # Create memories
    # Run consolidation
    # Verify changes
```

### Performance Tests (TODO)
```python
# Benchmark recall speed
def test_recall_performance():
    # Load 10K memories
    # Time various queries
    # Assert < 100ms
```

## 🔧 Common Tasks

### Store a Memory
```python
import requests

response = requests.post(
    "http://localhost:8001/memory",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    json={
        "content": "Decided to use FalkorDB over ArangoDB",
        "tags": ["decision", "database"],
        "importance": 0.9,
        "metadata": {
            "reasoning": "Better performance, lower cost"
        }
    }
)
```

### Recall Memories
```python
# Semantic search
response = requests.get(
    "http://localhost:8001/recall",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    params={
        "query": "database decisions",
        "limit": 10
    }
)

# Time-based recall
response = requests.get(
    "http://localhost:8001/recall",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    params={
        "time_query": "last week",
        "tags": "decision"
    }
)
```

### Trigger Consolidation
```python
response = requests.post(
    "http://localhost:8001/consolidate",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    json={
        "mode": "creative",  # or "decay", "cluster", "forget"
        "dry_run": True      # Test without changes
    }
)
```

## 🐛 Debugging Tips

### Check Logs
```bash
# Local development
make logs

# Production (Railway)
railway logs
```

### Common Issues

1. **"FalkorDB is unavailable"**
   - Check Docker is running
   - Verify FALKORDB_HOST in .env

2. **"Embedding must contain exactly 768 values"**
   - Either provide full vector or omit field
   - Check OPENAI_API_KEY if using real embeddings

3. **"401 Unauthorized"**
   - Verify AUTOMEM_API_TOKEN matches
   - Check header format: "Authorization: Bearer TOKEN"

## 📈 Monitoring

### Health Check
```bash
# Should return all services as "connected"
curl http://localhost:8001/health
```

### Consolidation Status
```bash
curl -H "Authorization: Bearer TOKEN" \
     http://localhost:8001/consolidate/status
```

### Graph Statistics
```bash
curl -H "Authorization: Bearer TOKEN" \
     http://localhost:8001/analyze
```

## 🚢 Deployment

### Railway Deployment (Production)

The project auto-deploys from the main branch to Railway:

1. **Push to main**: Triggers automatic deployment
2. **Monitor deployment**: 
   ```bash
   # Check logs
   railway logs
   
   # Or use Railway dashboard (you have access!)
   https://railway.app
   ```
3. **Rollback**: Use Railway dashboard if needed

### Environment Variables

Managed in Railway dashboard (never commit to git!):
- `AUTOMEM_API_TOKEN` - API authentication
- `ADMIN_API_TOKEN` - Admin operations
- `OPENAI_API_KEY` - For embeddings
- `QDRANT_URL` - Cloud Qdrant endpoint
- `QDRANT_API_KEY` - Qdrant authentication
- `FALKORDB_*` - Graph database settings

Local development uses `.env` file (not committed).

### Service URLs
- **Production API**: https://automem.up.railway.app
- **Health Check**: https://automem.up.railway.app/health
- **Railway Dashboard**: https://railway.app (you have access)
- **Qdrant Cloud**: https://cloud.qdrant.io (you have access)

## 📞 Communication & Support

### Questions?
1. Check existing documentation (README, CLAUDE.md, this guide)
2. Review test files for examples
3. Ask Jack via WhatsApp bridge (when connected)
4. Use Claude Desktop with memory context

### Key Resources Summary
- **Production Service**: https://automem.up.railway.app
- **Railway Dashboard**: https://railway.app (you have access)
- **Qdrant Cloud**: https://cloud.qdrant.io (you have access)
- **MCP Server GitHub**: https://github.com/verygoodplugins/mcp-automem
- **Local Repos**:
  - AutoMem: `/Users/jgarturo/Projects/OpenAI/automem`
  - MCP Server: `/Users/jgarturo/Projects/OpenAI/mcp-servers/mcp-automem`

### Reporting Issues
1. Create detailed bug report with:
   - Steps to reproduce
   - Expected vs actual behavior
   - Relevant logs
   - Test case if possible

## 🎓 Learning Resources

### Project Documentation
- `README.md` - Project overview and API reference
- `CLAUDE.md` - AI assistance context
- `full-scope.md` - Complete vision and roadmap
- `CONSOLIDATION_FEATURES.md` - Memory processing details
- `PKG_FEATURES.md` - Personal Knowledge Graph capabilities
- `ideas.md` - Future architecture plans

### Code Examples
- `scripts/consolidation_demo.py` - Consolidation examples
- `scripts/recall_performance_demo.py` - Query patterns
- `tests/` - Test implementations

### External Resources
- [FalkorDB Documentation](https://docs.falkordb.com/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Flask Best Practices](https://flask.palletsprojects.com/)
- [Railway Deployment Guide](https://docs.railway.app/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [MCP AutoMem NPM Package](https://www.npmjs.com/package/@verygoodplugins/mcp-automem)

## 🚀 Production Troubleshooting

### Railway Issues
- **Check deployment status**: Railway dashboard → Deployments tab
- **View logs**: `railway logs` or dashboard → Logs tab  
- **Restart service**: Dashboard → Settings → Restart
- **Environment variables**: Dashboard → Variables tab

### Qdrant Cloud Issues
- **Check cluster status**: Qdrant dashboard → Clusters
- **API key issues**: Regenerate in dashboard → API Keys
- **Collection missing**: AutoMem auto-creates on first write

### MCP Connection Issues
- **Claude not finding memories**: Check MCP config file location
- **Authentication errors**: Verify `AUTOMEM_API_KEY` matches Railway
- **Timeout errors**: Ensure production service is running

## 🏁 Next Steps

1. **Access Production Services**
   - Log into Railway dashboard (check invite email)
   - Access Qdrant Cloud dashboard (check invite email)
   - Test production health endpoint: `curl https://automem.up.railway.app/health`

2. **Get the code running locally**
   ```bash
   cd /Users/jgarturo/Projects/OpenAI/automem
   make dev  # Starts local FalkorDB + Qdrant + API
   ```

3. **Test MCP integration**
   ```bash
   # Install MCP server globally
   npm install -g @verygoodplugins/mcp-automem
   
   # Configure Claude Desktop (see MCP Integration section)
   # Test by asking Claude to store and recall memories
   ```

4. **Run the test suite** 
   ```bash
   make test  # Understand current coverage
   ```

5. **Make your first contribution**
   - Pick a small security or monitoring task
   - Write tests first (TDD)
   - Submit PR with documentation
   - Deploy automatically on merge to main

6. **Study the patterns**
   - Review how AutoMem handles service degradation
   - Understand the consolidation scheduler design
   - Learn the authentication middleware pattern
   - These patterns will be reused in future services

Welcome aboard! You're now part of building a system that will fundamentally change how AI assistants understand and emulate human behavior. The goal is ambitious: create an AI that knows you so well it can make decisions exactly as you would.

Let's build something amazing! 🚀
