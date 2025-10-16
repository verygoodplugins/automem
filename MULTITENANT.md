# AutoMem Multi-Tenant Setup

This branch adds multi-tenancy to AutoMem so you can offer it as a managed service.

## What's Been Added

- **`tenant_manager.py`** - Tenant lifecycle management, authentication, isolation
- **`app_multitenant.py`** - Multi-tenant Flask app with admin endpoints
- **`test_multitenant.sh`** - Test script to verify everything works

## Quick Test (Local)

```bash
# 1. Start services (FalkorDB + Qdrant)
docker-compose up -d

# 2. Set admin token
export ADMIN_API_TOKEN="your-super-secret-admin-token"

# 3. Run multi-tenant app
python app_multitenant.py

# 4. Test in another terminal
./test_multitenant.sh
```

## Deploy to Railway

### Option 1: Environment Variables (Easiest)
```bash
# Railway dashboard → Environment Variables:
ADMIN_API_TOKEN=your-super-secret-admin-token
FALKORDB_HOST=${{FALKORDB_HOST}}
FALKORDB_PORT=6379
QDRANT_URL=${{QDRANT.QDRANT_URL}}
QDRANT_API_KEY=${{QDRANT.QDRANT_API_KEY}}
OPENAI_API_KEY=your-openai-key

# Change start command to use multitenant app:
python app_multitenant.py
```

### Option 2: Modify Dockerfile
```dockerfile
# Change CMD in Dockerfile:
CMD ["python", "app_multitenant.py"]
```

Then deploy:
```bash
railway up
```

## Usage Examples

### Create Tenants
```bash
# Create Fernando's tenant
curl -X POST https://your-automem.railway.app/admin/tenants \
  -H "Authorization: Bearer your-super-secret-admin-token" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "fernando",
    "name": "Fernando Client",
    "api_token": "fernando_unique_secure_token_here"
  }'

# Create Claudio's tenant  
curl -X POST https://your-automem.railway.app/admin/tenants \
  -H "Authorization: Bearer your-super-secret-admin-token" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "claudio",
    "name": "Claudio Client",
    "api_token": "claudio_unique_secure_token_here"
  }'
```

### Customers Use Their Tokens
```bash
# Fernando stores a memory (isolated to his namespace)
curl -X POST https://your-automem.railway.app/memory \
  -H "Authorization: Bearer fernando_unique_secure_token_here" \
  -H "Content-Type: application/json" \
  -d '{"content": "My project uses PostgreSQL"}'

# Claudio stores a memory (completely isolated from Fernando)
curl -X POST https://your-automem.railway.app/memory \
  -H "Authorization: Bearer claudio_unique_secure_token_here" \
  -H "Content-Type: application/json" \
  -d '{"content": "My project uses MongoDB"}'
```

### Monitor Usage
```bash
# List all tenants
curl https://your-automem.railway.app/admin/tenants?include_stats=true \
  -H "Authorization: Bearer your-super-secret-admin-token"

# Get specific tenant stats
curl https://your-automem.railway.app/admin/tenants/fernando/stats \
  -H "Authorization: Bearer your-super-secret-admin-token"

# Response:
{
  "tenant_id": "fernando",
  "memory_count": 156,
  "graph_nodes": 156,
  "graph_relationships": 89,
  "vector_points": 156,
  "storage_mb": 2.3
}
```

## Tenant Isolation

Each tenant gets:
- **Separate FalkorDB graph database**: `memories_fernando`, `memories_claudio`
- **Separate Qdrant collection**: `memories_fernando`, `memories_claudio`
- **Unique API token**: Maps to their namespace
- **Independent consolidation**: No cross-tenant access possible

## Security

✅ **Complete isolation** - Tenants cannot access each other's data by design  
✅ **Token-based auth** - Each tenant has unique API token  
✅ **Admin-only provisioning** - Only admin token can create/delete tenants  
✅ **Scoped queries** - All memory operations automatically scoped to tenant

## Billing Integration

Track usage per tenant:
```python
import requests

ADMIN_TOKEN = "your-super-secret-admin-token"
BASE_URL = "https://your-automem.railway.app"

def calculate_monthly_bill(tenant_id):
    response = requests.get(
        f"{BASE_URL}/admin/tenants/{tenant_id}/stats",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"}
    )
    
    stats = response.json()
    memories = stats['memory_count']
    storage_mb = stats['storage_mb']
    
    # Example pricing:
    # $100/month base + $0.01 per memory over 1000
    base = 100
    overage = max(0, memories - 1000) * 0.01
    
    return base + overage
```

## Next Steps

To integrate with full AutoMem functionality:

1. **Modify core `app.py`** to accept tenant context
2. **Pass tenant's graph/collection names** to all memory operations
3. **Replace placeholder endpoints** in `app_multitenant.py` with real implementations
4. **Add consolidated memory operations** that respect tenant boundaries

Current code provides:
- ✅ Complete tenant management (create, list, stats, delete)
- ✅ Authentication and isolation framework
- ✅ Admin endpoints working
- ⏳ Memory endpoints (placeholders - need full integration)

## Testing Locally

```bash
# Terminal 1: Start dependencies
docker-compose up -d

# Terminal 2: Run multi-tenant app
export ADMIN_API_TOKEN="test-admin-token"
python app_multitenant.py

# Terminal 3: Test
export ADMIN_TOKEN="test-admin-token"
export BASE_URL="http://localhost:8001"

# Create tenant
curl -X POST $BASE_URL/admin/tenants \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"tenant_id":"test","api_token":"test-token"}'

# Test tenant auth
curl -X POST $BASE_URL/memory \
  -H "Authorization: Bearer test-token" \
  -d '{"content":"test memory"}'

# Check stats
curl $BASE_URL/admin/tenants/test/stats \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

## Architecture Diagram

```
Client Request with Token
         │
         ▼
  ┌──────────────────┐
  │  Flask App       │
  │  (multitenant)   │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ TenantManager    │──→ Authenticate token
  │                  │──→ Map to tenant_id
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ Isolated Storage │
  │                  │
  │ FalkorDB         │──→ Graph: memories_{tenant_id}
  │ Qdrant           │──→ Collection: memories_{tenant_id}
  └──────────────────┘
```

## Environment Variables

**Required:**
- `ADMIN_API_TOKEN` - Admin access for tenant provisioning
- `FALKORDB_HOST` - FalkorDB connection
- `FALKORDB_PORT` - FalkorDB port (default 6379)

**Optional:**
- `QDRANT_URL` - Enable vector search
- `QDRANT_API_KEY` - Qdrant authentication
- `OPENAI_API_KEY` - Real embeddings (vs placeholders)
- `FLASK_DEBUG` - Enable debug mode (development only)

## Production Checklist

- [ ] Generate strong `ADMIN_API_TOKEN` (32+ chars random)
- [ ] Generate unique API tokens per tenant (use `secrets.token_urlsafe(32)`)
- [ ] Set up FalkorDB (Railway, Docker, or hosted)
- [ ] Set up Qdrant (Railway, Docker, or Qdrant Cloud)
- [ ] Configure environment variables
- [ ] Deploy to Railway
- [ ] Test tenant creation
- [ ] Test tenant isolation (Fernando can't see Claudio's data)
- [ ] Set up billing/usage monitoring
- [ ] Document customer onboarding flow

## Support

This is a working prototype demonstrating multi-tenancy. For production:

1. **Integrate memory operations** - Wire up full AutoMem functionality
2. **Add rate limiting** - Per-tenant request limits
3. **Add resource quotas** - Storage limits per tenant
4. **Add audit logging** - Track all admin operations
5. **Add tenant suspension** - Disable without deleting data
6. **Add backup/export** - Per-tenant data export

Want help with full integration? Let me know!
