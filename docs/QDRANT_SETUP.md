# Qdrant Setup Guide

Step-by-step guide to setting up Qdrant for AutoMem's vector storage and semantic search.

## Why Qdrant?

Qdrant stores vector embeddings of your memories, enabling:
- **Semantic search**: Find memories by meaning, not just keywords
- **Similarity matching**: Discover related memories automatically
- **Fast retrieval**: Sub-millisecond search across thousands of memories

**Without Qdrant**: AutoMem uses placeholder embeddings (hash-based). This works for testing but provides no semantic search capability.

---

## Option A: Self-Hosted on Railway (Recommended)

Run Qdrant inside your Railway project for the lowest latency and simplest setup — no external accounts needed.

### Setup

1. In your Railway project, click **"+ New Service"** → **"Docker Image"**
2. Image: `qdrant/qdrant:v1.11.3`
3. **Add persistent volume**: Settings → Volumes → Mount path: `/qdrant/storage`
4. **Set environment variables** on the Qdrant service:

   ```bash
   PORT=6333
   QDRANT__SERVICE__HOST=::
   ```

   > **⚠️ `QDRANT__SERVICE__HOST=::` is critical.** Railway's internal networking uses IPv6. Qdrant defaults to `0.0.0.0` (IPv4 only), which silently refuses all internal connections. `::` enables dual-stack (IPv6 + IPv4).

5. **Set on the AutoMem API service**:

   ```bash
   QDRANT_HOST=qdrant
   ```

   AutoMem auto-constructs `http://qdrant:6333`. No API key needed for internal networking.

6. Redeploy both services.

### Verify

```bash
curl https://your-automem.up.railway.app/health
# Should show: "qdrant": "connected"
```

### Benefits over Qdrant Cloud

- **Lower latency**: Internal networking (~1ms) vs external HTTPS (20-80ms)
- **No external account**: Everything in one Railway project
- **No API key management**: Internal networking doesn't need auth
- **Cost**: ~$3-5/mo on Railway vs $25/mo for Qdrant Cloud paid tier

### Troubleshooting

If health shows `qdrant: "disconnected"` with "Connection refused" in logs:

1. **Check `QDRANT__SERVICE__HOST=::`** on the Qdrant service — this is the #1 cause
2. Verify `QDRANT_HOST=qdrant` on the AutoMem API service (not `QDRANT_URL`)
3. Confirm both services are in the same Railway project/environment
4. Check Qdrant service is running (Railway dashboard → service status)

See [Railway Deployment Guide — Troubleshooting](RAILWAY_DEPLOYMENT.md#qdrant-connection-refused-on-internal-networking) for detailed diagnostics.

---

## Option B: Qdrant Cloud (Managed)

Use Qdrant's hosted service for zero-ops vector storage.

## Quick Start

### Step 1: Create a Qdrant Cloud Account

1. Go to [cloud.qdrant.io](https://cloud.qdrant.io)
2. Sign up with GitHub, Google, or email
3. Verify your email if required

### Step 2: Create a Cluster

1. Click **"Create Cluster"** (or "Free Cluster" for the free tier)
2. Choose a region close to your Railway deployment
3. **Free tier**: 1GB storage, perfect for getting started
4. Wait for provisioning (~30 seconds)

### Step 3: Create a Collection

1. Click on your cluster to open the dashboard
2. Click **"Create Collection"**
3. Configure as follows:

#### Collection Name
```
memories
```
Use `memories` (default) or a custom name like `memories-projectname`.

> **Note**: If you use a custom name, set `QDRANT_COLLECTION` in AutoMem to match.

#### Use Case
Select: **Global search**

AutoMem searches across all memories with optional tag filters. It's not multi-tenant.

#### Search Configuration
Select: **Simple Single embedding**

AutoMem uses dense text embeddings (Voyage/OpenAI/etc.) for semantic search. Keyword matching is handled separately by FalkorDB, so sparse vectors are not needed.

#### Vector Configuration

| Setting | Value | Notes |
|---------|-------|-------|
| **Dense vector name** | Leave as default or use `memories` | Field name for embeddings |
| **Dimensions** | `1024` | For `voyage-4` (default) |
| **Metric** | `Cosine` | Best for text embeddings |

<a href="img/qdrant-configuration.jpg" target="_blank"><img src="img/qdrant-configuration.jpg" alt="Qdrant collection configuration" width="400"></a>

*Click image to view full size*

> **Using a smaller model?** If you set `EMBEDDING_MODEL=text-embedding-3-small`, use `768` dimensions instead and set `VECTOR_SIZE=768` in AutoMem.
>
> **Using FastEmbed or Ollama?** Set dimensions to match your local model output (e.g., 384/768/1024 for FastEmbed). Ollama model dimensions vary—verify with a test embedding and set `VECTOR_SIZE` accordingly.

#### Payload Indexes (Recommended)

After vector configuration, you'll see a **Payload indexes** section. These speed up filtered searches.

Click **+ Add** and create these indexes:

| Field name | Field type |
|------------|------------|
| `tags` | **keyword** |
| `tag_prefixes` | **keyword** |

Without indexes, Qdrant does full-scan filtering which slows down as your collection grows. With indexes, tag filtering stays fast even with 100k+ memories.

> **Note**: Indexes are optional for small collections (<1000 memories) but recommended for production use.

4. Click **"Finish"**

### Step 4: Get Your API Credentials

1. In your cluster dashboard, click **"API Keys"** or **"Data Access"**
2. Copy your:
   - **Cluster URL**: `https://xxxxx-xxxxx.aws.cloud.qdrant.io`
   - **API Key**: `xxxxxxxxxxxxxxxxxxxxxx`

### Step 5: Configure AutoMem

Add these to your AutoMem environment variables:

```bash
QDRANT_URL="https://xxxxx-xxxxx.aws.cloud.qdrant.io"
QDRANT_API_KEY="your-api-key-here"
QDRANT_COLLECTION="memories"  # Only if using custom name
```

**Railway**: Add these in `AutoMem` → Variables, then redeploy.

**Local**: Add to your `.env` file.

---

## Verify Connection

After configuring, check the health endpoint:

```bash
curl https://your-automem.up.railway.app/health
```

You should see:
```json
{
  "status": "healthy",
  "falkordb": "connected",
  "qdrant": "connected",
  ...
}
```

If `qdrant` shows `"disconnected"` or `"not configured"`:
- Verify `QDRANT_URL` includes `https://`
- Check API key is correct
- Ensure collection exists

---

## Configuration Options

### Embedding Models & Dimensions

| Provider / Model | Dimensions | Cost | Quality |
|------------------|------------|------|---------|
| `voyage-4` (recommended) | 1024 | ~$0.05/1M tokens | Excellent for short text |
| `text-embedding-3-small` | 1536 native (truncatable) | $0.02/1M tokens | Good OpenAI fallback |
| `text-embedding-3-large` | 3072 native (truncatable) | $0.13/1M tokens | Maximum precision |

**To switch providers**:
1. Set `EMBEDDING_PROVIDER` and any required API key
2. Set `VECTOR_SIZE` to match the provider's output dimension
3. Create a new Qdrant collection with matching dimensions (or use `VECTOR_SIZE_AUTODETECT=true`)
4. Redeploy AutoMem

> ⚠️ **Warning**: Changing embedding models requires re-embedding all existing memories. See [MIGRATIONS.md](MIGRATIONS.md) for the reembed script.

### Custom Collection Names

If you're running multiple AutoMem instances on the same Qdrant cluster:

```bash
QDRANT_COLLECTION="memories-production"
QDRANT_COLLECTION="memories-staging"
QDRANT_COLLECTION="memories-luka"
```

Each collection is isolated—memories in one won't appear in another.

---

## Free Tier Limits

Qdrant Cloud free tier includes:
- **1GB storage** (~50,000-100,000 memories depending on content)
- **Unlimited API calls**
- **No time limit**

For most personal and small team use cases, the free tier is plenty.

### Upgrading

If you exceed 1GB:
1. Go to Qdrant Cloud dashboard
2. Click "Upgrade" on your cluster
3. Choose a paid plan ($25/month for 10GB)

---

## Troubleshooting

### "qdrant: disconnected" in health check

1. **Check URL format**: Must include `https://`
   ```bash
   # ✅ Correct
   QDRANT_URL="https://abc-123.aws.cloud.qdrant.io"

   # ❌ Wrong
   QDRANT_URL="abc-123.aws.cloud.qdrant.io"
   ```

2. **Verify API key**: Copy fresh from Qdrant dashboard

3. **Check collection exists**: Create it if missing

### "Vector dimension mismatch"

AutoMem expects dimensions to match `VECTOR_SIZE`:
- `voyage-4` → `VECTOR_SIZE=1024` (default)
- `text-embedding-3-small` → `VECTOR_SIZE` ≤ 1536 (default: 768; auto-upgrades to `text-embedding-3-large` if exceeded)
- `text-embedding-3-large` → `VECTOR_SIZE` ≤ 3072 (truncatable via Matryoshka)

If you created the collection with wrong dimensions:
1. Delete the collection in Qdrant dashboard
2. Recreate with correct dimensions
3. Re-store your memories (or run reembed script)

### Memories not appearing in semantic search

1. **Check `OPENAI_API_KEY`**: Required for generating embeddings
2. **Verify embeddings exist**: Check `/health` shows `qdrant: connected`
3. **Wait for enrichment**: New memories are embedded async (few seconds)

---

## Security Notes

- **API keys are secret**: Never commit to git or share publicly
- **Use Railway variables**: Reference via `${{secret()}}` for auto-generation
- **Rotate periodically**: Generate new API keys in Qdrant dashboard monthly

---

## Next Steps

- [Railway Deployment Guide](RAILWAY_DEPLOYMENT.md) — Full deployment walkthrough
- [Environment Variables](ENVIRONMENT_VARIABLES.md) — All configuration options
- [Testing Guide](TESTING.md) — Verify your setup

---

**Questions?** Open an issue: https://github.com/verygoodplugins/automem/issues
