# Graph Visualization Clustering - Phase 2 Implementation Plan

## Current State (Phase 1 Complete)

Quick wins implemented in `3dffb49`:
- Tag-based initial positioning (nodes with same tag start clustered)
- Entity tag prioritization (`entity:*` tags preferred for clustering)
- Tighter force physics (chargeStrength -60, linkStrength 0.75, linkDistance 40)
- Semantic clustering threshold lowered to 0.3

**Result**: Clusters now visually contain their nodes instead of spanning the entire graph.

---

## Phase 2: UMAP-Based Embedding Positioning

### The Problem
Current clustering uses tags and explicit edges. Memories with similar *content* but no shared tags/edges still scatter. True semantic similarity from embeddings isn't used for positioning.

### Solution Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Qdrant        │────▶│  UMAP Projection │────▶│  Initial (x,y,z)│
│   768d vectors  │     │  768d → 3d       │     │  positions      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │  D3 Force Sim   │
                                                 │  (refinement)   │
                                                 └─────────────────┘
```

### Implementation Options

#### Option A: Server-Side UMAP (Recommended)
- Add `/graph/umap` endpoint to Flask API
- Use `umap-learn` Python package
- Pre-compute and cache projections
- Return 3D positions with graph data

**Pros**: Fast on client, leverages existing Qdrant access, cacheable
**Cons**: Requires API change, server compute cost

#### Option B: Client-Side UMAP
- Fetch embeddings from Qdrant via API
- Run UMAP in browser with umap-js
- Cache results in localStorage

**Pros**: No backend changes, works offline
**Cons**: Slow for 1000+ nodes, requires embedding fetch

#### Option C: Hybrid t-SNE/UMAP
- Server pre-computes global layout
- Client refines with force simulation
- Periodic background recomputation

### Recommended Implementation (Option A)

1. **New API Endpoint**: `GET /graph/projected`
   ```python
   @app.route('/graph/projected')
   def graph_projected():
       # Get memories with embeddings from Qdrant
       # Run UMAP(n_components=3, n_neighbors=15, min_dist=0.1)
       # Return nodes with (x, y, z) positions
       return {
           "nodes": [...],  # with x,y,z from UMAP
           "edges": [...],
           "projection": {
               "method": "umap",
               "n_neighbors": 15,
               "min_dist": 0.1,
               "cached_at": "2025-12-28T..."
           }
       }
   ```

2. **Cache Strategy**:
   - Recompute when memory count changes by >10%
   - Store projections in FalkorDB node properties
   - Background job during consolidation

3. **Client Changes**:
   - New `fetchProjectedGraph()` in api.ts
   - Skip force simulation initial phase if UMAP positions present
   - Fallback to tag-based positioning if no projection

---

## Phase 3: Multi-Agent Architecture

### Agent Responsibilities

```
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator Agent                        │
│  Coordinates tasks, manages shared state, routes requests    │
└─────────────────────────────────────────────────────────────┘
        │           │           │           │           │
        ▼           ▼           ▼           ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│  Testing │ │ Storage  │ │ Retrieval│ │  Perf    │ │  Visual  │
│  Agent   │ │  Agent   │ │  Agent   │ │  Agent   │ │  Agent   │
└──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

#### Testing Agent
- Owns: `tests/`, Playwright tests, visual regression
- Responsibilities:
  - Unit tests for UMAP projection
  - Integration tests for API changes
  - Visual snapshot testing for cluster layouts
  - Performance benchmarks (time to first meaningful render)

#### Storage Agent
- Owns: `app.py`, FalkorDB schema, Qdrant collection
- Responsibilities:
  - Add projection cache to graph schema
  - UMAP projection storage and retrieval
  - Embedding dimension migration (768 → 3072 if needed)
  - Cache invalidation logic

#### Retrieval Agent
- Owns: Recall logic, neighbor expansion, semantic search
- Responsibilities:
  - Efficient projection fetch for subgraph views
  - Incremental projection for new memories
  - Neighbor-aware positioning (connected nodes stay close)

#### Performance Agent
- Owns: Caching, batching, streaming
- Responsibilities:
  - UMAP computation optimization (approximate NN, GPU)
  - WebGL instancing for 1000+ nodes
  - Level-of-detail for zoom levels
  - Memory footprint management

#### Visual Agent
- Owns: `packages/graph-viewer/`, React components, shaders
- Responsibilities:
  - UMAP position consumption
  - 2D fallback mode (Obsidian-style)
  - Cluster boundary rendering
  - Interactive pan/zoom/select

---

## Phase 4: 2D Obsidian-Style Mode

### Design Goals
- "At a glance" overview of memory landscape
- Fast navigation to memory clusters
- Works on mobile/tablet
- Optional 3D depth for power users

### Implementation
```typescript
// New DisplayConfig option
interface DisplayConfig {
  // ...existing
  dimensionMode: '2d' | '3d' | 'auto'  // auto = 3d on desktop, 2d on mobile
}
```

### 2D Layout Changes
- UMAP with `n_components=2`
- Replace Three.js with SVG or Canvas2D
- Obsidian-style node shapes (circles with importance-based size)
- Hover reveals connections with curved bezier edges

---

## Timeline Estimate

| Phase | Scope | Effort |
|-------|-------|--------|
| Phase 1 | Quick wins (DONE) | ✅ |
| Phase 2A | Server-side UMAP endpoint | 4-6 hours |
| Phase 2B | Client UMAP consumption | 2-3 hours |
| Phase 3 | Multi-agent scaffolding | 2-4 hours |
| Phase 4 | 2D mode | 4-6 hours |

---

## Next Immediate Steps

1. **Add UMAP to requirements.txt**
   ```
   umap-learn>=0.5.5
   ```

2. **Create projection endpoint** in `app.py`

3. **Add projection cache** to FalkorDB Memory nodes

4. **Update graph-viewer** to consume projected positions

5. **Add 2D/3D toggle** to settings panel
