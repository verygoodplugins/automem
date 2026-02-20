# AutoMem Visualizer Enhancement Plan

## Overview

This plan focuses on improving memory display and organization in the graph visualizer, inspired by Obsidian's graph view UX. **Hand gesture features are retained** as a differentiating capability. The goal is better clustering, navigation, and context-aware selection UI.

## Current State Analysis

### Existing Capabilities (Keep All)
- **3D Force Layout** - d3-force-3d with Fibonacci sphere initialization
- **Hand Gestures** - MediaPipe webcam + iPhone hand tracking integration
- **Post-processing** - Bloom, vignette (performance mode toggle)
- **Inspector Panel** - Node details with relationships
- **API Integration** - React Query for data fetching

### Data Available from AutoMem API

**Memory Nodes:**
- `id`, `content`, `type` (8 types), `importance`, `confidence`
- `tags` (categorical grouping)
- `timestamp`, `metadata`
- Computed: `color`, `radius`, `opacity`

**11 Relationship Types:**
| Type | Category | Use for Clustering |
|------|----------|-------------------|
| RELATES_TO | General | Medium weight |
| LEADS_TO | Causal | High weight |
| OCCURRED_BEFORE | Temporal | Sequence flow |
| PREFERS_OVER | Preference | User patterns |
| EXEMPLIFIES | Pattern | Strong cluster signal |
| CONTRADICTS | Conflict | Separate clusters |
| REINFORCES | Support | Strong cluster signal |
| INVALIDATED_BY | Superseded | Temporal layering |
| EVOLVED_INTO | Evolution | Flow direction |
| DERIVED_FROM | Source | Hierarchy |
| PART_OF | Hierarchical | Sub-clustering |

**Semantic Similarity:**
- Vector embeddings (3072-dim) from Qdrant
- Cosine similarity scores for any pair
- Accessible via `/graph/neighbors` endpoint

### What Needs Improvement
1. **Node clustering** - Currently random placement; no visual grouping
2. **Relationship visualization** - All edges look the same
3. **Selection context** - Inspector exists but not like Obsidian's focus mode
4. **Settings/controls** - Limited to filter dropdown; no force controls
5. **Navigation** - Relies heavily on gestures; needs mouse/keyboard fallbacks

---

## Proposed Enhancements

### Phase 1: Obsidian-Style Settings Panel (Right-Docked)

Create `SettingsPanel.tsx` with collapsible sections:

```
┌─────────────────────────────────┐
│ ⚙ Graph Settings            ✕  │
├─────────────────────────────────┤
│ ▼ Filters                       │
│   Memory Types   [Decision ▼]   │
│   Tags           [Select... ▼]  │
│   Min Importance ━━━━●━━━  0.3  │
│   Show Orphans          [✓]     │
│   Show Only Connected   [ ]     │
├─────────────────────────────────┤
│ ▼ Relationships                 │
│   [✓] RELATES_TO    ──────      │
│   [✓] LEADS_TO      ━━━━━━      │
│   [✓] EXEMPLIFIES   ··········  │
│   [✓] CONTRADICTS   ─ ─ ─ ─     │
│   [ ] SIMILAR_TO    ∿∿∿∿∿∿      │
│   ... (toggles for all 11+)     │
├─────────────────────────────────┤
│ ▼ Display                       │
│   Show Labels          [✓]      │
│   Label Fade Distance  ━━●━  80 │
│   Show Arrows          [ ]      │
│   Node Size Scale     ━●━━  1.0 │
│   Link Thickness      ━●━━  1.0 │
│   Link Opacity        ━━●━  0.6 │
├─────────────────────────────────┤
│ ▼ Clustering                    │
│   Mode: ○ Type ● Tags ○ Semantic│
│   Show Boundaries      [✓]      │
│   Cluster Strength    ━━●━  0.5 │
│   [Detect Clusters]             │
├─────────────────────────────────┤
│ ▼ Forces                        │
│   Center Force       ━●━━  0.05 │
│   Repel Force        ━━●━  -100 │
│   Link Force         ━●━━  0.5  │
│   Link Distance      ━━●━  50   │
│   Collision Radius   ━●━━  2.0  │
│                                 │
│   [Reheat] [Reset to Defaults]  │
└─────────────────────────────────┘
```

### Phase 2: Multi-Layer Relationship Visualization

**Edge Styling by Type:**

```typescript
const EDGE_STYLES: Record<RelationType, EdgeStyle> = {
  // Causal/Flow (solid, directional)
  LEADS_TO:        { stroke: '#3B82F6', dash: null, width: 2, arrow: true },
  EVOLVED_INTO:    { stroke: '#06B6D4', dash: null, width: 1.5, arrow: true },
  DERIVED_FROM:    { stroke: '#A855F7', dash: null, width: 1.5, arrow: true },

  // Temporal (dashed)
  OCCURRED_BEFORE: { stroke: '#6B7280', dash: [4, 2], width: 1, arrow: true },
  INVALIDATED_BY:  { stroke: '#F97316', dash: [4, 2], width: 1, arrow: true },

  // Semantic/Association (dotted)
  RELATES_TO:      { stroke: '#94A3B8', dash: [2, 2], width: 1, arrow: false },
  EXEMPLIFIES:     { stroke: '#10B981', dash: [2, 2], width: 1.5, arrow: false },
  REINFORCES:      { stroke: '#22C55E', dash: [2, 2], width: 1.5, arrow: false },

  // Conflict (red, special)
  CONTRADICTS:     { stroke: '#EF4444', dash: [6, 3], width: 2, arrow: false },

  // Preference/Hierarchy
  PREFERS_OVER:    { stroke: '#8B5CF6', dash: null, width: 1, arrow: true },
  PART_OF:         { stroke: '#64748B', dash: null, width: 1, arrow: true },

  // Semantic similarity (virtual edges from Qdrant)
  SIMILAR_TO:      { stroke: '#94A3B8', dash: [1, 3], width: 0.5, arrow: false, opacity: 0.3 },
}
```

**Relationship Layers:**
- Toggle visibility per relationship type
- Option to show semantic similarity edges (from Qdrant vectors)
- Edge labels on hover showing type and strength

### Phase 3: Smart Clustering

**Three Clustering Modes:**

1. **By Memory Type** (default)
   - Group nodes by type (Decision, Pattern, Preference, etc.)
   - Use type colors
   - Add subtle force to pull same-type nodes together

2. **By Tags**
   - Analyze shared tags across memories
   - Create cluster groups for common tag combinations
   - Color nodes by dominant tag cluster
   - Show tag labels on cluster boundaries

3. **By Semantic Similarity**
   - Use vector embeddings to compute pairwise distances
   - Apply community detection (Louvain or similar)
   - Color by detected cluster
   - Label clusters by common themes/keywords

**Visual Cluster Boundaries:**
- Subtle dotted circles around cluster groups (like Obsidian screenshot)
- Fade in/out based on zoom level
- Click boundary to select all nodes in cluster
- Hover to see cluster statistics

**Implementation:**

```typescript
// hooks/useClusterDetection.ts
interface ClusterConfig {
  mode: 'type' | 'tags' | 'semantic';
  minClusterSize: number;
  showBoundaries: boolean;
  clusterStrength: number; // Additional force pulling cluster members together
}

interface Cluster {
  id: string;
  label: string;
  nodeIds: Set<string>;
  center: { x: number; y: number; z: number };
  radius: number;
  color: string;
}
```

### Phase 4: Enhanced Selection & Focus Mode (Obsidian-Style)

When a node is selected:

**Visual Changes:**
1. Selected node: Bright highlight, larger size, pulsing glow
2. Connected nodes (1-hop): Highlighted at 80% intensity
3. Connected nodes (2-hop): Highlighted at 50% intensity (optional)
4. Unconnected nodes: Fade to 20% opacity
5. Relevant edges: Highlighted, thicker
6. Irrelevant edges: Fade to 10% opacity

**Inspector Panel Enhancement:**

```
┌─────────────────────────────────────┐
│ 🔵 Decision                    ✕    │
│ UUID: abc-123...                    │
├─────────────────────────────────────┤
│ Content                             │
│ ┌─────────────────────────────────┐ │
│ │ User prefers TypeScript over    │ │
│ │ JavaScript for new projects...  │ │
│ └─────────────────────────────────┘ │
├─────────────────────────────────────┤
│ 📊 Importance ━━━━━━━●━━━  0.85     │
│ 🎯 Confidence ━━━━━━━━●━  0.92      │
├─────────────────────────────────────┤
│ 🏷️ Tags                             │
│ [typescript] [preferences] [coding] │
├─────────────────────────────────────┤
│ ▼ Graph Relationships (5)           │
│ ┌─────────────────────────────────┐ │
│ │ → LEADS_TO                      │ │
│ │   "Adopted ESLint strict..."    │ │
│ │   Strength: 0.8  [Navigate →]   │ │
│ ├─────────────────────────────────┤ │
│ │ ← REINFORCES                    │ │
│ │   "Team agreed on TS for..."    │ │
│ │   Strength: 0.9  [Navigate →]   │ │
│ └─────────────────────────────────┘ │
├─────────────────────────────────────┤
│ ▼ Similar Memories (3)              │
│ ┌─────────────────────────────────┐ │
│ │ 🟣 Preference  92% similar      │ │
│ │   "Prefers React over Vue..."   │ │
│ │   [Navigate →] [Show Edge]      │ │
│ └─────────────────────────────────┘ │
├─────────────────────────────────────┤
│ [Edit Importance] [Delete Memory]   │
└─────────────────────────────────────┘
```

**Focus Mode Features:**
- "Show Edge" button to temporarily display semantic similarity edge
- Clicking related memory navigates + updates selection
- Breadcrumb trail of navigation history
- "Back" button to return to previous selection

### Phase 5: Improved Force Layout Configuration

**Enhanced `useForceLayout.ts`:**

```typescript
interface ForceConfig {
  // Core forces
  centerStrength: number;      // 0.01 - 0.2, default 0.05
  chargeStrength: number;      // -200 to -50, default -100
  linkStrength: number;        // 0.1 - 1.0, default 0.5
  linkDistance: number;        // 20 - 100, default 50
  collisionRadius: number;     // 1.0 - 4.0, default 2.0

  // Clustering forces (new)
  clusterStrength: number;     // 0 - 1.0, additional pull toward cluster center

  // Relationship-weighted links (new)
  relationshipWeights: Record<RelationType, number>;
  // e.g., LEADS_TO: 1.0, RELATES_TO: 0.3, CONTRADICTS: -0.2

  // Animation
  alphaDecay: number;          // 0.01 - 0.05
  velocityDecay: number;       // 0.2 - 0.5
}
```

**Relationship-Weighted Forces:**
- Stronger relationships (LEADS_TO, REINFORCES) pull nodes closer
- Weaker relationships (RELATES_TO) have looser coupling
- CONTRADICTS can have slight repulsion within cluster

### Phase 6: Keyboard & Mouse Navigation (Supplement to Gestures)

**Keyboard Shortcuts:**
| Key | Action |
|-----|--------|
| `Escape` | Deselect current node |
| `R` | Reset view to initial position |
| `F` | Fit all nodes in view |
| `Space` | Reheat simulation |
| `Tab` | Cycle through connected nodes |
| `1-8` | Filter to memory type 1-8 |
| `Ctrl+F` | Focus search bar |
| `?` | Show keyboard shortcuts overlay |

**Mouse Enhancements:**
- Double-click node: Focus + zoom to node
- Double-click empty: Reset view
- Right-click node: Context menu (navigate, edit, delete)
- Ctrl+click: Add to multi-selection
- Drag selection box: Select multiple nodes

---

## Component Architecture

```
App.tsx
├── Header
│   ├── Logo + Title
│   ├── SearchBar (enhanced with suggestions)
│   └── StatsBar
│
├── Main Content (PanelGroup - horizontal)
│   │
│   ├── GraphCanvas
│   │   ├── Scene
│   │   │   ├── NodeInstances (instanced mesh)
│   │   │   ├── EdgeLines (batched, styled by type)
│   │   │   ├── SemanticEdges (optional similarity edges)
│   │   │   ├── Labels (billboard, LOD-based)
│   │   │   ├── ClusterBoundaries (dotted circles)
│   │   │   └── SelectionHighlight (focus mode overlay)
│   │   │
│   │   ├── OrbitControls (always active as fallback)
│   │   │
│   │   └── Hand Gesture Layer (existing - unchanged)
│   │       ├── Hand2DOverlay
│   │       ├── GestureDebugOverlay
│   │       └── HandControlOverlay
│   │
│   ├── Inspector (enhanced - right panel)
│   │   ├── NodeDetails
│   │   ├── RelationshipList (grouped by type)
│   │   ├── SemanticNeighbors
│   │   └── ActionButtons
│   │
│   └── SettingsPanel (new - right panel, collapsible)
│       ├── FiltersSection
│       ├── RelationshipsSection
│       ├── DisplaySection
│       ├── ClusteringSection
│       └── ForcesSection
│
└── Overlays
    ├── KeyboardShortcutsHelp
    └── ClusterInfoTooltip
```

---

## New Files to Create

| File | Purpose |
|------|---------|
| `components/SettingsPanel.tsx` | Main settings panel with all sections |
| `components/SettingsSection.tsx` | Collapsible section wrapper |
| `components/SliderControl.tsx` | Labeled slider with value display |
| `components/ToggleControl.tsx` | Labeled toggle switch |
| `components/RelationshipToggles.tsx` | Per-relationship visibility controls |
| `components/ClusterBoundaries.tsx` | 3D cluster boundary visualization |
| `components/SelectionHighlight.tsx` | Focus mode visual overlay |
| `components/EdgeRenderer.tsx` | Styled edges by relationship type |
| `components/SemanticEdges.tsx` | Optional similarity-based edges |
| `hooks/useClusterDetection.ts` | Cluster analysis logic |
| `hooks/useSelectionFocus.ts` | Focus mode state management |
| `hooks/useKeyboardShortcuts.ts` | Keyboard navigation |
| `lib/clusterUtils.ts` | Clustering algorithms (tag, semantic) |
| `lib/edgeStyles.ts` | Edge styling configuration |

## Files to Modify

| File | Changes |
|------|---------|
| `App.tsx` | Add SettingsPanel, keyboard handler |
| `GraphCanvas.tsx` | Add cluster boundaries, edge styling, focus mode |
| `useForceLayout.ts` | Add configurable forces, clustering force |
| `Inspector.tsx` | Enhanced relationship display, navigation |
| `FilterPanel.tsx` | Move content into SettingsPanel |
| `types.ts` | Add ClusterConfig, ForceConfig, EdgeStyle types |

---

## Implementation Order

### Sprint 1: Settings Panel Foundation
1. Create SettingsSection collapsible component
2. Create SliderControl and ToggleControl
3. Create SettingsPanel shell with sections
4. Wire up force configuration to useForceLayout
5. Add reheat button functionality

### Sprint 2: Relationship Visualization
1. Define edge styles in edgeStyles.ts
2. Update EdgeLines rendering with styles
3. Add RelationshipToggles to settings
4. Implement edge visibility filtering
5. Add edge labels on hover

### Sprint 3: Clustering
1. Implement useClusterDetection hook
2. Create ClusterBoundaries component
3. Add clustering mode selector to settings
4. Implement tag-based clustering
5. Implement semantic clustering (using Qdrant data)
6. Add cluster force to simulation

### Sprint 4: Selection & Focus Mode
1. Implement useSelectionFocus hook
2. Add SelectionHighlight component
3. Enhance Inspector with navigation
4. Add focus mode opacity/highlight logic
5. Implement "Show Edge" for semantic neighbors

### Sprint 5: Navigation Polish
1. Add useKeyboardShortcuts hook
2. Implement all keyboard shortcuts
3. Add double-click behaviors
4. Add multi-selection support
5. Add navigation breadcrumbs

---

## Success Criteria

1. **Better Clustering** - Related memories visually grouped with optional boundaries
2. **Relationship Clarity** - Edge types distinguishable by style/color
3. **Intuitive Selection** - Obsidian-style focus mode with context highlighting
4. **Configurable Forces** - Users can tune simulation to their preference
5. **Keyboard Navigation** - Full functionality without mouse/gestures
6. **Hand Gestures Preserved** - All existing gesture features work alongside new controls
7. **Performance** - Smooth 60fps with 500+ nodes

---

## Research Sources

- [yFiles Knowledge Graph Guide](https://www.yfiles.com/resources/how-to/guide-to-visualizing-knowledge-graphs) - Layout matching, visual encoding
- [Cambridge Intelligence - Graph Layouts](https://cambridge-intelligence.com/automatic-graph-layouts/) - Force-directed best practices
- [Datavid - Knowledge Graph Visualization](https://datavid.com/blog/knowledge-graph-visualization) - Multi-layer visualization
- [FalkorDB Visualization](https://www.falkordb.com/blog/knowledge-graph-visualization-insights/) - Real-time graph manipulation
- [2024 Graph Layout Evaluation](https://onlinelibrary.wiley.com/doi/10.1111/cgf.15073) - Evaluation methodologies
