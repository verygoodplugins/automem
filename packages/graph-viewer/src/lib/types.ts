export interface GraphNode {
  id: string
  content: string
  type: MemoryType
  importance: number
  confidence: number
  tags: string[]
  timestamp: string
  updated_at?: string
  metadata?: Record<string, unknown>
  color: string
  radius: number
  opacity: number
  // 3D position (computed by force layout)
  x?: number
  y?: number
  z?: number
  // Velocity for force simulation
  vx?: number
  vy?: number
  vz?: number
}

export interface GraphEdge {
  id: string
  source: string
  target: string
  type: RelationType
  strength: number
  color: string
  properties?: Record<string, unknown>
}

export type MemoryType =
  | 'Decision'
  | 'Pattern'
  | 'Preference'
  | 'Style'
  | 'Habit'
  | 'Insight'
  | 'Context'
  | 'Memory'

export type RelationType =
  | 'RELATES_TO'
  | 'LEADS_TO'
  | 'OCCURRED_BEFORE'
  | 'PREFERS_OVER'
  | 'EXEMPLIFIES'
  | 'CONTRADICTS'
  | 'REINFORCES'
  | 'INVALIDATED_BY'
  | 'EVOLVED_INTO'
  | 'DERIVED_FROM'
  | 'PART_OF'

export interface GraphSnapshot {
  nodes: GraphNode[]
  edges: GraphEdge[]
  stats: {
    total_nodes: number
    total_edges: number
    returned_nodes: number
    returned_edges: number
    sampled: boolean
    sample_ratio: number
  }
  meta: {
    type_colors: Record<string, string>
    relation_colors: Record<string, string>
    query_time_ms: number
  }
}

export interface GraphNeighbors {
  center: GraphNode
  graph_neighbors: GraphNode[]
  semantic_neighbors: (GraphNode & { similarity: number })[]
  edges: GraphEdge[]
  meta: {
    depth: number
    query_time_ms: number
  }
}

export interface GraphStats {
  totals: {
    nodes: number
    edges: number
  }
  by_type: Record<string, number>
  by_relationship: Record<string, number>
  importance_distribution: {
    high: number
    medium: number
    low: number
  }
  recent_activity: Array<{
    date: string
    count: number
  }>
  meta: {
    type_colors: Record<string, string>
    relation_colors: Record<string, string>
    query_time_ms: number
  }
}

export interface FilterState {
  types: MemoryType[]
  minImportance: number
  maxNodes: number
}

export interface SimulationNode extends GraphNode {
  fx?: number | null
  fy?: number | null
  fz?: number | null
}

export interface SimulationLink {
  source: string | SimulationNode
  target: string | SimulationNode
  strength: number
  type: RelationType
}

// Force layout configuration
export interface ForceConfig {
  centerStrength: number      // 0.01 - 0.2, default 0.05
  chargeStrength: number      // -200 to -50, default -100
  linkStrength: number        // 0.1 - 1.0, default 0.5
  linkDistance: number        // 20 - 100, default 50
  collisionRadius: number     // 1.0 - 4.0, default 2.0
}

// Display settings
export interface DisplayConfig {
  showLabels: boolean
  labelFadeDistance: number   // Distance at which labels start fading
  showArrows: boolean
  nodeSizeScale: number       // Multiplier for node sizes
  linkThickness: number       // Base link thickness
  linkOpacity: number         // 0-1
}

// Clustering configuration
export type ClusterMode = 'type' | 'tags' | 'semantic' | 'none'

export interface ClusterConfig {
  mode: ClusterMode
  showBoundaries: boolean
  clusterStrength: number     // Additional force pulling cluster members together
  useUMAP: boolean            // Use UMAP embedding projection for positioning
}

// Relationship visibility
export type RelationshipVisibility = Record<RelationType, boolean>

// Combined settings state
export interface GraphSettings {
  forces: ForceConfig
  display: DisplayConfig
  clustering: ClusterConfig
  relationshipVisibility: RelationshipVisibility
}

// Default values
export const DEFAULT_FORCE_CONFIG: ForceConfig = {
  centerStrength: 0.05,
  chargeStrength: -60,      // Reduced from -100 for tighter clusters
  linkStrength: 0.75,       // Increased from 0.5 for stronger grouping
  linkDistance: 40,         // Reduced from 50 for denser clusters
  collisionRadius: 2.0,
}

export const DEFAULT_DISPLAY_CONFIG: DisplayConfig = {
  showLabels: true,
  labelFadeDistance: 80,
  showArrows: false,
  nodeSizeScale: 1.0,
  linkThickness: 1.0,
  linkOpacity: 0.6,
}

export const DEFAULT_CLUSTER_CONFIG: ClusterConfig = {
  mode: 'type',
  showBoundaries: false,
  clusterStrength: 0.3,
  useUMAP: false,
}

export const DEFAULT_RELATIONSHIP_VISIBILITY: RelationshipVisibility = {
  RELATES_TO: true,
  LEADS_TO: true,
  OCCURRED_BEFORE: true,
  PREFERS_OVER: true,
  EXEMPLIFIES: true,
  CONTRADICTS: true,
  REINFORCES: true,
  INVALIDATED_BY: true,
  EVOLVED_INTO: true,
  DERIVED_FROM: true,
  PART_OF: true,
}
