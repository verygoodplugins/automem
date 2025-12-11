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
