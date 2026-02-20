/**
 * usePathfinding - Find shortest paths between memory nodes
 *
 * Implements Dijkstra's algorithm with edge weights based on relationship strength.
 * Stronger relationships = lower weight = preferred path.
 */

import { useMemo, useCallback, useState } from 'react'
import type { GraphEdge, SimulationNode } from '../lib/types'

export interface PathStep {
  nodeId: string
  edgeType?: string
  strength?: number
}

export interface PathResult {
  path: PathStep[]
  totalStrength: number
  hopCount: number
}

interface UsePathfindingOptions {
  nodes: SimulationNode[]
  edges: GraphEdge[]
}

interface PathfindingState {
  sourceId: string | null
  targetId: string | null
  isSelectingTarget: boolean
  paths: PathResult[]
  activePath: number // Index of currently displayed path
}

/**
 * Build adjacency list from edges
 */
function buildAdjacency(edges: GraphEdge[]): Map<string, { nodeId: string; weight: number; type: string; strength: number }[]> {
  const adjacency = new Map<string, { nodeId: string; weight: number; type: string; strength: number }[]>()

  edges.forEach(edge => {
    // Weight is inverse of strength (stronger = lower weight = preferred)
    const weight = 1 - (edge.strength ?? 0.5)

    // Add both directions (undirected graph)
    if (!adjacency.has(edge.source)) {
      adjacency.set(edge.source, [])
    }
    adjacency.get(edge.source)!.push({
      nodeId: edge.target,
      weight,
      type: edge.type,
      strength: edge.strength ?? 0.5,
    })

    if (!adjacency.has(edge.target)) {
      adjacency.set(edge.target, [])
    }
    adjacency.get(edge.target)!.push({
      nodeId: edge.source,
      weight,
      type: edge.type,
      strength: edge.strength ?? 0.5,
    })
  })

  return adjacency
}

/**
 * Dijkstra's algorithm to find shortest path
 */
function dijkstra(
  adjacency: Map<string, { nodeId: string; weight: number; type: string; strength: number }[]>,
  source: string,
  target: string
): PathResult | null {
  const distances = new Map<string, number>()
  const previous = new Map<string, { nodeId: string; edgeType: string; strength: number } | null>()
  const visited = new Set<string>()

  // Priority queue (simple implementation - for large graphs, use a proper heap)
  const queue: { nodeId: string; distance: number }[] = []

  // Initialize
  distances.set(source, 0)
  previous.set(source, null)
  queue.push({ nodeId: source, distance: 0 })

  while (queue.length > 0) {
    // Get node with smallest distance
    queue.sort((a, b) => a.distance - b.distance)
    const current = queue.shift()!

    if (visited.has(current.nodeId)) continue
    visited.add(current.nodeId)

    // Found target
    if (current.nodeId === target) {
      break
    }

    // Process neighbors
    const neighbors = adjacency.get(current.nodeId) || []
    for (const neighbor of neighbors) {
      if (visited.has(neighbor.nodeId)) continue

      const newDist = (distances.get(current.nodeId) ?? Infinity) + neighbor.weight

      if (newDist < (distances.get(neighbor.nodeId) ?? Infinity)) {
        distances.set(neighbor.nodeId, newDist)
        previous.set(neighbor.nodeId, {
          nodeId: current.nodeId,
          edgeType: neighbor.type,
          strength: neighbor.strength,
        })
        queue.push({ nodeId: neighbor.nodeId, distance: newDist })
      }
    }
  }

  // Check if path exists
  if (!previous.has(target)) {
    return null
  }

  // Reconstruct path
  const path: PathStep[] = []
  let current: string | null = target
  let totalStrength = 0
  let hopCount = 0

  while (current) {
    const prev = previous.get(current)
    path.unshift({
      nodeId: current,
      edgeType: prev?.edgeType,
      strength: prev?.strength,
    })

    if (prev?.strength) {
      totalStrength += prev.strength
      hopCount++
    }

    current = prev?.nodeId ?? null
  }

  return {
    path,
    totalStrength: hopCount > 0 ? totalStrength / hopCount : 0,
    hopCount,
  }
}

/**
 * Find alternative paths by temporarily removing edges from the primary path
 */
function findAlternativePaths(
  edges: GraphEdge[],
  source: string,
  target: string,
  primaryPath: PathResult,
  maxAlternatives: number = 2
): PathResult[] {
  const alternatives: PathResult[] = []
  const primaryEdges = new Set<string>()

  // Identify edges in primary path
  for (let i = 0; i < primaryPath.path.length - 1; i++) {
    const from = primaryPath.path[i].nodeId
    const to = primaryPath.path[i + 1].nodeId
    primaryEdges.add(`${from}-${to}`)
    primaryEdges.add(`${to}-${from}`)
  }

  // Try removing each edge and finding alternative
  for (let i = 0; i < primaryPath.path.length - 1 && alternatives.length < maxAlternatives; i++) {
    const from = primaryPath.path[i].nodeId
    const to = primaryPath.path[i + 1].nodeId

    // Filter out this edge
    const filteredEdges = edges.filter(e => {
      const edgeKey1 = `${e.source}-${e.target}`
      const edgeKey2 = `${e.target}-${e.source}`
      return edgeKey1 !== `${from}-${to}` && edgeKey2 !== `${from}-${to}` &&
             edgeKey1 !== `${to}-${from}` && edgeKey2 !== `${to}-${from}`
    })

    const altAdjacency = buildAdjacency(filteredEdges)
    const altPath = dijkstra(altAdjacency, source, target)

    if (altPath && !pathsEqual(altPath.path, primaryPath.path)) {
      // Check if we already have this alternative
      const isDuplicate = alternatives.some(a => pathsEqual(a.path, altPath.path))
      if (!isDuplicate) {
        alternatives.push(altPath)
      }
    }
  }

  return alternatives
}

function pathsEqual(path1: PathStep[], path2: PathStep[]): boolean {
  if (path1.length !== path2.length) return false
  return path1.every((step, i) => step.nodeId === path2[i].nodeId)
}

export function usePathfinding({ nodes: _nodes, edges }: UsePathfindingOptions) {
  const [state, setState] = useState<PathfindingState>({
    sourceId: null,
    targetId: null,
    isSelectingTarget: false,
    paths: [],
    activePath: 0,
  })

  // Build adjacency list (memoized)
  const adjacency = useMemo(() => buildAdjacency(edges), [edges])

  // Start path selection from a node
  const startPathSelection = useCallback((nodeId: string) => {
    setState(prev => ({
      ...prev,
      sourceId: nodeId,
      targetId: null,
      isSelectingTarget: true,
      paths: [],
      activePath: 0,
    }))
  }, [])

  // Complete path selection to a target node
  const completePathSelection = useCallback((targetId: string) => {
    if (!state.sourceId || state.sourceId === targetId) return

    const primaryPath = dijkstra(adjacency, state.sourceId, targetId)

    if (!primaryPath) {
      // No path found
      setState(prev => ({
        ...prev,
        targetId,
        isSelectingTarget: false,
        paths: [],
      }))
      return
    }

    // Find alternative paths
    const alternatives = findAlternativePaths(edges, state.sourceId, targetId, primaryPath)

    setState(prev => ({
      ...prev,
      targetId,
      isSelectingTarget: false,
      paths: [primaryPath, ...alternatives],
      activePath: 0,
    }))
  }, [state.sourceId, adjacency, edges])

  // Cancel path selection
  const cancelPathSelection = useCallback(() => {
    setState({
      sourceId: null,
      targetId: null,
      isSelectingTarget: false,
      paths: [],
      activePath: 0,
    })
  }, [])

  // Clear current path (but keep source selected)
  const clearPath = useCallback(() => {
    setState(prev => ({
      ...prev,
      targetId: null,
      isSelectingTarget: false,
      paths: [],
      activePath: 0,
    }))
  }, [])

  // Cycle through alternative paths
  const nextPath = useCallback(() => {
    setState(prev => ({
      ...prev,
      activePath: (prev.activePath + 1) % Math.max(1, prev.paths.length),
    }))
  }, [])

  const previousPath = useCallback(() => {
    setState(prev => ({
      ...prev,
      activePath: (prev.activePath - 1 + prev.paths.length) % Math.max(1, prev.paths.length),
    }))
  }, [])

  // Get node IDs in the current active path
  const pathNodeIds = useMemo(() => {
    if (state.paths.length === 0) return new Set<string>()
    const currentPath = state.paths[state.activePath]
    return new Set(currentPath?.path.map(step => step.nodeId) ?? [])
  }, [state.paths, state.activePath])

  // Get edge keys in the current active path (for highlighting)
  const pathEdgeKeys = useMemo(() => {
    if (state.paths.length === 0) return new Set<string>()
    const currentPath = state.paths[state.activePath]
    if (!currentPath) return new Set<string>()

    const keys = new Set<string>()
    for (let i = 0; i < currentPath.path.length - 1; i++) {
      const from = currentPath.path[i].nodeId
      const to = currentPath.path[i + 1].nodeId
      keys.add(`${from}-${to}`)
      keys.add(`${to}-${from}`)
    }
    return keys
  }, [state.paths, state.activePath])

  return {
    // State
    sourceId: state.sourceId,
    targetId: state.targetId,
    isSelectingTarget: state.isSelectingTarget,
    paths: state.paths,
    activePath: state.activePath,
    currentPath: state.paths[state.activePath] ?? null,
    pathNodeIds,
    pathEdgeKeys,

    // Actions
    startPathSelection,
    completePathSelection,
    cancelPathSelection,
    clearPath,
    nextPath,
    previousPath,

    // Computed
    hasPath: state.paths.length > 0,
    pathCount: state.paths.length,
  }
}
