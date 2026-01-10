/**
 * useFocusMode - Calculate depth-based opacity for spotlight effect
 *
 * When focus mode is active, nodes are dimmed based on their
 * graph distance from the selected node:
 * - Selected node: 100% opacity
 * - 1st degree neighbors: 100% opacity
 * - 2nd degree neighbors: 60% opacity
 * - 3rd degree neighbors: 30% opacity
 * - Beyond: 10% opacity
 */

import { useMemo } from 'react'
import type { GraphNode, GraphEdge } from '../lib/types'

export interface FocusModeConfig {
  enabled: boolean
  selectedNodeId: string | null
  transitionProgress: number // 0-1, for smooth fade in/out
}

export interface NodeFocusState {
  depth: number // -1 if no selection, 0 for selected, 1+ for neighbors
  opacity: number // Computed opacity based on depth
  isInFocus: boolean // True if depth <= 3
}

// Opacity values for each depth level
const DEPTH_OPACITY = [
  1.0,  // depth 0 (selected)
  1.0,  // depth 1 (direct neighbors)
  0.6,  // depth 2
  0.3,  // depth 3
]
const DEFAULT_OPACITY = 0.08 // Beyond depth 3

/**
 * Build adjacency map from edges
 */
function buildAdjacencyMap(edges: GraphEdge[]): Map<string, Set<string>> {
  const adjacency = new Map<string, Set<string>>()

  edges.forEach(edge => {
    if (!adjacency.has(edge.source)) {
      adjacency.set(edge.source, new Set())
    }
    if (!adjacency.has(edge.target)) {
      adjacency.set(edge.target, new Set())
    }
    adjacency.get(edge.source)!.add(edge.target)
    adjacency.get(edge.target)!.add(edge.source)
  })

  return adjacency
}

/**
 * BFS to compute depth from source node
 */
function computeDepths(
  sourceId: string,
  adjacency: Map<string, Set<string>>,
  maxDepth: number = 3
): Map<string, number> {
  const depths = new Map<string, number>()
  depths.set(sourceId, 0)

  const queue: { id: string; depth: number }[] = [{ id: sourceId, depth: 0 }]

  while (queue.length > 0) {
    const { id, depth } = queue.shift()!

    if (depth >= maxDepth) continue

    const neighbors = adjacency.get(id)
    if (!neighbors) continue

    for (const neighborId of neighbors) {
      if (!depths.has(neighborId)) {
        depths.set(neighborId, depth + 1)
        queue.push({ id: neighborId, depth: depth + 1 })
      }
    }
  }

  return depths
}

/**
 * Hook to compute focus mode state for all nodes
 */
export function useFocusMode(
  nodes: GraphNode[],
  edges: GraphEdge[],
  selectedNodeId: string | null,
  enabled: boolean,
  transitionProgress: number = 1
): Map<string, NodeFocusState> {
  return useMemo(() => {
    const result = new Map<string, NodeFocusState>()

    // If not enabled or no selection, all nodes are fully visible
    if (!enabled || !selectedNodeId) {
      nodes.forEach(node => {
        result.set(node.id, {
          depth: -1,
          opacity: 1.0,
          isInFocus: true,
        })
      })
      return result
    }

    // Build adjacency and compute depths
    const adjacency = buildAdjacencyMap(edges)
    const depths = computeDepths(selectedNodeId, adjacency, 3)

    // Compute opacity for each node
    nodes.forEach(node => {
      const depth = depths.get(node.id) ?? Infinity
      const isInFocus = depth <= 3

      // Get target opacity based on depth
      let targetOpacity: number
      if (depth < DEPTH_OPACITY.length) {
        targetOpacity = DEPTH_OPACITY[depth]
      } else {
        targetOpacity = DEFAULT_OPACITY
      }

      // Interpolate with transition progress (for smooth fade)
      // When transitioning IN: go from 1.0 to target
      // When transitioning OUT: go from target to 1.0
      const opacity = 1.0 + (targetOpacity - 1.0) * transitionProgress

      result.set(node.id, {
        depth: depth === Infinity ? -1 : depth,
        opacity,
        isInFocus,
      })
    })

    return result
  }, [nodes, edges, selectedNodeId, enabled, transitionProgress])
}

/**
 * Get opacity for a specific node
 */
export function getNodeFocusOpacity(
  focusStates: Map<string, NodeFocusState>,
  nodeId: string
): number {
  return focusStates.get(nodeId)?.opacity ?? 1.0
}

/**
 * Check if node is in focus (within 3 degrees of selected)
 */
export function isNodeInFocus(
  focusStates: Map<string, NodeFocusState>,
  nodeId: string
): boolean {
  return focusStates.get(nodeId)?.isInFocus ?? true
}
