import { useMemo } from 'react'
import type { SimulationNode, ClusterMode, GraphEdge } from '../lib/types'

export interface Cluster {
  id: string
  label: string
  color: string
  nodeIds: Set<string>
  // Computed from node positions
  centroid: { x: number; y: number; z: number }
  radius: number
}

interface UseClusterDetectionOptions {
  nodes: SimulationNode[]
  edges: GraphEdge[]
  mode: ClusterMode
  typeColors?: Record<string, string>
}

// Generate consistent colors for arbitrary cluster keys
function hashColor(str: string): string {
  let hash = 0
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash)
  }
  const hue = Math.abs(hash) % 360
  return `hsl(${hue}, 60%, 50%)`
}

/**
 * Detect clusters based on the selected mode
 * Returns cluster assignments and computed boundaries
 */
export function useClusterDetection({
  nodes,
  edges,
  mode,
  typeColors = {},
}: UseClusterDetectionOptions): Cluster[] {
  return useMemo(() => {
    if (mode === 'none' || nodes.length === 0) {
      return []
    }

    // Group nodes by cluster key
    const nodeGroups = new Map<string, SimulationNode[]>()

    if (mode === 'type') {
      // Group by memory type
      for (const node of nodes) {
        const key = node.type
        if (!nodeGroups.has(key)) {
          nodeGroups.set(key, [])
        }
        nodeGroups.get(key)!.push(node)
      }
    } else if (mode === 'tags') {
      // Group by shared tags - nodes sharing ANY tag cluster together
      // Priority: entity tags > project tags > other tags
      const tagToNodes = new Map<string, SimulationNode[]>()

      for (const node of nodes) {
        // Find best clustering tag (prefer entity: tags, then project-like tags)
        let bestTag = 'untagged'
        for (const tag of node.tags) {
          if (tag.startsWith('entity:')) {
            // Entity tags are highest priority (entity:people:jack, entity:organizations:automem)
            bestTag = tag
            break
          } else if (!bestTag.startsWith('entity:') && !tag.match(/^\d{4}-\d{2}$/) && tag !== 'cursor') {
            // Skip date tags (2025-12) and generic platform tags
            bestTag = tag
          }
        }

        if (!tagToNodes.has(bestTag)) {
          tagToNodes.set(bestTag, [])
        }
        tagToNodes.get(bestTag)!.push(node)
      }

      // Transfer to nodeGroups
      for (const [tag, tagNodes] of tagToNodes) {
        nodeGroups.set(tag, tagNodes)
      }
    } else if (mode === 'semantic') {
      // Group by connected components using edges
      // Nodes connected by strong relationships form clusters
      const visited = new Set<string>()
      const nodeById = new Map(nodes.map(n => [n.id, n]))

      // Build adjacency list from edges with strength >= 0.3 (lowered for tighter clusters)
      const adj = new Map<string, string[]>()
      for (const edge of edges) {
        if (edge.strength >= 0.3) {
          // GraphEdge source/target are always strings
          const source = edge.source
          const target = edge.target

          if (!adj.has(source)) adj.set(source, [])
          if (!adj.has(target)) adj.set(target, [])
          adj.get(source)!.push(target)
          adj.get(target)!.push(source)
        }
      }

      // Find connected components via BFS
      let clusterIndex = 0
      for (const node of nodes) {
        if (visited.has(node.id)) continue

        const queue = [node.id]
        const component: SimulationNode[] = []

        while (queue.length > 0) {
          const id = queue.shift()!
          if (visited.has(id)) continue
          visited.add(id)

          const n = nodeById.get(id)
          if (n) component.push(n)

          const neighbors = adj.get(id) || []
          for (const neighborId of neighbors) {
            if (!visited.has(neighborId) && nodeById.has(neighborId)) {
              queue.push(neighborId)
            }
          }
        }

        if (component.length > 0) {
          const key = `cluster-${clusterIndex++}`
          nodeGroups.set(key, component)
        }
      }
    }

    // Convert groups to Cluster objects with computed centroids
    const clusters: Cluster[] = []

    for (const [key, groupNodes] of nodeGroups) {
      if (groupNodes.length < 2) continue // Skip single-node clusters

      // Calculate centroid
      let cx = 0, cy = 0, cz = 0
      for (const node of groupNodes) {
        cx += node.x || 0
        cy += node.y || 0
        cz += node.z || 0
      }
      cx /= groupNodes.length
      cy /= groupNodes.length
      cz /= groupNodes.length

      // Calculate radius (max distance from centroid + padding)
      let maxDist = 0
      for (const node of groupNodes) {
        const dx = (node.x || 0) - cx
        const dy = (node.y || 0) - cy
        const dz = (node.z || 0) - cz
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz)
        maxDist = Math.max(maxDist, dist)
      }

      // Determine color
      let color: string
      if (mode === 'type' && typeColors[key]) {
        color = typeColors[key]
      } else {
        color = hashColor(key)
      }

      clusters.push({
        id: key,
        label: key,
        color,
        nodeIds: new Set(groupNodes.map(n => n.id)),
        centroid: { x: cx, y: cy, z: cz },
        radius: maxDist + 15, // Add padding for visual clarity
      })
    }

    return clusters
  }, [nodes, edges, mode, typeColors])
}

/**
 * Get cluster assignment for a node
 */
export function getNodeCluster(nodeId: string, clusters: Cluster[]): Cluster | null {
  for (const cluster of clusters) {
    if (cluster.nodeIds.has(nodeId)) {
      return cluster
    }
  }
  return null
}
