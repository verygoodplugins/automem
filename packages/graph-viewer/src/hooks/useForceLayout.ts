import { useState, useCallback, useMemo } from 'react'
import {
  forceSimulation,
  forceLink,
  forceManyBody,
  forceCenter,
  forceCollide,
  forceRadial,
} from 'd3-force-3d'
import type {
  GraphNode,
  GraphEdge,
  SimulationNode,
  SimulationLink,
  ForceConfig,
  ClusterMode,
} from '../lib/types'
import { DEFAULT_FORCE_CONFIG } from '../lib/types'

interface UseForceLayoutOptions {
  nodes: GraphNode[]
  edges: GraphEdge[]
  forceConfig?: ForceConfig
  useServerPositions?: boolean  // If true, use pre-computed x,y,z from server (UMAP)
  seedMode?: ClusterMode
}

interface LayoutState {
  nodes: SimulationNode[]
  isSimulating: boolean
}

// Module-level cache that survives React Strict Mode and HMR
// This is outside React's lifecycle so it persists across component recreation
const layoutCache = {
  signature: '',
  nodes: [] as SimulationNode[],
  simulation: null as ReturnType<typeof forceSimulation> | null,
  seedMode: 'tags' as ClusterMode,
}

// Helper to create data signature
function createDataSignature(nodes: GraphNode[]): string {
  if (nodes.length === 0) return ''
  return `${nodes.length}-${nodes[0]?.id}-${nodes[nodes.length - 1]?.id}`
}

// Get primary clustering tag for a node (prioritize entity tags over generic ones)
function getPrimaryTag(node: GraphNode): string {
  for (const tag of node.tags) {
    if (tag.startsWith('entity:')) return tag
  }
  for (const tag of node.tags) {
    if (!tag.match(/^\d{4}-\d{2}$/) && tag !== 'cursor') return tag
  }
  return node.tags[0] || 'untagged'
}

function getSeedKey(node: GraphNode, mode: ClusterMode): string | null {
  if (mode === 'tags') return getPrimaryTag(node)
  if (mode === 'type') return node.type
  return null
}

// Generate deterministic position offset for a group key (for initial clustering)
function getSeedPosition(key: string, _index: number): { tx: number; ty: number; tz: number } {
  // Hash the tag to get a deterministic angle
  let hash = 0
  for (let i = 0; i < key.length; i++) {
    hash = key.charCodeAt(i) + ((hash << 5) - hash)
  }
  const angle = (Math.abs(hash) % 360) * (Math.PI / 180)
  const radius = 100 + (Math.abs(hash) % 80) // 100-180 radius for group clusters

  return {
    tx: radius * Math.cos(angle),
    ty: radius * Math.sin(angle),
    tz: (Math.abs(hash) % 100) - 50, // -50 to 50 z offset
  }
}

// Evenly distributed points on a sphere (deterministic)
function getFibonacciPosition(index: number, total: number, radius: number) {
  const safeTotal = Math.max(total, 1)
  const phi = Math.acos(1 - (2 * (index + 0.5)) / safeTotal)
  const theta = Math.PI * (1 + Math.sqrt(5)) * index
  return {
    x: radius * Math.sin(phi) * Math.cos(theta),
    y: radius * Math.sin(phi) * Math.sin(theta),
    z: radius * Math.cos(phi),
  }
}

// Helper to run the force simulation (pure function, no React)
function computeLayout(
  nodes: GraphNode[],
  edges: GraphEdge[],
  forceConfig: ForceConfig,
  existingNodes: SimulationNode[],
  seedMode: ClusterMode
): SimulationNode[] {
  const normalizedSeedMode = seedMode === 'semantic' ? 'none' : seedMode
  const useGroupedSeeds = normalizedSeedMode === 'tags' || normalizedSeedMode === 'type'
  const seedGroups = new Map<string, GraphNode[]>()

  if (useGroupedSeeds) {
    for (const node of nodes) {
      const key = getSeedKey(node, normalizedSeedMode) || 'ungrouped'
      if (!seedGroups.has(key)) seedGroups.set(key, [])
      seedGroups.get(key)!.push(node)
    }
  }

  // Create simulation nodes with seeded positions
  const simNodes: SimulationNode[] = nodes.map((node, index) => {
    // Check if we have existing position for this node
    const existing = existingNodes.find((n) => n.id === node.id)
    if (existing) {
      return {
        ...node,
        x: existing.x,
        y: existing.y,
        z: existing.z,
        vx: existing.vx || 0,
        vy: existing.vy || 0,
        vz: existing.vz || 0,
      }
    }

    if (useGroupedSeeds) {
      const key = getSeedKey(node, normalizedSeedMode) || 'ungrouped'
      const groupNodes = seedGroups.get(key) || []
      const indexInGroup = groupNodes.indexOf(node)
      const { tx, ty, tz } = getSeedPosition(key, indexInGroup)

      // Add small deterministic offset within cluster (Fibonacci-like spiral)
      const localPhi = Math.acos(1 - (2 * (indexInGroup + 0.5)) / Math.max(groupNodes.length, 1))
      const localTheta = Math.PI * (1 + Math.sqrt(5)) * indexInGroup
      const localRadius = 3 + (1 - node.importance) * 20 // Tighter local spread

      return {
        ...node,
        x: tx + localRadius * Math.sin(localPhi) * Math.cos(localTheta),
        y: ty + localRadius * Math.sin(localPhi) * Math.sin(localTheta),
        z: tz + localRadius * Math.cos(localPhi),
        vx: 0,
        vy: 0,
        vz: 0,
      }
    }

    const baseRadius = 110
    const radius = baseRadius + (1 - node.importance) * 40
    const { x, y, z } = getFibonacciPosition(index, nodes.length, radius)

    return {
      ...node,
      x,
      y,
      z,
      vx: 0,
      vy: 0,
      vz: 0,
    }
  })

  // Create node lookup
  const nodeById = new Map(simNodes.map((n) => [n.id, n]))

  // Create links
  const links: SimulationLink[] = edges
    .filter((e) => nodeById.has(e.source) && nodeById.has(e.target))
    .map((e) => ({
      source: e.source,
      target: e.target,
      strength: e.strength,
      type: e.type,
    }))

  // Stop existing simulation
  if (layoutCache.simulation) {
    layoutCache.simulation.stop()
  }

  // Create 3D force simulation
  const simulation = forceSimulation(simNodes, 3)
    .force(
      'link',
      forceLink(links)
        .id((d: SimulationNode) => d.id)
        .distance((d: SimulationLink) => {
          const baseDistance = forceConfig.linkDistance
          return baseDistance + (1 - d.strength) * baseDistance
        })
        .strength((d: SimulationLink) => d.strength * forceConfig.linkStrength)
    )
    .force('charge', forceManyBody().strength(forceConfig.chargeStrength))
    .force('center', forceCenter(0, 0, 0).strength(forceConfig.centerStrength))
    .force(
      'collision',
      forceCollide()
        .radius((d: SimulationNode) => d.radius * forceConfig.collisionRadius)
        .strength(0.7)
    )
    .force(
      'radial',
      forceRadial(
        (d: SimulationNode) => 30 + (1 - d.importance) * 70,
        0,
        0,
        0
      ).strength(0.3)
    )
    .alphaDecay(0.02)
    .velocityDecay(0.3)

  // Store simulation reference in cache for reheat
  layoutCache.simulation = simulation

  // Run simulation synchronously for initial layout
  const INITIAL_TICKS = 120
  simulation.alpha(1)
  for (let i = 0; i < INITIAL_TICKS; i++) {
    simulation.tick()
  }

  return simNodes
}

export function useForceLayout({
  nodes,
  edges,
  forceConfig = DEFAULT_FORCE_CONFIG,
  useServerPositions = false,
  seedMode = 'tags',
}: UseForceLayoutOptions): LayoutState & { reheat: () => void } {
  const [isSimulating, setIsSimulating] = useState(false)

  // Use useMemo to compute layout synchronously, with module-level caching
  // This approach is immune to React Strict Mode double-invocation
  const layoutNodes = useMemo(() => {
    if (nodes.length === 0) {
      layoutCache.signature = ''
      layoutCache.nodes = []
      return []
    }

    // If server provided positions (UMAP), use them directly without force simulation
    if (useServerPositions) {
      const hasPositions = nodes.every((n) => n.x !== undefined && n.y !== undefined && n.z !== undefined)
      if (hasPositions) {
        const serverNodes: SimulationNode[] = nodes.map((node) => ({
          ...node,
          x: node.x!,
          y: node.y!,
          z: node.z!,
          vx: 0,
          vy: 0,
          vz: 0,
        }))
        // Update cache with server-provided positions
        layoutCache.signature = createDataSignature(nodes) + '-server'
        layoutCache.nodes = serverNodes
        layoutCache.seedMode = seedMode
        return serverNodes
      }
    }

    const signature = createDataSignature(nodes)

    // Check cache - if signature matches, return cached nodes
    if (
      signature === layoutCache.signature &&
      layoutCache.nodes.length > 0 &&
      layoutCache.seedMode === seedMode
    ) {
      return layoutCache.nodes
    }

    // Compute new layout using force simulation
    const reusePositions = layoutCache.seedMode === seedMode
    const computed = computeLayout(
      nodes,
      edges,
      forceConfig,
      reusePositions ? layoutCache.nodes : [],
      seedMode
    )

    // Update cache
    layoutCache.signature = signature
    layoutCache.nodes = computed
    layoutCache.seedMode = seedMode

    return computed
  }, [nodes, edges, forceConfig, useServerPositions, seedMode])

  // Reheat function uses module-level cache
  const reheat = useCallback(() => {
    if (layoutCache.simulation) {
      layoutCache.simulation.alpha(0.5).restart()
      setIsSimulating(true)
    }
  }, [])

  return { nodes: layoutNodes, isSimulating, reheat }
}
