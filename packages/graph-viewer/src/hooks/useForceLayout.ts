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
} from '../lib/types'
import { DEFAULT_FORCE_CONFIG } from '../lib/types'

interface UseForceLayoutOptions {
  nodes: GraphNode[]
  edges: GraphEdge[]
  forceConfig?: ForceConfig
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

// Generate deterministic position offset for a tag (for initial clustering)
function getTagPosition(tag: string, index: number): { tx: number; ty: number; tz: number } {
  // Hash the tag to get a deterministic angle
  let hash = 0
  for (let i = 0; i < tag.length; i++) {
    hash = tag.charCodeAt(i) + ((hash << 5) - hash)
  }
  const angle = (Math.abs(hash) % 360) * (Math.PI / 180)
  const radius = 80 + (Math.abs(hash) % 60) // 80-140 radius for tag clusters

  return {
    tx: radius * Math.cos(angle),
    ty: radius * Math.sin(angle),
    tz: (Math.abs(hash) % 80) - 40, // -40 to 40 z offset
  }
}

// Helper to run the force simulation (pure function, no React)
function computeLayout(
  nodes: GraphNode[],
  edges: GraphEdge[],
  forceConfig: ForceConfig,
  existingNodes: SimulationNode[]
): SimulationNode[] {
  // Group nodes by primary tag for initial positioning
  const tagGroups = new Map<string, GraphNode[]>()
  for (const node of nodes) {
    const tag = getPrimaryTag(node)
    if (!tagGroups.has(tag)) tagGroups.set(tag, [])
    tagGroups.get(tag)!.push(node)
  }

  // Create simulation nodes with initial positions clustered by tag
  const simNodes: SimulationNode[] = nodes.map((node) => {
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

    // Get tag-based cluster position
    const tag = getPrimaryTag(node)
    const tagNodes = tagGroups.get(tag) || []
    const indexInTag = tagNodes.indexOf(node)
    const { tx, ty, tz } = getTagPosition(tag, indexInTag)

    // Add small random offset within cluster (Fibonacci-like spiral)
    const localPhi = Math.acos(1 - (2 * (indexInTag + 0.5)) / Math.max(tagNodes.length, 1))
    const localTheta = Math.PI * (1 + Math.sqrt(5)) * indexInTag
    const localRadius = 5 + (1 - node.importance) * 25 // Tighter local spread

    return {
      ...node,
      x: tx + localRadius * Math.sin(localPhi) * Math.cos(localTheta),
      y: ty + localRadius * Math.sin(localPhi) * Math.sin(localTheta),
      z: tz + localRadius * Math.cos(localPhi),
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

    const signature = createDataSignature(nodes)

    // Check cache - if signature matches, return cached nodes
    if (signature === layoutCache.signature && layoutCache.nodes.length > 0) {
      return layoutCache.nodes
    }

    // Compute new layout
    const computed = computeLayout(nodes, edges, forceConfig, layoutCache.nodes)

    // Update cache
    layoutCache.signature = signature
    layoutCache.nodes = computed

    return computed
  }, [nodes, edges, forceConfig])

  // Reheat function uses module-level cache
  const reheat = useCallback(() => {
    if (layoutCache.simulation) {
      layoutCache.simulation.alpha(0.5).restart()
      setIsSimulating(true)
    }
  }, [])

  return { nodes: layoutNodes, isSimulating, reheat }
}
