import { useEffect, useRef, useState, useCallback } from 'react'
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

export function useForceLayout({
  nodes,
  edges,
  forceConfig = DEFAULT_FORCE_CONFIG,
}: UseForceLayoutOptions): LayoutState & { reheat: () => void } {
  const simulationRef = useRef<ReturnType<typeof forceSimulation> | null>(null)
  const [layoutNodes, setLayoutNodes] = useState<SimulationNode[]>([])
  const [isSimulating, setIsSimulating] = useState(false)
  const simNodesRef = useRef<SimulationNode[]>([])

  // Initialize simulation when nodes/edges change
  useEffect(() => {
    if (nodes.length === 0) {
      setLayoutNodes([])
      simNodesRef.current = []
      return
    }

    // Create simulation nodes with initial positions
    const simNodes: SimulationNode[] = nodes.map((node, i) => {
      // Check if we have existing position for this node
      const existing = simNodesRef.current.find((n) => n.id === node.id)
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

      // Use Fibonacci sphere for initial distribution of new nodes
      const phi = Math.acos(1 - (2 * (i + 0.5)) / nodes.length)
      const theta = Math.PI * (1 + Math.sqrt(5)) * i
      const radius = 50 + (1 - node.importance) * 100 // High importance = center

      return {
        ...node,
        x: radius * Math.sin(phi) * Math.cos(theta),
        y: radius * Math.sin(phi) * Math.sin(theta),
        z: radius * Math.cos(phi),
        vx: 0,
        vy: 0,
        vz: 0,
      }
    })

    simNodesRef.current = simNodes

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
    if (simulationRef.current) {
      simulationRef.current.stop()
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

    simulationRef.current = simulation
    setIsSimulating(true)

    // Update state on each tick
    simulation.on('tick', () => {
      setLayoutNodes([...simNodes])
    })

    simulation.on('end', () => {
      setIsSimulating(false)
    })

    // Run simulation for a bit then settle
    simulation.alpha(1).restart()

    return () => {
      simulation.stop()
    }
  }, [nodes, edges]) // Note: forceConfig changes handled separately

  // Update forces when config changes (without resetting positions)
  useEffect(() => {
    const simulation = simulationRef.current
    if (!simulation) return

    // Update charge force
    const charge = simulation.force('charge') as ReturnType<typeof forceManyBody> | undefined
    if (charge) {
      charge.strength(forceConfig.chargeStrength)
    }

    // Update center force
    const center = simulation.force('center') as ReturnType<typeof forceCenter> | undefined
    if (center) {
      center.strength(forceConfig.centerStrength)
    }

    // Update collision force
    const collision = simulation.force('collision') as ReturnType<typeof forceCollide> | undefined
    if (collision) {
      collision.radius((d: SimulationNode) => d.radius * forceConfig.collisionRadius)
    }

    // Update link force
    const link = simulation.force('link') as ReturnType<typeof forceLink> | undefined
    if (link) {
      link
        .distance((d: SimulationLink) => {
          const baseDistance = forceConfig.linkDistance
          return baseDistance + (1 - d.strength) * baseDistance
        })
        .strength((d: SimulationLink) => d.strength * forceConfig.linkStrength)
    }

    // Gently reheat to apply changes
    simulation.alpha(0.3).restart()
    setIsSimulating(true)
  }, [forceConfig])

  const reheat = useCallback(() => {
    if (simulationRef.current) {
      simulationRef.current.alpha(0.5).restart()
      setIsSimulating(true)
    }
  }, [])

  return { nodes: layoutNodes, isSimulating, reheat }
}
