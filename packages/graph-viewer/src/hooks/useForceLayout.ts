import { useEffect, useRef, useState, useCallback } from 'react'
import {
  forceSimulation,
  forceLink,
  forceManyBody,
  forceCenter,
  forceCollide,
  forceRadial,
} from 'd3-force-3d'
import type { GraphNode, GraphEdge, SimulationNode, SimulationLink } from '../lib/types'

interface UseForceLayoutOptions {
  nodes: GraphNode[]
  edges: GraphEdge[]
  strength?: number
  centerStrength?: number
  collisionRadius?: number
}

interface LayoutState {
  nodes: SimulationNode[]
  isSimulating: boolean
}

export function useForceLayout({
  nodes,
  edges,
  strength = -100,
  centerStrength = 0.05,
  collisionRadius = 2,
}: UseForceLayoutOptions): LayoutState & { reheat: () => void } {
  const simulationRef = useRef<ReturnType<typeof forceSimulation> | null>(null)
  const [layoutNodes, setLayoutNodes] = useState<SimulationNode[]>([])
  const [isSimulating, setIsSimulating] = useState(false)

  // Initialize simulation when nodes/edges change
  useEffect(() => {
    if (nodes.length === 0) {
      setLayoutNodes([])
      return
    }

    // Create simulation nodes with initial positions
    const simNodes: SimulationNode[] = nodes.map((node, i) => {
      // Use Fibonacci sphere for initial distribution
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
          .distance((d: SimulationLink) => 30 + (1 - d.strength) * 50)
          .strength((d: SimulationLink) => d.strength * 0.5)
      )
      .force('charge', forceManyBody().strength(strength))
      .force('center', forceCenter(0, 0, 0).strength(centerStrength))
      .force(
        'collision',
        forceCollide()
          .radius((d: SimulationNode) => d.radius * collisionRadius)
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
  }, [nodes, edges, strength, centerStrength, collisionRadius])

  const reheat = useCallback(() => {
    if (simulationRef.current) {
      simulationRef.current.alpha(0.5).restart()
      setIsSimulating(true)
    }
  }, [])

  return { nodes: layoutNodes, isSimulating, reheat }
}
