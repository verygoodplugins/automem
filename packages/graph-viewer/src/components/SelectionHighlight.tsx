import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import type { SimulationNode, GraphNode } from '../lib/types'

interface SelectionHighlightProps {
  node: SimulationNode | null
  color?: string
  innerRadius?: number
  outerRadius?: number
}

/**
 * Animated glowing ring around the selected node
 * Creates an Obsidian-like focus effect
 */
export function SelectionHighlight({
  node,
  color,
  innerRadius = 1.2,
  outerRadius = 1.8,
}: SelectionHighlightProps) {
  const ringRef = useRef<THREE.Mesh>(null)
  const glowRef = useRef<THREE.Mesh>(null)

  // Ring geometry
  const ringGeometry = useMemo(() => {
    return new THREE.RingGeometry(innerRadius, outerRadius, 32)
  }, [innerRadius, outerRadius])

  // Pulsing animation
  useFrame((state) => {
    if (!node || !ringRef.current || !glowRef.current) return

    const t = state.clock.elapsedTime

    // Pulse opacity
    const pulse = 0.6 + Math.sin(t * 2) * 0.2
    const material = ringRef.current.material as THREE.MeshBasicMaterial
    material.opacity = pulse

    // Slow rotation
    ringRef.current.rotation.z = t * 0.3

    // Glow pulse
    const glowMaterial = glowRef.current.material as THREE.MeshBasicMaterial
    glowMaterial.opacity = 0.3 + Math.sin(t * 3) * 0.1
    glowRef.current.scale.setScalar(1 + Math.sin(t * 2) * 0.1)
  })

  if (!node) return null

  const nodeColor = color || node.color || '#3B82F6'
  const nodeRadius = node.radius || 3

  return (
    <group position={[node.x || 0, node.y || 0, node.z || 0]}>
      {/* Inner glow sphere */}
      <mesh ref={glowRef}>
        <sphereGeometry args={[nodeRadius * 1.5, 16, 16]} />
        <meshBasicMaterial
          color={nodeColor}
          transparent
          opacity={0.3}
          side={THREE.BackSide}
          depthWrite={false}
        />
      </mesh>

      {/* Selection ring - XY plane */}
      <mesh ref={ringRef} rotation={[0, 0, 0]}>
        <primitive object={ringGeometry} />
        <meshBasicMaterial
          color={nodeColor}
          transparent
          opacity={0.7}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>

      {/* Selection ring - XZ plane */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <primitive object={ringGeometry.clone()} />
        <meshBasicMaterial
          color={nodeColor}
          transparent
          opacity={0.4}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
    </group>
  )
}

interface ConnectedPathsHighlightProps {
  selectedNode: GraphNode | null
  connectedNodes: SimulationNode[]
  color?: string
}

/**
 * Highlights paths from selected node to connected nodes
 * Creates animated flowing particles along the edges
 */
export function ConnectedPathsHighlight({
  selectedNode,
  connectedNodes,
  color,
}: ConnectedPathsHighlightProps) {
  const particlesRef = useRef<THREE.Points>(null)

  // Generate particle positions along paths
  const { positions, colors } = useMemo(() => {
    if (!selectedNode || connectedNodes.length === 0) {
      return { positions: new Float32Array(0), colors: new Float32Array(0) }
    }

    const particlesPerPath = 5
    const totalParticles = connectedNodes.length * particlesPerPath
    const positions = new Float32Array(totalParticles * 3)
    const colors = new Float32Array(totalParticles * 3)

    const selectedPos = {
      x: (selectedNode as SimulationNode).x || 0,
      y: (selectedNode as SimulationNode).y || 0,
      z: (selectedNode as SimulationNode).z || 0,
    }

    connectedNodes.forEach((node, nodeIndex) => {
      const targetPos = {
        x: node.x || 0,
        y: node.y || 0,
        z: node.z || 0,
      }

      const baseColor = new THREE.Color(color || selectedNode.color || '#3B82F6')

      for (let i = 0; i < particlesPerPath; i++) {
        const idx = (nodeIndex * particlesPerPath + i) * 3
        const t = (i + 1) / (particlesPerPath + 1)

        // Interpolate position
        positions[idx] = selectedPos.x + (targetPos.x - selectedPos.x) * t
        positions[idx + 1] = selectedPos.y + (targetPos.y - selectedPos.y) * t
        positions[idx + 2] = selectedPos.z + (targetPos.z - selectedPos.z) * t

        // Color with fade
        const fade = 1 - Math.abs(t - 0.5) * 0.5
        colors[idx] = baseColor.r * fade
        colors[idx + 1] = baseColor.g * fade
        colors[idx + 2] = baseColor.b * fade
      }
    })

    return { positions, colors }
  }, [selectedNode, connectedNodes, color])

  // Animate particles flowing along paths
  useFrame((state) => {
    if (!particlesRef.current || !selectedNode || connectedNodes.length === 0) return

    const t = state.clock.elapsedTime
    const geometry = particlesRef.current.geometry
    const positionAttr = geometry.getAttribute('position') as THREE.BufferAttribute

    if (!positionAttr || positionAttr.count === 0) return

    const selectedPos = {
      x: (selectedNode as SimulationNode).x || 0,
      y: (selectedNode as SimulationNode).y || 0,
      z: (selectedNode as SimulationNode).z || 0,
    }

    const particlesPerPath = 5

    connectedNodes.forEach((node, nodeIndex) => {
      const targetPos = {
        x: node.x || 0,
        y: node.y || 0,
        z: node.z || 0,
      }

      for (let i = 0; i < particlesPerPath; i++) {
        const idx = nodeIndex * particlesPerPath + i
        // Flow along path with offset per particle
        const baseT = (i + 1) / (particlesPerPath + 1)
        const flowT = (baseT + (t * 0.5) % 1) % 1

        positionAttr.setXYZ(
          idx,
          selectedPos.x + (targetPos.x - selectedPos.x) * flowT,
          selectedPos.y + (targetPos.y - selectedPos.y) * flowT,
          selectedPos.z + (targetPos.z - selectedPos.z) * flowT
        )
      }
    })

    positionAttr.needsUpdate = true
  })

  if (!selectedNode || connectedNodes.length === 0 || positions.length === 0) {
    return null
  }

  return (
    <points ref={particlesRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={positions.length / 3}
          array={positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={colors.length / 3}
          array={colors}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={2}
        vertexColors
        transparent
        opacity={0.8}
        sizeAttenuation
        depthWrite={false}
      />
    </points>
  )
}
