import { useRef, useMemo, useState, useCallback, useEffect } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Text, Billboard, Line } from '@react-three/drei'
import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing'
import * as THREE from 'three'
import { useForceLayout } from '../hooks/useForceLayout'
import { useHandGestures, GestureState } from '../hooks/useHandGestures'
// import { HandSkeletonOverlay } from './HandSkeletonOverlay' // Using 2D overlay instead
import type { GraphNode, GraphEdge, SimulationNode } from '../lib/types'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'

interface GraphCanvasProps {
  nodes: GraphNode[]
  edges: GraphEdge[]
  selectedNode: GraphNode | null
  hoveredNode: GraphNode | null
  searchTerm: string
  onNodeSelect: (node: GraphNode | null) => void
  onNodeHover: (node: GraphNode | null) => void
  gestureControlEnabled?: boolean
  onGestureStateChange?: (state: GestureState) => void
}

export function GraphCanvas({
  nodes,
  edges,
  selectedNode,
  hoveredNode,
  searchTerm,
  onNodeSelect,
  onNodeHover,
  gestureControlEnabled = false,
  onGestureStateChange,
}: GraphCanvasProps) {
  // Hand gesture tracking
  const { gestureState, isEnabled: gesturesActive } = useHandGestures({
    enabled: gestureControlEnabled,
    onGestureChange: onGestureStateChange,
  })

  return (
    <Canvas
      camera={{ position: [0, 0, 150], fov: 60 }}
      gl={{ antialias: true, alpha: true }}
      style={{ background: 'linear-gradient(to bottom, #0a0a0f 0%, #0f0f18 100%)' }}
    >
      <Scene
        nodes={nodes}
        edges={edges}
        selectedNode={selectedNode}
        hoveredNode={hoveredNode}
        searchTerm={searchTerm}
        onNodeSelect={onNodeSelect}
        onNodeHover={onNodeHover}
        gestureState={gestureState}
        gestureControlEnabled={gestureControlEnabled && gesturesActive}
      />
    </Canvas>
  )
}

interface SceneProps extends GraphCanvasProps {
  gestureState: GestureState
  gestureControlEnabled: boolean
}

function Scene({
  nodes,
  edges,
  selectedNode,
  hoveredNode,
  searchTerm,
  onNodeSelect,
  onNodeHover,
  gestureState,
  gestureControlEnabled,
}: SceneProps) {
  const { nodes: layoutNodes, isSimulating } = useForceLayout({ nodes, edges })
  const [autoRotate, setAutoRotate] = useState(true)
  const groupRef = useRef<THREE.Group>(null)
  const controlsRef = useRef<OrbitControlsImpl>(null)

  // Create node lookup for edges
  const nodeById = useMemo(
    () => new Map(layoutNodes.map((n) => [n.id, n])),
    [layoutNodes]
  )

  // Filter nodes based on search
  const searchLower = searchTerm.toLowerCase()
  const matchingIds = useMemo(() => {
    if (!searchTerm) return new Set<string>()
    return new Set(
      layoutNodes
        .filter(
          (n) =>
            n.content.toLowerCase().includes(searchLower) ||
            n.tags.some((t) => t.toLowerCase().includes(searchLower)) ||
            n.type.toLowerCase().includes(searchLower)
        )
        .map((n) => n.id)
    )
  }, [layoutNodes, searchLower, searchTerm])

  // Get connected node IDs when a node is selected
  const connectedIds = useMemo(() => {
    if (!selectedNode) return new Set<string>()
    const ids = new Set<string>([selectedNode.id])
    edges.forEach((e) => {
      if (e.source === selectedNode.id) ids.add(e.target)
      if (e.target === selectedNode.id) ids.add(e.source)
    })
    return ids
  }, [selectedNode, edges])

  // Stop auto-rotate on user interaction
  const handleInteractionStart = useCallback(() => {
    setAutoRotate(false)
  }, [])

  // Apply gesture controls to camera
  useEffect(() => {
    if (!gestureControlEnabled || !controlsRef.current) return
    if (!gestureState.isTracking || gestureState.handsDetected < 2) return

    const controls = controlsRef.current

    // Two-hand zoom: spread = zoom in, pinch = zoom out
    if (Math.abs(gestureState.zoomDelta) > 0.001) {
      // Dolly in/out based on hand spread
      const zoomFactor = 1 - gestureState.zoomDelta * 0.5
      controls.object.position.multiplyScalar(zoomFactor)
      setAutoRotate(false)
    }

    // Two-hand rotation: rotate the camera around the target
    if (Math.abs(gestureState.rotateDelta) > 0.01) {
      // Convert rotation delta to camera orbit
      const rotateSpeed = 2
      controls.object.position.applyAxisAngle(
        new THREE.Vector3(0, 1, 0),
        gestureState.rotateDelta * rotateSpeed
      )
      setAutoRotate(false)
    }

    // Two-hand pan: move target based on center movement
    if (
      Math.abs(gestureState.panDelta.x) > 0.001 ||
      Math.abs(gestureState.panDelta.y) > 0.001
    ) {
      const panSpeed = 50
      controls.target.x -= gestureState.panDelta.x * panSpeed
      controls.target.y += gestureState.panDelta.y * panSpeed
      setAutoRotate(false)
    }

    controls.update()
  }, [gestureControlEnabled, gestureState])

  return (
    <>
      {/* Ambient lighting */}
      <ambientLight intensity={0.4} />
      <pointLight position={[100, 100, 100]} intensity={0.8} />
      <pointLight position={[-100, -100, -100]} intensity={0.4} color="#8B5CF6" />

      {/* Camera controls */}
      <OrbitControls
        ref={controlsRef}
        enableDamping
        dampingFactor={0.05}
        autoRotate={autoRotate && !isSimulating && !gestureControlEnabled}
        autoRotateSpeed={0.5}
        onStart={handleInteractionStart}
        minDistance={20}
        maxDistance={500}
      />

      {/* 3D Hand skeleton overlay - disabled, using 2D overlay instead */}
      {/* {gestureControlEnabled && (
        <HandSkeletonOverlay gestureState={gestureState} enabled={true} />
      )} */}

      {/* Graph content */}
      <group ref={groupRef}>
        {/* Edges */}
        {edges.map((edge) => {
          const sourceNode = nodeById.get(edge.source)
          const targetNode = nodeById.get(edge.target)
          if (!sourceNode || !targetNode) return null

          const isHighlighted =
            selectedNode &&
            (edge.source === selectedNode.id || edge.target === selectedNode.id)

          const isDimmed =
            selectedNode &&
            !connectedIds.has(edge.source) &&
            !connectedIds.has(edge.target)

          return (
            <EdgeLine
              key={edge.id}
              source={sourceNode}
              target={targetNode}
              color={edge.color}
              strength={edge.strength}
              isHighlighted={!!isHighlighted}
              isDimmed={!!isDimmed}
            />
          )
        })}

        {/* Nodes */}
        {layoutNodes.map((node) => {
          const isSelected = selectedNode?.id === node.id
          const isHovered = hoveredNode?.id === node.id
          const isSearchMatch = !!searchTerm && matchingIds.has(node.id)
          const isDimmed = !!(
            (selectedNode && !connectedIds.has(node.id)) ||
            (searchTerm && !matchingIds.has(node.id))
          )

          return (
            <NodeSphere
              key={node.id}
              node={node}
              isSelected={isSelected}
              isHovered={isHovered}
              isSearchMatch={isSearchMatch}
              isDimmed={isDimmed}
              onSelect={() => onNodeSelect(isSelected ? null : node)}
              onHover={(hovered) => onNodeHover(hovered ? node : null)}
            />
          )
        })}
      </group>

      {/* Post-processing effects */}
      <EffectComposer>
        <Bloom
          luminanceThreshold={0.2}
          luminanceSmoothing={0.9}
          intensity={0.8}
          radius={0.8}
        />
        <Vignette eskil={false} offset={0.1} darkness={0.8} />
      </EffectComposer>
    </>
  )
}

interface NodeSphereProps {
  node: SimulationNode
  isSelected: boolean
  isHovered: boolean
  isSearchMatch: boolean
  isDimmed: boolean
  onSelect: () => void
  onHover: (hovered: boolean) => void
}

function NodeSphere({
  node,
  isSelected,
  isHovered,
  isSearchMatch,
  isDimmed,
  onSelect,
  onHover,
}: NodeSphereProps) {
  const meshRef = useRef<THREE.Mesh>(null)
  const [scale, setScale] = useState(1)

  // Animate scale on hover/select
  useFrame((_, delta) => {
    if (!meshRef.current) return

    const targetScale = isSelected ? 1.5 : isHovered ? 1.2 : 1
    const newScale = THREE.MathUtils.lerp(scale, targetScale, delta * 10)
    setScale(newScale)
    meshRef.current.scale.setScalar(newScale)
  })

  // Pulsing animation for search matches
  useFrame(({ clock }) => {
    if (!meshRef.current || !isSearchMatch) return
    const pulse = 1 + Math.sin(clock.elapsedTime * 4) * 0.15
    meshRef.current.scale.setScalar(scale * pulse)
  })

  const color = useMemo(() => new THREE.Color(node.color), [node.color])
  const emissiveIntensity = isSelected ? 0.8 : isHovered ? 0.5 : isSearchMatch ? 0.6 : 0.2
  const opacity = isDimmed ? 0.15 : node.opacity

  // Truncate content for label
  const label = useMemo(() => {
    const text = node.content.slice(0, 40)
    return text.length < node.content.length ? text + '...' : text
  }, [node.content])

  return (
    <group position={[node.x ?? 0, node.y ?? 0, node.z ?? 0]}>
      {/* Node sphere */}
      <mesh
        ref={meshRef}
        onClick={(e) => {
          e.stopPropagation()
          onSelect()
        }}
        onPointerOver={(e) => {
          e.stopPropagation()
          onHover(true)
          document.body.style.cursor = 'pointer'
        }}
        onPointerOut={() => {
          onHover(false)
          document.body.style.cursor = 'default'
        }}
      >
        <sphereGeometry args={[node.radius, 32, 32]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={emissiveIntensity}
          transparent
          opacity={opacity}
          roughness={0.3}
          metalness={0.1}
        />
      </mesh>

      {/* Outer glow ring for selected/hovered */}
      {(isSelected || isHovered) && (
        <mesh>
          <ringGeometry args={[node.radius * 1.3, node.radius * 1.5, 32]} />
          <meshBasicMaterial
            color={color}
            transparent
            opacity={0.3}
            side={THREE.DoubleSide}
          />
        </mesh>
      )}

      {/* Label (only show when hovered or selected) */}
      {(isHovered || isSelected) && (
        <Billboard>
          <Text
            position={[0, node.radius * 2 + 2, 0]}
            fontSize={2.5}
            color="#f1f5f9"
            anchorX="center"
            anchorY="bottom"
            outlineWidth={0.1}
            outlineColor="#000000"
          >
            {label}
          </Text>
          <Text
            position={[0, node.radius * 2, 0]}
            fontSize={1.5}
            color="#94a3b8"
            anchorX="center"
            anchorY="top"
          >
            {node.type}
          </Text>
        </Billboard>
      )}
    </group>
  )
}

interface EdgeLineProps {
  source: SimulationNode
  target: SimulationNode
  color: string
  strength: number
  isHighlighted: boolean
  isDimmed: boolean
}

function EdgeLine({
  source,
  target,
  color,
  strength,
  isHighlighted,
  isDimmed,
}: EdgeLineProps) {
  const points = useMemo(
    () => [
      new THREE.Vector3(source.x ?? 0, source.y ?? 0, source.z ?? 0),
      new THREE.Vector3(target.x ?? 0, target.y ?? 0, target.z ?? 0),
    ],
    [source.x, source.y, source.z, target.x, target.y, target.z]
  )

  const lineWidth = isHighlighted ? 2 : Math.max(0.5, strength * 1.5)
  const opacity = isDimmed ? 0.05 : isHighlighted ? 0.8 : 0.3

  return (
    <Line
      points={points}
      color={color}
      lineWidth={lineWidth}
      transparent
      opacity={opacity}
    />
  )
}
