/**
 * GraphCanvas - High-performance 3D memory visualization
 *
 * Performance optimizations:
 * - Instanced mesh rendering for nodes (1 draw call for all nodes)
 * - Batched LineSegments for edges (1 draw call for all edges)
 * - Reduced geometry complexity (12x12 segments vs 32x32)
 * - LOD for labels (only show labels for nearby/selected nodes)
 * - Optional post-processing (performance mode toggle)
 * - Single useFrame callback for all animations
 */

import { useRef, useMemo, useState, useCallback, useEffect } from 'react'
import { Canvas, useFrame, useThree, ThreeEvent } from '@react-three/fiber'
import { OrbitControls, Text, Billboard } from '@react-three/drei'
import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing'
import * as THREE from 'three'
import { useForceLayout } from '../hooks/useForceLayout'
import { useHandGestures, GestureState } from '../hooks/useHandGestures'
import type { GraphNode, GraphEdge, SimulationNode } from '../lib/types'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'

// Performance constants
const SPHERE_SEGMENTS = 12 // Reduced from 32 - good enough for small spheres
const LABEL_DISTANCE_THRESHOLD = 80 // Only show labels for nodes within this distance
const MAX_VISIBLE_LABELS = 10 // Maximum labels to show at once (for LOD)

// Gesture smoothing constants - prevent sudden movements
const GESTURE_SMOOTHING = 0.15 // Lower = smoother but laggier (0.1-0.3 recommended)
const GESTURE_DEADZONE = 0.005 // Ignore tiny movements
const MAX_TRANSLATE_SPEED = 3 // Cap cloud translation per frame
const MAX_ROTATE_SPEED = 0.08 // Cap rotation rate per frame (radians)
const RECENTER_STRENGTH = 0.01 // How strongly to pull cloud back to center
const PULL_SENSITIVITY = 150 // How much Z translation per unit of depth change

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
  performanceMode?: boolean // New prop for disabling post-processing
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
  performanceMode = false,
}: GraphCanvasProps) {
  // Hand gesture tracking
  const { gestureState, isEnabled: gesturesActive } = useHandGestures({
    enabled: gestureControlEnabled,
    onGestureChange: onGestureStateChange,
  })

  return (
    <Canvas
      camera={{ position: [0, 0, 150], fov: 60 }}
      gl={{ antialias: !performanceMode, alpha: true, powerPreference: 'high-performance' }}
      style={{ background: 'linear-gradient(to bottom, #0a0a0f 0%, #0f0f18 100%)' }}
      frameloop={performanceMode ? 'demand' : 'always'}
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
        performanceMode={performanceMode}
      />
    </Canvas>
  )
}

interface SceneProps extends Omit<GraphCanvasProps, 'onGestureStateChange'> {
  gestureState: GestureState
  gestureControlEnabled: boolean
  performanceMode: boolean
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
  performanceMode,
}: SceneProps) {
  const { nodes: layoutNodes, isSimulating } = useForceLayout({ nodes, edges })
  const [autoRotate, setAutoRotate] = useState(false) // Start still, not rotating
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

  // Track previous pinch state for delta calculations
  const prevPinchStateRef = useRef<{
    leftOrigin: { x: number; y: number; z: number } | null
    rightOrigin: { x: number; y: number; z: number } | null
    distance: number
    rotation: number
    center: { x: number; y: number }
  }>({
    leftOrigin: null,
    rightOrigin: null,
    distance: 0,
    rotation: 0,
    center: { x: 0.5, y: 0.5 },
  })

  // Smoothed gesture values (to prevent sudden movements)
  const smoothedGestureRef = useRef({
    rotateDelta: 0,
    translateX: 0,
    translateY: 0,
    translateZ: 0,
  })

  // Clamp helper
  const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value))

  // Apply gesture controls to move the CLOUD (not camera) with smoothing
  // Pinch + pull back = pull cloud closer (translate Z positive toward camera)
  // Spread + push forward = push cloud away (translate Z negative away from camera)
  useEffect(() => {
    if (!gestureControlEnabled || !groupRef.current) return
    if (!gestureState.isTracking) return

    const group = groupRef.current
    const leftRay = gestureState.leftPinchRay
    const rightRay = gestureState.rightPinchRay
    const smoothed = smoothedGestureRef.current

    // Two-hand pinch manipulation: Both hands must be gripping
    const bothGripping = leftRay?.isValid && rightRay?.isValid

    if (bothGripping && leftRay && rightRay) {
      const prev = prevPinchStateRef.current

      // Calculate current pinch positions
      const leftOrigin = leftRay.origin
      const rightOrigin = rightRay.origin

      // Distance between pinch points (for detecting pinch vs spread)
      const dx = rightOrigin.x - leftOrigin.x
      const dy = rightOrigin.y - leftOrigin.y
      const currentDistance = Math.sqrt(dx * dx + dy * dy)

      // Rotation angle between pinch points
      const currentRotation = Math.atan2(dy, dx)

      // Center point between pinch origins
      const currentCenter = {
        x: (leftOrigin.x + rightOrigin.x) / 2,
        y: (leftOrigin.y + rightOrigin.y) / 2,
      }

      // Average Z depth (hands moving toward/away from camera)
      const currentZ = (leftOrigin.z + rightOrigin.z) / 2

      // Average pinch strength (how tightly fingers are pinched)
      const avgPinchStrength = (leftRay.strength + rightRay.strength) / 2

      // Only apply deltas if we have valid previous state
      if (prev.leftOrigin && prev.rightOrigin) {
        // Calculate raw deltas
        let rawRotateDelta = currentRotation - prev.rotation
        while (rawRotateDelta > Math.PI) rawRotateDelta -= 2 * Math.PI
        while (rawRotateDelta < -Math.PI) rawRotateDelta += 2 * Math.PI

        // Z depth change (positive = hands moving toward camera = pulling)
        const prevZ = (prev.leftOrigin.z + prev.rightOrigin.z) / 2
        const rawZDelta = currentZ - prevZ

        // THE KEY GESTURE:
        // Pinch (fingers together, avgPinchStrength high) + Pull back (Z increases) = bring cloud closer
        // Spread (fingers apart) + Push forward (Z decreases) = push cloud away
        // Combine pinch strength with Z movement for the pull/push gesture
        // When pinched tight and pulling back: translate cloud toward camera (+Z in world)
        // When spreading and pushing forward: translate cloud away (-Z in world)

        const rawTranslateZ = -rawZDelta * PULL_SENSITIVITY * avgPinchStrength

        // Apply smoothing (lerp toward target)
        smoothed.rotateDelta += (rawRotateDelta - smoothed.rotateDelta) * GESTURE_SMOOTHING
        smoothed.translateZ += (rawTranslateZ - smoothed.translateZ) * GESTURE_SMOOTHING

        // Clamp to max speeds
        const rotateDelta = clamp(smoothed.rotateDelta, -MAX_ROTATE_SPEED, MAX_ROTATE_SPEED)
        const translateZ = clamp(smoothed.translateZ, -MAX_TRANSLATE_SPEED, MAX_TRANSLATE_SPEED)

        // ROTATE: Rotate hands around each other = rotate the cloud
        if (Math.abs(rotateDelta) > GESTURE_DEADZONE) {
          group.rotation.y += rotateDelta
        }

        // PULL/PUSH: Move hands toward/away from camera = translate cloud in Z
        if (Math.abs(translateZ) > GESTURE_DEADZONE) {
          group.position.z += translateZ
        }

        // Gentle recenter: slowly pull cloud back toward origin
        group.position.x *= (1 - RECENTER_STRENGTH)
        group.position.y *= (1 - RECENTER_STRENGTH)
        group.position.z *= (1 - RECENTER_STRENGTH)
      }

      // Update previous state
      prevPinchStateRef.current = {
        leftOrigin,
        rightOrigin,
        distance: currentDistance,
        rotation: currentRotation,
        center: currentCenter,
      }
    } else {
      // Reset previous state when not both gripping
      prevPinchStateRef.current = {
        leftOrigin: null,
        rightOrigin: null,
        distance: 0,
        rotation: 0,
        center: { x: 0.5, y: 0.5 },
      }

      // Decay smoothed values when not gripping
      smoothed.rotateDelta *= 0.9
      smoothed.translateX *= 0.9
      smoothed.translateY *= 0.9
      smoothed.translateZ *= 0.9

      // Always gently recenter the cloud when not gripping
      if (groupRef.current) {
        const group = groupRef.current
        if (group.position.x !== 0 || group.position.y !== 0 || group.position.z !== 0) {
          group.position.x *= (1 - RECENTER_STRENGTH)
          group.position.y *= (1 - RECENTER_STRENGTH)
          group.position.z *= (1 - RECENTER_STRENGTH)
        }
        // Also slowly reset rotation
        group.rotation.y *= (1 - RECENTER_STRENGTH)
      }
    }
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

      {/* Graph content */}
      <group ref={groupRef}>
        {/* Batched edges - single draw call for all edges */}
        <BatchedEdges
          edges={edges}
          nodeById={nodeById}
          selectedNode={selectedNode}
          connectedIds={connectedIds}
        />

        {/* Instanced nodes - single draw call for all nodes */}
        <InstancedNodes
          nodes={layoutNodes}
          selectedNode={selectedNode}
          hoveredNode={hoveredNode}
          searchTerm={searchTerm}
          matchingIds={matchingIds}
          connectedIds={connectedIds}
          onNodeSelect={onNodeSelect}
          onNodeHover={onNodeHover}
        />

        {/* LOD Labels - only for selected/hovered/nearby nodes */}
        <LODLabels
          nodes={layoutNodes}
          selectedNode={selectedNode}
          hoveredNode={hoveredNode}
          searchTerm={searchTerm}
          matchingIds={matchingIds}
        />
      </group>

      {/* Post-processing effects - conditional based on performance mode */}
      {!performanceMode && (
        <EffectComposer>
          <Bloom
            luminanceThreshold={0.2}
            luminanceSmoothing={0.9}
            intensity={0.8}
            radius={0.8}
          />
          <Vignette eskil={false} offset={0.1} darkness={0.8} />
        </EffectComposer>
      )}
    </>
  )
}

/**
 * Batched edge rendering using LineSegments
 * All edges rendered in a single draw call
 */
interface BatchedEdgesProps {
  edges: GraphEdge[]
  nodeById: Map<string, SimulationNode>
  selectedNode: GraphNode | null
  connectedIds: Set<string>
}

function BatchedEdges({ edges, nodeById, selectedNode, connectedIds }: BatchedEdgesProps) {
  const lineRef = useRef<THREE.LineSegments>(null)

  // Create geometry with all edge vertices
  const { positions, colors } = useMemo(() => {
    const positions: number[] = []
    const colors: number[] = []

    edges.forEach((edge) => {
      const sourceNode = nodeById.get(edge.source)
      const targetNode = nodeById.get(edge.target)
      if (!sourceNode || !targetNode) return

      const isHighlighted =
        selectedNode &&
        (edge.source === selectedNode.id || edge.target === selectedNode.id)

      const isDimmed =
        selectedNode &&
        !connectedIds.has(edge.source) &&
        !connectedIds.has(edge.target)

      // Source vertex
      positions.push(sourceNode.x ?? 0, sourceNode.y ?? 0, sourceNode.z ?? 0)
      // Target vertex
      positions.push(targetNode.x ?? 0, targetNode.y ?? 0, targetNode.z ?? 0)

      // Parse edge color
      const color = new THREE.Color(edge.color)
      const alpha = isDimmed ? 0.05 : isHighlighted ? 0.8 : 0.3

      // Apply alpha to color (approximate, since LineBasicMaterial doesn't support per-vertex alpha)
      const r = color.r * alpha
      const g = color.g * alpha
      const b = color.b * alpha

      // Color for source and target vertices
      colors.push(r, g, b, r, g, b)
    })

    return {
      positions: new Float32Array(positions),
      colors: new Float32Array(colors),
    }
  }, [edges, nodeById, selectedNode, connectedIds])

  // Update geometry when positions/colors change
  useEffect(() => {
    if (!lineRef.current) return

    const geometry = lineRef.current.geometry
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    geometry.attributes.position.needsUpdate = true
    geometry.attributes.color.needsUpdate = true
    geometry.computeBoundingSphere()
  }, [positions, colors])

  return (
    <lineSegments ref={lineRef}>
      <bufferGeometry />
      <lineBasicMaterial vertexColors transparent opacity={0.6} />
    </lineSegments>
  )
}

/**
 * Instanced node rendering
 * All nodes rendered in a single draw call using InstancedMesh
 */
interface InstancedNodesProps {
  nodes: SimulationNode[]
  selectedNode: GraphNode | null
  hoveredNode: GraphNode | null
  searchTerm: string
  matchingIds: Set<string>
  connectedIds: Set<string>
  onNodeSelect: (node: GraphNode | null) => void
  onNodeHover: (node: GraphNode | null) => void
}

function InstancedNodes({
  nodes,
  selectedNode,
  hoveredNode,
  searchTerm,
  matchingIds,
  connectedIds,
  onNodeSelect,
  onNodeHover,
}: InstancedNodesProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null)
  const { camera, raycaster, pointer } = useThree()

  // Shared geometry and material - created once
  const geometry = useMemo(() => new THREE.SphereGeometry(1, SPHERE_SEGMENTS, SPHERE_SEGMENTS), [])
  const material = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        roughness: 0.3,
        metalness: 0.1,
        transparent: true,
      }),
    []
  )

  // Node lookup for raycasting
  const nodeIndexMap = useMemo(() => {
    const map = new Map<number, SimulationNode>()
    nodes.forEach((node, index) => {
      map.set(index, node)
    })
    return map
  }, [nodes])

  // Animation state
  const scalesRef = useRef<Float32Array>(new Float32Array(nodes.length))
  const targetScalesRef = useRef<Float32Array>(new Float32Array(nodes.length))

  // Temp objects for matrix calculations (reused to avoid GC)
  const tempMatrix = useMemo(() => new THREE.Matrix4(), [])
  const tempColor = useMemo(() => new THREE.Color(), [])
  const tempPosition = useMemo(() => new THREE.Vector3(), [])
  const tempQuaternion = useMemo(() => new THREE.Quaternion(), [])
  const tempScale = useMemo(() => new THREE.Vector3(), [])

  // Update instance matrices and colors each frame
  useFrame((_, delta) => {
    if (!meshRef.current) return

    const mesh = meshRef.current

    nodes.forEach((node, i) => {
      const isSelected = selectedNode?.id === node.id
      const isHovered = hoveredNode?.id === node.id
      const isSearchMatch = !!searchTerm && matchingIds.has(node.id)
      const isDimmed = !!(
        (selectedNode && !connectedIds.has(node.id)) ||
        (searchTerm && !matchingIds.has(node.id))
      )

      // Target scale based on state
      const targetScale = isSelected ? 1.5 : isHovered ? 1.2 : 1
      targetScalesRef.current[i] = targetScale

      // Smooth scale animation
      const currentScale = scalesRef.current[i] || 1
      const newScale = THREE.MathUtils.lerp(currentScale, targetScale, delta * 10)
      scalesRef.current[i] = newScale

      // Apply pulsing for search matches
      let finalScale = newScale
      if (isSearchMatch) {
        const pulse = 1 + Math.sin(performance.now() * 0.004) * 0.15
        finalScale *= pulse
      }

      // Set position and scale
      tempPosition.set(node.x ?? 0, node.y ?? 0, node.z ?? 0)
      tempScale.setScalar(node.radius * finalScale)
      tempMatrix.compose(tempPosition, tempQuaternion, tempScale)
      mesh.setMatrixAt(i, tempMatrix)

      // Set color with dimming
      tempColor.set(node.color)
      if (isDimmed) {
        tempColor.multiplyScalar(0.15)
      } else if (isSelected || isHovered || isSearchMatch) {
        // Brighten selected/hovered nodes
        tempColor.multiplyScalar(1.2)
      }
      mesh.setColorAt(i, tempColor)
    })

    mesh.instanceMatrix.needsUpdate = true
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
  })

  // Handle click/hover via raycasting
  const handlePointerMove = useCallback(
    (_event: ThreeEvent<PointerEvent>) => {
      if (!meshRef.current) return

      raycaster.setFromCamera(pointer, camera)
      const intersects = raycaster.intersectObject(meshRef.current)

      if (intersects.length > 0) {
        const instanceId = intersects[0].instanceId
        if (instanceId !== undefined) {
          const node = nodeIndexMap.get(instanceId)
          if (node) {
            onNodeHover(node)
            document.body.style.cursor = 'pointer'
            return
          }
        }
      }

      onNodeHover(null)
      document.body.style.cursor = 'default'
    },
    [camera, pointer, raycaster, nodeIndexMap, onNodeHover]
  )

  const handleClick = useCallback(
    (event: ThreeEvent<MouseEvent>) => {
      if (!meshRef.current) return

      raycaster.setFromCamera(pointer, camera)
      const intersects = raycaster.intersectObject(meshRef.current)

      if (intersects.length > 0) {
        const instanceId = intersects[0].instanceId
        if (instanceId !== undefined) {
          const node = nodeIndexMap.get(instanceId)
          if (node) {
            event.stopPropagation()
            onNodeSelect(selectedNode?.id === node.id ? null : node)
          }
        }
      }
    },
    [camera, pointer, raycaster, nodeIndexMap, onNodeSelect, selectedNode]
  )

  return (
    <instancedMesh
      ref={meshRef}
      args={[geometry, material, nodes.length]}
      onPointerMove={handlePointerMove}
      onClick={handleClick}
      frustumCulled={true}
    />
  )
}

/**
 * LOD Labels - Only render labels for nearby/selected/hovered nodes
 * Uses distance-based culling and limits max visible labels
 */
interface LODLabelsProps {
  nodes: SimulationNode[]
  selectedNode: GraphNode | null
  hoveredNode: GraphNode | null
  searchTerm: string
  matchingIds: Set<string>
}

function LODLabels({
  nodes,
  selectedNode,
  hoveredNode,
  searchTerm,
  matchingIds,
}: LODLabelsProps) {
  const { camera } = useThree()
  const [visibleNodes, setVisibleNodes] = useState<SimulationNode[]>([])

  // Update visible labels based on camera distance
  useFrame(() => {
    const cameraPos = camera.position

    // Always show selected and hovered nodes
    const priorityNodes: SimulationNode[] = []
    const nearbyNodes: { node: SimulationNode; distance: number }[] = []

    nodes.forEach((node) => {
      const isSelected = selectedNode?.id === node.id
      const isHovered = hoveredNode?.id === node.id
      const isSearchMatch = !!searchTerm && matchingIds.has(node.id)

      if (isSelected || isHovered) {
        priorityNodes.push(node)
        return
      }

      // Calculate distance to camera
      const dx = (node.x ?? 0) - cameraPos.x
      const dy = (node.y ?? 0) - cameraPos.y
      const dz = (node.z ?? 0) - cameraPos.z
      const distance = Math.sqrt(dx * dx + dy * dy + dz * dz)

      // Include search matches and nearby nodes
      if (distance < LABEL_DISTANCE_THRESHOLD || isSearchMatch) {
        nearbyNodes.push({ node, distance })
      }
    })

    // Sort by distance and limit
    nearbyNodes.sort((a, b) => a.distance - b.distance)
    const nearbyToShow = nearbyNodes
      .slice(0, MAX_VISIBLE_LABELS - priorityNodes.length)
      .map((n) => n.node)

    setVisibleNodes([...priorityNodes, ...nearbyToShow])
  })

  return (
    <>
      {visibleNodes.map((node) => (
        <NodeLabel
          key={node.id}
          node={node}
          isSelected={selectedNode?.id === node.id}
          isHovered={hoveredNode?.id === node.id}
        />
      ))}
    </>
  )
}

interface NodeLabelProps {
  node: SimulationNode
  isSelected: boolean
  isHovered: boolean
}

function NodeLabel({ node, isSelected, isHovered }: NodeLabelProps) {
  // Truncate content for label
  const label = useMemo(() => {
    const text = node.content.slice(0, 40)
    return text.length < node.content.length ? text + '...' : text
  }, [node.content])

  return (
    <Billboard position={[node.x ?? 0, (node.y ?? 0) + node.radius * 2 + 2, node.z ?? 0]}>
      <Text
        fontSize={2.5}
        color="#f1f5f9"
        anchorX="center"
        anchorY="bottom"
        outlineWidth={0.1}
        outlineColor="#000000"
      >
        {label}
      </Text>
      {(isSelected || isHovered) && (
        <Text
          position={[0, -1.5, 0]}
          fontSize={1.5}
          color="#94a3b8"
          anchorX="center"
          anchorY="top"
        >
          {node.type}
        </Text>
      )}
    </Billboard>
  )
}
