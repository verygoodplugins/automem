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
 *
 * Hand interaction features:
 * - Stable pointer ray with arm model + One Euro Filter
 * - Accurate ray-sphere intersection for node selection
 * - Pinch-to-select with expansion animation
 * - Pull/push gestures for Z manipulation
 * - Two-hand rotation and zoom
 */

import { useRef, useMemo, useState, useCallback, useEffect } from 'react'
import { Canvas, useFrame, useThree, ThreeEvent } from '@react-three/fiber'
import { OrbitControls, Text, Billboard } from '@react-three/drei'
import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing'
import * as THREE from 'three'
import { useForceLayout } from '../hooks/useForceLayout'
import { useHandGestures, GestureState } from '../hooks/useHandGestures'
import { useIPhoneHandTracking } from '../hooks/useIPhoneHandTracking'
import { useHandInteraction } from '../hooks/useHandInteraction'
import { useHandLockAndGrab } from '../hooks/useHandLockAndGrab'
import { ExpandedNodeView } from './ExpandedNodeView'
import type {
  GraphNode,
  GraphEdge,
  SimulationNode,
  ForceConfig,
  DisplayConfig,
  ClusterConfig,
  RelationshipVisibility,
} from '../lib/types'
import { DEFAULT_FORCE_CONFIG, DEFAULT_DISPLAY_CONFIG, DEFAULT_CLUSTER_CONFIG, DEFAULT_RELATIONSHIP_VISIBILITY } from '../lib/types'
import { useClusterDetection } from '../hooks/useClusterDetection'
import { ClusterBoundaries } from './ClusterBoundaries'
import { SelectionHighlight, ConnectedPathsHighlight } from './SelectionHighlight'
import { getEdgeStyle } from '../lib/edgeStyles'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import { findNodeHit, type NodeSphere, type NodeHit } from '../hooks/useStablePointerRay'

// Check if we should use iPhone tracking (based on URL param or env)
function useTrackingSource() {
  const [source, setSource] = useState<'mediapipe' | 'iphone'>('mediapipe')
  const [iphoneUrl, setIphoneUrl] = useState('ws://localhost:8766/ws')

  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    if (params.get('iphone') === 'true') {
      setSource('iphone')
    }
    const url = params.get('iphone_url')
    if (url) {
      setIphoneUrl(url)
    }
  }, [])

  return { source, iphoneUrl, setSource }
}

// Performance constants
const SPHERE_SEGMENTS = 12 // Reduced from 32 - good enough for small spheres
const LABEL_DISTANCE_THRESHOLD = 80 // Only show labels for nodes within this distance
const MAX_VISIBLE_LABELS = 10 // Maximum labels to show at once (for LOD)

// Gesture smoothing constants - prevent sudden movements
const GESTURE_SMOOTHING = 0.15 // Lower = smoother but laggier (0.1-0.3 recommended)
const GESTURE_DEADZONE = 0.005 // Ignore tiny movements
const MAX_TRANSLATE_SPEED = 3 // Cap cloud translation per frame
const RECENTER_STRENGTH = 0.01 // How strongly to pull cloud back to center

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
  onTrackingInfoChange?: (info: {
    source: 'mediapipe' | 'iphone'
    iphoneUrl: string
    iphoneConnected: boolean
    hasLiDAR: boolean
    phoneConnected: boolean
    bridgeIps: string[]
    phonePort: number | null
  }) => void
  performanceMode?: boolean
  forceConfig?: ForceConfig
  displayConfig?: DisplayConfig
  clusterConfig?: ClusterConfig
  relationshipVisibility?: RelationshipVisibility
  typeColors?: Record<string, string>
  onReheatReady?: (reheat: () => void) => void
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
  onTrackingInfoChange,
  performanceMode = false,
  forceConfig = DEFAULT_FORCE_CONFIG,
  displayConfig = DEFAULT_DISPLAY_CONFIG,
  clusterConfig = DEFAULT_CLUSTER_CONFIG,
  relationshipVisibility = DEFAULT_RELATIONSHIP_VISIBILITY,
  typeColors = {},
  onReheatReady,
}: GraphCanvasProps) {
  // Determine tracking source
  const { source, iphoneUrl } = useTrackingSource()

  // MediaPipe hand tracking (webcam)
  const { gestureState: mediapipeState, isEnabled: mediapipeActive } = useHandGestures({
    enabled: gestureControlEnabled && source === 'mediapipe',
    onGestureChange: source === 'mediapipe' ? onGestureStateChange : undefined,
  })

  // iPhone hand tracking (WebSocket)
  const {
    gestureState: iphoneState,
    isConnected: iphoneConnected,
    hasLiDAR,
    phoneConnected,
    bridgeIps,
    phonePort,
  } = useIPhoneHandTracking({
    enabled: gestureControlEnabled && source === 'iphone',
    serverUrl: iphoneUrl,
    onGestureChange: source === 'iphone' ? onGestureStateChange : undefined,
  })

  // Use whichever source is active
  const gestureState = source === 'iphone' ? iphoneState : mediapipeState
  const gesturesActive = source === 'iphone' ? iphoneConnected : mediapipeActive

  useEffect(() => {
    onTrackingInfoChange?.({
      source,
      iphoneUrl,
      iphoneConnected,
      hasLiDAR,
      phoneConnected,
      bridgeIps,
      phonePort,
    })
  }, [onTrackingInfoChange, source, iphoneUrl, iphoneConnected, hasLiDAR, phoneConnected, bridgeIps, phonePort])

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
        forceConfig={forceConfig}
        displayConfig={displayConfig}
        clusterConfig={clusterConfig}
        relationshipVisibility={relationshipVisibility}
        typeColors={typeColors}
        onReheatReady={onReheatReady}
      />
    </Canvas>
  )
}

interface SceneProps extends Omit<GraphCanvasProps, 'onGestureStateChange' | 'onTrackingInfoChange'> {
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
  forceConfig = DEFAULT_FORCE_CONFIG,
  displayConfig = DEFAULT_DISPLAY_CONFIG,
  clusterConfig = DEFAULT_CLUSTER_CONFIG,
  relationshipVisibility = DEFAULT_RELATIONSHIP_VISIBILITY,
  typeColors = {},
  onReheatReady,
}: SceneProps) {
  const { nodes: layoutNodes, isSimulating, reheat } = useForceLayout({ nodes, edges, forceConfig })

  // Cluster detection
  const clusters = useClusterDetection({
    nodes: layoutNodes,
    edges,
    mode: clusterConfig.mode,
    typeColors,
  })

  // Expose reheat function to parent
  useEffect(() => {
    if (onReheatReady) {
      onReheatReady(reheat)
    }
  }, [onReheatReady, reheat])
  const [autoRotate, setAutoRotate] = useState(false)
  const groupRef = useRef<THREE.Group>(null)
  const controlsRef = useRef<OrbitControlsImpl>(null)
  const { camera } = useThree()

  // Expanded node state (for the bloom animation)
  const [expandedNodeId, setExpandedNodeId] = useState<string | null>(null)
  const [hitPoint, setHitPoint] = useState<{ x: number; y: number; z: number }>({ x: 0, y: 0, z: 0 })
  const [isExpanding, setIsExpanding] = useState(false)

  // Hand interaction (stable rays) - used for future two-hand gestures and internal metrics
  const { interactionState, processGestures } = useHandInteraction({
    nodes: layoutNodes,
    enableSelection: false,
    onNodeSelect: (nodeId) => {
      if (nodeId) {
        // Find the node and trigger expansion
        const node = layoutNodes.find(n => n.id === nodeId)
        if (node) {
          setExpandedNodeId(nodeId)
          setHitPoint({
            x: interactionState.hoveredNode?.point.x ?? node.x ?? 0,
            y: interactionState.hoveredNode?.point.y ?? node.y ?? 0,
            z: interactionState.hoveredNode?.point.z ?? node.z ?? 0,
          })
          setIsExpanding(true)
        }
        onNodeSelect(node ?? null)
      } else {
        setExpandedNodeId(null)
        setIsExpanding(false)
        onNodeSelect(null)
      }
    },
    // We'll drive hover from the explicit pointing ray (below)
    onNodeHover: undefined,
  })

  // Process gestures each frame
  useEffect(() => {
    if (gestureControlEnabled && gestureState.isTracking) {
      processGestures(gestureState)
    }
  }, [gestureControlEnabled, gestureState, processGestures])

  // New UI: open-palm acquire/lock + fist grab controls (single-hand for now)
  const { lock: handLock, deltas: grabDeltas } = useHandLockAndGrab(gestureState, gestureControlEnabled)

  // Pointing ray + pinch-click
  const [aimHit, setAimHit] = useState<NodeHit | null>(null)
  const [aimWorldPoint, setAimWorldPoint] = useState<{ x: number; y: number; z: number } | null>(null)
  const aimRayRef = useRef<{ origin: THREE.Vector3; direction: THREE.Vector3 } | null>(null)
  const pinchDownRef = useRef(false)
  const pressedNodeRef = useRef<string | null>(null)

  const nodeSpheres: NodeSphere[] = useMemo(() => {
    return layoutNodes.map((n) => ({
      id: n.id,
      x: n.x ?? 0,
      y: n.y ?? 0,
      z: n.z ?? 0,
      radius: (n.radius ?? 1) * 1.5,
    }))
  }, [layoutNodes])

  // Helper: select a node by id and animate expansion
  const selectNodeById = useCallback(
    (nodeId: string | null, hit?: { x: number; y: number; z: number }) => {
      if (nodeId) {
        const node = layoutNodes.find((n) => n.id === nodeId) ?? null
        if (node) {
          setExpandedNodeId(nodeId)
          setHitPoint({
            x: hit?.x ?? node.x ?? 0,
            y: hit?.y ?? node.y ?? 0,
            z: hit?.z ?? node.z ?? 0,
          })
          setIsExpanding(true)
        }
        onNodeSelect(node)
      } else {
        setExpandedNodeId(null)
        setIsExpanding(false)
        onNodeSelect(null)
      }
    },
    [layoutNodes, onNodeSelect]
  )

  // Create node lookup for edges
  const nodeById = useMemo(
    () => new Map(layoutNodes.map((n) => [n.id, n])),
    [layoutNodes]
  )

  // Pointing + pinch click (Meta-style: index points, pinch clicks; fist grabs)
  useEffect(() => {
    if (!gestureControlEnabled) return

    if (handLock.mode !== 'locked') {
      setAimHit(null)
      setAimWorldPoint(null)
      aimRayRef.current = null
      pinchDownRef.current = false
      pressedNodeRef.current = null
      onNodeHover(null)
      return
    }

    const m = handLock.metrics
    const isAimPose = !handLock.grabbed && m.point > 0.55
    if (!isAimPose) {
      setAimHit(null)
      setAimWorldPoint(null)
      aimRayRef.current = null
      pinchDownRef.current = false
      pressedNodeRef.current = null
      onNodeHover(null)
      return
    }

    const handData = handLock.hand === 'right' ? gestureState.rightHand : gestureState.leftHand
    if (!handData) return

    const indexTip = handData.landmarks[8]

    // Convert to screen-space (0..1, origin bottom-left)
    // Both MediaPipe and iPhone are in "camera image" coords (x left->right, y top->bottom).
    // For intuitive interaction, mirror X like a selfie preview, and invert Y to bottom-left origin.
    const screenX = 1 - indexTip.x
    const screenY = 1 - indexTip.y

    // Build a ray from the camera through the screen point
    const ndcX = screenX * 2 - 1
    const ndcY = screenY * 2 - 1
    const worldPoint = new THREE.Vector3(ndcX, ndcY, 0.5).unproject(camera)
    const direction = worldPoint.sub(camera.position).normalize()
    const origin = camera.position.clone()
    aimRayRef.current = { origin, direction: direction.clone() }

    // Hit test against spheres in graph GROUP local coordinates
    const group = groupRef.current
    let hit: NodeHit | null = null
    if (group) {
      const inv = group.matrixWorld.clone().invert()
      const localOrigin = origin.clone().applyMatrix4(inv)
      const localDir = direction.clone().transformDirection(inv)
      hit = findNodeHit(
        {
          origin: { x: localOrigin.x, y: localOrigin.y, z: localOrigin.z },
          direction: { x: localDir.x, y: localDir.y, z: localDir.z },
        },
        nodeSpheres,
        4000
      )
    }

    setAimHit(hit)

    // World-space aim point for rendering (hit point if present; otherwise plane intersection near graph center)
    let aimPointWorld: THREE.Vector3 | null = null
    if (group && hit) {
      aimPointWorld = new THREE.Vector3(hit.point.x, hit.point.y, hit.point.z).applyMatrix4(group.matrixWorld)
    } else if (group) {
      const center = group.getWorldPosition(new THREE.Vector3())
      const normal = camera.getWorldDirection(new THREE.Vector3()).normalize()
      const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(normal, center)
      const ray = new THREE.Ray(origin, direction)
      aimPointWorld = ray.intersectPlane(plane, new THREE.Vector3()) || null
    }
    if (!aimPointWorld) {
      aimPointWorld = origin.clone().add(direction.clone().multiplyScalar(250))
    }
    setAimWorldPoint({ x: aimPointWorld.x, y: aimPointWorld.y, z: aimPointWorld.z })

    // Hover highlight
    const hoverNode = hit ? (nodeById.get(hit.nodeId) ?? null) : null
    onNodeHover(hoverNode as GraphNode | null)

    // Pinch click (only when aiming). Trigger on release to allow cancel.
    const pinchActive = m.pinch > 0.75
    if (pinchActive && !pinchDownRef.current) {
      pinchDownRef.current = true
      pressedNodeRef.current = hit?.nodeId ?? null
      return
    }

    if (!pinchActive && pinchDownRef.current) {
      pinchDownRef.current = false
      const pressed = pressedNodeRef.current
      pressedNodeRef.current = null
      if (pressed && hit?.nodeId === pressed) {
        selectNodeById(pressed, hit.point)
      }
    }
  }, [
    gestureControlEnabled,
    camera,
    nodeSpheres,
    nodeById,
    onNodeHover,
    selectNodeById,
    gestureState.leftHand,
    gestureState.rightHand,
    handLock.mode,
    (handLock as any).hand,
    (handLock as any).grabbed,
    (handLock as any).metrics?.point,
    (handLock as any).metrics?.pinch,
  ])

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

  // Get connected nodes for selection highlight
  const connectedNodes = useMemo(() => {
    if (!selectedNode) return []
    const ids = new Set<string>()
    edges.forEach((e) => {
      if (e.source === selectedNode.id) ids.add(e.target)
      if (e.target === selectedNode.id) ids.add(e.source)
    })
    return layoutNodes.filter(n => ids.has(n.id))
  }, [selectedNode, edges, layoutNodes])

  // Get selected node from layout (with current position)
  const selectedLayoutNode = useMemo(() => {
    if (!selectedNode) return null
    return layoutNodes.find(n => n.id === selectedNode.id) ?? null
  }, [selectedNode, layoutNodes])

  // Get expanded node and its connections
  const expandedNode = useMemo(() => {
    if (!expandedNodeId) return null
    return layoutNodes.find(n => n.id === expandedNodeId) ?? null
  }, [expandedNodeId, layoutNodes])

  const connectedToExpanded = useMemo(() => {
    if (!expandedNodeId) return []
    const connectedNodeIds = new Set<string>()
    edges.forEach(e => {
      if (e.source === expandedNodeId) connectedNodeIds.add(e.target)
      if (e.target === expandedNodeId) connectedNodeIds.add(e.source)
    })
    return layoutNodes.filter(n => connectedNodeIds.has(n.id))
  }, [expandedNodeId, edges, layoutNodes])

  // Stop auto-rotate on user interaction
  const handleInteractionStart = useCallback(() => {
    setAutoRotate(false)
  }, [])

  // Close expanded node
  const handleCloseExpanded = useCallback(() => {
    setExpandedNodeId(null)
    setIsExpanding(false)
    onNodeSelect(null)
  }, [onNodeSelect])

  // Smoothed gesture values (to prevent sudden movements)
  const smoothedGestureRef = useRef({
    translateZ: 0,
    rotateX: 0,
    rotateY: 0,
  })

  // Clamp helper
  const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value))

  // Apply gesture controls to move the CLOUD (not camera) with smoothing
  // Now also uses the new interaction state for more precise control
  useEffect(() => {
    if (!gestureControlEnabled || !groupRef.current) return
    if (!gestureState.isTracking) return

    const group = groupRef.current
    const smoothed = smoothedGestureRef.current

    // Use the new interaction state for cloud manipulation
    const { rotationDelta, zoomDelta, dragDeltaZ, isDragging } = interactionState

    const usingGrabControls = handLock.mode === 'locked' && handLock.grabbed

    if (usingGrabControls) {
      // Exponential zoom velocity already computed; smooth + clamp
      smoothed.translateZ += (grabDeltas.zoom - smoothed.translateZ) * GESTURE_SMOOTHING
      const zVel = clamp(smoothed.translateZ, -MAX_TRANSLATE_SPEED, MAX_TRANSLATE_SPEED)
      if (Math.abs(zVel) > GESTURE_DEADZONE) {
        group.position.z += zVel
      }

      // Rotation (pitch/yaw)
      const rx = clamp(grabDeltas.rotateX, -0.08, 0.08)
      const ry = clamp(grabDeltas.rotateY, -0.08, 0.08)
      if (Math.abs(rx) > GESTURE_DEADZONE) group.rotation.x += rx
      if (Math.abs(ry) > GESTURE_DEADZONE) group.rotation.y += ry
    } else {
      // Apply zoom (two-hand spread/pinch)
      if (Math.abs(zoomDelta) > GESTURE_DEADZONE) {
        group.position.z += zoomDelta * 0.5
      }

      // Apply rotation (two-hand rotation)
      if (Math.abs(rotationDelta.x) > GESTURE_DEADZONE) {
        group.rotation.z += rotationDelta.x
      }

      // Apply Z drag (single hand push/pull when not selecting a node)
      if (!isDragging && Math.abs(dragDeltaZ) > GESTURE_DEADZONE) {
        smoothed.translateZ += (dragDeltaZ - smoothed.translateZ) * GESTURE_SMOOTHING
        const clamped = clamp(smoothed.translateZ, -MAX_TRANSLATE_SPEED, MAX_TRANSLATE_SPEED)
        group.position.z += clamped
      }
    }

    // Decay smoothed values
    smoothed.translateZ *= 0.9
    smoothed.rotateX *= 0.9
    smoothed.rotateY *= 0.9

    // Gentle recenter: slowly pull cloud back toward origin
    group.position.x *= (1 - RECENTER_STRENGTH)
    group.position.y *= (1 - RECENTER_STRENGTH)
    group.position.z *= (1 - RECENTER_STRENGTH)
    group.rotation.x *= (1 - RECENTER_STRENGTH)
    group.rotation.y *= (1 - RECENTER_STRENGTH)
  }, [gestureControlEnabled, gestureState, interactionState, handLock, grabDeltas])

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
        {/* Cluster boundaries (rendered behind edges) */}
        <ClusterBoundaries
          clusters={clusters}
          visible={clusterConfig.showBoundaries}
          opacity={0.25}
        />

        {/* Batched edges - single draw call for all edges */}
        <BatchedEdges
          edges={edges}
          nodeById={nodeById}
          selectedNode={selectedNode}
          connectedIds={connectedIds}
          relationshipVisibility={relationshipVisibility}
          linkThickness={displayConfig.linkThickness}
          linkOpacity={displayConfig.linkOpacity}
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
          nodeSizeScale={displayConfig.nodeSizeScale}
        />

        {/* Selection highlight - glowing ring around selected node */}
        {selectedLayoutNode && (
          <SelectionHighlight
            node={selectedLayoutNode}
            innerRadius={selectedLayoutNode.radius * displayConfig.nodeSizeScale * 1.3}
            outerRadius={selectedLayoutNode.radius * displayConfig.nodeSizeScale * 1.8}
          />
        )}

        {/* Connected paths highlight - particles flowing to connected nodes */}
        {selectedNode && connectedNodes.length > 0 && (
          <ConnectedPathsHighlight
            selectedNode={selectedNode}
            connectedNodes={connectedNodes}
          />
        )}

        {/* LOD Labels - only for selected/hovered/nearby nodes */}
        {displayConfig.showLabels && (
          <LODLabels
            nodes={layoutNodes}
            selectedNode={selectedNode}
            hoveredNode={hoveredNode}
            searchTerm={searchTerm}
            labelFadeDistance={displayConfig.labelFadeDistance}
            matchingIds={matchingIds}
          />
        )}

        {/* Expanded Node View - shows when a node is selected via hand */}
        {expandedNode && (
          <ExpandedNodeView
            node={expandedNode}
            connectedNodes={connectedToExpanded}
            edges={edges}
            hitPoint={hitPoint}
            onClose={handleCloseExpanded}
            isExpanding={isExpanding}
          />
        )}
      </group>

      {/* Aim cursor + laser (Meta-style: index aims, pinch clicks). Laser only appears while pinching. */}
      {gestureControlEnabled &&
        handLock.mode === 'locked' &&
        !handLock.grabbed &&
        handLock.metrics.point > 0.55 &&
        aimWorldPoint && (
          <>
            {/* Cursor at aim point (hit point when hovering a node; otherwise a plane in front of the cloud) */}
            <mesh position={[aimWorldPoint.x, aimWorldPoint.y, aimWorldPoint.z]}>
              <sphereGeometry args={[handLock.metrics.pinch > 0.75 ? (aimHit ? 0.9 : 0.7) : (aimHit ? 0.55 : 0.4), 16, 16]} />
              <meshBasicMaterial
                color={handLock.metrics.pinch > 0.75 ? '#ffffff' : aimHit ? '#fbbf24' : '#94a3b8'}
                transparent
                opacity={0.9}
              />
            </mesh>

            {/* Laser beam only while pinching */}
            {handLock.metrics.pinch > 0.75 && aimRayRef.current && (
              <line>
                <bufferGeometry>
                  <bufferAttribute
                    attach="attributes-position"
                    args={[
                      new Float32Array([
                        // start a bit in front of camera along the aim ray
                        aimRayRef.current.origin.x + aimRayRef.current.direction.x * 20,
                        aimRayRef.current.origin.y + aimRayRef.current.direction.y * 20,
                        aimRayRef.current.origin.z + aimRayRef.current.direction.z * 20,
                        // end at aim point
                        aimWorldPoint.x,
                        aimWorldPoint.y,
                        aimWorldPoint.z,
                      ]),
                      3,
                    ]}
                  />
                </bufferGeometry>
                <lineBasicMaterial color="#ffffff" transparent opacity={0.85} />
              </line>
            )}
          </>
        )}

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
 * All edges rendered in a single draw call with relationship-based styling
 */
interface BatchedEdgesProps {
  edges: GraphEdge[]
  nodeById: Map<string, SimulationNode>
  selectedNode: GraphNode | null
  connectedIds: Set<string>
  relationshipVisibility: RelationshipVisibility
  linkThickness: number
  linkOpacity: number
}

function BatchedEdges({
  edges,
  nodeById,
  selectedNode,
  connectedIds,
  relationshipVisibility,
  linkThickness,
  linkOpacity,
}: BatchedEdgesProps) {
  const lineRef = useRef<THREE.LineSegments>(null)

  // Filter edges by visibility and create geometry
  const { positions, colors, visibleCount } = useMemo(() => {
    const positions: number[] = []
    const colors: number[] = []
    let visibleCount = 0

    edges.forEach((edge) => {
      // Filter by relationship visibility
      if (!relationshipVisibility[edge.type]) return

      const sourceNode = nodeById.get(edge.source)
      const targetNode = nodeById.get(edge.target)
      if (!sourceNode || !targetNode) return

      visibleCount++

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

      // Get style for this edge type
      const style = getEdgeStyle(edge.type)

      // Use style color instead of edge.color
      const color = new THREE.Color(style.color)

      // Calculate alpha based on state and style
      let alpha = style.opacity * linkOpacity
      if (isDimmed) {
        alpha *= 0.1
      } else if (isHighlighted) {
        alpha = Math.min(1, alpha * 1.5)
      }

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
      visibleCount,
    }
  }, [edges, nodeById, selectedNode, connectedIds, relationshipVisibility, linkOpacity])

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

  if (visibleCount === 0) return null

  return (
    <lineSegments ref={lineRef}>
      <bufferGeometry />
      <lineBasicMaterial
        vertexColors
        transparent
        opacity={linkOpacity}
        linewidth={linkThickness}
      />
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
  nodeSizeScale?: number
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
  nodeSizeScale = 1.0,
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

      // Set position and scale (apply nodeSizeScale)
      tempPosition.set(node.x ?? 0, node.y ?? 0, node.z ?? 0)
      tempScale.setScalar(node.radius * finalScale * nodeSizeScale)
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
  labelFadeDistance?: number
}

function LODLabels({
  nodes,
  selectedNode,
  hoveredNode,
  searchTerm,
  matchingIds,
  labelFadeDistance = LABEL_DISTANCE_THRESHOLD,
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
      if (distance < labelFadeDistance || isSearchMatch) {
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
