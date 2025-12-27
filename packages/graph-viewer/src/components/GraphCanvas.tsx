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
 * Interaction model (simplified):
 * - Mouse: Click nodes to select, OrbitControls for navigation
 * - Hand gestures: Two-hand pinch to pan/zoom/rotate; one-hand fist grab to pan
 */

import { useRef, useMemo, useState, useCallback, useEffect } from 'react'
import { Canvas, useFrame, useThree, ThreeEvent } from '@react-three/fiber'
import { OrbitControls, Text, Billboard } from '@react-three/drei'
import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing'
import * as THREE from 'three'
import { useForceLayout } from '../hooks/useForceLayout'
import { useHandGestures, GestureState } from '../hooks/useHandGestures'
import { useIPhoneHandTracking } from '../hooks/useIPhoneHandTracking'
import { useHandLockAndGrab } from '../hooks/useHandLockAndGrab'
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
import { useFocusMode, type NodeFocusState } from '../hooks/useFocusMode'
import { ClusterBoundaries } from './ClusterBoundaries'
import { SelectionHighlight, ConnectedPathsHighlight, PinchPreSelectHighlight } from './SelectionHighlight'
import { getEdgeStyle } from '../lib/edgeStyles'
import { EdgeParticles } from './EdgeParticles'
import { MiniMap } from './MiniMap'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'

// Get iPhone WebSocket URL from URL params or default
function useIPhoneUrl() {
  const [iphoneUrl, setIphoneUrl] = useState('ws://localhost:8766/ws')

  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const url = params.get('iphone_url')
    if (url) {
      setIphoneUrl(url)
    }
  }, [])

  return iphoneUrl
}

// Performance constants
const SPHERE_SEGMENTS = 12 // Reduced from 32 - good enough for small spheres
const LABEL_DISTANCE_THRESHOLD = 80 // Only show labels for nodes within this distance
const MAX_VISIBLE_LABELS = 10 // Maximum labels to show at once (for LOD)

interface GraphCanvasProps {
  nodes: GraphNode[]
  edges: GraphEdge[]
  selectedNode: GraphNode | null
  hoveredNode: GraphNode | null
  searchTerm: string
  onNodeSelect: (node: GraphNode | null) => void
  onNodeHover: (node: GraphNode | null) => void
  onNodeContextMenu?: (node: GraphNode, screenPosition: { x: number; y: number }) => void
  gestureControlEnabled?: boolean
  trackingSource?: 'mediapipe' | 'iphone'
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
  onResetViewReady?: (resetView: () => void) => void
  focusModeEnabled?: boolean
  focusTransition?: number
  // Bookmarks: expose camera state and navigation to parent
  onCameraStateForBookmarks?: (state: { x: number; y: number; z: number; zoom: number }) => void
  onNavigateForBookmarks?: (fn: (x: number, y: number, z?: number) => void) => void
  // Pathfinding: highlight path nodes and edges
  pathNodeIds?: Set<string>
  pathEdgeKeys?: Set<string>
  pathSourceId?: string | null
  pathTargetId?: string | null
  isPathSelecting?: boolean
  // Time Travel: filter nodes by timestamp
  timeTravelActive?: boolean
  timeTravelVisibleNodes?: Set<string>
  // Lasso selection
  onGetNodesInPolygon?: (fn: (polygon: { x: number; y: number }[]) => string[]) => void
  lassoSelectedIds?: Set<string>
  // Tag cloud filtering
  tagFilteredNodeIds?: Set<string>
  hasTagFilter?: boolean
}

export function GraphCanvas({
  nodes,
  edges,
  selectedNode,
  hoveredNode,
  searchTerm,
  onNodeSelect,
  onNodeHover,
  onNodeContextMenu,
  gestureControlEnabled = false,
  trackingSource: source = 'mediapipe',
  onGestureStateChange,
  onTrackingInfoChange,
  performanceMode = false,
  forceConfig = DEFAULT_FORCE_CONFIG,
  displayConfig = DEFAULT_DISPLAY_CONFIG,
  clusterConfig = DEFAULT_CLUSTER_CONFIG,
  relationshipVisibility = DEFAULT_RELATIONSHIP_VISIBILITY,
  typeColors = {},
  onReheatReady,
  onResetViewReady,
  focusModeEnabled = false,
  focusTransition = 0,
  onCameraStateForBookmarks,
  onNavigateForBookmarks,
  pathNodeIds,
  pathEdgeKeys,
  pathSourceId,
  pathTargetId,
  isPathSelecting,
  timeTravelActive = false,
  timeTravelVisibleNodes,
  onGetNodesInPolygon,
  lassoSelectedIds,
  tagFilteredNodeIds,
  hasTagFilter = false,
}: GraphCanvasProps) {
  // MiniMap state
  const [cameraState, setCameraState] = useState({ x: 0, y: 0, z: 150, zoom: 1 })
  const [layoutNodesForMiniMap, setLayoutNodesForMiniMap] = useState<SimulationNode[]>([])
  // Bimanual grab state for visual feedback
  const [bimanualActive, setBimanualActive] = useState(false)
  const navigateToRef = useRef<((x: number, y: number) => void) | null>(null)

  const handleMiniMapNavigate = useCallback((x: number, y: number) => {
    navigateToRef.current?.(x, y)
  }, [])

  // Forward camera state to parent for bookmarks
  useEffect(() => {
    onCameraStateForBookmarks?.(cameraState)
  }, [cameraState, onCameraStateForBookmarks])

  // Callback to capture and expose navigation function
  const handleNavigateToReady = useCallback((fn: (x: number, y: number) => void) => {
    navigateToRef.current = fn
    onNavigateForBookmarks?.(fn)
  }, [onNavigateForBookmarks])

  // Get iPhone WebSocket URL (from URL param or default)
  const iphoneUrl = useIPhoneUrl()

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
    <div className={`relative w-full h-full transition-shadow duration-300 ${bimanualActive ? 'ring-2 ring-inset ring-purple-500/50 shadow-[inset_0_0_30px_rgba(168,85,247,0.15)]' : ''}`}>
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
          onNodeContextMenu={onNodeContextMenu}
        gestureState={gestureState}
        gestureControlEnabled={gestureControlEnabled && gesturesActive}
        performanceMode={performanceMode}
          forceConfig={forceConfig}
          displayConfig={displayConfig}
          clusterConfig={clusterConfig}
          relationshipVisibility={relationshipVisibility}
          typeColors={typeColors}
          onReheatReady={onReheatReady}
          onResetViewReady={onResetViewReady}
          focusModeEnabled={focusModeEnabled}
          focusTransition={focusTransition}
          onCameraStateChange={setCameraState}
          onLayoutNodesChange={setLayoutNodesForMiniMap}
          onNavigateToReady={handleNavigateToReady}
          pathNodeIds={pathNodeIds}
          pathEdgeKeys={pathEdgeKeys}
          pathSourceId={pathSourceId}
          pathTargetId={pathTargetId}
          isPathSelecting={isPathSelecting}
          timeTravelActive={timeTravelActive}
          timeTravelVisibleNodes={timeTravelVisibleNodes}
          onGetNodesInPolygon={onGetNodesInPolygon}
          lassoSelectedIds={lassoSelectedIds}
          tagFilteredNodeIds={tagFilteredNodeIds}
          hasTagFilter={hasTagFilter}
          onBimanualGrabChange={setBimanualActive}
      />
    </Canvas>

      {/* MiniMap Navigator */}
      <MiniMap
        nodes={layoutNodesForMiniMap}
        selectedNode={selectedNode}
        cameraPosition={cameraState}
        cameraZoom={cameraState.zoom}
        onNavigate={handleMiniMapNavigate}
        visible={!performanceMode && layoutNodesForMiniMap.length > 0}
        size={140}
      />
    </div>
  )
}

interface SceneProps extends Omit<GraphCanvasProps, 'onGestureStateChange' | 'onTrackingInfoChange' | 'onNodeContextMenu'> {
  onNodeContextMenu?: (node: GraphNode, screenPosition: { x: number; y: number }) => void
  gestureState: GestureState
  gestureControlEnabled: boolean
  performanceMode: boolean
  onResetViewReady?: (resetView: () => void) => void
  focusModeEnabled: boolean
  focusTransition: number
  onCameraStateChange?: (state: { x: number; y: number; z: number; zoom: number }) => void
  onLayoutNodesChange?: (nodes: SimulationNode[]) => void
  onNavigateToReady?: (fn: (x: number, y: number) => void) => void
  // Pathfinding
  pathNodeIds?: Set<string>
  pathEdgeKeys?: Set<string>
  pathSourceId?: string | null
  pathTargetId?: string | null
  isPathSelecting?: boolean
  // Time Travel
  timeTravelActive?: boolean
  timeTravelVisibleNodes?: Set<string>
  // Lasso selection
  onGetNodesInPolygon?: (fn: (polygon: { x: number; y: number }[]) => string[]) => void
  lassoSelectedIds?: Set<string>
  // Tag cloud filtering
  tagFilteredNodeIds?: Set<string>
  hasTagFilter?: boolean
  // Bimanual world-manipulation feedback
  onBimanualGrabChange?: (active: boolean) => void
}

function Scene({
  nodes,
  edges,
  selectedNode,
  hoveredNode,
  searchTerm,
  onNodeSelect,
  onNodeHover,
  onNodeContextMenu,
  gestureState,
  gestureControlEnabled,
  performanceMode,
  forceConfig = DEFAULT_FORCE_CONFIG,
  displayConfig = DEFAULT_DISPLAY_CONFIG,
  clusterConfig = DEFAULT_CLUSTER_CONFIG,
  relationshipVisibility = DEFAULT_RELATIONSHIP_VISIBILITY,
  typeColors = {},
  onReheatReady,
  onResetViewReady,
  focusModeEnabled,
  focusTransition,
  onCameraStateChange,
  onLayoutNodesChange,
  onNavigateToReady,
  pathNodeIds,
  pathEdgeKeys,
  pathSourceId,
  pathTargetId,
  isPathSelecting: _isPathSelecting,
  timeTravelActive = false,
  timeTravelVisibleNodes,
  onGetNodesInPolygon,
  lassoSelectedIds,
  tagFilteredNodeIds,
  hasTagFilter = false,
  onBimanualGrabChange,
}: SceneProps) {
  const { camera } = useThree()
  const { nodes: layoutNodes, isSimulating, reheat } = useForceLayout({ nodes, edges, forceConfig })

  // DEBUG: Log node counts
  // Focus mode - compute depth-based opacity for spotlight effect
  const focusStates = useFocusMode(
    layoutNodes,
    edges,
    selectedNode?.id ?? null,
    focusModeEnabled,
    focusTransition
  )

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

  // Reset view function - centers the graph and resets rotation
  const resetView = useCallback(() => {
    if (groupRef.current) {
      groupRef.current.position.set(0, 0, 0)
      groupRef.current.rotation.set(0, 0, 0)
    }
    if (controlsRef.current) {
      controlsRef.current.reset()
    }
  }, [])

  useEffect(() => {
    if (onResetViewReady) {
      onResetViewReady(resetView)
    }
  }, [onResetViewReady, resetView])

  // MiniMap: Send layout nodes when they change
  useEffect(() => {
    onLayoutNodesChange?.(layoutNodes)
  }, [layoutNodes, onLayoutNodesChange])

  // MiniMap: Navigate to function
  const navigateTo = useCallback((x: number, y: number) => {
    if (controlsRef.current) {
      // Smoothly animate the OrbitControls target
      const controls = controlsRef.current
      const startTarget = controls.target.clone()
      const endTarget = new THREE.Vector3(x, y, 0)
      const startTime = performance.now()
      const duration = 400

      const animate = () => {
        const elapsed = performance.now() - startTime
        const progress = Math.min(elapsed / duration, 1)
        const eased = 1 - Math.pow(1 - progress, 3) // ease out cubic

        controls.target.lerpVectors(startTarget, endTarget, eased)
        controls.update()

        if (progress < 1) {
          requestAnimationFrame(animate)
        }
      }
      requestAnimationFrame(animate)
    }
  }, [])

  useEffect(() => {
    onNavigateToReady?.(navigateTo)
  }, [navigateTo, onNavigateToReady])

  // Get nodes inside a screen-space polygon (for lasso selection)
  const getNodesInPolygon = useCallback((polygon: { x: number; y: number }[]) => {
    if (polygon.length < 3) return []

    // Get the canvas size from the renderer
    const canvas = document.querySelector('canvas')
    if (!canvas) return []
    const rect = canvas.getBoundingClientRect()

    // Point-in-polygon test using ray casting
    const isPointInPolygon = (point: { x: number; y: number }) => {
      let inside = false
      const n = polygon.length
      for (let i = 0, j = n - 1; i < n; j = i++) {
        const xi = polygon[i].x
        const yi = polygon[i].y
        const xj = polygon[j].x
        const yj = polygon[j].y
        if (yi > point.y !== yj > point.y && point.x < ((xj - xi) * (point.y - yi)) / (yj - yi) + xi) {
          inside = !inside
        }
      }
      return inside
    }

    // Project each node to screen space and check if inside polygon
    const result: string[] = []
    layoutNodes.forEach((node) => {
      const worldPos = new THREE.Vector3(node.x ?? 0, node.y ?? 0, node.z ?? 0)
      const projected = worldPos.project(camera)
      const screenX = ((projected.x + 1) / 2) * rect.width
      const screenY = ((-projected.y + 1) / 2) * rect.height

      if (isPointInPolygon({ x: screenX, y: screenY })) {
        result.push(node.id)
      }
    })

    return result
  }, [layoutNodes, camera])

  // Expose getNodesInPolygon to parent
  useEffect(() => {
    onGetNodesInPolygon?.(getNodesInPolygon)
  }, [getNodesInPolygon, onGetNodesInPolygon])

  const [autoRotate, setAutoRotate] = useState(false)
  const groupRef = useRef<THREE.Group>(null)
  const controlsRef = useRef<OrbitControlsImpl>(null)

  // MiniMap: Track camera state and update periodically
  const lastCameraUpdateRef = useRef(0)
  const lastCameraPosRef = useRef({ x: 0, y: 0, z: 150 })
  useFrame(() => {
    if (!onCameraStateChange) return

    const now = performance.now()
    // Only update every 100ms to avoid excessive rerenders
    if (now - lastCameraUpdateRef.current < 100) return

    // Get camera position (accounting for OrbitControls target)
    const target = controlsRef.current?.target ?? new THREE.Vector3(0, 0, 0)
    const pos = { x: target.x, y: target.y, z: camera.position.z }

    // Check if position changed significantly
    const lastPos = lastCameraPosRef.current
    const dist = Math.sqrt(
      Math.pow(pos.x - lastPos.x, 2) +
      Math.pow(pos.y - lastPos.y, 2) +
      Math.pow(pos.z - lastPos.z, 2)
    )

    if (dist > 0.5) {
      lastCameraPosRef.current = pos
      lastCameraUpdateRef.current = now

      // Calculate zoom from camera distance
      const zoom = 150 / Math.max(camera.position.z, 10)

      onCameraStateChange({
        x: pos.x,
        y: pos.y,
        z: pos.z,
        zoom,
      })
    }
  })

  // Hand controls: two-hand pinch world manipulation + single-hand lock/grab/pinch
  const {
    lock: handLock,
    deltas: grabDeltas,
    clearRequested,
    bimanualPinch,
    leftMetrics,
    rightMetrics,
  } = useHandLockAndGrab(gestureState, gestureControlEnabled)

  // Clear selection when user holds open palm for ~0.5 seconds
  const clearWasRequestedRef = useRef(false)
  useEffect(() => {
    if (clearRequested && !clearWasRequestedRef.current && selectedNode) {
      onNodeSelect(null)
    }
    clearWasRequestedRef.current = clearRequested
  }, [clearRequested, selectedNode, onNodeSelect])

  // Notify parent of bimanual grab state for visual feedback (border glow)
  useEffect(() => {
    onBimanualGrabChange?.(bimanualPinch)
  }, [bimanualPinch, onBimanualGrabChange])

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

  // Get hovered node from layout (for hand-tracking pre-select highlight)
  const hoveredLayoutNode = useMemo(() => {
    if (!hoveredNode) return null
    return layoutNodes.find(n => n.id === hoveredNode.id) ?? null
  }, [hoveredNode, layoutNodes])

  // Track pinch strength for visual feedback (updated in useFrame)
  const [pinchStrength, setPinchStrength] = useState(0)
  const pinchStrengthRef = useRef(0)

  // Stop auto-rotate on user interaction
  const handleInteractionStart = useCallback(() => {
    setAutoRotate(false)
  }, [])

  // Track world position at grab start for displacement-based movement
  const grabStartPosRef = useRef({ x: 0, y: 0, z: 0 })
  const grabPrevTargetRef = useRef(new THREE.Vector3())
  const grabVelocityRef = useRef(new THREE.Vector3())
  const wasGrabbingRef = useRef(false)
  const inertiaActiveRef = useRef(false)

  // Bimanual navigation: two-hand pinch to pan/zoom/rotate the cloud
  const wasBimanualRef = useRef(false)
  const bimanualAnchorRef = useRef<{
    distance: number
    angle: number
    center: { x: number; y: number }
    worldPos: { x: number; y: number; z: number }
    worldRotZ: number
  } | null>(null)

  // Direct pinch selection ("pick the berry")
  // Position pinchPoint over a node on screen, pinch to select
  const PINCH_SELECT_RADIUS = 50 // pixels - fixed radius for selection
  const handHoverIdRef = useRef<string | null>(null)
  const pinchWasActiveRef = useRef(false) // for edge detection
  const lastClickMsRef = useRef(0)

  // Temp objects for grab calculations
  const tmpTarget = useMemo(() => new THREE.Vector3(), [])
  const tmpInstVel = useMemo(() => new THREE.Vector3(), [])

  // Hand controls (grab inertia + point/pinch selection)
  useFrame((_, dt) => {
    if (!gestureControlEnabled || !groupRef.current) return
    if (!gestureState.isTracking) return

    const group = groupRef.current
    const isLocked = handLock.mode === 'locked'
    const isGrabbing = isLocked && handLock.grabbed

    // --- Bimanual pinch: two-point transform (pan/zoom/rotate) ---
    if (bimanualPinch && leftMetrics && rightMetrics) {
      const PAN_SPEED = 350 // world units per normalized screen unit
      const ZOOM_SPEED = 320 // world units per ln(distance ratio)
      const ROTATE_SPEED = 1.0 // radians per radian of pinch-line rotation

      const left = leftMetrics.pinchPoint
      const right = rightMetrics.pinchPoint

      const center = { x: (left.x + right.x) / 2, y: (left.y + right.y) / 2 }
      const dx = right.x - left.x
      const dyUp = -(right.y - left.y) // flip Y so "up" is positive for angles
      const distance = Math.sqrt(dx * dx + dyUp * dyUp)

      const canonicalSegmentAngle = (angle: number) => {
        // Treat the segment as undirected: wrap to [-pi/2, pi/2) so swapping endpoints doesn't jump by pi.
        let a = angle
        while (a >= Math.PI / 2) a -= Math.PI
        while (a < -Math.PI / 2) a += Math.PI
        return a
      }

      const normalizeDeltaPi = (delta: number) => {
        // Normalize to [-pi/2, pi/2] to match canonical segment angle range.
        let d = delta
        while (d > Math.PI / 2) d -= Math.PI
        while (d < -Math.PI / 2) d += Math.PI
        return d
      }

      const angle = canonicalSegmentAngle(Math.atan2(dyUp, dx))

      if (!wasBimanualRef.current) {
        bimanualAnchorRef.current = {
          distance: Math.max(1e-4, distance),
          angle,
          center,
          worldPos: { x: group.position.x, y: group.position.y, z: group.position.z },
          worldRotZ: group.rotation.z,
        }
      }

      const anchor = bimanualAnchorRef.current
      if (anchor) {
        const safeDt = Math.max(1e-4, dt)
        const follow = 1 - Math.exp(-18 * safeDt)

        const panDx = center.x - anchor.center.x
        const panDy = center.y - anchor.center.y
        const rotationDelta = normalizeDeltaPi(angle - anchor.angle)

        // Standard pinch zoom uses a distance ratio. log() makes it symmetric for in/out.
        const distRatio = Math.max(1e-4, distance) / Math.max(1e-4, anchor.distance)
        const zoomDelta = Math.log(distRatio)

        const targetX = anchor.worldPos.x + panDx * PAN_SPEED
        const targetY = anchor.worldPos.y - panDy * PAN_SPEED
        const targetZ = anchor.worldPos.z + zoomDelta * ZOOM_SPEED
        const targetRotZ = anchor.worldRotZ + rotationDelta * ROTATE_SPEED

        group.position.x = THREE.MathUtils.lerp(group.position.x, targetX, follow)
        group.position.y = THREE.MathUtils.lerp(group.position.y, targetY, follow)
        group.position.z = THREE.MathUtils.lerp(group.position.z, targetZ, follow)
        group.rotation.z = THREE.MathUtils.lerp(group.rotation.z, targetRotZ, follow)
      }

      wasBimanualRef.current = true
      wasGrabbingRef.current = false
      return
    } else {
      wasBimanualRef.current = false
      bimanualAnchorRef.current = null
    }

    // --- Grab: follow target with damping + inertial coast on release ---
    if (isGrabbing) {
      // On first frame of grab, capture current world position
      if (grabDeltas.grabStart || !wasGrabbingRef.current) {
        grabStartPosRef.current = {
          x: group.position.x,
          y: group.position.y,
          z: group.position.z,
        }
        grabPrevTargetRef.current.set(group.position.x, group.position.y, group.position.z)
        grabVelocityRef.current.set(0, 0, 0)
        inertiaActiveRef.current = false
      }

      // Target position relative to grab start
      const startPos = grabStartPosRef.current
      tmpTarget.set(startPos.x + grabDeltas.panX, startPos.y + grabDeltas.panY, startPos.z + grabDeltas.panZ)

      // Estimate target velocity (used for inertial release)
      const safeDt = Math.max(1e-4, dt)
      tmpInstVel.copy(tmpTarget).sub(grabPrevTargetRef.current).multiplyScalar(1 / safeDt)
      grabVelocityRef.current.lerp(tmpInstVel, 0.35)
      grabPrevTargetRef.current.copy(tmpTarget)

      // Follow target with a critically-damped feel (reduces jitter while still feeling 1:1)
      const follow = 1 - Math.exp(-28 * safeDt)
      group.position.lerp(tmpTarget, follow)
    } else {
      // Released: coast briefly with exponential decay (iOS-style momentum)
      if (wasGrabbingRef.current) inertiaActiveRef.current = true

      if (inertiaActiveRef.current) {
        const safeDt = Math.max(1e-4, dt)
        group.position.x += grabVelocityRef.current.x * safeDt
        group.position.y += grabVelocityRef.current.y * safeDt
        group.position.z += grabVelocityRef.current.z * safeDt

        const decay = Math.exp(-6.5 * safeDt)
        grabVelocityRef.current.multiplyScalar(decay)

        if (grabVelocityRef.current.lengthSq() < 1) {
          grabVelocityRef.current.set(0, 0, 0)
          inertiaActiveRef.current = false
        }
      }
    }
    wasGrabbingRef.current = isGrabbing

    // --- Direct pinch selection ("pick the berry") ---
    // Only active when locked and not grabbing
    const pinchActive = isLocked && !isGrabbing

    // Update pinch strength for visual feedback
    const currentPinchStrength = isLocked ? handLock.metrics.pinch : 0
    if (Math.abs(currentPinchStrength - pinchStrengthRef.current) > 0.02) {
      pinchStrengthRef.current = currentPinchStrength
      setPinchStrength(currentPinchStrength)
    }

    if (!pinchActive) {
      // Clear hover when not in selection mode
      if (handHoverIdRef.current !== null) {
        onNodeHover(null)
        handHoverIdRef.current = null
      }
      pinchWasActiveRef.current = false
      // Reset pinch strength when not active
      if (pinchStrengthRef.current > 0.01) {
        pinchStrengthRef.current = 0
        setPinchStrength(0)
      }
      return
    }

    // Use the locked hand's pinch point when available, otherwise prefer right then left.
    const pinchPoint =
      handLock.mode === 'locked'
        ? handLock.metrics.pinchPoint
        : rightMetrics?.pinchPoint ?? leftMetrics?.pinchPoint ?? null
    if (!pinchPoint) {
      if (handHoverIdRef.current !== null) {
        onNodeHover(null)
        handHoverIdRef.current = null
      }
      return
    }

    // Get canvas size for screen-space calculations
    const canvas = document.querySelector('canvas')
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()

    // Convert pinchPoint (0-1 normalized) to screen pixels
    const pinchScreenX = pinchPoint.x * rect.width
    const pinchScreenY = pinchPoint.y * rect.height

    // Find nearest node to pinchPoint in screen space
    let nearestNode: SimulationNode | null = null
    let nearestDist = Infinity

    for (const n of layoutNodes) {
      // Get node world position (accounting for group transform)
      const worldPos = new THREE.Vector3(n.x ?? 0, n.y ?? 0, n.z ?? 0)
      group.localToWorld(worldPos)

      // Project to screen coordinates
      const projected = worldPos.project(camera)
      const screenX = ((projected.x + 1) / 2) * rect.width
      const screenY = ((-projected.y + 1) / 2) * rect.height

      // Calculate distance to pinch point
      const dx = screenX - pinchScreenX
      const dy = screenY - pinchScreenY
      const dist = Math.sqrt(dx * dx + dy * dy)

      // Check if within selection radius and closer than current best
      if (dist < PINCH_SELECT_RADIUS && dist < nearestDist) {
        nearestDist = dist
        nearestNode = n
      }
    }

    // Update hover state based on nearest node
    if (nearestNode) {
      if (handHoverIdRef.current !== nearestNode.id) {
        onNodeHover(nearestNode)
        handHoverIdRef.current = nearestNode.id
      }
    } else if (handHoverIdRef.current !== null) {
      onNodeHover(null)
      handHoverIdRef.current = null
    }

    // Get pinch activation state (with hysteresis from useHandLockAndGrab)
    const pinchActivated = handLock.mode === 'locked' && handLock.pinchActivated

    // Pinch selection (edge triggered: select on rising edge of pinchActivated)
    if (pinchActivated && !pinchWasActiveRef.current && nearestNode) {
      const nowMs = performance.now()
      // Debounce to prevent rapid double-selects
      if (nowMs - lastClickMsRef.current > 250) {
        lastClickMsRef.current = nowMs
        onNodeSelect(nearestNode)
      }
    }
    pinchWasActiveRef.current = pinchActivated
  })

  return (
    <>
      {/* Ambient lighting */}
      <ambientLight intensity={0.4} />
      <pointLight position={[100, 100, 100]} intensity={0.8} />
      <pointLight position={[-100, -100, -100]} intensity={0.4} color="#8B5CF6" />

      {/* Camera controls */}
      <OrbitControls
        ref={controlsRef}
        makeDefault
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
          focusStates={focusStates}
          pathEdgeKeys={pathEdgeKeys}
          timeTravelActive={timeTravelActive}
          timeTravelVisibleNodes={timeTravelVisibleNodes}
          tagFilteredNodeIds={tagFilteredNodeIds}
          hasTagFilter={hasTagFilter}
        />

        {/* Ambient edge particles - flowing along edges */}
        <EdgeParticles
          edges={edges}
          nodes={layoutNodes}
          enabled={!performanceMode}
          particlesPerEdge={2}
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
          onNodeContextMenu={onNodeContextMenu}
          nodeSizeScale={displayConfig.nodeSizeScale}
          focusStates={focusStates}
          pathNodeIds={pathNodeIds}
          pathSourceId={pathSourceId}
          pathTargetId={pathTargetId}
          timeTravelActive={timeTravelActive}
          timeTravelVisibleNodes={timeTravelVisibleNodes}
          lassoSelectedIds={lassoSelectedIds}
          tagFilteredNodeIds={tagFilteredNodeIds}
          hasTagFilter={hasTagFilter}
        />

        {/* Selection highlight - glowing ring around selected node */}
        {selectedLayoutNode && (
          <SelectionHighlight
            node={selectedLayoutNode}
            innerRadius={selectedLayoutNode.radius * displayConfig.nodeSizeScale * 1.3}
            outerRadius={selectedLayoutNode.radius * displayConfig.nodeSizeScale * 1.8}
          />
        )}

        {/* Pinch pre-select highlight - tightening ring for "pick the berry" selection */}
        {gestureControlEnabled && (
          <PinchPreSelectHighlight
            node={hoveredLayoutNode}
            pinchStrength={pinchStrength}
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
  focusStates: Map<string, NodeFocusState>
  pathEdgeKeys?: Set<string>
  timeTravelActive?: boolean
  timeTravelVisibleNodes?: Set<string>
  tagFilteredNodeIds?: Set<string>
  hasTagFilter?: boolean
}

function BatchedEdges({
  edges,
  nodeById,
  selectedNode,
  connectedIds,
  relationshipVisibility,
  linkThickness,
  linkOpacity,
  focusStates,
  pathEdgeKeys,
  timeTravelActive = false,
  timeTravelVisibleNodes,
  tagFilteredNodeIds,
  hasTagFilter = false,
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

      // Time Travel: hide edges when either endpoint is outside time window
      if (timeTravelActive && timeTravelVisibleNodes) {
        const sourceVisible = timeTravelVisibleNodes.has(edge.source)
        const targetVisible = timeTravelVisibleNodes.has(edge.target)
        if (!sourceVisible || !targetVisible) return
      }

      // Tag filtering: dim edges when both endpoints are not in the filtered set
      // Only hide if BOTH are outside the filter to keep edges from matching nodes visible
      if (hasTagFilter && tagFilteredNodeIds) {
        const sourceInFilter = tagFilteredNodeIds.has(edge.source)
        const targetInFilter = tagFilteredNodeIds.has(edge.target)
        if (!sourceInFilter && !targetInFilter) return
      }

      visibleCount++

      // Check if this edge is part of the pathfinding result
      const edgeKey1 = `${edge.source}-${edge.target}`
      const edgeKey2 = `${edge.target}-${edge.source}`
      const isInPath = pathEdgeKeys?.has(edgeKey1) || pathEdgeKeys?.has(edgeKey2)
      const hasActivePath = pathEdgeKeys && pathEdgeKeys.size > 0

      const isHighlighted =
        selectedNode &&
        (edge.source === selectedNode.id || edge.target === selectedNode.id)

      const isDimmed =
        (selectedNode &&
        !connectedIds.has(edge.source) &&
        !connectedIds.has(edge.target)) ||
        // Dim non-path edges when path is active
        (hasActivePath && !isInPath)

      // Source vertex
      positions.push(sourceNode.x ?? 0, sourceNode.y ?? 0, sourceNode.z ?? 0)
      // Target vertex
      positions.push(targetNode.x ?? 0, targetNode.y ?? 0, targetNode.z ?? 0)

      // Get style for this edge type
      const style = getEdgeStyle(edge.type)

      // Use style color, or bright cyan for path edges
      const color = isInPath
        ? new THREE.Color('#00d4ff')  // Bright electric cyan for path
        : new THREE.Color(style.color)

      // Get focus mode opacity for both endpoints (use minimum)
      const sourceFocus = focusStates.get(edge.source)?.opacity ?? 1
      const targetFocus = focusStates.get(edge.target)?.opacity ?? 1
      const focusOpacity = Math.min(sourceFocus, targetFocus)

      // Calculate alpha based on state and style
      let alpha = style.opacity * linkOpacity * focusOpacity
      if (isInPath) {
        // Path edges are always bright
        alpha = 1.0
      } else if (isDimmed) {
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
  }, [edges, nodeById, selectedNode, connectedIds, relationshipVisibility, linkOpacity, focusStates, pathEdgeKeys, timeTravelActive, timeTravelVisibleNodes, tagFilteredNodeIds, hasTagFilter])

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
  onNodeContextMenu?: (node: GraphNode, screenPosition: { x: number; y: number }) => void
  nodeSizeScale?: number
  focusStates: Map<string, NodeFocusState>
  pathNodeIds?: Set<string>
  pathSourceId?: string | null
  pathTargetId?: string | null
  timeTravelActive?: boolean
  timeTravelVisibleNodes?: Set<string>
  lassoSelectedIds?: Set<string>
  tagFilteredNodeIds?: Set<string>
  hasTagFilter?: boolean
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
  onNodeContextMenu,
  nodeSizeScale = 1.0,
  focusStates,
  pathNodeIds,
  pathSourceId,
  pathTargetId,
  timeTravelActive = false,
  timeTravelVisibleNodes,
  lassoSelectedIds,
  tagFilteredNodeIds,
  hasTagFilter = false,
}: InstancedNodesProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null)
  const { camera, raycaster, pointer, gl } = useThree()

  // Node lookup for raycasting - must be defined before useEffect that uses it
  const nodeIndexMap = useMemo(() => {
    const map = new Map<number, SimulationNode>()
    nodes.forEach((node, index) => {
      map.set(index, node)
    })
    return map
  }, [nodes])

  // Track pointer for click detection (distinguish click vs drag)
  const pointerDownRef = useRef<{ x: number; y: number; time: number } | null>(null)

  // DOM-level click handling (bypasses R3F's event system which doesn't work with OrbitControls)
  useEffect(() => {
    const canvas = gl.domElement

    const handlePointerDown = (e: PointerEvent) => {
      pointerDownRef.current = { x: e.clientX, y: e.clientY, time: Date.now() }
    }

    const handlePointerUp = (e: PointerEvent) => {
      if (!meshRef.current || !pointerDownRef.current) return

      const dx = e.clientX - pointerDownRef.current.x
      const dy = e.clientY - pointerDownRef.current.y
      const dt = Date.now() - pointerDownRef.current.time
      const distance = Math.sqrt(dx * dx + dy * dy)

      // Consider it a click if moved less than 5px and less than 300ms
      const isClick = distance < 5 && dt < 300

      console.log('🖱️ DOM PointerUp - distance:', distance.toFixed(1), 'dt:', dt, 'isClick:', isClick)

      if (isClick) {
        // Calculate NDC from event coordinates (R3F's pointer isn't updated for DOM events)
        const rect = canvas.getBoundingClientRect()
        const x = ((e.clientX - rect.left) / rect.width) * 2 - 1
        const y = -((e.clientY - rect.top) / rect.height) * 2 + 1

        // Force update the mesh's world matrix for accurate raycasting
        meshRef.current.updateMatrixWorld(true)

        const ndcVector = new THREE.Vector2(x, y)
        raycaster.setFromCamera(ndcVector, camera)
        const intersects = raycaster.intersectObject(meshRef.current)

        console.log('🖱️ Raycast - NDC:', x.toFixed(2), y.toFixed(2), 'intersects:', intersects.length, 'nodes:', nodeIndexMap.size)

        if (intersects.length > 0) {
          const instanceId = intersects[0].instanceId
          console.log('🖱️ Raycast hit instanceId:', instanceId)
          if (instanceId !== undefined) {
            const node = nodeIndexMap.get(instanceId)
            console.log('🖱️ Found node:', node?.id)
            if (node) {
              // Toggle selection
              onNodeSelect(selectedNode?.id === node.id ? null : node)
            }
          }
        }
      }

      pointerDownRef.current = null
    }

    canvas.addEventListener('pointerdown', handlePointerDown)
    canvas.addEventListener('pointerup', handlePointerUp)

    return () => {
      canvas.removeEventListener('pointerdown', handlePointerDown)
      canvas.removeEventListener('pointerup', handlePointerUp)
    }
  }, [gl, camera, raycaster, nodeIndexMap, onNodeSelect, selectedNode])

  // Refs to hold latest time travel state (needed for useFrame closure)
  const timeTravelActiveRef = useRef(timeTravelActive)
  const timeTravelVisibleNodesRef = useRef(timeTravelVisibleNodes)
  timeTravelActiveRef.current = timeTravelActive
  timeTravelVisibleNodesRef.current = timeTravelVisibleNodes

  // Refs for tag filtering state (needed for useFrame closure)
  const hasTagFilterRef = useRef(hasTagFilter)
  const tagFilteredNodeIdsRef = useRef(tagFilteredNodeIds)
  hasTagFilterRef.current = hasTagFilter
  tagFilteredNodeIdsRef.current = tagFilteredNodeIds

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

  // Animation state - recreate when node count changes
  const nodeCount = nodes.length
  const scalesRef = useRef<Float32Array>(new Float32Array(0))
  const targetScalesRef = useRef<Float32Array>(new Float32Array(0))
  // Deep dive: z-offset for selected node (pulls toward camera)
  const zOffsetsRef = useRef<Float32Array>(new Float32Array(0))
  const targetZOffsetsRef = useRef<Float32Array>(new Float32Array(0))

  // Resize animation arrays when node count changes
  useEffect(() => {
    if (scalesRef.current.length !== nodeCount) {
      scalesRef.current = new Float32Array(nodeCount)
      targetScalesRef.current = new Float32Array(nodeCount)
      zOffsetsRef.current = new Float32Array(nodeCount)
      targetZOffsetsRef.current = new Float32Array(nodeCount)
      // Initialize scales to 1 and z-offsets to 0
      for (let i = 0; i < nodeCount; i++) {
        scalesRef.current[i] = 1
        targetScalesRef.current[i] = 1
        zOffsetsRef.current[i] = 0
        targetZOffsetsRef.current[i] = 0
      }
    }
  }, [nodeCount])

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
      const isLassoSelected = lassoSelectedIds?.has(node.id) ?? false

      // Pathfinding state
      const isPathSource = pathSourceId === node.id
      const isPathTarget = pathTargetId === node.id
      const isInPath = pathNodeIds?.has(node.id) ?? false
      const hasActivePath = pathNodeIds && pathNodeIds.size > 0

      // Time Travel visibility - hide nodes outside the time window (use refs for fresh values)
      const isVisibleInTimeTravel = !timeTravelActiveRef.current || (timeTravelVisibleNodesRef.current?.has(node.id) ?? true)

      // Tag cloud filtering - use refs for fresh values
      const isMatchingTagFilter = !hasTagFilterRef.current || (tagFilteredNodeIdsRef.current?.has(node.id) ?? true)

      const isDimmed = !!(
        (selectedNode && !connectedIds.has(node.id)) ||
        (searchTerm && !matchingIds.has(node.id)) ||
        // Dim non-path nodes when path is active
        (hasActivePath && !isInPath) ||
        // Dim nodes not matching tag filter
        (hasTagFilterRef.current && !isMatchingTagFilter)
      )

      // Get focus mode opacity
      const focusOpacity = focusStates.get(node.id)?.opacity ?? 1

      // Target scale based on state - path nodes get a size boost
      // Time travel: nodes outside the time window scale to 0
      let targetScale: number
      if (!isVisibleInTimeTravel) {
        targetScale = 0 // Hide node by scaling to 0
      } else {
        targetScale = isSelected ? 1.5 : isHovered ? 1.2 : 1
        if (isPathSource || isPathTarget) {
          targetScale = Math.max(targetScale, 1.4)
        } else if (isInPath) {
          targetScale = Math.max(targetScale, 1.2)
        }
        // Lasso selected nodes get a slight boost
        if (isLassoSelected && !isSelected) {
          targetScale = Math.max(targetScale, 1.15)
        }
      }
      targetScalesRef.current[i] = targetScale

      // Smooth scale animation
      const currentScale = scalesRef.current[i] || 1
      const newScale = THREE.MathUtils.lerp(currentScale, targetScale, delta * 10)
      scalesRef.current[i] = newScale

      // Deep dive z-offset: selected node pops toward camera, connected nodes follow slightly
      // This creates a "focus" effect where the selected node comes forward
      const DEEP_DIVE_DISTANCE = 25 // How far forward selected node moves
      const CONNECTED_DIVE_DISTANCE = 10 // How far connected nodes follow
      let targetZOffset = 0
      if (isSelected) {
        targetZOffset = DEEP_DIVE_DISTANCE
      } else if (selectedNode && connectedIds.has(node.id)) {
        targetZOffset = CONNECTED_DIVE_DISTANCE
      }
      targetZOffsetsRef.current[i] = targetZOffset

      // Smooth z-offset animation (slightly slower for dramatic effect)
      const currentZOffset = zOffsetsRef.current[i] || 0
      const newZOffset = THREE.MathUtils.lerp(currentZOffset, targetZOffset, delta * 6)
      zOffsetsRef.current[i] = newZOffset

      // Apply pulsing for search matches, path nodes, and lasso selected
      let finalScale = newScale
      if (isSearchMatch) {
        const pulse = 1 + Math.sin(performance.now() * 0.004) * 0.15
        finalScale *= pulse
      }
      if (isInPath && !isPathSource && !isPathTarget) {
        // Subtle pulse for intermediate path nodes
        const pulse = 1 + Math.sin(performance.now() * 0.003) * 0.08
        finalScale *= pulse
      }
      if (isLassoSelected && !isSelected) {
        // Gentle pulse for lasso selected nodes
        const pulse = 1 + Math.sin(performance.now() * 0.0025) * 0.06
        finalScale *= pulse
      }

      // Node breathing - ambient pulse based on importance
      // Phase offset based on node ID to prevent synchronized breathing
      const nodePhase = (node.id.charCodeAt(0) + node.id.charCodeAt(node.id.length - 1)) * 0.1
      const breathingSpeed = 0.6 + node.importance * 0.2 // Faster for important nodes
      const breathingAmplitude = 0.015 + node.importance * 0.025 // Bigger pulse for important nodes
      const breathingTime = performance.now() * 0.001 * breathingSpeed
      const breathing = 1 + Math.sin(breathingTime + nodePhase) * breathingAmplitude
      finalScale *= breathing

      // Set position and scale (apply nodeSizeScale and deep-dive z-offset)
      // z-offset moves node toward camera (positive z in screen space)
      tempPosition.set(node.x ?? 0, node.y ?? 0, (node.z ?? 0) + newZOffset)
      tempScale.setScalar(node.radius * finalScale * nodeSizeScale)
      tempMatrix.compose(tempPosition, tempQuaternion, tempScale)
      mesh.setMatrixAt(i, tempMatrix)

      // Set color with special handling for path nodes and lasso selection
      if (isPathSource) {
        // Source node: bright green
        tempColor.set('#22c55e')
      } else if (isPathTarget) {
        // Target node: bright red/orange
        tempColor.set('#ef4444')
      } else if (isInPath) {
        // Intermediate path nodes: electric cyan
        tempColor.set('#00d4ff')
      } else if (isLassoSelected) {
        // Lasso selected nodes: blue tint
      tempColor.set(node.color)
        // Add blue tint by lerping toward blue
        const blueColor = new THREE.Color('#3b82f6')
        tempColor.lerp(blueColor, 0.35)
      } else {
        // Normal node color
        tempColor.set(node.color)
      }

      if (isDimmed && !isInPath && !isLassoSelected) {
        tempColor.multiplyScalar(0.35) // was 0.15 - too aggressive
      } else if (isSelected || isHovered || isSearchMatch || isInPath || isLassoSelected) {
        // Brighten selected/hovered/path/lasso nodes
        tempColor.multiplyScalar(isInPath ? 1.3 : isLassoSelected ? 1.15 : 1.2)
      } else {
        // Recent nodes glow brighter - subtle pulsing brightness
        const nodeTimestamp = node.timestamp ? new Date(node.timestamp).getTime() : 0
        const daysSinceCreation = (Date.now() - nodeTimestamp) / (1000 * 60 * 60 * 24)
        if (daysSinceCreation < 7) {
          // Nodes within last 7 days get a subtle brightness boost
          const recentnessFactor = 1 - (daysSinceCreation / 7) // 1 for brand new, 0 for 7 days old
          const glowPulse = 1 + Math.sin(performance.now() * 0.002 + nodePhase) * 0.1 * recentnessFactor
          tempColor.multiplyScalar(1 + recentnessFactor * 0.15 * glowPulse)
        }
      }
      // Apply focus mode opacity (but don't dim path or lasso selected nodes)
      if (!isInPath && !isLassoSelected) {
        tempColor.multiplyScalar(focusOpacity)
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

  const handleContextMenu = useCallback(
    (event: ThreeEvent<MouseEvent>) => {
      if (!meshRef.current || !onNodeContextMenu) return

      // Prevent the browser's default context menu
      event.nativeEvent.preventDefault()

      raycaster.setFromCamera(pointer, camera)
      const intersects = raycaster.intersectObject(meshRef.current)

      if (intersects.length > 0) {
        const instanceId = intersects[0].instanceId
        if (instanceId !== undefined) {
          const node = nodeIndexMap.get(instanceId)
          if (node) {
            event.stopPropagation()
            // Get screen position from the native event
            const screenPosition = {
              x: event.nativeEvent.clientX,
              y: event.nativeEvent.clientY,
            }
            onNodeContextMenu(node, screenPosition)
          }
        }
      }
    },
    [camera, pointer, raycaster, nodeIndexMap, onNodeContextMenu]
  )

  // R3F onClick handler - uses R3F's event system which works with OrbitControls
  const handleClick = useCallback(
    (event: ThreeEvent<MouseEvent>) => {
      if (!meshRef.current) return

      console.log('🖱️ R3F onClick triggered')

      raycaster.setFromCamera(pointer, camera)
      const intersects = raycaster.intersectObject(meshRef.current)

      console.log('🖱️ R3F onClick - intersects:', intersects.length)

      if (intersects.length > 0) {
        const instanceId = intersects[0].instanceId
        console.log('🖱️ R3F onClick hit instanceId:', instanceId)
        if (instanceId !== undefined) {
          const node = nodeIndexMap.get(instanceId)
          console.log('🖱️ R3F onClick found node:', node?.id)
          if (node) {
            event.stopPropagation()
            // Toggle selection
            onNodeSelect(selectedNode?.id === node.id ? null : node)
          }
        }
      }
    },
    [camera, pointer, raycaster, nodeIndexMap, onNodeSelect, selectedNode]
  )

  return (
    <instancedMesh
      key={`nodes-${nodeCount}`}
      ref={meshRef}
      args={[geometry, material, nodeCount]}
      onClick={handleClick}
      onPointerMove={handlePointerMove}
      onContextMenu={handleContextMenu}
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
