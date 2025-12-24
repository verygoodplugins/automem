import { useState, useCallback, useEffect, useRef, useMemo } from 'react'
import { Settings } from 'lucide-react'

// Build version - update this when making significant changes
const BUILD_VERSION = '2024-12-23-obsidian-settings-v1'
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels'
import { useGraphSnapshot } from './hooks/useGraphData'
import { useAuth } from './hooks/useAuth'
import { GraphCanvas } from './components/GraphCanvas'
import { Inspector } from './components/Inspector'
import { SearchBar } from './components/SearchBar'
import { TokenPrompt } from './components/TokenPrompt'
import { StatsBar } from './components/StatsBar'
import { GestureDebugOverlay } from './components/GestureDebugOverlay'
import { Hand2DOverlay } from './components/Hand2DOverlay'
import { HandControlOverlay } from './components/HandControlOverlay'
import { SettingsPanel } from './components/settings'
import { BookmarksPanel } from './components/BookmarksPanel'
import { PathfindingOverlay } from './components/PathfindingOverlay'
import { TimelineBar } from './components/TimelineBar'
import { RadialMenu } from './components/RadialMenu'
import { LassoOverlay } from './components/LassoOverlay'
import { SelectionActions } from './components/SelectionActions'
import { TagCloud } from './components/TagCloud'
import { useHandLockAndGrab } from './hooks/useHandLockAndGrab'
import { useTagCloud } from './hooks/useTagCloud'
import { useKeyboardNavigation } from './hooks/useKeyboardNavigation'
import { useBookmarks, type Bookmark } from './hooks/useBookmarks'
import { usePathfinding } from './hooks/usePathfinding'
import { useTimeTravel } from './hooks/useTimeTravel'
import { useSoundEffects } from './hooks/useSoundEffects'
import type {
  GraphNode,
  FilterState,
  ForceConfig,
  DisplayConfig,
  ClusterConfig,
  RelationshipVisibility,
} from './lib/types'
import {
  DEFAULT_FORCE_CONFIG,
  DEFAULT_DISPLAY_CONFIG,
  DEFAULT_CLUSTER_CONFIG,
  DEFAULT_RELATIONSHIP_VISIBILITY,
} from './lib/types'
import type { GestureState } from './hooks/useHandGestures'

// Default gesture state for when not tracking
const DEFAULT_GESTURE_STATE: GestureState = {
  isTracking: false,
  handsDetected: 0,
  leftHand: null,
  rightHand: null,
  twoHandDistance: 0.5,
  twoHandRotation: 0,
  twoHandCenter: { x: 0.5, y: 0.5 },
  pointingHand: null,
  pointDirection: null,
  pinchStrength: 0,
  grabStrength: 0,
  leftPinchRay: null,
  rightPinchRay: null,
  activePinchRay: null,
  zoomDelta: 0,
  rotateDelta: 0,
  panDelta: { x: 0, y: 0 },
}

// Hand icon SVG component
function HandIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M18 11V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2v0" />
      <path d="M14 10V4a2 2 0 0 0-2-2a2 2 0 0 0-2 2v6" />
      <path d="M10 10.5V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2v8" />
      <path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15" />
    </svg>
  )
}

// Bug/Debug icon SVG component
function BugIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M8 2l1.88 1.88" />
      <path d="M14.12 3.88L16 2" />
      <path d="M9 7.13v-1a3.003 3.003 0 116 0v1" />
      <path d="M12 20c-3.3 0-6-2.7-6-6v-3a4 4 0 014-4h4a4 4 0 014 4v3c0 3.3-2.7 6-6 6" />
      <path d="M12 20v-9" />
      <path d="M6.53 9C4.6 8.8 3 7.1 3 5" />
      <path d="M6 13H2" />
      <path d="M3 21c0-2.1 1.7-3.9 3.8-4" />
      <path d="M20.97 5c0 2.1-1.6 3.8-3.5 4" />
      <path d="M22 13h-4" />
      <path d="M17.2 17c2.1.1 3.8 1.9 3.8 4" />
    </svg>
  )
}

// Bolt/Performance icon SVG component
function BoltIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
    </svg>
  )
}

// Stable empty arrays to prevent creating new references on every render
const EMPTY_NODES: GraphNode[] = []
const EMPTY_EDGES: import('./lib/types').GraphEdge[] = []

export default function App() {
  const { setToken, isAuthenticated } = useAuth()
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [gestureControlEnabled, setGestureControlEnabled] = useState(false)
  const [debugOverlayVisible, setDebugOverlayVisible] = useState(false)
  const [performanceMode, setPerformanceMode] = useState(false)
  const [settingsPanelOpen, setSettingsPanelOpen] = useState(false)

  // Focus/Spotlight mode state
  const [focusModeEnabled, setFocusModeEnabled] = useState(false)
  const [focusTransition, setFocusTransition] = useState(0) // 0-1 for smooth transition
  const focusTransitionRef = useRef<number>(0)
  const focusAnimationRef = useRef<number | null>(null)

  // Radial menu state
  const [radialMenuState, setRadialMenuState] = useState<{
    isOpen: boolean
    node: GraphNode | null
    position: { x: number; y: number }
  }>({
    isOpen: false,
    node: null,
    position: { x: 0, y: 0 },
  })

  // Lasso selection state
  const [lassoState, setLassoState] = useState<{
    isDrawing: boolean
    points: { x: number; y: number }[]
    selectedIds: Set<string>
  }>({
    isDrawing: false,
    points: [],
    selectedIds: new Set(),
  })
  const canvasContainerRef = useRef<HTMLDivElement>(null)
  const getNodesInPolygonRef = useRef<((polygon: { x: number; y: number }[]) => string[]) | null>(null)

  // Tag cloud state
  const [tagCloudVisible, setTagCloudVisible] = useState(false)

  // Cleanup focus animation on unmount
  useEffect(() => {
    return () => {
      if (focusAnimationRef.current) {
        cancelAnimationFrame(focusAnimationRef.current)
      }
    }
  }, [])

  const [gestureState, setGestureState] = useState<GestureState>(DEFAULT_GESTURE_STATE)
  // Tracking source - check URL param on mount, then allow UI toggle
  const [trackingSource, setTrackingSource] = useState<'mediapipe' | 'iphone'>(() => {
    const params = new URLSearchParams(window.location.search)
    return params.get('iphone') === 'true' ? 'iphone' : 'mediapipe'
  })

  const [trackingInfo, setTrackingInfo] = useState<{
    source: 'mediapipe' | 'iphone'
    iphoneUrl: string
    iphoneConnected: boolean
    hasLiDAR: boolean
    phoneConnected: boolean
    bridgeIps: string[]
    phonePort: number | null
  }>({
    source: trackingSource,
    iphoneUrl: 'ws://localhost:8766/ws',
    iphoneConnected: false,
    hasLiDAR: false,
    phoneConnected: false,
    bridgeIps: [],
    phonePort: null,
  })

  const handleSourceChange = useCallback((source: 'mediapipe' | 'iphone') => {
    setTrackingSource(source)
  }, [])

  // Filter state
  const [filters, setFilters] = useState<FilterState>({
    types: [],
    minImportance: 0,
    maxNodes: 500,
  })

  // Force configuration state
  const [forceConfig, setForceConfig] = useState<ForceConfig>(DEFAULT_FORCE_CONFIG)

  // Display configuration state
  const [displayConfig, setDisplayConfig] = useState<DisplayConfig>(DEFAULT_DISPLAY_CONFIG)

  // Clustering configuration state
  const [clusterConfig, setClusterConfig] = useState<ClusterConfig>(DEFAULT_CLUSTER_CONFIG)

  // Relationship visibility state
  const [relationshipVisibility, setRelationshipVisibility] = useState<RelationshipVisibility>(
    DEFAULT_RELATIONSHIP_VISIBILITY
  )

  // Reheat callback - will be set by GraphCanvas
  const [reheatFn, setReheatFn] = useState<(() => void) | null>(null)

  // Reset view callback - will be set by GraphCanvas
  const [resetViewFn, setResetViewFn] = useState<(() => void) | null>(null)

  // Bookmarks
  const {
    bookmarks,
    addBookmark,
    updateBookmark,
    deleteBookmark,
    getBookmarkByIndex,
  } = useBookmarks()

  // Camera state and navigation for bookmarks
  const [cameraStateForBookmarks, setCameraStateForBookmarks] = useState({ x: 0, y: 0, z: 150, zoom: 1 })
  const navigateForBookmarksRef = useRef<((x: number, y: number) => void) | null>(null)

  const handleGestureStateChange = useCallback((state: GestureState) => {
    setGestureState(state)
  }, [])

  const { lock: handLock } = useHandLockAndGrab(gestureState, gestureControlEnabled)

  const { data, isLoading, error, refetch } = useGraphSnapshot({
    limit: filters.maxNodes,
    minImportance: filters.minImportance,
    types: filters.types.length > 0 ? filters.types : undefined,
    enabled: isAuthenticated,
  })

  // Stable data references - use EMPTY constants when data not loaded
  const nodes = data?.nodes ?? EMPTY_NODES
  const edges = data?.edges ?? EMPTY_EDGES

  // Tag Cloud
  const tagCloud = useTagCloud({
    nodes,
    typeColors: data?.meta?.type_colors,
  })

  // Sound Effects
  const sound = useSoundEffects()

  // Pathfinding
  const pathfinding = usePathfinding({
    nodes: nodes as any,
    edges,
  })

  // Time Travel
  const timeTravel = useTimeTravel({
    nodes,
    enabled: isAuthenticated,
  })

  // Play sound when time travel is activated
  const prevTimeTravelActive = useRef(timeTravel.isActive)
  useEffect(() => {
    if (timeTravel.isActive !== prevTimeTravelActive.current) {
      if (timeTravel.isActive) {
        sound.playTimeTravel()
      }
      prevTimeTravelActive.current = timeTravel.isActive
    }
  }, [timeTravel.isActive, sound.playTimeTravel])

  // Get source and target nodes for pathfinding overlay
  const pathSourceNode = useMemo(() => {
    if (!pathfinding.sourceId || nodes.length === 0) return null
    return nodes.find(n => n.id === pathfinding.sourceId) ?? null
  }, [pathfinding.sourceId, nodes])

  const pathTargetNode = useMemo(() => {
    if (!pathfinding.targetId || nodes.length === 0) return null
    return nodes.find(n => n.id === pathfinding.targetId) ?? null
  }, [pathfinding.targetId, nodes])

  // Bookmark handlers (must be after data is defined)
  const handleSaveBookmark = useCallback(() => {
    addBookmark(
      { x: cameraStateForBookmarks.x, y: cameraStateForBookmarks.y, z: cameraStateForBookmarks.z },
      cameraStateForBookmarks.zoom,
      selectedNode?.id
    )
    sound.playBookmark()
  }, [addBookmark, cameraStateForBookmarks, selectedNode, sound.playBookmark])

  const handleNavigateToBookmark = useCallback((bookmark: Bookmark) => {
    navigateForBookmarksRef.current?.(bookmark.position.x, bookmark.position.y)
    // If bookmark has a selected node, select it
    if (bookmark.selectedNodeId && nodes.length > 0) {
      const node = nodes.find(n => n.id === bookmark.selectedNodeId)
      if (node) {
        setSelectedNode(node)
      }
    }
  }, [nodes])

  const handleRenameBookmark = useCallback((id: string, name: string) => {
    updateBookmark(id, { name })
  }, [updateBookmark])

  // Quick navigate to bookmark by number (1-9)
  const handleQuickNavigate = useCallback((index: number) => {
    const bookmark = getBookmarkByIndex(index)
    if (bookmark) {
      handleNavigateToBookmark(bookmark)
    }
  }, [getBookmarkByIndex, handleNavigateToBookmark])

  const handleNodeSelect = useCallback((node: GraphNode | null) => {
    // If we're in path selection mode and a node is clicked, complete the path
    if (pathfinding.isSelectingTarget && node) {
      pathfinding.completePathSelection(node.id)
      sound.playPathFound()
      return
    }
    if (node) {
      sound.playSelect(node.importance ?? 0.5)
    }
    setSelectedNode(node)
  }, [pathfinding.isSelectingTarget, pathfinding.completePathSelection, sound.playPathFound, sound.playSelect])

  const handleNodeHover = useCallback((node: GraphNode | null) => {
    if (node) {
      sound.playHover()
    }
    setHoveredNode(node)
  }, [sound.playHover])

  // Radial menu handlers
  const handleNodeContextMenu = useCallback((node: GraphNode, screenPosition: { x: number; y: number }) => {
    setRadialMenuState({
      isOpen: true,
      node,
      position: screenPosition,
    })
    setSelectedNode(node) // Also select the node
  }, [])

  const handleCloseRadialMenu = useCallback(() => {
    setRadialMenuState(prev => ({
      ...prev,
      isOpen: false,
    }))
  }, [])

  const handleCopyNodeId = useCallback((nodeId: string) => {
    // Could show a toast notification here
    console.log('Copied node ID:', nodeId)
  }, [])

  const handleViewNodeContent = useCallback((node: GraphNode) => {
    // Select the node to show in inspector
    setSelectedNode(node)
  }, [])

  // Lasso selection handlers
  const handleLassoStart = useCallback((x: number, y: number) => {
    setLassoState(prev => ({
      ...prev,
      isDrawing: true,
      points: [{ x, y }],
    }))
  }, [])

  const handleLassoMove = useCallback((x: number, y: number) => {
    setLassoState(prev => {
      if (!prev.isDrawing) return prev
      // Only add point if moved enough to avoid too many points
      const lastPoint = prev.points[prev.points.length - 1]
      if (lastPoint) {
        const dist = Math.sqrt(Math.pow(x - lastPoint.x, 2) + Math.pow(y - lastPoint.y, 2))
        if (dist < 3) return prev
      }
      return {
        ...prev,
        points: [...prev.points, { x, y }],
      }
    })
  }, [])

  const handleLassoEnd = useCallback(() => {
    setLassoState(prev => {
      if (!prev.isDrawing || prev.points.length < 3) {
        return { ...prev, isDrawing: false, points: [] }
      }

      // Call GraphCanvas to find nodes in the polygon
      const nodesInPolygon = getNodesInPolygonRef.current?.(prev.points) ?? []
      const newSelectedIds = new Set(prev.selectedIds)
      nodesInPolygon.forEach(id => newSelectedIds.add(id))

      // Play lasso sound if nodes were selected
      if (nodesInPolygon.length > 0) {
        sound.playLasso()
      }

      return {
        isDrawing: false,
        points: [],
        selectedIds: newSelectedIds,
      }
    })
  }, [sound.playLasso])

  const handleLassoCancel = useCallback(() => {
    setLassoState(prev => ({
      ...prev,
      isDrawing: false,
      points: [],
    }))
  }, [])

  const handleClearLassoSelection = useCallback(() => {
    setLassoState(prev => ({
      ...prev,
      selectedIds: new Set(),
    }))
  }, [])

  // Get selected nodes from lasso
  const lassoSelectedNodes = useMemo(() => {
    if (nodes.length === 0 || lassoState.selectedIds.size === 0) return []
    return nodes.filter(n => lassoState.selectedIds.has(n.id))
  }, [nodes, lassoState.selectedIds])

  const handleSearch = useCallback((term: string) => {
    // Play search sound on typing (only if term changed and is not empty)
    if (term.length > 0) {
      sound.playSearch()
    }
    setSearchTerm(term)
  }, [sound.playSearch])

  const handleFilterChange = useCallback((newFilters: Partial<FilterState>) => {
    setFilters(prev => ({ ...prev, ...newFilters }))
  }, [])

  const handleForceConfigChange = useCallback((config: Partial<ForceConfig>) => {
    setForceConfig(prev => ({ ...prev, ...config }))
  }, [])

  const handleDisplayConfigChange = useCallback((config: Partial<DisplayConfig>) => {
    setDisplayConfig(prev => ({ ...prev, ...config }))
  }, [])

  const handleClusterConfigChange = useCallback((config: Partial<ClusterConfig>) => {
    setClusterConfig(prev => ({ ...prev, ...config }))
  }, [])

  const handleRelationshipVisibilityChange = useCallback((visibility: Partial<RelationshipVisibility>) => {
    setRelationshipVisibility(prev => ({ ...prev, ...visibility }))
  }, [])

  const handleReheat = useCallback(() => {
    reheatFn?.()
  }, [reheatFn])

  const handleResetForces = useCallback(() => {
    setForceConfig(DEFAULT_FORCE_CONFIG)
  }, [])

  const handleToggleLabels = useCallback(() => {
    setDisplayConfig(prev => ({ ...prev, showLabels: !prev.showLabels }))
  }, [])

  // Focus mode toggle with smooth transition animation
  const handleToggleFocusMode = useCallback(() => {
    setFocusModeEnabled(prev => {
      const newEnabled = !prev

      // Cancel any existing animation
      if (focusAnimationRef.current) {
        cancelAnimationFrame(focusAnimationRef.current)
      }

      const startTime = performance.now()
      const duration = 400 // 400ms transition
      const startValue = focusTransitionRef.current
      const endValue = newEnabled ? 1 : 0

      const animate = (currentTime: number) => {
        const elapsed = currentTime - startTime
        const progress = Math.min(elapsed / duration, 1)

        // Ease out cubic for smooth deceleration
        const eased = 1 - Math.pow(1 - progress, 3)
        const newTransition = startValue + (endValue - startValue) * eased

        focusTransitionRef.current = newTransition
        setFocusTransition(newTransition)

        if (progress < 1) {
          focusAnimationRef.current = requestAnimationFrame(animate)
        } else {
          focusAnimationRef.current = null
        }
      }

      focusAnimationRef.current = requestAnimationFrame(animate)
      return newEnabled
    })
  }, [])

  // Keyboard navigation
  const handleStartPathfindingFromKeyboard = useCallback(() => {
    if (selectedNode) {
      pathfinding.startPathSelection(selectedNode.id)
    }
  }, [selectedNode, pathfinding.startPathSelection])

  const { shortcuts } = useKeyboardNavigation({
    nodes: nodes as any,
    selectedNode,
    onNodeSelect: handleNodeSelect,
    onReheat: handleReheat,
    onToggleSettings: () => setSettingsPanelOpen(prev => !prev),
    onToggleLabels: handleToggleLabels,
    onToggleFocus: handleToggleFocusMode,
    onSaveBookmark: handleSaveBookmark,
    onQuickNavigate: handleQuickNavigate,
    onStartPathfinding: handleStartPathfindingFromKeyboard,
    onCancelPathfinding: pathfinding.cancelPathSelection,
    isPathSelecting: pathfinding.isSelectingTarget,
    enabled: true,
  })

  // Log available shortcuts for debugging (remove in production)
  void shortcuts

  // Toggle tag cloud with 'T' key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 't' || e.key === 'T') {
        setTagCloudVisible(prev => !prev)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  if (!isAuthenticated) {
    return <TokenPrompt onSubmit={setToken} />
  }

  return (
    <div className="h-screen w-screen bg-[#0a0a0f] text-slate-100 flex flex-col overflow-hidden">
      {/* Top Bar */}
      <header className="h-14 flex-shrink-0 glass border-b border-white/5 flex items-center px-4 gap-4 z-50">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
            <span className="text-white font-bold text-sm">AM</span>
          </div>
          <h1 className="text-lg font-semibold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            AutoMem
          </h1>
        </div>

        <SearchBar
          value={searchTerm}
          onChange={handleSearch}
          className="flex-1 max-w-xl"
        />

        <StatsBar stats={data?.stats} isLoading={isLoading} />

        {/* Version indicator - helps verify deployment */}
        <span className="text-xs text-slate-500 hidden lg:inline" title="Build version">
          {BUILD_VERSION}
        </span>

        {/* Focus/Spotlight Mode Toggle */}
        <button
          onClick={handleToggleFocusMode}
          className={`
            flex items-center gap-2 px-3 py-1.5 rounded-lg transition-all duration-200
            ${focusModeEnabled
              ? 'bg-gradient-to-r from-amber-500 to-yellow-500 text-white shadow-lg shadow-amber-500/25'
              : 'bg-white/5 hover:bg-white/10 text-slate-400 hover:text-white'
            }
          `}
          title={focusModeEnabled ? 'Disable focus mode (F)' : 'Enable focus mode - spotlight selected node (F)'}
        >
          <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="3" />
            <path d="M12 2v2" />
            <path d="M12 20v2" />
            <path d="M2 12h2" />
            <path d="M20 12h2" />
            <path d="m4.93 4.93 1.41 1.41" />
            <path d="m17.66 17.66 1.41 1.41" />
            <path d="m17.66 6.34 1.41-1.41" />
            <path d="m4.93 19.07 1.41-1.41" />
          </svg>
          <span className="text-sm font-medium hidden sm:inline">
            {focusModeEnabled ? 'Focus' : 'Focus'}
          </span>
        </button>

        {/* Performance Mode Toggle */}
        <button
          onClick={() => setPerformanceMode(!performanceMode)}
          className={`
            flex items-center gap-2 px-3 py-1.5 rounded-lg transition-all duration-200
            ${performanceMode
              ? 'bg-gradient-to-r from-yellow-500 to-orange-500 text-white shadow-lg shadow-yellow-500/25'
              : 'bg-white/5 hover:bg-white/10 text-slate-400 hover:text-white'
            }
          `}
          title={performanceMode ? 'Disable performance mode (enable effects)' : 'Enable performance mode (disable bloom/vignette for faster rendering)'}
        >
          <BoltIcon className="w-5 h-5" />
          <span className="text-sm font-medium hidden sm:inline">
            {performanceMode ? 'Perf ON' : 'Perf'}
          </span>
        </button>

        {/* Gesture Control Toggle */}
        <button
          onClick={() => setGestureControlEnabled(!gestureControlEnabled)}
          className={`
            flex items-center gap-2 px-3 py-1.5 rounded-lg transition-all duration-200
            ${gestureControlEnabled
              ? 'bg-gradient-to-r from-cyan-500 to-purple-500 text-white shadow-lg shadow-cyan-500/25'
              : 'bg-white/5 hover:bg-white/10 text-slate-400 hover:text-white'
            }
          `}
          title={gestureControlEnabled ? 'Disable hand gestures' : 'Enable hand gestures (requires camera)'}
        >
          <HandIcon className="w-5 h-5" />
          <span className="text-sm font-medium hidden sm:inline">
            {gestureControlEnabled ? 'Gestures ON' : 'Gestures'}
          </span>
        </button>

        {/* Debug Overlay Toggle (only show when gestures enabled) */}
        {gestureControlEnabled && (
          <button
            onClick={() => setDebugOverlayVisible(!debugOverlayVisible)}
            className={`
              flex items-center gap-2 px-3 py-1.5 rounded-lg transition-all duration-200
              ${debugOverlayVisible
                ? 'bg-gradient-to-r from-green-500 to-emerald-500 text-white shadow-lg shadow-green-500/25'
                : 'bg-white/5 hover:bg-white/10 text-slate-400 hover:text-white'
              }
            `}
            title={debugOverlayVisible ? 'Hide debug overlay' : 'Show gesture debug overlay'}
          >
            <BugIcon className="w-5 h-5" />
            <span className="text-sm font-medium hidden sm:inline">
              {debugOverlayVisible ? 'Debug ON' : 'Debug'}
            </span>
          </button>
        )}

        {/* Settings Panel Toggle */}
        <button
          onClick={() => setSettingsPanelOpen(!settingsPanelOpen)}
          className={`
            flex items-center gap-2 px-3 py-1.5 rounded-lg transition-all duration-200
            ${settingsPanelOpen
              ? 'bg-gradient-to-r from-blue-500 to-indigo-500 text-white shadow-lg shadow-blue-500/25'
              : 'bg-white/5 hover:bg-white/10 text-slate-400 hover:text-white'
            }
          `}
          title={settingsPanelOpen ? 'Hide settings' : 'Show graph settings'}
        >
          <Settings className="w-5 h-5" />
          <span className="text-sm font-medium hidden sm:inline">
            Settings
          </span>
        </button>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        <PanelGroup direction="horizontal" className="flex-1">
          {/* Graph Canvas */}
          <Panel defaultSize={settingsPanelOpen ? 50 : 75} minSize={40}>
            <div ref={canvasContainerRef} className="h-full relative">
              {isLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-10">
                  <div className="flex flex-col items-center gap-3">
                    <div className="w-12 h-12 border-4 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
                    <span className="text-slate-400">Loading memories...</span>
                  </div>
                </div>
              )}

              {error && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-10">
                  <div className="glass p-6 rounded-xl max-w-md text-center">
                    <div className="text-red-400 text-lg mb-2">Connection Error</div>
                    <div className="text-slate-400 text-sm mb-4">{(error as Error).message}</div>
                    <button
                      onClick={() => refetch()}
                      className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg transition-colors"
                    >
                      Retry
                    </button>
                  </div>
                </div>
              )}

              <GraphCanvas
                nodes={nodes}
                edges={edges}
                selectedNode={selectedNode}
                hoveredNode={hoveredNode}
                searchTerm={searchTerm}
                onNodeSelect={handleNodeSelect}
                onNodeHover={handleNodeHover}
                onNodeContextMenu={handleNodeContextMenu}
                gestureControlEnabled={gestureControlEnabled}
                trackingSource={trackingSource}
                onGestureStateChange={handleGestureStateChange}
                onTrackingInfoChange={setTrackingInfo}
                performanceMode={performanceMode}
                forceConfig={forceConfig}
                displayConfig={displayConfig}
                clusterConfig={clusterConfig}
                relationshipVisibility={relationshipVisibility}
                typeColors={data?.meta?.type_colors}
                onReheatReady={setReheatFn}
                onResetViewReady={setResetViewFn}
                focusModeEnabled={focusModeEnabled}
                focusTransition={focusTransition}
                onCameraStateForBookmarks={setCameraStateForBookmarks}
                onNavigateForBookmarks={(fn) => { navigateForBookmarksRef.current = fn }}
                pathNodeIds={pathfinding.pathNodeIds}
                pathEdgeKeys={pathfinding.pathEdgeKeys}
                pathSourceId={pathfinding.sourceId}
                pathTargetId={pathfinding.targetId}
                isPathSelecting={pathfinding.isSelectingTarget}
                timeTravelActive={timeTravel.isActive}
                timeTravelVisibleNodes={timeTravel.visibleNodes}
                onGetNodesInPolygon={(fn) => { getNodesInPolygonRef.current = fn }}
                lassoSelectedIds={lassoState.selectedIds}
                tagFilteredNodeIds={tagCloud.filteredNodeIds}
                hasTagFilter={tagCloud.hasActiveFilter}
              />

              {/* 2D Hand Overlay (on top of canvas, life-size) */}
              <Hand2DOverlay
                gestureState={gestureState}
                enabled={gestureControlEnabled}
              />

              {/* Gesture Debug Overlay */}
              <GestureDebugOverlay
                gestureState={gestureState}
                visible={debugOverlayVisible && gestureControlEnabled}
              />

              {/* Hand Control Overlay (lock/grab metrics) */}
              <HandControlOverlay
                enabled={gestureControlEnabled}
                lock={handLock}
                source={trackingSource}
                onSourceChange={handleSourceChange}
                onResetView={resetViewFn ?? undefined}
                iphoneConnected={trackingInfo.iphoneConnected}
                hasLiDAR={trackingInfo.hasLiDAR}
                iphoneUrl={trackingInfo.iphoneUrl}
                phoneConnected={trackingInfo.phoneConnected}
                bridgeIps={trackingInfo.bridgeIps}
                phonePort={trackingInfo.phonePort}
              />

              {/* Bookmarks Panel */}
              <BookmarksPanel
                bookmarks={bookmarks}
                onNavigate={handleNavigateToBookmark}
                onDelete={deleteBookmark}
                onRename={handleRenameBookmark}
                onSaveBookmark={handleSaveBookmark}
                visible={true}
              />

              {/* Pathfinding Overlay */}
              <PathfindingOverlay
                isSelectingTarget={pathfinding.isSelectingTarget}
                sourceNode={pathSourceNode}
                targetNode={pathTargetNode}
                currentPath={pathfinding.currentPath}
                pathCount={pathfinding.pathCount}
                activePath={pathfinding.activePath}
                onNextPath={pathfinding.nextPath}
                onPreviousPath={pathfinding.previousPath}
                onCancel={pathfinding.cancelPathSelection}
                onClear={pathfinding.clearPath}
                visible={pathfinding.isSelectingTarget || pathfinding.hasPath}
              />

              {/* Time Travel Timeline */}
              <TimelineBar
                isActive={timeTravel.isActive}
                isPlaying={timeTravel.isPlaying}
                currentTime={timeTravel.currentTime}
                minTime={timeTravel.minTime}
                maxTime={timeTravel.maxTime}
                progress={timeTravel.progress}
                playbackSpeed={timeTravel.playbackSpeed}
                visibleCount={timeTravel.visibleCount}
                totalCount={timeTravel.totalCount}
                onToggleActive={timeTravel.toggleActive}
                onTogglePlay={timeTravel.togglePlay}
                onSetProgress={timeTravel.setProgress}
                onStepForward={timeTravel.stepForward}
                onStepBackward={timeTravel.stepBackward}
                onCycleSpeed={timeTravel.cycleSpeed}
                onGoToStart={timeTravel.goToStart}
                onGoToEnd={timeTravel.goToEnd}
              />

              {/* Lasso Selection Overlay */}
              <LassoOverlay
                isDrawing={lassoState.isDrawing}
                points={lassoState.points}
                selectedCount={lassoState.selectedIds.size}
                onStartDraw={handleLassoStart}
                onMoveDraw={handleLassoMove}
                onEndDraw={handleLassoEnd}
                onCancelDraw={handleLassoCancel}
              />

              {/* Selection Actions (bulk operations) */}
              <SelectionActions
                selectedNodes={lassoSelectedNodes}
                allEdges={edges}
                onClearSelection={handleClearLassoSelection}
              />
            </div>
          </Panel>

          {/* Resize Handle */}
          <PanelResizeHandle className="w-1 bg-white/5 hover:bg-blue-500/50 transition-colors cursor-col-resize" />

          {/* Inspector Panel */}
          <Panel defaultSize={25} minSize={15} maxSize={40}>
            <Inspector
              node={selectedNode}
              onClose={() => setSelectedNode(null)}
              onNavigate={handleNodeSelect}
              onStartPathfinding={pathfinding.startPathSelection}
              isPathSelecting={pathfinding.isSelectingTarget}
            />
          </Panel>
        </PanelGroup>

        {/* Settings Panel (right-docked) */}
        <SettingsPanel
          isOpen={settingsPanelOpen}
          onClose={() => setSettingsPanelOpen(false)}
          filters={filters}
          onFiltersChange={handleFilterChange}
          typeColors={data?.meta?.type_colors}
          forceConfig={forceConfig}
          onForceConfigChange={handleForceConfigChange}
          onReheat={handleReheat}
          onResetForces={handleResetForces}
          displayConfig={displayConfig}
          onDisplayConfigChange={handleDisplayConfigChange}
          clusterConfig={clusterConfig}
          onClusterConfigChange={handleClusterConfigChange}
          relationshipVisibility={relationshipVisibility}
          onRelationshipVisibilityChange={handleRelationshipVisibilityChange}
          soundEnabled={sound.settings.enabled}
          onSoundEnabledChange={sound.setEnabled}
          soundVolume={sound.settings.masterVolume}
          onSoundVolumeChange={sound.setMasterVolume}
        />
      </div>

      {/* Radial Menu (context menu for nodes) */}
      {radialMenuState.isOpen && radialMenuState.node && (
        <RadialMenu
          node={radialMenuState.node}
          position={radialMenuState.position}
          onClose={handleCloseRadialMenu}
          onToggleFocus={handleToggleFocusMode}
          onStartPath={pathfinding.startPathSelection}
          onViewContent={handleViewNodeContent}
          onCopyId={handleCopyNodeId}
          focusModeEnabled={focusModeEnabled}
        />
      )}

      {/* Tag Cloud (press 'T' to toggle) */}
      <TagCloud
        tags={tagCloud.tags}
        filteredTags={tagCloud.filteredTags}
        selectedTags={tagCloud.selectedTags}
        filterMode={tagCloud.filterMode}
        filteredCount={tagCloud.filteredNodeIds.size}
        totalCount={nodes.length}
        onToggleTag={tagCloud.toggleTag}
        onClearSelection={tagCloud.clearSelection}
        onToggleFilterMode={tagCloud.toggleFilterMode}
        onSearchChange={tagCloud.setSearchTerm}
        searchTerm={tagCloud.searchTerm}
        typeColors={data?.meta?.type_colors}
        visible={tagCloudVisible}
        onClose={() => setTagCloudVisible(false)}
      />
    </div>
  )
}
