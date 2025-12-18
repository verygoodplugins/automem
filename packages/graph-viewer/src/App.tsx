import { useState, useCallback } from 'react'

// Build version - update this when making significant changes
const BUILD_VERSION = '2024-12-11-masterhand-v7'
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels'
import { useGraphSnapshot } from './hooks/useGraphData'
import { useAuth } from './hooks/useAuth'
import { GraphCanvas } from './components/GraphCanvas'
import { Inspector } from './components/Inspector'
import { SearchBar } from './components/SearchBar'
import { FilterPanel } from './components/FilterPanel'
import { TokenPrompt } from './components/TokenPrompt'
import { StatsBar } from './components/StatsBar'
import { GestureDebugOverlay } from './components/GestureDebugOverlay'
import { Hand2DOverlay } from './components/Hand2DOverlay'
import { HandControlOverlay } from './components/HandControlOverlay'
import { useHandLockAndGrab } from './hooks/useHandLockAndGrab'
import type { GraphNode, FilterState } from './lib/types'
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

export default function App() {
  const { setToken, isAuthenticated } = useAuth()
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [gestureControlEnabled, setGestureControlEnabled] = useState(false)
  const [debugOverlayVisible, setDebugOverlayVisible] = useState(false)
  const [performanceMode, setPerformanceMode] = useState(false)
  const [gestureState, setGestureState] = useState<GestureState>(DEFAULT_GESTURE_STATE)
  const [trackingInfo, setTrackingInfo] = useState<{
    source: 'mediapipe' | 'iphone'
    iphoneUrl: string
    iphoneConnected: boolean
    hasLiDAR: boolean
    phoneConnected: boolean
    bridgeIps: string[]
    phonePort: number | null
  }>({
    source: 'mediapipe',
    iphoneUrl: 'ws://localhost:8766/ws',
    iphoneConnected: false,
    hasLiDAR: false,
    phoneConnected: false,
    bridgeIps: [],
    phonePort: null,
  })
  const [filters, setFilters] = useState<FilterState>({
    types: [],
    minImportance: 0,
    maxNodes: 500,
  })

  const handleGestureStateChange = useCallback((state: GestureState) => {
    setGestureState(state)
  }, [])

  const { lock: handLock } = useHandLockAndGrab(gestureState, gestureControlEnabled)
  // Note: GraphCanvas owns the actual tracking source selection via URL params.
  // We mirror it here via onTrackingInfoChange so overlays can show accurate status.

  const { data, isLoading, error, refetch } = useGraphSnapshot({
    limit: filters.maxNodes,
    minImportance: filters.minImportance,
    types: filters.types.length > 0 ? filters.types : undefined,
    enabled: isAuthenticated,
  })

  const handleNodeSelect = useCallback((node: GraphNode | null) => {
    setSelectedNode(node)
  }, [])

  const handleNodeHover = useCallback((node: GraphNode | null) => {
    setHoveredNode(node)
  }, [])

  const handleSearch = useCallback((term: string) => {
    setSearchTerm(term)
  }, [])

  const handleFilterChange = useCallback((newFilters: Partial<FilterState>) => {
    setFilters(prev => ({ ...prev, ...newFilters }))
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

        <FilterPanel
          filters={filters}
          onChange={handleFilterChange}
          typeColors={data?.meta?.type_colors}
        />

        <StatsBar stats={data?.stats} isLoading={isLoading} />

        {/* Version indicator - helps verify deployment */}
        <span className="text-xs text-slate-500 hidden lg:inline" title="Build version">
          {BUILD_VERSION}
        </span>

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
      </header>

      {/* Main Content */}
      <PanelGroup direction="horizontal" className="flex-1">
        {/* Graph Canvas */}
        <Panel defaultSize={75} minSize={50}>
          <div className="h-full relative">
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
              nodes={data?.nodes ?? []}
              edges={data?.edges ?? []}
              selectedNode={selectedNode}
              hoveredNode={hoveredNode}
              searchTerm={searchTerm}
              onNodeSelect={handleNodeSelect}
              onNodeHover={handleNodeHover}
              gestureControlEnabled={gestureControlEnabled}
              onGestureStateChange={handleGestureStateChange}
              onTrackingInfoChange={setTrackingInfo}
              performanceMode={performanceMode}
            />

            {/* 2D Hand Overlay (on top of canvas, life-size) */}
            <Hand2DOverlay
              gestureState={gestureState}
              enabled={gestureControlEnabled}
              showLaser={false}
              handLock={handLock}
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
              source={trackingInfo.source}
              iphoneConnected={trackingInfo.iphoneConnected}
              hasLiDAR={trackingInfo.hasLiDAR}
              iphoneUrl={trackingInfo.iphoneUrl}
              phoneConnected={trackingInfo.phoneConnected}
              bridgeIps={trackingInfo.bridgeIps}
              phonePort={trackingInfo.phonePort}
            />
          </div>
        </Panel>

        {/* Resize Handle */}
        <PanelResizeHandle className="w-1 bg-white/5 hover:bg-blue-500/50 transition-colors cursor-col-resize" />

        {/* Inspector Panel */}
        <Panel defaultSize={25} minSize={20} maxSize={40}>
          <Inspector
            node={selectedNode}
            onClose={() => setSelectedNode(null)}
            onNavigate={handleNodeSelect}
          />
        </Panel>
      </PanelGroup>
    </div>
  )
}
