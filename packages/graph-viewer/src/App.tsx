import { useState, useCallback } from 'react'
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels'
import { useGraphSnapshot } from './hooks/useGraphData'
import { useAuth } from './hooks/useAuth'
import { GraphCanvas } from './components/GraphCanvas'
import { Inspector } from './components/Inspector'
import { SearchBar } from './components/SearchBar'
import { FilterPanel } from './components/FilterPanel'
import { TokenPrompt } from './components/TokenPrompt'
import { StatsBar } from './components/StatsBar'
import type { GraphNode, FilterState } from './lib/types'

export default function App() {
  const { setToken, isAuthenticated } = useAuth()
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [filters, setFilters] = useState<FilterState>({
    types: [],
    minImportance: 0,
    maxNodes: 500,
  })

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
