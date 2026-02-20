/**
 * PathfindingOverlay - UI for path selection and path info display
 *
 * Shows:
 * - Instructions when selecting target
 * - Path info panel when path is found
 * - Controls for cycling between alternative paths
 */

import type { PathResult } from '../hooks/usePathfinding'
import type { SimulationNode } from '../lib/types'

interface PathfindingOverlayProps {
  isSelectingTarget: boolean
  sourceNode: SimulationNode | null
  targetNode: SimulationNode | null
  currentPath: PathResult | null
  pathCount: number
  activePath: number
  onNextPath: () => void
  onPreviousPath: () => void
  onCancel: () => void
  onClear: () => void
  visible?: boolean
}

export function PathfindingOverlay({
  isSelectingTarget,
  sourceNode,
  targetNode,
  currentPath,
  pathCount,
  activePath,
  onNextPath,
  onPreviousPath,
  onCancel,
  onClear,
  visible = true,
}: PathfindingOverlayProps) {
  if (!visible) return null

  // Show selection prompt when selecting target
  if (isSelectingTarget && sourceNode) {
    return (
      <div className="absolute top-20 left-1/2 -translate-x-1/2 z-50">
        <div className="bg-slate-800/95 backdrop-blur-sm border border-blue-500/50 rounded-lg px-4 py-3 shadow-lg">
          <div className="flex items-center gap-3">
            <div className="animate-pulse">
              <svg className="w-5 h-5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <div>
              <div className="text-white font-medium">Select destination node</div>
              <div className="text-slate-400 text-sm">
                Click another node to find path from "{truncate(sourceNode.content, 30)}"
              </div>
            </div>
            <button
              onClick={onCancel}
              className="ml-4 px-2 py-1 rounded text-sm text-slate-400 hover:text-white hover:bg-slate-700 transition-colors"
            >
              Cancel (Esc)
            </button>
          </div>
        </div>
      </div>
    )
  }

  // Show path info when path is found
  if (currentPath && sourceNode && targetNode) {
    return (
      <div className="absolute bottom-20 left-1/2 -translate-x-1/2 z-50">
        <div className="bg-slate-800/95 backdrop-blur-sm border border-slate-700/50 rounded-lg shadow-xl overflow-hidden min-w-[320px]">
          {/* Header */}
          <div className="px-4 py-3 bg-gradient-to-r from-blue-600/20 to-purple-600/20 border-b border-slate-700/50">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <svg className="w-5 h-5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                <span className="text-white font-medium">Path Found</span>
              </div>
              <button
                onClick={onClear}
                className="p-1 rounded hover:bg-slate-700/50 text-slate-400 hover:text-white transition-colors"
                title="Clear path"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>

          {/* Path details */}
          <div className="px-4 py-3 space-y-2">
            {/* Source and Target */}
            <div className="flex items-center gap-2 text-sm">
              <span className="px-2 py-0.5 bg-green-500/20 text-green-400 rounded text-xs font-medium">FROM</span>
              <span className="text-slate-300 truncate max-w-[120px]">{truncate(sourceNode.content, 25)}</span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <span className="px-2 py-0.5 bg-red-500/20 text-red-400 rounded text-xs font-medium">TO</span>
              <span className="text-slate-300 truncate max-w-[120px]">{truncate(targetNode.content, 25)}</span>
            </div>

            {/* Stats */}
            <div className="flex items-center gap-4 pt-2 border-t border-slate-700/50">
              <div className="text-center">
                <div className="text-2xl font-bold text-white">{currentPath.hopCount}</div>
                <div className="text-xs text-slate-400">Hops</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-white">{(currentPath.totalStrength * 100).toFixed(0)}%</div>
                <div className="text-xs text-slate-400">Avg Strength</div>
              </div>
              <div className="text-center flex-1">
                <div className="text-sm text-slate-300">
                  {getUniqueTypes(currentPath).join(' → ')}
                </div>
                <div className="text-xs text-slate-400">Relationship types</div>
              </div>
            </div>
          </div>

          {/* Alternative paths navigation */}
          {pathCount > 1 && (
            <div className="px-4 py-2 bg-slate-900/50 border-t border-slate-700/50 flex items-center justify-between">
              <button
                onClick={onPreviousPath}
                className="p-1 rounded hover:bg-slate-700/50 text-slate-400 hover:text-white transition-colors"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
              </button>
              <span className="text-sm text-slate-400">
                Path {activePath + 1} of {pathCount}
              </span>
              <button
                onClick={onNextPath}
                className="p-1 rounded hover:bg-slate-700/50 text-slate-400 hover:text-white transition-colors"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </button>
            </div>
          )}
        </div>
      </div>
    )
  }

  return null
}

function truncate(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text
  return text.slice(0, maxLength) + '...'
}

function getUniqueTypes(path: PathResult): string[] {
  const types: string[] = []
  for (const step of path.path) {
    if (step.edgeType && !types.includes(step.edgeType)) {
      types.push(step.edgeType)
    }
  }
  return types.length > 0 ? types : ['—']
}
