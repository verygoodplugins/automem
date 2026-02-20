/**
 * SelectionActions - Bulk action buttons for selected nodes
 *
 * Actions:
 * - Find common tags
 * - Show connections between selected
 * - Export selection (JSON/CSV)
 * - Clear selection
 * - Delete all (with confirmation)
 */

import { useState, useMemo } from 'react'
import {
  Tags,
  GitBranch,
  Download,
  X,
  Trash2,
  AlertTriangle,
} from 'lucide-react'
import type { GraphNode, GraphEdge } from '../lib/types'

interface SelectionActionsProps {
  selectedNodes: GraphNode[]
  allEdges: GraphEdge[]
  onClearSelection: () => void
  onHighlightConnections?: (nodeIds: string[]) => void
  onExportSelection?: (nodes: GraphNode[], format: 'json' | 'csv') => void
  onDeleteSelected?: (nodes: GraphNode[]) => void
}

export function SelectionActions({
  selectedNodes,
  allEdges,
  onClearSelection,
  onHighlightConnections,
  onExportSelection,
  onDeleteSelected,
}: SelectionActionsProps) {
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [showCommonTags, setShowCommonTags] = useState(false)

  // Find common tags among selected nodes
  const commonTags = useMemo(() => {
    if (selectedNodes.length === 0) return []

    // Get all tags from first node
    const firstNodeTags = new Set(selectedNodes[0].tags)

    // Filter to only tags present in ALL selected nodes
    const common: string[] = []
    firstNodeTags.forEach((tag) => {
      if (selectedNodes.every((n) => n.tags.includes(tag))) {
        common.push(tag)
      }
    })

    return common.sort()
  }, [selectedNodes])

  // Count connections between selected nodes
  const internalConnections = useMemo(() => {
    const selectedIds = new Set(selectedNodes.map((n) => n.id))
    return allEdges.filter(
      (e) => selectedIds.has(e.source) && selectedIds.has(e.target)
    ).length
  }, [selectedNodes, allEdges])

  // Export selection as JSON
  const handleExportJSON = () => {
    if (onExportSelection) {
      onExportSelection(selectedNodes, 'json')
    } else {
      const data = selectedNodes.map((n) => ({
        id: n.id,
        content: n.content,
        type: n.type,
        tags: n.tags,
        importance: n.importance,
        timestamp: n.timestamp,
      }))
      const blob = new Blob([JSON.stringify(data, null, 2)], {
        type: 'application/json',
      })
      downloadBlob(blob, `memory-selection-${Date.now()}.json`)
    }
  }

  // Export selection as CSV
  const handleExportCSV = () => {
    if (onExportSelection) {
      onExportSelection(selectedNodes, 'csv')
    } else {
      const headers = ['id', 'content', 'type', 'tags', 'importance', 'timestamp']
      const rows = selectedNodes.map((n) =>
        [
          n.id,
          `"${n.content.replace(/"/g, '""')}"`,
          n.type,
          `"${n.tags.join(', ')}"`,
          n.importance,
          n.timestamp,
        ].join(',')
      )
      const csv = [headers.join(','), ...rows].join('\n')
      const blob = new Blob([csv], { type: 'text/csv' })
      downloadBlob(blob, `memory-selection-${Date.now()}.csv`)
    }
  }

  // Handle delete with confirmation
  const handleDelete = () => {
    if (showDeleteConfirm) {
      onDeleteSelected?.(selectedNodes)
      setShowDeleteConfirm(false)
    } else {
      setShowDeleteConfirm(true)
    }
  }

  if (selectedNodes.length === 0) return null

  return (
    <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-50">
      <div className="bg-slate-900/95 backdrop-blur-sm rounded-2xl border border-slate-700/50 shadow-2xl p-3">
        <div className="flex items-center gap-2">
          {/* Common Tags Button */}
          <button
            onClick={() => setShowCommonTags(!showCommonTags)}
            className={`
              flex items-center gap-2 px-3 py-2 rounded-lg
              transition-colors text-sm font-medium
              ${showCommonTags
                ? 'bg-blue-600 text-white'
                : 'bg-slate-800 text-slate-300 hover:bg-slate-700'}
            `}
            title="Show common tags"
          >
            <Tags className="w-4 h-4" />
            <span>Tags</span>
            {commonTags.length > 0 && (
              <span className="px-1.5 py-0.5 bg-slate-600/50 rounded text-xs">
                {commonTags.length}
              </span>
            )}
          </button>

          {/* Show Connections Button */}
          <button
            onClick={() =>
              onHighlightConnections?.(selectedNodes.map((n) => n.id))
            }
            className="
              flex items-center gap-2 px-3 py-2 rounded-lg
              bg-slate-800 text-slate-300 hover:bg-slate-700
              transition-colors text-sm font-medium
            "
            title="Highlight connections between selected nodes"
          >
            <GitBranch className="w-4 h-4" />
            <span>Connections</span>
            <span className="px-1.5 py-0.5 bg-slate-600/50 rounded text-xs">
              {internalConnections}
            </span>
          </button>

          {/* Export Dropdown */}
          <div className="relative group">
            <button
              className="
                flex items-center gap-2 px-3 py-2 rounded-lg
                bg-slate-800 text-slate-300 hover:bg-slate-700
                transition-colors text-sm font-medium
              "
              title="Export selection"
            >
              <Download className="w-4 h-4" />
              <span>Export</span>
            </button>
            <div className="
              absolute bottom-full left-0 mb-2
              bg-slate-800 rounded-lg border border-slate-700
              opacity-0 invisible group-hover:opacity-100 group-hover:visible
              transition-all duration-150 shadow-xl
              min-w-[120px]
            ">
              <button
                onClick={handleExportJSON}
                className="w-full px-3 py-2 text-left text-sm text-slate-300 hover:bg-slate-700 rounded-t-lg"
              >
                Export JSON
              </button>
              <button
                onClick={handleExportCSV}
                className="w-full px-3 py-2 text-left text-sm text-slate-300 hover:bg-slate-700 rounded-b-lg"
              >
                Export CSV
              </button>
            </div>
          </div>

          {/* Divider */}
          <div className="w-px h-6 bg-slate-700" />

          {/* Delete Button */}
          {onDeleteSelected && (
            <button
              onClick={handleDelete}
              className={`
                flex items-center gap-2 px-3 py-2 rounded-lg
                transition-colors text-sm font-medium
                ${showDeleteConfirm
                  ? 'bg-red-600 text-white animate-pulse'
                  : 'bg-slate-800 text-red-400 hover:bg-red-900/50'}
              `}
              title={showDeleteConfirm ? 'Click again to confirm' : 'Delete selected'}
            >
              {showDeleteConfirm ? (
                <>
                  <AlertTriangle className="w-4 h-4" />
                  <span>Confirm?</span>
                </>
              ) : (
                <>
                  <Trash2 className="w-4 h-4" />
                  <span>Delete</span>
                </>
              )}
            </button>
          )}

          {/* Clear Selection Button */}
          <button
            onClick={() => {
              setShowDeleteConfirm(false)
              onClearSelection()
            }}
            className="
              flex items-center gap-2 px-3 py-2 rounded-lg
              bg-slate-800 text-slate-400 hover:bg-slate-700
              transition-colors text-sm font-medium
            "
            title="Clear selection"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Common Tags Panel */}
        {showCommonTags && commonTags.length > 0 && (
          <div className="mt-3 pt-3 border-t border-slate-700/50">
            <div className="text-xs text-slate-500 mb-2">Common tags:</div>
            <div className="flex flex-wrap gap-1.5">
              {commonTags.map((tag) => (
                <span
                  key={tag}
                  className="px-2 py-1 bg-blue-600/20 text-blue-300 rounded-md text-xs"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        )}

        {showCommonTags && commonTags.length === 0 && (
          <div className="mt-3 pt-3 border-t border-slate-700/50">
            <div className="text-xs text-slate-500">No common tags found</div>
          </div>
        )}
      </div>

      {/* Cancel delete confirmation on click outside */}
      {showDeleteConfirm && (
        <div
          className="fixed inset-0 z-[-1]"
          onClick={() => setShowDeleteConfirm(false)}
        />
      )}
    </div>
  )
}

// Helper to trigger download
function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}
