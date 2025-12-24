import { useState } from 'react'
import { X, Clock, Tag, ArrowRight, Sparkles, Edit2, Save, Trash2, Route } from 'lucide-react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useGraphNeighbors } from '../hooks/useGraphData'
import { updateMemory, deleteMemory } from '../api/client'
import type { GraphNode } from '../lib/types'

interface InspectorProps {
  node: GraphNode | null
  onClose: () => void
  onNavigate: (node: GraphNode) => void
  onStartPathfinding?: (nodeId: string) => void
  isPathSelecting?: boolean
}

export function Inspector({ node, onClose, onNavigate, onStartPathfinding, isPathSelecting }: InspectorProps) {
  const [isEditing, setIsEditing] = useState(false)
  const [editedImportance, setEditedImportance] = useState(0)
  const queryClient = useQueryClient()

  const { data: neighbors } = useGraphNeighbors(node?.id ?? null, {
    depth: 1,
    includeSemantic: true,
    semanticLimit: 5,
  })

  const updateMutation = useMutation({
    mutationFn: ({ id, updates }: { id: string; updates: { importance?: number } }) =>
      updateMemory(id, updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['graph'] })
      setIsEditing(false)
    },
  })

  const deleteMutation = useMutation({
    mutationFn: (id: string) => deleteMemory(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['graph'] })
      onClose()
    },
  })

  const handleStartEdit = () => {
    if (node) {
      setEditedImportance(node.importance)
      setIsEditing(true)
    }
  }

  const handleSave = () => {
    if (node) {
      updateMutation.mutate({
        id: node.id,
        updates: { importance: editedImportance },
      })
    }
  }

  const handleDelete = () => {
    if (node && confirm('Delete this memory? This cannot be undone.')) {
      deleteMutation.mutate(node.id)
    }
  }

  if (!node) {
    return (
      <div className="h-full glass flex flex-col items-center justify-center p-6 text-center">
        <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center mb-4">
          <Sparkles className="w-8 h-8 text-slate-600" />
        </div>
        <h3 className="text-lg font-medium text-slate-400 mb-2">No Memory Selected</h3>
        <p className="text-sm text-slate-500">
          Click a node in the graph to view its details
        </p>
      </div>
    )
  }

  const formattedDate = new Date(node.timestamp).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })

  return (
    <div className="h-full glass flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex-shrink-0 p-4 border-b border-white/5">
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <div
                className="w-3 h-3 rounded-full flex-shrink-0"
                style={{ backgroundColor: node.color }}
              />
              <span className="text-sm font-medium text-slate-300">{node.type}</span>
              <span className="text-xs text-slate-500">
                {(node.confidence * 100).toFixed(0)}% conf
              </span>
            </div>
            <div className="text-xs text-slate-500 font-mono truncate">{node.id}</div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-white/10 transition-colors text-slate-400 hover:text-slate-200"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Content */}
        <div>
          <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">
            Content
          </h4>
          <p className="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">
            {node.content}
          </p>
        </div>

        {/* Tags */}
        {node.tags.length > 0 && (
          <div>
            <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">
              Tags
            </h4>
            <div className="flex flex-wrap gap-1.5">
              {node.tags.map((tag) => (
                <span
                  key={tag}
                  className="inline-flex items-center gap-1 px-2 py-0.5 bg-white/5 rounded text-xs text-slate-300"
                >
                  <Tag className="w-3 h-3" />
                  {tag}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Importance */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wider">
              Importance
            </h4>
            {!isEditing ? (
              <button
                onClick={handleStartEdit}
                className="p-1 rounded hover:bg-white/10 text-slate-500 hover:text-slate-300"
              >
                <Edit2 className="w-3 h-3" />
              </button>
            ) : (
              <button
                onClick={handleSave}
                disabled={updateMutation.isPending}
                className="p-1 rounded hover:bg-white/10 text-blue-400 hover:text-blue-300"
              >
                <Save className="w-3 h-3" />
              </button>
            )}
          </div>
          {isEditing ? (
            <div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={editedImportance}
                onChange={(e) => setEditedImportance(parseFloat(e.target.value))}
                className="w-full accent-blue-500"
              />
              <div className="text-right text-xs text-slate-400">
                {editedImportance.toFixed(2)}
              </div>
            </div>
          ) : (
            <div className="relative h-2 bg-white/10 rounded-full overflow-hidden">
              <div
                className="absolute inset-y-0 left-0 rounded-full"
                style={{
                  width: `${node.importance * 100}%`,
                  backgroundColor: node.color,
                }}
              />
            </div>
          )}
          <div className="flex justify-between text-xs text-slate-500 mt-1">
            <span>Low</span>
            <span>{(node.importance * 100).toFixed(0)}%</span>
            <span>Critical</span>
          </div>
        </div>

        {/* Timestamp */}
        <div>
          <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">
            Created
          </h4>
          <div className="flex items-center gap-2 text-sm text-slate-300">
            <Clock className="w-4 h-4 text-slate-500" />
            {formattedDate}
          </div>
        </div>

        {/* Graph Relationships */}
        {neighbors?.edges && neighbors.edges.length > 0 && (
          <div>
            <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">
              Relationships ({neighbors.edges.length})
            </h4>
            <div className="space-y-2">
              {neighbors.graph_neighbors.slice(0, 5).map((neighbor) => {
                const edge = neighbors.edges.find(
                  (e) =>
                    (e.source === node.id && e.target === neighbor.id) ||
                    (e.target === node.id && e.source === neighbor.id)
                )

                return (
                  <button
                    key={neighbor.id}
                    onClick={() => onNavigate(neighbor)}
                    className="w-full flex items-start gap-2 p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors text-left group"
                  >
                    <div
                      className="w-2 h-2 rounded-full mt-1.5 flex-shrink-0"
                      style={{ backgroundColor: neighbor.color }}
                    />
                    <div className="flex-1 min-w-0">
                      <div className="text-xs text-slate-400 mb-0.5">
                        {edge?.type.replace(/_/g, ' ')}
                      </div>
                      <div className="text-sm text-slate-200 line-clamp-2">
                        {neighbor.content.slice(0, 80)}
                        {neighbor.content.length > 80 && '...'}
                      </div>
                    </div>
                    <ArrowRight className="w-4 h-4 text-slate-600 group-hover:text-slate-400 flex-shrink-0" />
                  </button>
                )
              })}
            </div>
          </div>
        )}

        {/* Semantic Neighbors */}
        {neighbors?.semantic_neighbors && neighbors.semantic_neighbors.length > 0 && (
          <div>
            <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">
              Similar Memories
            </h4>
            <div className="space-y-2">
              {neighbors.semantic_neighbors.map((neighbor) => (
                <button
                  key={neighbor.id}
                  onClick={() => onNavigate(neighbor)}
                  className="w-full flex items-start gap-2 p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors text-left group"
                >
                  <div
                    className="w-2 h-2 rounded-full mt-1.5 flex-shrink-0"
                    style={{ backgroundColor: neighbor.color }}
                  />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-0.5">
                      <span className="text-xs text-slate-400">{neighbor.type}</span>
                      <span className="text-xs text-green-400">
                        {(neighbor.similarity * 100).toFixed(0)}% similar
                      </span>
                    </div>
                    <div className="text-sm text-slate-200 line-clamp-2">
                      {neighbor.content.slice(0, 80)}
                      {neighbor.content.length > 80 && '...'}
                    </div>
                  </div>
                  <ArrowRight className="w-4 h-4 text-slate-600 group-hover:text-slate-400 flex-shrink-0" />
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Footer Actions */}
      <div className="flex-shrink-0 p-4 border-t border-white/5 space-y-2">
        {/* Find Path Button */}
        {onStartPathfinding && (
          <button
            onClick={() => onStartPathfinding(node.id)}
            disabled={isPathSelecting}
            className={`w-full flex items-center justify-center gap-2 py-2 text-sm rounded-lg transition-colors ${
              isPathSelecting
                ? 'text-cyan-400 bg-cyan-500/20 cursor-not-allowed'
                : 'text-cyan-400 hover:text-cyan-300 hover:bg-cyan-500/10'
            }`}
          >
            <Route className="w-4 h-4" />
            {isPathSelecting ? 'Click destination node...' : 'Find Path To...'}
          </button>
        )}

        <button
          onClick={handleDelete}
          disabled={deleteMutation.isPending}
          className="w-full flex items-center justify-center gap-2 py-2 text-sm text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded-lg transition-colors"
        >
          <Trash2 className="w-4 h-4" />
          {deleteMutation.isPending ? 'Deleting...' : 'Delete Memory'}
        </button>
      </div>
    </div>
  )
}
