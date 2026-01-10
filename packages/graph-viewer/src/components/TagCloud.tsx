/**
 * TagCloud - Interactive floating tag cloud for filtering memories
 *
 * Features:
 * - Tags sized by frequency
 * - Colored by dominant memory type
 * - Click to select/deselect tags
 * - AND/OR filter mode toggle
 * - Search to filter visible tags
 * - Floating animation effect
 */

import { useState, useEffect, useMemo, useRef } from 'react'
import { X, Search, ToggleLeft, ToggleRight, Trash2 } from 'lucide-react'
import type { TagData } from '../hooks/useTagCloud'

interface TagCloudProps {
  tags: TagData[]
  filteredTags: TagData[]
  selectedTags: Set<string>
  filterMode: 'AND' | 'OR'
  filteredCount: number
  totalCount: number
  onToggleTag: (tag: string) => void
  onClearSelection: () => void
  onToggleFilterMode: () => void
  onSearchChange: (term: string) => void
  searchTerm: string
  typeColors?: Record<string, string>
  visible: boolean
  onClose: () => void
}

// Default colors for memory types
const DEFAULT_TYPE_COLORS: Record<string, string> = {
  Decision: '#22c55e',
  Pattern: '#8b5cf6',
  Preference: '#f59e0b',
  Style: '#ec4899',
  Habit: '#06b6d4',
  Insight: '#3b82f6',
  Context: '#64748b',
  Memory: '#6366f1',
}

export function TagCloud({
  filteredTags,
  selectedTags,
  filterMode,
  filteredCount,
  totalCount,
  onToggleTag,
  onClearSelection,
  onToggleFilterMode,
  onSearchChange,
  searchTerm,
  typeColors = {},
  visible,
  onClose,
}: TagCloudProps) {
  const [isAnimating, setIsAnimating] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  // Merge type colors with defaults
  const colors = useMemo(() => ({ ...DEFAULT_TYPE_COLORS, ...typeColors }), [typeColors])

  // Animate in/out
  useEffect(() => {
    if (visible) {
      setIsAnimating(true)
    }
  }, [visible])

  // Handle escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && visible) {
        onClose()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [visible, onClose])

  if (!visible && !isAnimating) return null

  return (
    <div
      className={`
        fixed inset-0 z-50 flex items-center justify-center
        transition-all duration-300
        ${visible ? 'bg-black/50 backdrop-blur-sm' : 'bg-transparent pointer-events-none'}
      `}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose()
      }}
    >
      <div
        ref={containerRef}
        className={`
          relative max-w-3xl w-full mx-4 max-h-[80vh]
          bg-slate-900/95 backdrop-blur-lg rounded-2xl
          border border-slate-700/50 shadow-2xl
          transition-all duration-300 ease-out
          ${visible ? 'opacity-100 scale-100' : 'opacity-0 scale-95'}
        `}
        onTransitionEnd={() => {
          if (!visible) setIsAnimating(false)
        }}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-slate-700/50">
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-semibold text-white">Tag Cloud</h2>
            {selectedTags.size > 0 && (
              <span className="px-2 py-0.5 bg-blue-600/30 text-blue-300 rounded-full text-sm">
                {selectedTags.size} selected
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {/* Filter Mode Toggle */}
            <button
              onClick={onToggleFilterMode}
              className={`
                flex items-center gap-2 px-3 py-1.5 rounded-lg
                transition-colors text-sm font-medium
                ${filterMode === 'AND'
                  ? 'bg-purple-600/30 text-purple-300 border border-purple-500/50'
                  : 'bg-blue-600/30 text-blue-300 border border-blue-500/50'}
              `}
              title={filterMode === 'AND' ? 'Nodes must have ALL selected tags' : 'Nodes must have ANY selected tag'}
            >
              {filterMode === 'AND' ? (
                <ToggleRight className="w-4 h-4" />
              ) : (
                <ToggleLeft className="w-4 h-4" />
              )}
              {filterMode}
            </button>

            {/* Clear Selection */}
            {selectedTags.size > 0 && (
              <button
                onClick={onClearSelection}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 text-slate-400 hover:text-white transition-colors text-sm"
              >
                <Trash2 className="w-4 h-4" />
                Clear
              </button>
            )}

            {/* Close Button */}
            <button
              onClick={onClose}
              className="p-2 rounded-lg bg-slate-800 text-slate-400 hover:text-white transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Search */}
        <div className="p-4 border-b border-slate-700/50">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => onSearchChange(e.target.value)}
              placeholder="Search tags..."
              className="w-full pl-10 pr-4 py-2 bg-slate-800/50 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 transition-colors"
            />
          </div>
        </div>

        {/* Tags */}
        <div className="p-4 overflow-y-auto max-h-[50vh]">
          {filteredTags.length === 0 ? (
            <div className="text-center text-slate-500 py-8">
              No tags found
            </div>
          ) : (
            <div className="flex flex-wrap gap-2 justify-center">
              {filteredTags.map((tagData, index) => {
                const isSelected = selectedTags.has(tagData.tag)
                const color = colors[tagData.dominantType] || colors.Memory

                // Calculate font size based on frequency (0.75rem to 1.5rem)
                const fontSize = 0.75 + tagData.frequency * 0.75

                // Slight animation delay for staggered appearance
                const animationDelay = `${index * 20}ms`

                return (
                  <button
                    key={tagData.tag}
                    onClick={() => onToggleTag(tagData.tag)}
                    className={`
                      relative px-3 py-1.5 rounded-lg
                      transition-all duration-200 ease-out
                      hover:scale-105 hover:shadow-lg
                      ${isSelected
                        ? 'ring-2 ring-white/50 shadow-lg'
                        : 'hover:ring-1 hover:ring-white/20'}
                    `}
                    style={{
                      fontSize: `${fontSize}rem`,
                      backgroundColor: isSelected
                        ? color
                        : `${color}30`,
                      color: isSelected ? 'white' : color,
                      animationDelay,
                    }}
                    title={`${tagData.count} memor${tagData.count === 1 ? 'y' : 'ies'} • ${tagData.dominantType}`}
                  >
                    <span className="relative z-10">{tagData.tag}</span>
                    {/* Count badge */}
                    <span
                      className={`
                        absolute -top-1 -right-1 min-w-[1.25rem] h-5
                        flex items-center justify-center
                        text-xs font-medium rounded-full
                        ${isSelected
                          ? 'bg-white/20 text-white'
                          : 'bg-slate-800 text-slate-400'}
                      `}
                    >
                      {tagData.count}
                    </span>
                  </button>
                )
              })}
            </div>
          )}
        </div>

        {/* Footer - Results count */}
        <div className="p-4 border-t border-slate-700/50">
          <div className="flex items-center justify-between text-sm text-slate-400">
            <span>
              {filteredTags.length} tag{filteredTags.length !== 1 ? 's' : ''} shown
            </span>
            {selectedTags.size > 0 && (
              <span className="text-blue-400">
                {filteredCount} of {totalCount} memories match
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
