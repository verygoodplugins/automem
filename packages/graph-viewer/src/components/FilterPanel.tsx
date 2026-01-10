import { useState } from 'react'
import { Filter, ChevronDown, Check } from 'lucide-react'
import type { FilterState, MemoryType } from '../lib/types'

const MEMORY_TYPES: MemoryType[] = [
  'Decision',
  'Pattern',
  'Preference',
  'Style',
  'Habit',
  'Insight',
  'Context',
  'Memory',
]

interface FilterPanelProps {
  filters: FilterState
  onChange: (filters: Partial<FilterState>) => void
  typeColors?: Record<string, string>
}

export function FilterPanel({ filters, onChange, typeColors = {} }: FilterPanelProps) {
  const [isOpen, setIsOpen] = useState(false)

  const toggleType = (type: MemoryType) => {
    const types = filters.types.includes(type)
      ? filters.types.filter((t) => t !== type)
      : [...filters.types, type]
    onChange({ types })
  }

  const selectedCount = filters.types.length

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`flex items-center gap-2 px-3 py-2 rounded-lg border transition-colors ${
          selectedCount > 0
            ? 'bg-blue-500/20 border-blue-500/50 text-blue-400'
            : 'bg-black/30 border-white/10 text-slate-400 hover:border-white/20'
        }`}
      >
        <Filter className="w-4 h-4" />
        <span className="text-sm">
          {selectedCount > 0 ? `${selectedCount} types` : 'Filter'}
        </span>
        <ChevronDown
          className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`}
        />
      </button>

      {isOpen && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-40"
            onClick={() => setIsOpen(false)}
          />

          {/* Dropdown */}
          <div className="absolute right-0 top-full mt-2 w-64 glass rounded-xl shadow-xl z-50 p-3">
            <div className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">
              Memory Types
            </div>
            <div className="space-y-1">
              {MEMORY_TYPES.map((type) => {
                const isSelected = filters.types.includes(type)
                const color = typeColors[type] || '#94A3B8'

                return (
                  <button
                    key={type}
                    onClick={() => toggleType(type)}
                    className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg transition-colors ${
                      isSelected
                        ? 'bg-white/10'
                        : 'hover:bg-white/5'
                    }`}
                  >
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: color }}
                    />
                    <span className="flex-1 text-left text-sm text-slate-200">
                      {type}
                    </span>
                    {isSelected && (
                      <Check className="w-4 h-4 text-blue-400" />
                    )}
                  </button>
                )
              })}
            </div>

            <div className="mt-3 pt-3 border-t border-white/10">
              <div className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">
                Importance
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={filters.minImportance}
                onChange={(e) =>
                  onChange({ minImportance: parseFloat(e.target.value) })
                }
                className="w-full accent-blue-500"
              />
              <div className="flex justify-between text-xs text-slate-500">
                <span>All</span>
                <span>{filters.minImportance.toFixed(1)}</span>
                <span>Critical</span>
              </div>
            </div>

            <div className="mt-3 pt-3 border-t border-white/10">
              <div className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">
                Max Nodes
              </div>
              <div className="flex gap-2">
                {[100, 250, 500, 1000].map((n) => (
                  <button
                    key={n}
                    onClick={() => onChange({ maxNodes: n })}
                    className={`flex-1 py-1 text-xs rounded ${
                      filters.maxNodes === n
                        ? 'bg-blue-500 text-white'
                        : 'bg-white/5 text-slate-400 hover:bg-white/10'
                    }`}
                  >
                    {n}
                  </button>
                ))}
              </div>
            </div>

            {selectedCount > 0 && (
              <button
                onClick={() => onChange({ types: [] })}
                className="mt-3 w-full py-2 text-sm text-slate-400 hover:text-slate-200 transition-colors"
              >
                Clear filters
              </button>
            )}
          </div>
        </>
      )}
    </div>
  )
}
