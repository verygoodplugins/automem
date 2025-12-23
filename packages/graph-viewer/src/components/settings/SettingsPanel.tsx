import { X, RotateCcw, Zap } from 'lucide-react'
import { SettingsSection } from './SettingsSection'
import { SliderControl } from './SliderControl'
import { ToggleControl } from './ToggleControl'
import type {
  ForceConfig,
  DisplayConfig,
  ClusterConfig,
  ClusterMode,
  RelationType,
  RelationshipVisibility,
  MemoryType,
  FilterState,
} from '../../lib/types'

// Relationship type metadata for display
const RELATIONSHIP_INFO: Record<RelationType, { label: string; color: string; style: string }> = {
  RELATES_TO: { label: 'Relates To', color: '#94A3B8', style: 'dotted' },
  LEADS_TO: { label: 'Leads To', color: '#3B82F6', style: 'solid' },
  OCCURRED_BEFORE: { label: 'Occurred Before', color: '#6B7280', style: 'dashed' },
  PREFERS_OVER: { label: 'Prefers Over', color: '#8B5CF6', style: 'solid' },
  EXEMPLIFIES: { label: 'Exemplifies', color: '#10B981', style: 'dotted' },
  CONTRADICTS: { label: 'Contradicts', color: '#EF4444', style: 'dashed' },
  REINFORCES: { label: 'Reinforces', color: '#22C55E', style: 'dotted' },
  INVALIDATED_BY: { label: 'Invalidated By', color: '#F97316', style: 'dashed' },
  EVOLVED_INTO: { label: 'Evolved Into', color: '#06B6D4', style: 'solid' },
  DERIVED_FROM: { label: 'Derived From', color: '#A855F7', style: 'solid' },
  PART_OF: { label: 'Part Of', color: '#64748B', style: 'solid' },
}

const MEMORY_TYPES: MemoryType[] = [
  'Decision', 'Pattern', 'Preference', 'Style',
  'Habit', 'Insight', 'Context', 'Memory',
]

const CLUSTER_MODES: { value: ClusterMode; label: string }[] = [
  { value: 'none', label: 'None' },
  { value: 'type', label: 'By Type' },
  { value: 'tags', label: 'By Tags' },
  { value: 'semantic', label: 'Semantic' },
]

interface SettingsPanelProps {
  isOpen: boolean
  onClose: () => void
  // Filter state
  filters: FilterState
  onFiltersChange: (filters: Partial<FilterState>) => void
  typeColors?: Record<string, string>
  // Force configuration
  forceConfig: ForceConfig
  onForceConfigChange: (config: Partial<ForceConfig>) => void
  onReheat: () => void
  onResetForces: () => void
  // Display settings
  displayConfig: DisplayConfig
  onDisplayConfigChange: (config: Partial<DisplayConfig>) => void
  // Clustering
  clusterConfig: ClusterConfig
  onClusterConfigChange: (config: Partial<ClusterConfig>) => void
  // Relationship visibility
  relationshipVisibility: RelationshipVisibility
  onRelationshipVisibilityChange: (visibility: Partial<RelationshipVisibility>) => void
}

export function SettingsPanel({
  isOpen,
  onClose,
  filters,
  onFiltersChange,
  typeColors = {},
  forceConfig,
  onForceConfigChange,
  onReheat,
  onResetForces,
  displayConfig,
  onDisplayConfigChange,
  clusterConfig,
  onClusterConfigChange,
  relationshipVisibility,
  onRelationshipVisibilityChange,
}: SettingsPanelProps) {
  if (!isOpen) return null

  const toggleType = (type: MemoryType) => {
    const types = filters.types.includes(type)
      ? filters.types.filter((t) => t !== type)
      : [...filters.types, type]
    onFiltersChange({ types })
  }

  return (
    <div className="h-full w-72 glass flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex-shrink-0 flex items-center justify-between px-4 py-3 border-b border-white/5">
        <h2 className="text-sm font-semibold text-slate-200">Graph Settings</h2>
        <button
          onClick={onClose}
          className="p-1 rounded hover:bg-white/10 text-slate-400 hover:text-slate-200 transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto">
        {/* Filters Section */}
        <SettingsSection title="Filters" defaultOpen={true}>
          {/* Memory Types */}
          <div className="space-y-1.5">
            <label className="text-xs text-slate-400">Memory Types</label>
            <div className="flex flex-wrap gap-1">
              {MEMORY_TYPES.map((type) => {
                const isSelected = filters.types.length === 0 || filters.types.includes(type)
                const color = typeColors[type] || '#94A3B8'
                return (
                  <button
                    key={type}
                    onClick={() => toggleType(type)}
                    className={`
                      flex items-center gap-1.5 px-2 py-1 rounded text-xs transition-all
                      ${isSelected
                        ? 'bg-white/10 text-slate-200'
                        : 'bg-white/5 text-slate-500 opacity-50'
                      }
                    `}
                  >
                    <div
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: color }}
                    />
                    {type}
                  </button>
                )
              })}
            </div>
            {filters.types.length > 0 && (
              <button
                onClick={() => onFiltersChange({ types: [] })}
                className="text-xs text-slate-500 hover:text-slate-300"
              >
                Clear filter
              </button>
            )}
          </div>

          <SliderControl
            label="Min Importance"
            value={filters.minImportance}
            min={0}
            max={1}
            step={0.1}
            onChange={(v) => onFiltersChange({ minImportance: v })}
            formatValue={(v) => v.toFixed(1)}
          />

          <div className="space-y-1.5">
            <label className="text-xs text-slate-400">Max Nodes</label>
            <div className="flex gap-1">
              {[100, 250, 500, 1000].map((n) => (
                <button
                  key={n}
                  onClick={() => onFiltersChange({ maxNodes: n })}
                  className={`
                    flex-1 py-1 text-xs rounded transition-colors
                    ${filters.maxNodes === n
                      ? 'bg-blue-500 text-white'
                      : 'bg-white/5 text-slate-400 hover:bg-white/10'
                    }
                  `}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>
        </SettingsSection>

        {/* Relationships Section */}
        <SettingsSection title="Relationships" defaultOpen={false}>
          <div className="space-y-1">
            {(Object.keys(RELATIONSHIP_INFO) as RelationType[]).map((rel) => {
              const info = RELATIONSHIP_INFO[rel]
              const isVisible = relationshipVisibility[rel]
              return (
                <button
                  key={rel}
                  onClick={() => onRelationshipVisibilityChange({ [rel]: !isVisible })}
                  className={`
                    w-full flex items-center gap-2 px-2 py-1.5 rounded text-xs transition-all
                    ${isVisible
                      ? 'bg-white/5 text-slate-300'
                      : 'text-slate-600'
                    }
                  `}
                >
                  <div
                    className={`w-4 h-0.5 ${
                      info.style === 'dashed' ? 'border-t border-dashed' :
                      info.style === 'dotted' ? 'border-t border-dotted' :
                      ''
                    }`}
                    style={{
                      backgroundColor: info.style === 'solid' ? info.color : 'transparent',
                      borderColor: info.color,
                    }}
                  />
                  <span className="flex-1 text-left">{info.label}</span>
                  <div
                    className={`w-3 h-3 rounded border transition-colors ${
                      isVisible
                        ? 'bg-blue-500 border-blue-500'
                        : 'border-slate-600'
                    }`}
                  >
                    {isVisible && (
                      <svg className="w-full h-full text-white" viewBox="0 0 12 12">
                        <path
                          d="M2 6l3 3 5-5"
                          stroke="currentColor"
                          strokeWidth="2"
                          fill="none"
                        />
                      </svg>
                    )}
                  </div>
                </button>
              )
            })}
          </div>
        </SettingsSection>

        {/* Display Section */}
        <SettingsSection title="Display" defaultOpen={true}>
          <ToggleControl
            label="Show Labels"
            checked={displayConfig.showLabels}
            onChange={(v) => onDisplayConfigChange({ showLabels: v })}
          />

          {displayConfig.showLabels && (
            <SliderControl
              label="Label Fade Distance"
              value={displayConfig.labelFadeDistance}
              min={20}
              max={200}
              step={10}
              onChange={(v) => onDisplayConfigChange({ labelFadeDistance: v })}
              formatValue={(v) => v.toFixed(0)}
            />
          )}

          <ToggleControl
            label="Show Arrows"
            checked={displayConfig.showArrows}
            onChange={(v) => onDisplayConfigChange({ showArrows: v })}
            description="Directional arrows on edges"
          />

          <SliderControl
            label="Node Size"
            value={displayConfig.nodeSizeScale}
            min={0.5}
            max={2}
            step={0.1}
            onChange={(v) => onDisplayConfigChange({ nodeSizeScale: v })}
            formatValue={(v) => `${v.toFixed(1)}x`}
          />

          <SliderControl
            label="Link Thickness"
            value={displayConfig.linkThickness}
            min={0.5}
            max={3}
            step={0.25}
            onChange={(v) => onDisplayConfigChange({ linkThickness: v })}
            formatValue={(v) => v.toFixed(2)}
          />

          <SliderControl
            label="Link Opacity"
            value={displayConfig.linkOpacity}
            min={0.1}
            max={1}
            step={0.1}
            onChange={(v) => onDisplayConfigChange({ linkOpacity: v })}
            formatValue={(v) => `${Math.round(v * 100)}%`}
          />
        </SettingsSection>

        {/* Clustering Section */}
        <SettingsSection title="Clustering" defaultOpen={false}>
          <div className="space-y-1.5">
            <label className="text-xs text-slate-400">Cluster Mode</label>
            <div className="grid grid-cols-2 gap-1">
              {CLUSTER_MODES.map(({ value, label }) => (
                <button
                  key={value}
                  onClick={() => onClusterConfigChange({ mode: value })}
                  className={`
                    py-1.5 text-xs rounded transition-colors
                    ${clusterConfig.mode === value
                      ? 'bg-blue-500 text-white'
                      : 'bg-white/5 text-slate-400 hover:bg-white/10'
                    }
                  `}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {clusterConfig.mode !== 'none' && (
            <>
              <ToggleControl
                label="Show Boundaries"
                checked={clusterConfig.showBoundaries}
                onChange={(v) => onClusterConfigChange({ showBoundaries: v })}
                description="Dotted circles around clusters"
              />

              <SliderControl
                label="Cluster Strength"
                value={clusterConfig.clusterStrength}
                min={0}
                max={1}
                step={0.05}
                onChange={(v) => onClusterConfigChange({ clusterStrength: v })}
                formatValue={(v) => v.toFixed(2)}
              />
            </>
          )}
        </SettingsSection>

        {/* Forces Section */}
        <SettingsSection title="Forces" defaultOpen={false}>
          <SliderControl
            label="Center Force"
            value={forceConfig.centerStrength}
            min={0.01}
            max={0.2}
            step={0.01}
            onChange={(v) => onForceConfigChange({ centerStrength: v })}
          />

          <SliderControl
            label="Repel Force"
            value={forceConfig.chargeStrength}
            min={-200}
            max={-20}
            step={10}
            onChange={(v) => onForceConfigChange({ chargeStrength: v })}
            formatValue={(v) => v.toFixed(0)}
          />

          <SliderControl
            label="Link Force"
            value={forceConfig.linkStrength}
            min={0.1}
            max={1}
            step={0.05}
            onChange={(v) => onForceConfigChange({ linkStrength: v })}
          />

          <SliderControl
            label="Link Distance"
            value={forceConfig.linkDistance}
            min={20}
            max={100}
            step={5}
            onChange={(v) => onForceConfigChange({ linkDistance: v })}
            formatValue={(v) => v.toFixed(0)}
          />

          <SliderControl
            label="Collision Radius"
            value={forceConfig.collisionRadius}
            min={1}
            max={4}
            step={0.25}
            onChange={(v) => onForceConfigChange({ collisionRadius: v })}
          />

          <div className="flex gap-2 pt-2">
            <button
              onClick={onReheat}
              className="flex-1 flex items-center justify-center gap-1.5 py-2 bg-blue-500 hover:bg-blue-400 text-white text-xs rounded transition-colors"
            >
              <Zap className="w-3 h-3" />
              Reheat
            </button>
            <button
              onClick={onResetForces}
              className="flex-1 flex items-center justify-center gap-1.5 py-2 bg-white/10 hover:bg-white/20 text-slate-300 text-xs rounded transition-colors"
            >
              <RotateCcw className="w-3 h-3" />
              Reset
            </button>
          </div>
        </SettingsSection>
      </div>
    </div>
  )
}
