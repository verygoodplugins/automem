import type { RelationType } from './types'

export interface EdgeStyle {
  color: string           // Hex color
  opacity: number         // 0-1
  width: number          // Line width multiplier
  dashPattern: number[] | null  // [dashSize, gapSize] or null for solid
  arrow: boolean          // Whether to show directional arrow
  category: 'causal' | 'temporal' | 'associative' | 'conflict' | 'hierarchical'
  label: string           // Human-readable name
}

/**
 * Edge styles by relationship type
 * Organized by semantic category for visual coherence
 */
export const EDGE_STYLES: Record<RelationType, EdgeStyle> = {
  // Causal/Flow relationships (solid, bold, directional)
  LEADS_TO: {
    color: '#3B82F6',      // Blue
    opacity: 0.8,
    width: 2.0,
    dashPattern: null,
    arrow: true,
    category: 'causal',
    label: 'Leads To',
  },
  EVOLVED_INTO: {
    color: '#06B6D4',      // Cyan
    opacity: 0.7,
    width: 1.5,
    dashPattern: null,
    arrow: true,
    category: 'causal',
    label: 'Evolved Into',
  },
  DERIVED_FROM: {
    color: '#A855F7',      // Purple
    opacity: 0.7,
    width: 1.5,
    dashPattern: null,
    arrow: true,
    category: 'causal',
    label: 'Derived From',
  },

  // Temporal relationships (dashed, directional)
  OCCURRED_BEFORE: {
    color: '#6B7280',      // Gray
    opacity: 0.5,
    width: 1.0,
    dashPattern: [4, 2],
    arrow: true,
    category: 'temporal',
    label: 'Occurred Before',
  },
  INVALIDATED_BY: {
    color: '#F97316',      // Orange
    opacity: 0.6,
    width: 1.5,
    dashPattern: [4, 2],
    arrow: true,
    category: 'temporal',
    label: 'Invalidated By',
  },

  // Associative relationships (dotted, bidirectional)
  RELATES_TO: {
    color: '#94A3B8',      // Slate
    opacity: 0.4,
    width: 1.0,
    dashPattern: [2, 2],
    arrow: false,
    category: 'associative',
    label: 'Relates To',
  },
  EXEMPLIFIES: {
    color: '#10B981',      // Emerald
    opacity: 0.6,
    width: 1.5,
    dashPattern: [2, 2],
    arrow: false,
    category: 'associative',
    label: 'Exemplifies',
  },
  REINFORCES: {
    color: '#22C55E',      // Green
    opacity: 0.6,
    width: 1.5,
    dashPattern: [2, 2],
    arrow: false,
    category: 'associative',
    label: 'Reinforces',
  },

  // Conflict relationships (red, prominent)
  CONTRADICTS: {
    color: '#EF4444',      // Red
    opacity: 0.7,
    width: 2.0,
    dashPattern: [6, 3],
    arrow: false,
    category: 'conflict',
    label: 'Contradicts',
  },

  // Preference/Hierarchy (solid, thinner)
  PREFERS_OVER: {
    color: '#8B5CF6',      // Violet
    opacity: 0.6,
    width: 1.0,
    dashPattern: null,
    arrow: true,
    category: 'hierarchical',
    label: 'Prefers Over',
  },
  PART_OF: {
    color: '#64748B',      // Slate darker
    opacity: 0.5,
    width: 1.0,
    dashPattern: null,
    arrow: true,
    category: 'hierarchical',
    label: 'Part Of',
  },
}

/**
 * Get style for a relationship type with fallback
 */
export function getEdgeStyle(type: RelationType): EdgeStyle {
  return EDGE_STYLES[type] || EDGE_STYLES.RELATES_TO
}

/**
 * Category colors for grouping in UI
 */
export const CATEGORY_COLORS: Record<EdgeStyle['category'], string> = {
  causal: '#3B82F6',
  temporal: '#F97316',
  associative: '#10B981',
  conflict: '#EF4444',
  hierarchical: '#8B5CF6',
}

/**
 * Get edges grouped by category for UI display
 */
export function getEdgesByCategory(): Record<EdgeStyle['category'], RelationType[]> {
  const groups: Record<EdgeStyle['category'], RelationType[]> = {
    causal: [],
    temporal: [],
    associative: [],
    conflict: [],
    hierarchical: [],
  }

  for (const [type, style] of Object.entries(EDGE_STYLES)) {
    groups[style.category].push(type as RelationType)
  }

  return groups
}
