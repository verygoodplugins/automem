/**
 * useTagCloud - Aggregate and manage tags for the interactive tag cloud
 *
 * Features:
 * - Aggregates tags from all nodes
 * - Calculates tag frequency and dominant type
 * - Supports multi-select filtering (AND/OR modes)
 */

import { useMemo, useState, useCallback } from 'react'
import type { GraphNode } from '../lib/types'

export interface TagData {
  tag: string
  count: number
  frequency: number // 0-1 based on max count
  dominantType: string
  types: Record<string, number> // Type -> count mapping
}

export interface UseTagCloudOptions {
  nodes: GraphNode[]
  typeColors?: Record<string, string>
  maxTags?: number
}

export interface UseTagCloudReturn {
  // Data
  tags: TagData[]
  filteredTags: TagData[]

  // Selection state
  selectedTags: Set<string>
  filterMode: 'AND' | 'OR'

  // Filtered nodes based on selection
  filteredNodeIds: Set<string>
  hasActiveFilter: boolean

  // Actions
  toggleTag: (tag: string) => void
  selectTag: (tag: string) => void
  deselectTag: (tag: string) => void
  clearSelection: () => void
  toggleFilterMode: () => void
  setFilterMode: (mode: 'AND' | 'OR') => void

  // Search
  searchTerm: string
  setSearchTerm: (term: string) => void
}

export function useTagCloud({
  nodes,
  typeColors: _typeColors = {},
  maxTags = 50,
}: UseTagCloudOptions): UseTagCloudReturn {
  const [selectedTags, setSelectedTags] = useState<Set<string>>(new Set())
  const [filterMode, setFilterMode] = useState<'AND' | 'OR'>('OR')
  const [searchTerm, setSearchTerm] = useState('')

  // Aggregate tags from all nodes
  const tags = useMemo(() => {
    const tagMap = new Map<string, { count: number; types: Record<string, number> }>()

    nodes.forEach((node) => {
      node.tags.forEach((tag) => {
        const existing = tagMap.get(tag)
        if (existing) {
          existing.count++
          existing.types[node.type] = (existing.types[node.type] || 0) + 1
        } else {
          tagMap.set(tag, {
            count: 1,
            types: { [node.type]: 1 },
          })
        }
      })
    })

    // Convert to array and calculate frequencies
    const maxCount = Math.max(...Array.from(tagMap.values()).map((t) => t.count), 1)

    const tagArray: TagData[] = Array.from(tagMap.entries())
      .map(([tag, data]) => {
        // Find dominant type (most common type for this tag)
        const dominantType = Object.entries(data.types).sort((a, b) => b[1] - a[1])[0]?.[0] || 'Memory'

        return {
          tag,
          count: data.count,
          frequency: data.count / maxCount,
          dominantType,
          types: data.types,
        }
      })
      .sort((a, b) => b.count - a.count)
      .slice(0, maxTags)

    return tagArray
  }, [nodes, maxTags])

  // Filter tags by search term
  const filteredTags = useMemo(() => {
    if (!searchTerm) return tags
    const lower = searchTerm.toLowerCase()
    return tags.filter((t) => t.tag.toLowerCase().includes(lower))
  }, [tags, searchTerm])

  // Get filtered node IDs based on selected tags
  const filteredNodeIds = useMemo(() => {
    if (selectedTags.size === 0) {
      return new Set<string>(nodes.map((n) => n.id))
    }

    const selectedTagsArray = Array.from(selectedTags)

    return new Set(
      nodes
        .filter((node) => {
          if (filterMode === 'AND') {
            // Node must have ALL selected tags
            return selectedTagsArray.every((tag) => node.tags.includes(tag))
          } else {
            // Node must have ANY selected tag
            return selectedTagsArray.some((tag) => node.tags.includes(tag))
          }
        })
        .map((n) => n.id)
    )
  }, [nodes, selectedTags, filterMode])

  // Actions
  const toggleTag = useCallback((tag: string) => {
    setSelectedTags((prev) => {
      const next = new Set(prev)
      if (next.has(tag)) {
        next.delete(tag)
      } else {
        next.add(tag)
      }
      return next
    })
  }, [])

  const selectTag = useCallback((tag: string) => {
    setSelectedTags((prev) => new Set(prev).add(tag))
  }, [])

  const deselectTag = useCallback((tag: string) => {
    setSelectedTags((prev) => {
      const next = new Set(prev)
      next.delete(tag)
      return next
    })
  }, [])

  const clearSelection = useCallback(() => {
    setSelectedTags(new Set())
  }, [])

  const toggleFilterMode = useCallback(() => {
    setFilterMode((prev) => (prev === 'AND' ? 'OR' : 'AND'))
  }, [])

  return {
    tags,
    filteredTags,
    selectedTags,
    filterMode,
    filteredNodeIds,
    hasActiveFilter: selectedTags.size > 0,
    toggleTag,
    selectTag,
    deselectTag,
    clearSelection,
    toggleFilterMode,
    setFilterMode,
    searchTerm,
    setSearchTerm,
  }
}
