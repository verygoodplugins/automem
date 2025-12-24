/**
 * useLassoSelection - Select multiple nodes by drawing around them
 *
 * Features:
 * - Hold Shift + Drag to draw a lasso
 * - Point-in-polygon check for nodes
 * - Projects 3D positions to 2D screen space
 */

import { useState, useCallback, useRef, useMemo } from 'react'
import type { SimulationNode, GraphNode } from '../lib/types'
import * as THREE from 'three'

interface LassoPoint {
  x: number
  y: number
}

export interface LassoSelectionState {
  isDrawing: boolean
  points: LassoPoint[]
  selectedIds: Set<string>
}

interface UseLassoSelectionOptions {
  nodes: SimulationNode[]
  camera: THREE.Camera | null
  canvasRect: DOMRect | null
  enabled?: boolean
  onSelectionChange?: (selectedNodes: GraphNode[]) => void
}

interface UseLassoSelectionReturn {
  // State
  state: LassoSelectionState
  isDrawing: boolean
  points: LassoPoint[]
  selectedIds: Set<string>
  selectedNodes: GraphNode[]

  // Actions
  startDrawing: (x: number, y: number) => void
  addPoint: (x: number, y: number) => void
  finishDrawing: () => void
  cancelDrawing: () => void
  clearSelection: () => void
  toggleNodeSelection: (nodeId: string) => void
}

// Point-in-polygon check using ray casting algorithm
function isPointInPolygon(point: LassoPoint, polygon: LassoPoint[]): boolean {
  if (polygon.length < 3) return false

  let inside = false
  const n = polygon.length

  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = polygon[i].x
    const yi = polygon[i].y
    const xj = polygon[j].x
    const yj = polygon[j].y

    if (yi > point.y !== yj > point.y && point.x < ((xj - xi) * (point.y - yi)) / (yj - yi) + xi) {
      inside = !inside
    }
  }

  return inside
}

// Project 3D position to 2D screen coordinates
function projectToScreen(
  position: THREE.Vector3,
  camera: THREE.Camera,
  width: number,
  height: number
): LassoPoint {
  const projected = position.clone().project(camera)

  return {
    x: ((projected.x + 1) / 2) * width,
    y: ((-projected.y + 1) / 2) * height,
  }
}

export function useLassoSelection({
  nodes,
  camera,
  canvasRect,
  enabled = true,
  onSelectionChange,
}: UseLassoSelectionOptions): UseLassoSelectionReturn {
  const [state, setState] = useState<LassoSelectionState>({
    isDrawing: false,
    points: [],
    selectedIds: new Set(),
  })

  // Track the last added point to avoid duplicates
  const lastPointRef = useRef<LassoPoint | null>(null)

  // Start drawing the lasso
  const startDrawing = useCallback(
    (x: number, y: number) => {
      if (!enabled) return

      const point = { x, y }
      lastPointRef.current = point
      setState((prev) => ({
        ...prev,
        isDrawing: true,
        points: [point],
      }))
    },
    [enabled]
  )

  // Add a point to the lasso path
  const addPoint = useCallback(
    (x: number, y: number) => {
      if (!enabled) return

      const point = { x, y }
      const last = lastPointRef.current

      // Only add if moved enough (avoid too many points)
      if (last) {
        const dist = Math.sqrt(Math.pow(x - last.x, 2) + Math.pow(y - last.y, 2))
        if (dist < 3) return
      }

      lastPointRef.current = point
      setState((prev) => {
        if (!prev.isDrawing) return prev
        return {
          ...prev,
          points: [...prev.points, point],
        }
      })
    },
    [enabled]
  )

  // Finish drawing and select nodes inside the polygon
  const finishDrawing = useCallback(() => {
    if (!enabled || !camera || !canvasRect) {
      setState((prev) => ({ ...prev, isDrawing: false, points: [] }))
      return
    }

    setState((prev) => {
      if (!prev.isDrawing || prev.points.length < 3) {
        return { ...prev, isDrawing: false, points: [] }
      }

      // Find all nodes inside the lasso polygon
      const newSelectedIds = new Set<string>(prev.selectedIds)

      nodes.forEach((node) => {
        const worldPos = new THREE.Vector3(node.x ?? 0, node.y ?? 0, node.z ?? 0)
        const screenPos = projectToScreen(worldPos, camera, canvasRect.width, canvasRect.height)

        if (isPointInPolygon(screenPos, prev.points)) {
          newSelectedIds.add(node.id)
        }
      })

      return {
        isDrawing: false,
        points: [],
        selectedIds: newSelectedIds,
      }
    })

    lastPointRef.current = null
  }, [enabled, camera, canvasRect, nodes])

  // Cancel drawing without selecting
  const cancelDrawing = useCallback(() => {
    lastPointRef.current = null
    setState((prev) => ({
      ...prev,
      isDrawing: false,
      points: [],
    }))
  }, [])

  // Clear all selected nodes
  const clearSelection = useCallback(() => {
    setState((prev) => ({
      ...prev,
      selectedIds: new Set(),
    }))
    onSelectionChange?.([])
  }, [onSelectionChange])

  // Toggle a single node's selection
  const toggleNodeSelection = useCallback((nodeId: string) => {
    setState((prev) => {
      const newIds = new Set(prev.selectedIds)
      if (newIds.has(nodeId)) {
        newIds.delete(nodeId)
      } else {
        newIds.add(nodeId)
      }
      return { ...prev, selectedIds: newIds }
    })
  }, [])

  // Get selected nodes as full GraphNode objects
  const selectedNodes = useMemo(() => {
    return nodes.filter((n) => state.selectedIds.has(n.id)) as GraphNode[]
  }, [nodes, state.selectedIds])

  // Notify parent when selection changes
  const prevSelectionRef = useRef<Set<string>>(new Set())
  if (
    onSelectionChange &&
    !state.isDrawing &&
    !areSetsEqual(prevSelectionRef.current, state.selectedIds)
  ) {
    prevSelectionRef.current = new Set(state.selectedIds)
    onSelectionChange(selectedNodes)
  }

  return {
    state,
    isDrawing: state.isDrawing,
    points: state.points,
    selectedIds: state.selectedIds,
    selectedNodes,
    startDrawing,
    addPoint,
    finishDrawing,
    cancelDrawing,
    clearSelection,
    toggleNodeSelection,
  }
}

// Helper to compare sets
function areSetsEqual<T>(a: Set<T>, b: Set<T>): boolean {
  if (a.size !== b.size) return false
  for (const item of a) {
    if (!b.has(item)) return false
  }
  return true
}
