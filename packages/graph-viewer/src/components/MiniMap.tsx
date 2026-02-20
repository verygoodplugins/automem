/**
 * MiniMap - 2D overview navigator for the 3D graph
 *
 * Shows a simplified bird's-eye view of all nodes with:
 * - Nodes colored by type
 * - Current viewport rectangle
 * - Click-to-navigate
 * - Selected node highlight
 */

import { useRef, useEffect, useCallback, useMemo } from 'react'
import type { SimulationNode, GraphNode } from '../lib/types'

interface MiniMapProps {
  nodes: SimulationNode[]
  selectedNode: GraphNode | null
  cameraPosition: { x: number; y: number; z: number }
  cameraZoom: number
  onNavigate: (x: number, y: number) => void
  visible?: boolean
  size?: number
}

// Type colors (simplified from main graph)
const TYPE_COLORS: Record<string, string> = {
  Decision: '#f59e0b',
  Pattern: '#8b5cf6',
  Preference: '#3b82f6',
  Style: '#ec4899',
  Habit: '#10b981',
  Insight: '#06b6d4',
  Context: '#6366f1',
  Memory: '#6b7280',
}

export function MiniMap({
  nodes,
  selectedNode,
  cameraPosition,
  cameraZoom,
  onNavigate,
  visible = true,
  size = 150,
}: MiniMapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  // Calculate bounds of all nodes
  const bounds = useMemo(() => {
    if (nodes.length === 0) {
      return { minX: -100, maxX: 100, minY: -100, maxY: 100 }
    }

    let minX = Infinity, maxX = -Infinity
    let minY = Infinity, maxY = -Infinity

    nodes.forEach(node => {
      const x = node.x ?? 0
      const y = node.y ?? 0
      minX = Math.min(minX, x)
      maxX = Math.max(maxX, x)
      minY = Math.min(minY, y)
      maxY = Math.max(maxY, y)
    })

    // Add padding
    const padding = 20
    return {
      minX: minX - padding,
      maxX: maxX + padding,
      minY: minY - padding,
      maxY: maxY + padding,
    }
  }, [nodes])

  // Convert world coordinates to canvas coordinates
  const worldToCanvas = useCallback((x: number, y: number) => {
    const rangeX = bounds.maxX - bounds.minX
    const rangeY = bounds.maxY - bounds.minY
    const scale = Math.max(rangeX, rangeY)

    const canvasX = ((x - bounds.minX) / scale) * size
    const canvasY = ((bounds.maxY - y) / scale) * size // Flip Y axis

    return { x: canvasX, y: canvasY }
  }, [bounds, size])

  // Convert canvas coordinates to world coordinates
  const canvasToWorld = useCallback((canvasX: number, canvasY: number) => {
    const rangeX = bounds.maxX - bounds.minX
    const rangeY = bounds.maxY - bounds.minY
    const scale = Math.max(rangeX, rangeY)

    const x = (canvasX / size) * scale + bounds.minX
    const y = bounds.maxY - (canvasY / size) * scale // Flip Y axis

    return { x, y }
  }, [bounds, size])

  // Draw the mini-map
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !visible) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear canvas
    ctx.fillStyle = 'rgba(10, 10, 20, 0.85)'
    ctx.fillRect(0, 0, size, size)

    // Draw border
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)'
    ctx.lineWidth = 1
    ctx.strokeRect(0, 0, size, size)

    // Draw nodes
    nodes.forEach(node => {
      const pos = worldToCanvas(node.x ?? 0, node.y ?? 0)
      const isSelected = selectedNode?.id === node.id
      const nodeColor = TYPE_COLORS[node.type] || TYPE_COLORS.Memory

      ctx.beginPath()
      ctx.arc(pos.x, pos.y, isSelected ? 4 : 2, 0, Math.PI * 2)
      ctx.fillStyle = isSelected ? '#ffffff' : nodeColor
      ctx.fill()

      // Glow effect for selected node
      if (isSelected) {
        ctx.beginPath()
        ctx.arc(pos.x, pos.y, 8, 0, Math.PI * 2)
        ctx.fillStyle = 'rgba(255, 255, 255, 0.2)'
        ctx.fill()
      }
    })

    // Draw viewport rectangle
    const viewportSize = 100 / Math.max(cameraZoom, 0.1) // Rough estimate
    const viewCenter = worldToCanvas(cameraPosition.x, cameraPosition.y)

    const halfSize = (viewportSize / (bounds.maxX - bounds.minX)) * size / 2
    const rectSize = Math.min(halfSize * 2, size * 0.8)

    ctx.strokeStyle = 'rgba(59, 130, 246, 0.8)'
    ctx.lineWidth = 2
    ctx.strokeRect(
      viewCenter.x - rectSize / 2,
      viewCenter.y - rectSize / 2,
      rectSize,
      rectSize
    )

    // Draw camera center dot
    ctx.beginPath()
    ctx.arc(viewCenter.x, viewCenter.y, 3, 0, Math.PI * 2)
    ctx.fillStyle = 'rgba(59, 130, 246, 1)'
    ctx.fill()

  }, [nodes, selectedNode, cameraPosition, cameraZoom, bounds, size, visible, worldToCanvas])

  // Handle click to navigate
  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const canvasX = e.clientX - rect.left
    const canvasY = e.clientY - rect.top

    const worldPos = canvasToWorld(canvasX, canvasY)
    onNavigate(worldPos.x, worldPos.y)
  }, [canvasToWorld, onNavigate])

  if (!visible) return null

  return (
    <div
      ref={containerRef}
      className="absolute bottom-4 left-4 z-40 rounded-lg overflow-hidden shadow-xl"
      style={{
        width: size,
        height: size,
        backdropFilter: 'blur(8px)',
      }}
    >
      <canvas
        ref={canvasRef}
        width={size}
        height={size}
        onClick={handleClick}
        className="cursor-crosshair hover:opacity-100 opacity-80 transition-opacity"
        title="Click to navigate"
      />
      {/* Zoom indicator */}
      <div className="absolute bottom-1 right-1 text-[10px] text-white/50 font-mono">
        {(cameraZoom * 100).toFixed(0)}%
      </div>
    </div>
  )
}
