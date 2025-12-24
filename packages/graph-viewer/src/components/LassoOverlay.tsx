/**
 * LassoOverlay - SVG overlay for drawing lasso selection path
 *
 * Features:
 * - Draws the lasso path as user drags
 * - Animated dashed stroke
 * - Shows selection count badge
 */

import { useRef, useCallback, useEffect } from 'react'

interface LassoPoint {
  x: number
  y: number
}

interface LassoOverlayProps {
  isDrawing: boolean
  points: LassoPoint[]
  selectedCount: number
  onStartDraw: (x: number, y: number) => void
  onMoveDraw: (x: number, y: number) => void
  onEndDraw: () => void
  onCancelDraw: () => void
}

export function LassoOverlay({
  isDrawing,
  points,
  selectedCount,
  onStartDraw,
  onMoveDraw,
  onEndDraw,
  onCancelDraw,
}: LassoOverlayProps) {
  const overlayRef = useRef<HTMLDivElement>(null)
  const isShiftPressedRef = useRef(false)

  // Track Shift key state
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Shift') {
        isShiftPressedRef.current = true
      }
      if (e.key === 'Escape' && isDrawing) {
        onCancelDraw()
      }
    }

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.key === 'Shift') {
        isShiftPressedRef.current = false
        // If we were drawing when Shift is released, finish the drawing
        if (isDrawing) {
          onEndDraw()
        }
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    window.addEventListener('keyup', handleKeyUp)
    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
    }
  }, [isDrawing, onCancelDraw, onEndDraw])

  // Handle mouse events
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      // Only start if Shift is held
      if (!isShiftPressedRef.current) return

      const rect = overlayRef.current?.getBoundingClientRect()
      if (!rect) return

      onStartDraw(e.clientX - rect.left, e.clientY - rect.top)
    },
    [onStartDraw]
  )

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!isDrawing) return

      const rect = overlayRef.current?.getBoundingClientRect()
      if (!rect) return

      onMoveDraw(e.clientX - rect.left, e.clientY - rect.top)
    },
    [isDrawing, onMoveDraw]
  )

  const handleMouseUp = useCallback(() => {
    if (isDrawing) {
      onEndDraw()
    }
  }, [isDrawing, onEndDraw])

  // Build SVG path
  const pathD =
    points.length > 0
      ? points.reduce((d, point, i) => {
          if (i === 0) return `M ${point.x} ${point.y}`
          return `${d} L ${point.x} ${point.y}`
        }, '') + ' Z' // Close the path
      : ''

  return (
    <div
      ref={overlayRef}
      className="absolute inset-0 pointer-events-auto"
      style={{
        cursor: isShiftPressedRef.current || isDrawing ? 'crosshair' : 'default',
        zIndex: 40,
      }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {/* SVG overlay for drawing */}
      {isDrawing && points.length > 0 && (
        <svg className="absolute inset-0 w-full h-full pointer-events-none">
          {/* Fill area */}
          <path
            d={pathD}
            fill="rgba(59, 130, 246, 0.1)"
            stroke="none"
          />
          {/* Animated dashed border */}
          <path
            d={pathD}
            fill="none"
            stroke="#3b82f6"
            strokeWidth="2"
            strokeDasharray="8 4"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="animate-dash"
          />
          {/* Points indicator */}
          <circle
            cx={points[0].x}
            cy={points[0].y}
            r="4"
            fill="#3b82f6"
            stroke="white"
            strokeWidth="2"
          />
        </svg>
      )}

      {/* Selection count badge */}
      {selectedCount > 0 && !isDrawing && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 px-4 py-2 bg-blue-600/90 backdrop-blur-sm rounded-full text-white font-medium text-sm shadow-lg">
          {selectedCount} node{selectedCount !== 1 ? 's' : ''} selected
          <span className="ml-2 text-blue-200 text-xs">(Shift+Drag to add more)</span>
        </div>
      )}

      {/* Drawing hint */}
      {isDrawing && (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 px-3 py-1.5 bg-slate-900/80 backdrop-blur-sm rounded-lg text-white text-xs">
          Release to select nodes • ESC to cancel
        </div>
      )}
    </div>
  )
}
