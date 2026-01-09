/**
 * Hand Cursor Component
 *
 * A screen-space cursor controlled by hand position with:
 * - Visual feedback (pulse, glow, color warmth)
 * - Smooth interpolation
 * - Pinch indicator
 *
 * Works with the existing selection logic in GraphCanvas - this just
 * provides visual feedback for where the hand is pointing.
 */

import { useState, useEffect, useRef, useCallback } from 'react'
import type { GestureState } from '../hooks/useHandGestures'
import type { HandLockState } from '../hooks/useHandLockAndGrab'

// Cursor configuration
const SMOOTHING = 0.12 // Lower = smoother but laggier
const PINCH_THRESHOLD = 0.6 // Visual feedback threshold

interface HandCursorProps {
  gestureState: GestureState
  lock: HandLockState
  enabled: boolean
  hasTarget: boolean // Is there a node being hovered (from GraphCanvas)
  containerRef?: React.RefObject<HTMLElement>
  onPinchStart?: () => void
  onPinchEnd?: () => void
}

interface CursorState {
  // Smoothed position (0-1 normalized)
  smoothX: number
  smoothY: number
  // Screen position (pixels)
  screenX: number
  screenY: number
  // Visual states
  isVisible: boolean
  pinchStrength: number
}

export function HandCursor({
  gestureState,
  lock,
  enabled,
  hasTarget,
  containerRef,
  onPinchStart,
  onPinchEnd,
}: HandCursorProps) {
  const [cursor, setCursor] = useState<CursorState>({
    smoothX: 0.5,
    smoothY: 0.5,
    screenX: 0,
    screenY: 0,
    isVisible: false,
    pinchStrength: 0,
  })

  const prevPinchingRef = useRef(false)
  const animationRef = useRef<number>()

  // Get container dimensions
  const getContainerSize = useCallback(() => {
    if (containerRef?.current) {
      const rect = containerRef.current.getBoundingClientRect()
      return { width: rect.width, height: rect.height }
    }
    return { width: window.innerWidth, height: window.innerHeight }
  }, [containerRef])

  // Main update loop
  useEffect(() => {
    if (!enabled) {
      setCursor(prev => ({ ...prev, isVisible: false }))
      return
    }

    const updateCursor = () => {
      const { width, height } = getContainerSize()

      setCursor(prev => {
        // Get active hand based on lock state
        const activeHand = lock.mode !== 'idle'
          ? (lock.hand === 'left' ? gestureState.leftHand : gestureState.rightHand)
          : (gestureState.rightHand || gestureState.leftHand)

        if (!activeHand || !gestureState.isTracking) {
          return { ...prev, isVisible: false }
        }

        // Only show cursor when hand is locked (acquired)
        if (lock.mode !== 'locked') {
          return { ...prev, isVisible: false }
        }

        // Calculate pinch point (midpoint of thumb and index tips)
        const thumbTip = activeHand.landmarks[4]
        const indexTip = activeHand.landmarks[8]
        const rawX = (thumbTip.x + indexTip.x) / 2
        const rawY = (thumbTip.y + indexTip.y) / 2

        // Smooth the position
        const smoothX = prev.smoothX + (rawX - prev.smoothX) * (1 - SMOOTHING)
        const smoothY = prev.smoothY + (rawY - prev.smoothY) * (1 - SMOOTHING)

        // Convert to screen coordinates (NOT mirrored - raw coords)
        // The mirroring happens in the landmark processing
        const screenX = smoothX * width
        const screenY = smoothY * height

        // Calculate pinch strength
        const pinchDist = Math.sqrt(
          Math.pow(thumbTip.x - indexTip.x, 2) +
          Math.pow(thumbTip.y - indexTip.y, 2)
        )
        const pinchStrength = Math.max(0, Math.min(1, 1 - (pinchDist - 0.02) / 0.13))

        return {
          smoothX,
          smoothY,
          screenX,
          screenY,
          isVisible: true,
          pinchStrength,
        }
      })

      animationRef.current = requestAnimationFrame(updateCursor)
    }

    animationRef.current = requestAnimationFrame(updateCursor)

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [enabled, gestureState, lock, getContainerSize])

  // Handle pinch callbacks
  useEffect(() => {
    if (!enabled) return

    const isPinching = cursor.pinchStrength > PINCH_THRESHOLD
    const wasPinching = prevPinchingRef.current

    if (isPinching && !wasPinching) {
      onPinchStart?.()
    } else if (!isPinching && wasPinching) {
      onPinchEnd?.()
    }

    prevPinchingRef.current = isPinching
  }, [cursor.pinchStrength, enabled, onPinchStart, onPinchEnd])

  if (!enabled || !cursor.isVisible) return null

  const isPinching = cursor.pinchStrength > PINCH_THRESHOLD

  // Color: cool (no target) -> warm (has target) -> hot (pinching)
  const baseHue = hasTarget ? 35 : 200 // cyan -> orange
  const saturation = hasTarget ? 90 : 70
  const lightness = isPinching ? 65 : (hasTarget ? 55 : 50)
  const cursorColor = `hsl(${baseHue}, ${saturation}%, ${lightness}%)`

  // Size based on pinch strength
  const baseSize = 20
  const size = baseSize + cursor.pinchStrength * 16

  // Glow intensity
  const glowIntensity = hasTarget ? 0.6 : 0.3

  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden z-40">
      <div
        className="absolute transform -translate-x-1/2 -translate-y-1/2"
        style={{
          left: cursor.screenX,
          top: cursor.screenY,
          transition: 'left 0.05s ease-out, top 0.05s ease-out',
        }}
      >
        {/* Outer glow */}
        <div
          className="absolute rounded-full transform -translate-x-1/2 -translate-y-1/2"
          style={{
            width: size * 3,
            height: size * 3,
            left: '50%',
            top: '50%',
            background: `radial-gradient(circle, ${cursorColor}${Math.round(glowIntensity * 50).toString(16).padStart(2, '0')} 0%, transparent 70%)`,
            filter: `blur(8px)`,
          }}
        />

        {/* Target acquired ring */}
        {hasTarget && (
          <div
            className="absolute rounded-full border-2 transform -translate-x-1/2 -translate-y-1/2"
            style={{
              width: size * 2.2,
              height: size * 2.2,
              left: '50%',
              top: '50%',
              borderColor: cursorColor,
              opacity: 0.6,
              animation: 'pulse 1s ease-in-out infinite',
            }}
          />
        )}

        {/* Pinch progress ring */}
        {cursor.pinchStrength > 0.2 && (
          <svg
            className="absolute transform -translate-x-1/2 -translate-y-1/2"
            style={{
              width: size * 2,
              height: size * 2,
              left: '50%',
              top: '50%',
            }}
            viewBox="0 0 100 100"
          >
            <circle
              cx="50"
              cy="50"
              r="42"
              fill="none"
              stroke={cursorColor}
              strokeWidth="4"
              strokeLinecap="round"
              strokeDasharray={`${cursor.pinchStrength * 264} 264`}
              transform="rotate(-90 50 50)"
              opacity={0.7}
            />
          </svg>
        )}

        {/* Inner cursor */}
        <div
          className="absolute rounded-full transform -translate-x-1/2 -translate-y-1/2"
          style={{
            width: size,
            height: size,
            left: '50%',
            top: '50%',
            background: `radial-gradient(circle at 30% 30%, white 0%, ${cursorColor} 100%)`,
            boxShadow: `0 0 ${hasTarget ? 20 : 10}px ${cursorColor}`,
            transform: `scale(${isPinching ? 0.75 : 1})`,
            transition: 'transform 0.1s ease-out, box-shadow 0.15s ease-out',
          }}
        />

        {/* Crosshair when pinching */}
        {isPinching && (
          <>
            <div
              className="absolute bg-white/80"
              style={{ width: 2, height: 12, left: '50%', top: -size/2 - 14, transform: 'translateX(-50%)' }}
            />
            <div
              className="absolute bg-white/80"
              style={{ width: 2, height: 12, left: '50%', top: size/2 + 2, transform: 'translateX(-50%)' }}
            />
            <div
              className="absolute bg-white/80"
              style={{ width: 12, height: 2, left: -size/2 - 14, top: '50%', transform: 'translateY(-50%)' }}
            />
            <div
              className="absolute bg-white/80"
              style={{ width: 12, height: 2, left: size/2 + 2, top: '50%', transform: 'translateY(-50%)' }}
            />
          </>
        )}
      </div>

      {/* Pulse animation keyframes */}
      <style>{`
        @keyframes pulse {
          0%, 100% { transform: translate(-50%, -50%) scale(1); opacity: 0.6; }
          50% { transform: translate(-50%, -50%) scale(1.1); opacity: 0.3; }
        }
      `}</style>
    </div>
  )
}

export default HandCursor
