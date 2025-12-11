/**
 * Hand 2D Overlay
 *
 * Renders hands as a 2D overlay on top of the canvas with:
 * - Smoothing/interpolation (ghost effect when hand disappears)
 * - Laser beams pointing toward the memory nexus center
 * - Pinch grip indicator (lights up when gripped)
 * - Support for two-hand manipulation
 */

import { useState, useEffect, useRef } from 'react'
import type { GestureState, PinchRay } from '../hooks/useHandGestures'

// Hand skeleton connections (pairs of landmark indices)
const HAND_CONNECTIONS = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index finger
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle finger
  [0, 9], [9, 10], [10, 11], [11, 12],
  // Ring finger
  [0, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [0, 17], [17, 18], [18, 19], [19, 20],
  // Palm
  [5, 9], [9, 13], [13, 17], [0, 17],
]

// Fingertip indices for larger dots
const FINGERTIPS = [4, 8, 12, 16, 20]

// Smoothing configuration
const SMOOTHING_FACTOR = 0.15 // Lower = smoother but laggier
const GHOST_FADE_DURATION = 500 // ms to fade out ghost hand
const GHOST_PERSIST_DURATION = 300 // ms to keep ghost before fading

interface SmoothedHand {
  landmarks: { x: number; y: number; z: number }[]
  lastSeen: number
  isGhost: boolean
  opacity: number
}

interface Hand2DOverlayProps {
  gestureState: GestureState
  enabled?: boolean
  showLaser?: boolean
}

export function Hand2DOverlay({ gestureState, enabled = true, showLaser = true }: Hand2DOverlayProps) {
  // Track smoothed hand positions with ghost effect
  const [leftSmoothed, setLeftSmoothed] = useState<SmoothedHand | null>(null)
  const [rightSmoothed, setRightSmoothed] = useState<SmoothedHand | null>(null)
  const animationRef = useRef<number>()

  // Smoothing and ghost effect
  useEffect(() => {
    if (!enabled) return

    const now = Date.now()

    // Process left hand
    if (gestureState.leftHand) {
      setLeftSmoothed(prev => {
        const newLandmarks = gestureState.leftHand!.landmarks.map((lm, i) => {
          const prevLm = prev?.landmarks[i]
          if (prevLm && !prev.isGhost) {
            // Interpolate toward new position
            return {
              x: prevLm.x + (lm.x - prevLm.x) * SMOOTHING_FACTOR,
              y: prevLm.y + (lm.y - prevLm.y) * SMOOTHING_FACTOR,
              z: prevLm.z + ((lm.z || 0) - prevLm.z) * SMOOTHING_FACTOR,
            }
          }
          return { x: lm.x, y: lm.y, z: lm.z || 0 }
        })
        return { landmarks: newLandmarks, lastSeen: now, isGhost: false, opacity: 1 }
      })
    } else if (leftSmoothed && !leftSmoothed.isGhost) {
      // Hand disappeared - start ghost mode
      setLeftSmoothed(prev => prev ? { ...prev, isGhost: true, lastSeen: now } : null)
    }

    // Process right hand
    if (gestureState.rightHand) {
      setRightSmoothed(prev => {
        const newLandmarks = gestureState.rightHand!.landmarks.map((lm, i) => {
          const prevLm = prev?.landmarks[i]
          if (prevLm && !prev.isGhost) {
            return {
              x: prevLm.x + (lm.x - prevLm.x) * SMOOTHING_FACTOR,
              y: prevLm.y + (lm.y - prevLm.y) * SMOOTHING_FACTOR,
              z: prevLm.z + ((lm.z || 0) - prevLm.z) * SMOOTHING_FACTOR,
            }
          }
          return { x: lm.x, y: lm.y, z: lm.z || 0 }
        })
        return { landmarks: newLandmarks, lastSeen: now, isGhost: false, opacity: 1 }
      })
    } else if (rightSmoothed && !rightSmoothed.isGhost) {
      setRightSmoothed(prev => prev ? { ...prev, isGhost: true, lastSeen: now } : null)
    }
  }, [gestureState, enabled])

  // Ghost fade animation
  useEffect(() => {
    const animate = () => {
      const now = Date.now()

      // Fade left ghost
      if (leftSmoothed?.isGhost) {
        const elapsed = now - leftSmoothed.lastSeen
        if (elapsed > GHOST_PERSIST_DURATION) {
          const fadeProgress = (elapsed - GHOST_PERSIST_DURATION) / GHOST_FADE_DURATION
          if (fadeProgress >= 1) {
            setLeftSmoothed(null)
          } else {
            setLeftSmoothed(prev => prev ? { ...prev, opacity: 1 - fadeProgress } : null)
          }
        }
      }

      // Fade right ghost
      if (rightSmoothed?.isGhost) {
        const elapsed = now - rightSmoothed.lastSeen
        if (elapsed > GHOST_PERSIST_DURATION) {
          const fadeProgress = (elapsed - GHOST_PERSIST_DURATION) / GHOST_FADE_DURATION
          if (fadeProgress >= 1) {
            setRightSmoothed(null)
          } else {
            setRightSmoothed(prev => prev ? { ...prev, opacity: 1 - fadeProgress } : null)
          }
        }
      }

      animationRef.current = requestAnimationFrame(animate)
    }

    animationRef.current = requestAnimationFrame(animate)
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
    }
  }, [leftSmoothed?.isGhost, rightSmoothed?.isGhost])

  if (!enabled || !gestureState.isTracking) return null

  // Check if both hands are gripping (for two-hand manipulation)
  const leftGripping = gestureState.leftPinchRay?.isValid
  const rightGripping = gestureState.rightPinchRay?.isValid
  const bothGripping = leftGripping && rightGripping

  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden">
      <svg
        className="w-full h-full"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
      >
        {/* Define glow filter for grip effect */}
        <defs>
          <filter id="glow-cyan" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="1" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <filter id="glow-magenta" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="1" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <filter id="grip-glow" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="2" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Left hand - cyan */}
        {leftSmoothed && (
          <g opacity={leftSmoothed.opacity} filter={leftSmoothed.isGhost ? 'url(#glow-cyan)' : undefined}>
            <HandSkeleton
              landmarks={leftSmoothed.landmarks}
              color="#4ecdc4"
              isGhost={leftSmoothed.isGhost}
            />
          </g>
        )}

        {/* Right hand - magenta */}
        {rightSmoothed && (
          <g opacity={rightSmoothed.opacity} filter={rightSmoothed.isGhost ? 'url(#glow-magenta)' : undefined}>
            <HandSkeleton
              landmarks={rightSmoothed.landmarks}
              color="#f72585"
              isGhost={rightSmoothed.isGhost}
            />
          </g>
        )}

        {/* Connection line between hands when both gripping */}
        {bothGripping && gestureState.leftPinchRay && gestureState.rightPinchRay && (
          <line
            x1={gestureState.leftPinchRay.origin.x * 100}
            y1={gestureState.leftPinchRay.origin.y * 100}
            x2={gestureState.rightPinchRay.origin.x * 100}
            y2={gestureState.rightPinchRay.origin.y * 100}
            stroke="#ffffff"
            strokeWidth={0.3}
            strokeDasharray="2 1"
            opacity={0.5}
          />
        )}

        {/* Left laser */}
        {showLaser && gestureState.leftPinchRay && gestureState.leftPinchRay.strength > 0.3 && (
          <LaserBeam
            ray={gestureState.leftPinchRay}
            color="#4ecdc4"
            isGripped={leftGripping || false}
            otherRay={bothGripping ? gestureState.rightPinchRay : undefined}
          />
        )}

        {/* Right laser */}
        {showLaser && gestureState.rightPinchRay && gestureState.rightPinchRay.strength > 0.3 && (
          <LaserBeam
            ray={gestureState.rightPinchRay}
            color="#f72585"
            isGripped={rightGripping || false}
            otherRay={bothGripping ? gestureState.leftPinchRay : undefined}
          />
        )}

        {/* Center nexus indicator when gripping */}
        {(leftGripping || rightGripping) && (
          <g>
            <circle cx={50} cy={50} r={3} fill="none" stroke="#ffffff" strokeWidth={0.2} opacity={0.3} />
            <circle cx={50} cy={50} r={1.5} fill="#ffffff" opacity={0.2} />
          </g>
        )}
      </svg>
    </div>
  )
}

interface HandSkeletonProps {
  landmarks: { x: number; y: number; z: number }[]
  color: string
  isGhost?: boolean
}

function HandSkeleton({ landmarks, color, isGhost = false }: HandSkeletonProps) {
  // Calculate scale based on hand depth (z of wrist)
  const wristZ = landmarks[0].z || 0
  const depthScale = 1 + wristZ * 5
  const clampedScale = Math.max(0.3, Math.min(1.5, depthScale))

  // Base stroke width that scales with depth
  const baseStroke = 0.3 * clampedScale
  const jointRadius = 0.4 * clampedScale
  const fingertipRadius = 0.6 * clampedScale

  // Ghost hands are more transparent and have a blur effect
  const baseOpacity = isGhost ? 0.4 : 0.8

  const toSvg = (lm: { x: number; y: number }) => ({
    x: lm.x * 100,
    y: lm.y * 100,
  })

  const points = landmarks.map(toSvg)

  return (
    <g>
      {/* Connection lines */}
      {HAND_CONNECTIONS.map(([i, j], idx) => {
        const p1 = points[i]
        const p2 = points[j]
        return (
          <line
            key={`line-${idx}`}
            x1={p1.x}
            y1={p1.y}
            x2={p2.x}
            y2={p2.y}
            stroke={color}
            strokeWidth={baseStroke}
            strokeLinecap="round"
            opacity={baseOpacity}
          />
        )
      })}

      {/* Joint dots */}
      {points.map((p, idx) => {
        const isFingertip = FINGERTIPS.includes(idx)
        const radius = isFingertip ? fingertipRadius : jointRadius
        return (
          <circle
            key={`joint-${idx}`}
            cx={p.x}
            cy={p.y}
            r={radius}
            fill={color}
            opacity={isFingertip ? baseOpacity + 0.2 : baseOpacity - 0.1}
          />
        )
      })}

      {/* Glow effect for fingertips */}
      {!isGhost && FINGERTIPS.map((idx) => {
        const p = points[idx]
        return (
          <circle
            key={`glow-${idx}`}
            cx={p.x}
            cy={p.y}
            r={fingertipRadius * 2}
            fill={color}
            opacity={0.15}
          />
        )
      })}
    </g>
  )
}

interface LaserBeamProps {
  ray: PinchRay
  color: string
  isGripped: boolean
  otherRay?: PinchRay | null
}

function LaserBeam({ ray, color, isGripped }: LaserBeamProps) {
  const originX = ray.origin.x * 100
  const originY = ray.origin.y * 100

  // Laser always points toward center of screen (the nexus)
  const centerX = 50
  const centerY = 50

  // Direction toward center
  const toCenterX = centerX - originX
  const toCenterY = centerY - originY
  const dist = Math.sqrt(toCenterX * toCenterX + toCenterY * toCenterY)
  const normX = dist > 0 ? toCenterX / dist : 0
  const normY = dist > 0 ? toCenterY / dist : 0

  // Laser extends to center, not beyond (it "hits" the nexus)
  const laserLength = dist
  const endX = originX + normX * laserLength
  const endY = originY + normY * laserLength

  // Visual properties - much more intense when gripped
  const baseStrokeWidth = 0.2 + ray.strength * 0.3
  const strokeWidth = isGripped ? baseStrokeWidth * 2.5 : baseStrokeWidth
  const baseOpacity = 0.3 + ray.strength * 0.4
  const opacity = isGripped ? Math.min(1, baseOpacity * 1.8) : baseOpacity
  const glowRadius = isGripped ? 2 + ray.strength * 2 : 0.8 + ray.strength * 0.8

  // Grip indicator color - brighter white-ish when gripped
  const gripColor = isGripped ? '#ffffff' : color

  return (
    <g filter={isGripped ? 'url(#grip-glow)' : undefined}>
      {/* Outer glow when gripped */}
      {isGripped && (
        <line
          x1={originX}
          y1={originY}
          x2={endX}
          y2={endY}
          stroke={color}
          strokeWidth={strokeWidth * 4}
          strokeLinecap="round"
          opacity={0.2}
        />
      )}

      {/* Laser glow (wider, more transparent) */}
      <line
        x1={originX}
        y1={originY}
        x2={endX}
        y2={endY}
        stroke={color}
        strokeWidth={strokeWidth * 2.5}
        strokeLinecap="round"
        opacity={opacity * 0.3}
      />

      {/* Main laser beam */}
      <line
        x1={originX}
        y1={originY}
        x2={endX}
        y2={endY}
        stroke={gripColor}
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        opacity={opacity}
      />

      {/* Origin glow sphere */}
      <circle
        cx={originX}
        cy={originY}
        r={glowRadius * 1.5}
        fill={color}
        opacity={0.2}
      />
      <circle
        cx={originX}
        cy={originY}
        r={glowRadius}
        fill={gripColor}
        opacity={isGripped ? 1 : 0.7}
      />

      {/* Impact point at center - "warm spot" where laser hits the nexus */}
      <circle
        cx={endX}
        cy={endY}
        r={isGripped ? 2.5 : 1.5}
        fill={color}
        opacity={isGripped ? 0.6 : 0.3}
      />
      {isGripped && (
        <>
          <circle
            cx={endX}
            cy={endY}
            r={4}
            fill={color}
            opacity={0.2}
          />
          <circle
            cx={endX}
            cy={endY}
            r={1}
            fill="#ffffff"
            opacity={0.8}
          />
        </>
      )}

      {/* Pulsing ring at origin when gripped */}
      {isGripped && (
        <circle
          cx={originX}
          cy={originY}
          r={glowRadius * 2}
          fill="none"
          stroke={gripColor}
          strokeWidth={0.3}
          opacity={0.7}
          className="animate-ping"
        />
      )}
    </g>
  )
}

export default Hand2DOverlay
