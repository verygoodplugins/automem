/**
 * Hand 2D Overlay
 *
 * Renders hands as a 2D SVG overlay with:
 * - Ghost 3D hand effect (translucent, glowing Master Hand style)
 * - Smoothing/interpolation (ghost persists briefly when hand disappears)
 * - Depth-aware scaling ("reach through screen" paradigm)
 *
 * SIMPLIFIED: No lasers, no center target, just visual hand feedback.
 */

import { useState, useEffect, useRef } from 'react'
import type { GestureState } from '../hooks/useHandGestures'
import type { HandLockState } from '../hooks/useHandLockAndGrab'

// Fingertip indices
const FINGERTIPS = [4, 8, 12, 16, 20]

// Knuckle indices (base of fingers)
const KNUCKLES = [5, 9, 13, 17]

// Smoothing configuration
const SMOOTHING_FACTOR = 0.2 // Lower = smoother but laggier
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
  lock?: HandLockState
  leftLock?: { mode: 'idle' } | { mode: 'candidate'; frames: number } | { mode: 'locked' }
  rightLock?: { mode: 'idle' } | { mode: 'candidate'; frames: number } | { mode: 'locked' }
}

export function Hand2DOverlay({
  gestureState,
  enabled = true,
  lock: _lock, // Legacy single-hand lock (deprecated, use leftLock/rightLock)
  leftLock,
  rightLock,
}: Hand2DOverlayProps) {
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
        const hand = gestureState.leftHand!
        const newLandmarks = hand.landmarks.map((lm, i) => {
          // Prefer world Z (meters for iPhone LiDAR) when available; fall back to normalized z.
          const zTarget = (hand.worldLandmarks?.[i]?.z ?? (lm.z || 0)) as number
          const prevLm = prev?.landmarks[i]
          if (prevLm && !prev.isGhost) {
            // Interpolate toward new position
            return {
              x: prevLm.x + (lm.x - prevLm.x) * SMOOTHING_FACTOR,
              y: prevLm.y + (lm.y - prevLm.y) * SMOOTHING_FACTOR,
              z: prevLm.z + (zTarget - prevLm.z) * SMOOTHING_FACTOR,
            }
          }
          return { x: lm.x, y: lm.y, z: zTarget }
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
        const hand = gestureState.rightHand!
        const newLandmarks = hand.landmarks.map((lm, i) => {
          // Prefer world Z (meters for iPhone LiDAR) when available; fall back to normalized z.
          const zTarget = (hand.worldLandmarks?.[i]?.z ?? (lm.z || 0)) as number
          const prevLm = prev?.landmarks[i]
          if (prevLm && !prev.isGhost) {
            return {
              x: prevLm.x + (lm.x - prevLm.x) * SMOOTHING_FACTOR,
              y: prevLm.y + (lm.y - prevLm.y) * SMOOTHING_FACTOR,
              z: prevLm.z + (zTarget - prevLm.z) * SMOOTHING_FACTOR,
            }
          }
          return { x: lm.x, y: lm.y, z: zTarget }
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

  // Per-hand visibility: each hand has its own lock state
  // - Locked = bright (0.85)
  // - Candidate = medium (0.25)
  // - Idle = faint (0.06)
  const getHandOpacity = (handLock?: { mode: string }) => {
    const mode = handLock?.mode ?? 'idle'
    return mode === 'locked' ? 0.85 :
           mode === 'candidate' ? 0.25 :
           0.06
  }

  const leftOpacity = getHandOpacity(leftLock)
  const rightOpacity = getHandOpacity(rightLock)

  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden">
      <svg
        className="w-full h-full"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
      >

        {/* Left hand - cyan lines */}
        {leftSmoothed && (
          <g opacity={leftSmoothed.opacity}>
            <LineHand
              landmarks={leftSmoothed.landmarks}
              color="#4ecdc4"
              isGhost={leftSmoothed.isGhost}
              opacityMultiplier={leftOpacity}
            />
          </g>
        )}

        {/* Right hand - magenta lines */}
        {rightSmoothed && (
          <g opacity={rightSmoothed.opacity}>
            <LineHand
              landmarks={rightSmoothed.landmarks}
              color="#f72585"
              isGhost={rightSmoothed.isGhost}
              opacityMultiplier={rightOpacity}
            />
          </g>
        )}
      </svg>
    </div>
  )
}

interface LineHandProps {
  landmarks: { x: number; y: number; z: number }[]
  color: string
  isGhost?: boolean
  opacityMultiplier?: number
}

/**
 * LineHand - Minimal line-based hand visualization
 * Just lines connecting joints with small dots at key points.
 * Clean, intuitive, and non-distracting.
 */
function LineHand({
  landmarks,
  color,
  isGhost = false,
  opacityMultiplier = 1,
}: LineHandProps) {
  const wristZ = landmarks[0].z || 0

  // Detect if Z is in meters (LiDAR) vs normalized (MediaPipe-style)
  const isMeters = wristZ >= 0.1 && wristZ <= 8.0

  // Depth-based scaling (reach through portal paradigm)
  let scaleFactor = 1.0
  let depthOpacity = 1.0

  if (isMeters) {
    const t = Math.max(0, Math.min(1, (wristZ - 0.25) / 0.85))
    scaleFactor = 0.4 + t * 1.2
    depthOpacity = 0.6 + t * 0.4
  } else {
    const t = Math.max(0, Math.min(1, (wristZ + 0.25) / 0.35))
    scaleFactor = 0.4 + t * 1.2
    depthOpacity = 0.6 + t * 0.4
  }

  const effectiveOpacity = opacityMultiplier * depthOpacity * (isGhost ? 0.5 : 0.9)
  const clampedScale = Math.max(0.4, Math.min(1.8, scaleFactor))

  // Convert landmarks to SVG coordinates
  const toSvg = (lm: { x: number; y: number }) => ({
    x: lm.x * 100,
    y: lm.y * 100,
  })

  const rawPoints = landmarks.map(toSvg)
  const cx = rawPoints[0]?.x ?? 50
  const cy = rawPoints[0]?.y ?? 50

  // Apply depth warp per-landmark
  const warpGain = isMeters ? 2.2 : 6.0
  const clampRange = (v: number, min: number, max: number) => Math.max(min, Math.min(max, v))

  const points = rawPoints.map((p, i) => {
    const zi = landmarks[i]?.z ?? wristZ
    const dz = zi - wristZ
    const pointScale = clampRange(1 + dz * warpGain, 0.65, 1.45)
    const s = clampedScale * pointScale
    return {
      x: cx + (p.x - cx) * s,
      y: cy + (p.y - cy) * s,
    }
  })

  // Line thickness based on scale
  const lineWidth = 0.25 * clampedScale
  const jointRadius = 0.4 * clampedScale
  const tipRadius = 0.5 * clampedScale

  // Finger bone connections: each finger is a chain of joints
  const fingerChains = [
    [0, 1, 2, 3, 4],       // Thumb: wrist -> CMC -> MCP -> IP -> tip
    [0, 5, 6, 7, 8],       // Index: wrist -> MCP -> PIP -> DIP -> tip
    [0, 9, 10, 11, 12],    // Middle
    [0, 13, 14, 15, 16],   // Ring
    [0, 17, 18, 19, 20],   // Pinky
  ]

  // Palm connections (connect finger bases)
  const palmConnections = [
    [5, 9], [9, 13], [13, 17], // Across top of palm
    [1, 5],                    // Thumb to index
  ]

  return (
    <g opacity={effectiveOpacity}>
      {/* Glow filter for the lines */}
      <defs>
        <filter id={`line-glow-${color}`} x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="0.3" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      <g filter={`url(#line-glow-${color})`}>
        {/* Finger bones */}
        {fingerChains.map((chain, fingerIdx) => (
          <g key={`finger-${fingerIdx}`}>
            {chain.slice(1).map((jointIdx, i) => {
              const prevIdx = chain[i]
              const p1 = points[prevIdx]
              const p2 = points[jointIdx]
              return (
                <line
                  key={`bone-${fingerIdx}-${i}`}
                  x1={p1.x}
                  y1={p1.y}
                  x2={p2.x}
                  y2={p2.y}
                  stroke={color}
                  strokeWidth={lineWidth}
                  strokeLinecap="round"
                />
              )
            })}
          </g>
        ))}

        {/* Palm connections */}
        {palmConnections.map(([i1, i2], idx) => {
          const p1 = points[i1]
          const p2 = points[i2]
          return (
            <line
              key={`palm-${idx}`}
              x1={p1.x}
              y1={p1.y}
              x2={p2.x}
              y2={p2.y}
              stroke={color}
              strokeWidth={lineWidth * 0.8}
              strokeLinecap="round"
              strokeOpacity={0.6}
            />
          )
        })}

        {/* Joint dots (knuckles) */}
        {KNUCKLES.map((idx) => {
          const p = points[idx]
          return (
            <circle
              key={`joint-${idx}`}
              cx={p.x}
              cy={p.y}
              r={jointRadius}
              fill={color}
              fillOpacity={0.7}
            />
          )
        })}

        {/* Fingertip dots (brighter) */}
        {FINGERTIPS.map((idx) => {
          const p = points[idx]
          return (
            <circle
              key={`tip-${idx}`}
              cx={p.x}
              cy={p.y}
              r={tipRadius}
              fill={color}
            />
          )
        })}

        {/* Wrist dot */}
        <circle
          cx={points[0].x}
          cy={points[0].y}
          r={jointRadius * 1.2}
          fill={color}
          fillOpacity={0.5}
        />
      </g>
    </g>
  )
}

export default Hand2DOverlay
