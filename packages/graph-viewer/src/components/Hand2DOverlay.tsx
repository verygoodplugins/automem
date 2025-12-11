/**
 * Hand 2D Overlay
 *
 * Renders hands as a 2D overlay on top of the canvas with:
 * - Ghost 3D hand effect (translucent, glowing)
 * - Smoothing/interpolation (ghost persists when hand disappears)
 * - Laser beams that default toward center with slight deviation
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
  // Palm connections
  [5, 9], [9, 13], [13, 17], [0, 17],
]

// Finger groups for palm fill
const PALM_OUTLINE = [0, 5, 9, 13, 17, 0] // Wrist and base of each finger

// Fingertip indices
const FINGERTIPS = [4, 8, 12, 16, 20]

// Knuckle indices (base of fingers)
const KNUCKLES = [5, 9, 13, 17]

// Smoothing configuration
const SMOOTHING_FACTOR = 0.2 // Lower = smoother but laggier
const GHOST_FADE_DURATION = 500 // ms to fade out ghost hand
const GHOST_PERSIST_DURATION = 300 // ms to keep ghost before fading

// Laser configuration
const LASER_CENTER_BIAS = 0.7 // How strongly laser aims at center (0 = follow hand, 1 = always center)
const LASER_DEVIATION_SCALE = 0.3 // How much hand position affects laser direction

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
        {/* Define filters and gradients */}
        <defs>
          {/* Ghost glow filter */}
          <filter id="ghost-glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="0.8" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>

          {/* Grip glow filter */}
          <filter id="grip-glow" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="1.5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>

          {/* Gradient for 3D depth effect - cyan */}
          <radialGradient id="hand-gradient-cyan" cx="50%" cy="30%" r="70%">
            <stop offset="0%" stopColor="#4ecdc4" stopOpacity="0.6" />
            <stop offset="100%" stopColor="#2a7a75" stopOpacity="0.2" />
          </radialGradient>

          {/* Gradient for 3D depth effect - magenta */}
          <radialGradient id="hand-gradient-magenta" cx="50%" cy="30%" r="70%">
            <stop offset="0%" stopColor="#f72585" stopOpacity="0.6" />
            <stop offset="100%" stopColor="#a01850" stopOpacity="0.2" />
          </radialGradient>
        </defs>

        {/* Left hand - cyan ghost */}
        {leftSmoothed && (
          <g opacity={leftSmoothed.opacity} filter="url(#ghost-glow)">
            <GhostHand
              landmarks={leftSmoothed.landmarks}
              color="#4ecdc4"
              gradientId="hand-gradient-cyan"
              isGhost={leftSmoothed.isGhost}
            />
          </g>
        )}

        {/* Right hand - magenta ghost */}
        {rightSmoothed && (
          <g opacity={rightSmoothed.opacity} filter="url(#ghost-glow)">
            <GhostHand
              landmarks={rightSmoothed.landmarks}
              color="#f72585"
              gradientId="hand-gradient-magenta"
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
          />
        )}

        {/* Right laser */}
        {showLaser && gestureState.rightPinchRay && gestureState.rightPinchRay.strength > 0.3 && (
          <LaserBeam
            ray={gestureState.rightPinchRay}
            color="#f72585"
            isGripped={rightGripping || false}
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

interface GhostHandProps {
  landmarks: { x: number; y: number; z: number }[]
  color: string
  gradientId: string
  isGhost?: boolean
}

function GhostHand({ landmarks, color, gradientId, isGhost = false }: GhostHandProps) {
  // Calculate scale based on hand depth (z of wrist)
  const wristZ = landmarks[0].z || 0
  const depthScale = 1 + wristZ * 5
  const clampedScale = Math.max(0.3, Math.min(1.5, depthScale))

  const toSvg = (lm: { x: number; y: number }) => ({
    x: lm.x * 100,
    y: lm.y * 100,
  })

  const points = landmarks.map(toSvg)

  // Base properties scaled with depth
  const strokeWidth = 0.25 * clampedScale
  const jointRadius = 0.35 * clampedScale
  const fingertipRadius = 0.5 * clampedScale
  const palmOpacity = isGhost ? 0.15 : 0.25
  const lineOpacity = isGhost ? 0.3 : 0.5

  // Create palm fill path
  const palmPath = PALM_OUTLINE.map((idx, i) => {
    const p = points[idx]
    return i === 0 ? `M ${p.x} ${p.y}` : `L ${p.x} ${p.y}`
  }).join(' ') + ' Z'

  // Create finger fill paths for 3D effect
  const createFingerPath = (base: number, _tip: number) => {
    const indices = [base, base + 1, base + 2, base + 3]
    const fingerPoints = indices.map(i => points[i])
    // Create a rounded path along the finger
    return `M ${fingerPoints[0].x} ${fingerPoints[0].y} ` +
           `Q ${fingerPoints[1].x} ${fingerPoints[1].y} ${fingerPoints[2].x} ${fingerPoints[2].y} ` +
           `Q ${fingerPoints[2].x} ${fingerPoints[2].y} ${fingerPoints[3].x} ${fingerPoints[3].y}`
  }

  return (
    <g>
      {/* Palm fill - gives 3D depth appearance */}
      <path
        d={palmPath}
        fill={`url(#${gradientId})`}
        opacity={palmOpacity}
      />

      {/* Finger strokes with gradient for 3D effect */}
      {[[1, 4], [5, 8], [9, 12], [13, 16], [17, 20]].map(([base], idx) => (
        <path
          key={`finger-fill-${idx}`}
          d={createFingerPath(base === 1 ? 1 : base, base === 1 ? 4 : base + 3)}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth * 3}
          strokeLinecap="round"
          opacity={0.15}
        />
      ))}

      {/* Skeleton lines - thin glowing */}
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
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            opacity={lineOpacity}
          />
        )
      })}

      {/* Knuckle highlights - larger for 3D effect */}
      {KNUCKLES.map((idx) => {
        const p = points[idx]
        return (
          <g key={`knuckle-${idx}`}>
            <circle
              cx={p.x}
              cy={p.y}
              r={jointRadius * 1.5}
              fill={color}
              opacity={0.2}
            />
            <circle
              cx={p.x}
              cy={p.y}
              r={jointRadius}
              fill={color}
              opacity={0.4}
            />
          </g>
        )
      })}

      {/* Joint dots */}
      {points.map((p, idx) => {
        const isFingertip = FINGERTIPS.includes(idx)
        const isKnuckle = KNUCKLES.includes(idx)
        if (isKnuckle) return null // Already rendered

        const radius = isFingertip ? fingertipRadius : jointRadius
        return (
          <circle
            key={`joint-${idx}`}
            cx={p.x}
            cy={p.y}
            r={radius}
            fill={color}
            opacity={isFingertip ? 0.7 : 0.4}
          />
        )
      })}

      {/* Fingertip glow - the main "ghost" effect */}
      {FINGERTIPS.map((idx) => {
        const p = points[idx]
        return (
          <g key={`fingertip-glow-${idx}`}>
            {/* Outer glow */}
            <circle
              cx={p.x}
              cy={p.y}
              r={fingertipRadius * 3}
              fill={color}
              opacity={0.1}
            />
            {/* Inner glow */}
            <circle
              cx={p.x}
              cy={p.y}
              r={fingertipRadius * 1.8}
              fill={color}
              opacity={0.2}
            />
            {/* Core */}
            <circle
              cx={p.x}
              cy={p.y}
              r={fingertipRadius}
              fill="#ffffff"
              opacity={0.5}
            />
          </g>
        )
      })}

      {/* Wrist indicator */}
      <circle
        cx={points[0].x}
        cy={points[0].y}
        r={jointRadius * 2}
        fill={color}
        opacity={0.3}
      />
    </g>
  )
}

interface LaserBeamProps {
  ray: PinchRay
  color: string
  isGripped: boolean
}

function LaserBeam({ ray, color, isGripped }: LaserBeamProps) {
  const originX = ray.origin.x * 100
  const originY = ray.origin.y * 100

  // Center of screen (the nexus)
  const centerX = 50
  const centerY = 50

  // Calculate direction: blend between center and hand-influenced direction
  // Hand position deviation from center
  const handDeviationX = (originX - 50) * LASER_DEVIATION_SCALE
  const handDeviationY = (originY - 50) * LASER_DEVIATION_SCALE

  // Target point: mostly center, slightly influenced by hand position
  const targetX = centerX - handDeviationX * (1 - LASER_CENTER_BIAS)
  const targetY = centerY - handDeviationY * (1 - LASER_CENTER_BIAS)

  // Direction toward target
  const toCenterX = targetX - originX
  const toCenterY = targetY - originY
  const dist = Math.sqrt(toCenterX * toCenterX + toCenterY * toCenterY)
  const normX = dist > 0 ? toCenterX / dist : 0
  const normY = dist > 0 ? toCenterY / dist : 0

  // Laser extends to target (the nexus area)
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
