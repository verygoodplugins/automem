/**
 * Hand 2D Overlay
 *
 * Renders hands as a 2D overlay on top of the canvas with:
 * - Ghost 3D hand effect (translucent, glowing)
 * - Smoothing/interpolation (ghost persists when hand disappears)
 * - Accurate laser beams using stable pointer ray with arm model
 * - Hit indicator when pointing at a node
 * - Pinch grip indicator (lights up when gripped)
 * - Support for two-hand manipulation
 */

import { useState, useEffect, useRef } from 'react'
import type { GestureState, PinchRay } from '../hooks/useHandGestures'
import type { StableRay, NodeHit } from '../hooks/useStablePointerRay'

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
  /** Stable left ray from arm model + One Euro Filter */
  leftStableRay?: StableRay | null
  /** Stable right ray from arm model + One Euro Filter */
  rightStableRay?: StableRay | null
  /** Current node hit (if any) */
  hoveredNode?: NodeHit | null
}

export function Hand2DOverlay({
  gestureState,
  enabled = true,
  showLaser = true,
  leftStableRay,
  rightStableRay,
  hoveredNode,
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
            x1={(1 - gestureState.leftPinchRay.origin.x) * 100}
            y1={gestureState.leftPinchRay.origin.y * 100}
            x2={(1 - gestureState.rightPinchRay.origin.x) * 100}
            y2={gestureState.rightPinchRay.origin.y * 100}
            stroke="#ffffff"
            strokeWidth={0.3}
            strokeDasharray="2 1"
            opacity={0.5}
          />
        )}

        {/* Left laser - use stable ray if available, otherwise fall back to basic ray */}
        {showLaser && gestureState.leftPinchRay && gestureState.leftPinchRay.strength > 0.3 && (
          <LaserBeam
            ray={gestureState.leftPinchRay}
            stableRay={leftStableRay}
            color="#4ecdc4"
            isGripped={leftGripping || false}
            hasHit={hoveredNode !== null && leftStableRay !== null &&
              (leftStableRay?.confidence ?? 0) >= (rightStableRay?.confidence ?? 0)}
          />
        )}

        {/* Right laser - use stable ray if available */}
        {showLaser && gestureState.rightPinchRay && gestureState.rightPinchRay.strength > 0.3 && (
          <LaserBeam
            ray={gestureState.rightPinchRay}
            stableRay={rightStableRay}
            color="#f72585"
            isGripped={rightGripping || false}
            hasHit={hoveredNode !== null && rightStableRay !== null &&
              (rightStableRay?.confidence ?? 0) > (leftStableRay?.confidence ?? 0)}
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

/**
 * MasterHand - Smash Bros Master Hand / Crazy Hand style
 * Volumetric filled shapes with soft gradients and ambient occlusion
 */
function GhostHand({ landmarks, color: _color, gradientId: _gradientId, isGhost = false }: GhostHandProps) {
  // INVERTED depth scaling: hand closer to camera (negative Z) = smaller (farther in 3D space)
  const wristZ = landmarks[0].z || 0
  const depthScale = 1 - wristZ * 4
  const clampedScale = Math.max(0.5, Math.min(2.0, depthScale))

  // Un-mirror the X coordinate
  const toSvg = (lm: { x: number; y: number }) => ({
    x: (1 - lm.x) * 100,
    y: lm.y * 100,
  })

  const points = landmarks.map(toSvg)
  const gloveOpacity = isGhost ? 0.5 : 0.85

  // Finger width based on scale - fatter fingers for Master Hand look
  const fingerWidth = 1.8 * clampedScale

  // Unique ID for this hand's gradients/filters
  const handId = Math.round(points[0].x * 10)

  // Helper: get perpendicular offset for finger width
  const getPerpendicular = (p1: {x: number, y: number}, p2: {x: number, y: number}, width: number) => {
    const dx = p2.x - p1.x
    const dy = p2.y - p1.y
    const len = Math.sqrt(dx * dx + dy * dy) || 1
    return { x: -dy / len * width, y: dx / len * width }
  }

  // Create filled finger shape (capsule/sausage shape)
  const createFingerShape = (indices: number[], width: number) => {
    const pts = indices.map(i => points[i])
    if (pts.length < 2) return ''

    // Build outline going down one side and back up the other
    const leftSide: string[] = []
    const rightSide: string[] = []

    for (let i = 0; i < pts.length - 1; i++) {
      const perp = getPerpendicular(pts[i], pts[i + 1], width)
      leftSide.push(`${pts[i].x + perp.x},${pts[i].y + perp.y}`)
      rightSide.unshift(`${pts[i].x - perp.x},${pts[i].y - perp.y}`)
    }

    // Add rounded tip
    const lastPt = pts[pts.length - 1]
    const prevPt = pts[pts.length - 2]
    const tipPerp = getPerpendicular(prevPt, lastPt, width)

    // Rounded end cap using arc
    const tipLeft = `${lastPt.x + tipPerp.x},${lastPt.y + tipPerp.y}`
    const tipRight = `${lastPt.x - tipPerp.x},${lastPt.y - tipPerp.y}`

    return `M ${leftSide[0]} L ${leftSide.join(' L ')} L ${tipLeft} A ${width} ${width} 0 0 1 ${tipRight} L ${rightSide.join(' L ')} Z`
  }

  // Create palm shape connecting all finger bases
  const createPalmShape = () => {
    // Palm outline: wrist -> thumb base -> around finger bases -> back to wrist
    const wrist = points[0]
    const thumbBase = points[1]
    const indexBase = points[5]
    const middleBase = points[9]
    const ringBase = points[13]
    const pinkyBase = points[17]

    // Offset points outward for palm width
    const palmWidth = fingerWidth * 1.2

    return `
      M ${wrist.x} ${wrist.y + palmWidth}
      Q ${thumbBase.x - palmWidth} ${thumbBase.y} ${thumbBase.x} ${thumbBase.y - palmWidth * 0.5}
      L ${indexBase.x - palmWidth * 0.3} ${indexBase.y - palmWidth * 0.5}
      Q ${(indexBase.x + middleBase.x) / 2} ${Math.min(indexBase.y, middleBase.y) - palmWidth * 0.8}
        ${middleBase.x} ${middleBase.y - palmWidth * 0.5}
      Q ${(middleBase.x + ringBase.x) / 2} ${Math.min(middleBase.y, ringBase.y) - palmWidth * 0.6}
        ${ringBase.x} ${ringBase.y - palmWidth * 0.5}
      Q ${(ringBase.x + pinkyBase.x) / 2} ${Math.min(ringBase.y, pinkyBase.y) - palmWidth * 0.5}
        ${pinkyBase.x + palmWidth * 0.3} ${pinkyBase.y - palmWidth * 0.3}
      L ${pinkyBase.x + palmWidth * 0.5} ${pinkyBase.y + palmWidth * 0.5}
      Q ${wrist.x + palmWidth * 1.5} ${(wrist.y + pinkyBase.y) / 2}
        ${wrist.x} ${wrist.y + palmWidth}
      Z
    `
  }

  // Finger definitions: [landmark indices]
  const fingers = [
    { indices: [1, 2, 3, 4], width: fingerWidth * 0.9 },      // Thumb (slightly thinner)
    { indices: [5, 6, 7, 8], width: fingerWidth },            // Index
    { indices: [9, 10, 11, 12], width: fingerWidth * 1.05 },  // Middle (slightly thicker)
    { indices: [13, 14, 15, 16], width: fingerWidth * 0.95 }, // Ring
    { indices: [17, 18, 19, 20], width: fingerWidth * 0.85 }, // Pinky (thinnest)
  ]

  return (
    <g opacity={gloveOpacity}>
      {/* Definitions */}
      <defs>
        {/* Main hand gradient - white to soft lavender */}
        <radialGradient id={`hand-fill-${handId}`} cx="30%" cy="25%" r="80%">
          <stop offset="0%" stopColor="#ffffff" />
          <stop offset="40%" stopColor="#f5f5fa" />
          <stop offset="70%" stopColor="#e8e8f0" />
          <stop offset="100%" stopColor="#d8d8e5" />
        </radialGradient>

        {/* Ambient occlusion gradient for creases */}
        <radialGradient id={`ao-gradient-${handId}`} cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#000000" stopOpacity="0" />
          <stop offset="70%" stopColor="#000000" stopOpacity="0.1" />
          <stop offset="100%" stopColor="#000000" stopOpacity="0.25" />
        </radialGradient>

        {/* Rim light gradient */}
        <linearGradient id={`rim-light-${handId}`} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#ffffff" stopOpacity="0.8" />
          <stop offset="50%" stopColor="#ffffff" stopOpacity="0.1" />
          <stop offset="100%" stopColor="#ffffff" stopOpacity="0.4" />
        </linearGradient>

        {/* Soft blur for glow effect */}
        <filter id={`hand-glow-${handId}`} x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur in="SourceGraphic" stdDeviation="0.5" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>

        {/* Drop shadow */}
        <filter id={`hand-shadow-${handId}`} x="-50%" y="-50%" width="200%" height="200%">
          <feDropShadow dx="0.3" dy="0.5" stdDeviation="0.8" floodColor="#000000" floodOpacity="0.3" />
        </filter>
      </defs>

      {/* Shadow layer */}
      <g filter={`url(#hand-shadow-${handId})`}>
        {/* Palm base shape */}
        <path
          d={createPalmShape()}
          fill={`url(#hand-fill-${handId})`}
        />

        {/* Fingers - rendered back to front for proper overlapping */}
        {[...fingers].reverse().map((finger, idx) => (
          <path
            key={`finger-base-${idx}`}
            d={createFingerShape(finger.indices, finger.width)}
            fill={`url(#hand-fill-${handId})`}
          />
        ))}
      </g>

      {/* Ambient occlusion in creases (between fingers) */}
      {[5, 9, 13].map((baseIdx, idx) => {
        const p1 = points[baseIdx]
        const p2 = points[baseIdx + 4]
        return (
          <ellipse
            key={`ao-${idx}`}
            cx={(p1.x + p2.x) / 2}
            cy={(p1.y + p2.y) / 2 - fingerWidth * 0.3}
            rx={fingerWidth * 0.6}
            ry={fingerWidth * 0.4}
            fill={`url(#ao-gradient-${handId})`}
            opacity={0.5}
          />
        )
      })}

      {/* Knuckle definition shadows */}
      {KNUCKLES.map((idx) => {
        const p = points[idx]
        return (
          <circle
            key={`knuckle-shadow-${idx}`}
            cx={p.x}
            cy={p.y + fingerWidth * 0.2}
            r={fingerWidth * 0.5}
            fill="#000000"
            opacity={0.08}
          />
        )
      })}

      {/* Highlight layer - rim lighting effect */}
      <g filter={`url(#hand-glow-${handId})`}>
        {/* Palm highlight */}
        <path
          d={createPalmShape()}
          fill="none"
          stroke={`url(#rim-light-${handId})`}
          strokeWidth={0.3 * clampedScale}
        />

        {/* Finger highlights */}
        {fingers.map((finger, idx) => (
          <path
            key={`finger-highlight-${idx}`}
            d={createFingerShape(finger.indices, finger.width * 0.7)}
            fill="none"
            stroke="#ffffff"
            strokeWidth={0.2 * clampedScale}
            strokeOpacity={0.4}
          />
        ))}

        {/* Fingertip highlights - small specular dots */}
        {FINGERTIPS.map((idx) => {
          const p = points[idx]
          return (
            <circle
              key={`tip-highlight-${idx}`}
              cx={p.x - fingerWidth * 0.15}
              cy={p.y - fingerWidth * 0.15}
              r={fingerWidth * 0.2}
              fill="#ffffff"
              opacity={0.7}
            />
          )
        })}
      </g>

      {/* Wrist cuff */}
      <ellipse
        cx={points[0].x}
        cy={points[0].y + fingerWidth * 0.5}
        rx={fingerWidth * 1.5}
        ry={fingerWidth * 0.8}
        fill="none"
        stroke="#ffffff"
        strokeWidth={fingerWidth * 0.4}
        opacity={0.3}
      />
    </g>
  )
}

interface LaserBeamProps {
  ray: PinchRay
  stableRay?: StableRay | null
  color: string
  isGripped: boolean
  hasHit?: boolean
}

function LaserBeam({ ray, stableRay, color, isGripped, hasHit = false }: LaserBeamProps) {
  // Use stable ray screen hit if available, otherwise fall back to basic calculation
  let originX: number, originY: number, endX: number, endY: number

  if (stableRay?.screenHit) {
    // Use the stable ray (arm model + One Euro Filter)
    // Un-mirror the X coordinate (webcam is mirrored)
    originX = (1 - stableRay.pinchPoint.x) * 100
    originY = stableRay.pinchPoint.y * 100

    // End point from ray intersection with screen plane
    endX = (1 - stableRay.screenHit.x) * 100
    endY = stableRay.screenHit.y * 100

    // Clamp to viewport
    endX = Math.max(0, Math.min(100, endX))
    endY = Math.max(0, Math.min(100, endY))
  } else {
    // Fallback: original basic ray calculation
    originX = (1 - ray.origin.x) * 100
    originY = ray.origin.y * 100

    const centerX = 50
    const centerY = 50
    const handDeviationX = (originX - 50) * LASER_DEVIATION_SCALE
    const handDeviationY = (originY - 50) * LASER_DEVIATION_SCALE
    const targetX = centerX - handDeviationX * (1 - LASER_CENTER_BIAS)
    const targetY = centerY - handDeviationY * (1 - LASER_CENTER_BIAS)

    const toCenterX = targetX - originX
    const toCenterY = targetY - originY
    const dist = Math.sqrt(toCenterX * toCenterX + toCenterY * toCenterY)
    const normX = dist > 0 ? toCenterX / dist : 0
    const normY = dist > 0 ? toCenterY / dist : 0
    const laserLength = dist

    endX = originX + normX * laserLength
    endY = originY + normY * laserLength
  }

  // Visual properties - intensify based on state
  const pinchStrength = stableRay?.pinchStrength ?? ray.strength
  const baseStrokeWidth = 0.2 + pinchStrength * 0.3
  const strokeWidth = isGripped ? baseStrokeWidth * 2.5 : hasHit ? baseStrokeWidth * 1.5 : baseStrokeWidth
  const baseOpacity = 0.3 + pinchStrength * 0.4
  const opacity = isGripped ? Math.min(1, baseOpacity * 1.8) : hasHit ? baseOpacity * 1.3 : baseOpacity
  const glowRadius = isGripped ? 2 + pinchStrength * 2 : hasHit ? 1.5 + pinchStrength : 0.8 + pinchStrength * 0.8

  // Color changes based on state
  const activeColor = isGripped ? '#ffffff' : hasHit ? '#fbbf24' : color // Golden when hitting
  const confidence = stableRay?.confidence ?? 0.8

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
        stroke={activeColor}
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        opacity={opacity * confidence}
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
        fill={activeColor}
        opacity={isGripped ? 1 : hasHit ? 0.9 : 0.7}
      />

      {/* Hit indicator - pulsing crosshair when pointing at a node */}
      {hasHit && (
        <g className="animate-pulse">
          {/* Outer ring */}
          <circle
            cx={endX}
            cy={endY}
            r={3}
            fill="none"
            stroke="#fbbf24"
            strokeWidth={0.3}
            opacity={0.8}
          />
          {/* Inner target */}
          <circle
            cx={endX}
            cy={endY}
            r={1.5}
            fill="#fbbf24"
            opacity={0.6}
          />
          {/* Crosshair lines */}
          <line x1={endX - 4} y1={endY} x2={endX - 2} y2={endY} stroke="#fbbf24" strokeWidth={0.2} opacity={0.7} />
          <line x1={endX + 2} y1={endY} x2={endX + 4} y2={endY} stroke="#fbbf24" strokeWidth={0.2} opacity={0.7} />
          <line x1={endX} y1={endY - 4} x2={endX} y2={endY - 2} stroke="#fbbf24" strokeWidth={0.2} opacity={0.7} />
          <line x1={endX} y1={endY + 2} x2={endX} y2={endY + 4} stroke="#fbbf24" strokeWidth={0.2} opacity={0.7} />
        </g>
      )}

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
          stroke={activeColor}
          strokeWidth={0.3}
          opacity={0.7}
          className="animate-ping"
        />
      )}
    </g>
  )
}

export default Hand2DOverlay
