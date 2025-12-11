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

/**
 * PuffyGlove - Mario-style white glove with puffy, rounded appearance
 * Semi-transparent with soft edges and volumetric feel
 */
function GhostHand({ landmarks, color: _color, gradientId: _gradientId, isGhost = false }: GhostHandProps) {
  // INVERTED depth scaling: hand closer to camera (negative Z) = smaller (farther in 3D space)
  // Hand pulled back (positive Z) = bigger (closer to viewer)
  const wristZ = landmarks[0].z || 0
  // Invert: positive Z (hand far from camera) = bigger, negative Z (close to camera) = smaller
  const depthScale = 1 - wristZ * 4 // Inverted from before
  const clampedScale = Math.max(0.4, Math.min(1.8, depthScale))

  // Un-mirror the X coordinate (webcam is mirrored, so flip it back)
  const toSvg = (lm: { x: number; y: number }) => ({
    x: (1 - lm.x) * 100, // Flip X to un-mirror
    y: lm.y * 100,
  })

  const points = landmarks.map(toSvg)

  // Puffy glove sizing - everything is rounder and bigger
  const baseFingerWidth = 1.2 * clampedScale
  const fingertipRadius = 0.9 * clampedScale
  const knuckleRadius = 0.7 * clampedScale
  const gloveOpacity = isGhost ? 0.4 : 0.6

  // Finger segment indices: [base, mid1, mid2, tip]
  const FINGERS = [
    [1, 2, 3, 4],     // Thumb
    [5, 6, 7, 8],     // Index
    [9, 10, 11, 12],  // Middle
    [13, 14, 15, 16], // Ring
    [17, 18, 19, 20], // Pinky
  ]

  // Create puffy finger path with rounded capsule segments
  const createPuffyFingerPath = (fingerIndices: number[]) => {
    const fingerPoints = fingerIndices.map(i => points[i])
    // Create a thick rounded path
    let path = `M ${fingerPoints[0].x} ${fingerPoints[0].y}`
    for (let i = 1; i < fingerPoints.length; i++) {
      path += ` L ${fingerPoints[i].x} ${fingerPoints[i].y}`
    }
    return path
  }

  // Palm center for radial gradient
  const palmCenter = {
    x: (points[0].x + points[5].x + points[9].x + points[13].x + points[17].x) / 5,
    y: (points[0].y + points[5].y + points[9].y + points[13].y + points[17].y) / 5,
  }

  // Create smooth palm outline
  const palmPath = `
    M ${points[0].x} ${points[0].y}
    Q ${points[1].x} ${points[1].y} ${points[2].x} ${points[2].y}
    L ${points[5].x} ${points[5].y}
    Q ${(points[5].x + points[9].x) / 2} ${Math.min(points[5].y, points[9].y) - 1}
      ${points[9].x} ${points[9].y}
    Q ${(points[9].x + points[13].x) / 2} ${Math.min(points[9].y, points[13].y) - 0.5}
      ${points[13].x} ${points[13].y}
    Q ${(points[13].x + points[17].x) / 2} ${Math.min(points[13].y, points[17].y) - 0.5}
      ${points[17].x} ${points[17].y}
    L ${points[0].x} ${points[0].y}
    Z
  `

  return (
    <g>
      {/* Definitions for this hand */}
      <defs>
        {/* Puffy white glove gradient */}
        <radialGradient id={`glove-gradient-${palmCenter.x.toFixed(0)}`} cx="30%" cy="30%" r="70%">
          <stop offset="0%" stopColor="#ffffff" stopOpacity="0.9" />
          <stop offset="50%" stopColor="#f0f0f5" stopOpacity="0.7" />
          <stop offset="100%" stopColor="#d0d0e0" stopOpacity="0.5" />
        </radialGradient>

        {/* Soft shadow/depth filter */}
        <filter id={`glove-shadow-${palmCenter.x.toFixed(0)}`} x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur in="SourceAlpha" stdDeviation="0.8" result="blur" />
          <feOffset in="blur" dx="0.3" dy="0.5" result="shadow" />
          <feComposite in="SourceGraphic" in2="shadow" operator="over" />
        </filter>

        {/* Soft glow for volumetric effect */}
        <filter id={`glove-glow-${palmCenter.x.toFixed(0)}`} x="-100%" y="-100%" width="300%" height="300%">
          <feGaussianBlur stdDeviation="1" result="glow" />
          <feMerge>
            <feMergeNode in="glow" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Main glove group with shadow */}
      <g filter={`url(#glove-shadow-${palmCenter.x.toFixed(0)})`} opacity={gloveOpacity}>

        {/* Palm - puffy rounded shape */}
        <path
          d={palmPath}
          fill={`url(#glove-gradient-${palmCenter.x.toFixed(0)})`}
          stroke="#ffffff"
          strokeWidth={0.3 * clampedScale}
          strokeOpacity={0.5}
        />

        {/* Fingers - thick puffy tubes */}
        {FINGERS.map((fingerIndices, idx) => (
          <g key={`finger-${idx}`}>
            {/* Finger tube - thick white stroke for puffy look */}
            <path
              d={createPuffyFingerPath(fingerIndices)}
              fill="none"
              stroke="#ffffff"
              strokeWidth={baseFingerWidth * 2.5}
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeOpacity={0.5}
            />
            {/* Inner highlight */}
            <path
              d={createPuffyFingerPath(fingerIndices)}
              fill="none"
              stroke="#f8f8ff"
              strokeWidth={baseFingerWidth * 1.5}
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeOpacity={0.7}
            />
            {/* Core line for definition */}
            <path
              d={createPuffyFingerPath(fingerIndices)}
              fill="none"
              stroke="#ffffff"
              strokeWidth={baseFingerWidth * 0.8}
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeOpacity={0.9}
            />
          </g>
        ))}

        {/* Fingertip puffs - round ball ends like Mario gloves */}
        {FINGERTIPS.map((idx) => {
          const p = points[idx]
          return (
            <g key={`tip-${idx}`}>
              {/* Outer soft glow */}
              <circle
                cx={p.x}
                cy={p.y}
                r={fingertipRadius * 2}
                fill="#ffffff"
                opacity={0.3}
              />
              {/* Main puff */}
              <circle
                cx={p.x}
                cy={p.y}
                r={fingertipRadius * 1.4}
                fill="#f8f8ff"
                opacity={0.7}
              />
              {/* Highlight dot */}
              <circle
                cx={p.x - fingertipRadius * 0.3}
                cy={p.y - fingertipRadius * 0.3}
                r={fingertipRadius * 0.5}
                fill="#ffffff"
                opacity={0.9}
              />
            </g>
          )
        })}

        {/* Knuckle bumps - subtle rounded protrusions */}
        {KNUCKLES.map((idx) => {
          const p = points[idx]
          return (
            <g key={`knuckle-${idx}`}>
              <circle
                cx={p.x}
                cy={p.y}
                r={knuckleRadius * 1.8}
                fill="#f0f0f5"
                opacity={0.5}
              />
              <circle
                cx={p.x}
                cy={p.y}
                r={knuckleRadius}
                fill="#ffffff"
                opacity={0.7}
              />
            </g>
          )
        })}

        {/* Wrist cuff - puffy ring */}
        <circle
          cx={points[0].x}
          cy={points[0].y}
          r={knuckleRadius * 2.5}
          fill="none"
          stroke="#ffffff"
          strokeWidth={knuckleRadius * 1.5}
          opacity={0.4}
        />
      </g>
    </g>
  )
}

interface LaserBeamProps {
  ray: PinchRay
  color: string
  isGripped: boolean
}

function LaserBeam({ ray, color, isGripped }: LaserBeamProps) {
  // Un-mirror the X coordinate (webcam is mirrored, so flip it back)
  const originX = (1 - ray.origin.x) * 100
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
