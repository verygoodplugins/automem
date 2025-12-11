/**
 * Hand 2D Overlay
 *
 * Renders the hand as a 2D overlay on top of the canvas (not inside the 3D scene).
 * The hand appears life-size relative to the screen - closer to camera = larger.
 * Uses SVG for crisp rendering at any size.
 */

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

interface Hand2DOverlayProps {
  gestureState: GestureState
  enabled?: boolean
  showLaser?: boolean
}

export function Hand2DOverlay({ gestureState, enabled = true, showLaser = true }: Hand2DOverlayProps) {
  if (!enabled || !gestureState.isTracking) return null

  // Get the dominant hand (prefer right, fallback to left)
  const hand = gestureState.rightHand || gestureState.leftHand
  const pinchRay = gestureState.rightPinchRay || gestureState.leftPinchRay
  const isRightHand = !!gestureState.rightHand
  const color = isRightHand ? '#f72585' : '#4ecdc4'

  if (!hand) return null

  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden">
      <svg
        className="w-full h-full"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
      >
        {/* Hand skeleton */}
        <HandSkeleton
          landmarks={hand.landmarks}
          color={color}
        />

        {/* Laser beam from pinch point */}
        {showLaser && pinchRay && pinchRay.strength > 0.3 && (
          <LaserBeam ray={pinchRay} color={color} />
        )}
      </svg>
    </div>
  )
}

interface HandSkeletonProps {
  landmarks: { x: number; y: number; z?: number }[]
  color: string
}

function HandSkeleton({ landmarks, color }: HandSkeletonProps) {
  // Calculate scale based on hand depth (z of wrist)
  // Closer to camera = SMALLER (further into scene), far from camera = LARGER (closer to viewer)
  const wristZ = landmarks[0].z || 0
  // Z typically ranges from -0.1 (close to camera) to 0.1 (far from camera)
  // Invert: close to camera = smaller (going into screen), far = larger
  const depthScale = 1 + wristZ * 5
  const clampedScale = Math.max(0.3, Math.min(1.5, depthScale))

  // Base stroke width that scales with depth
  const baseStroke = 0.3 * clampedScale
  const jointRadius = 0.4 * clampedScale
  const fingertipRadius = 0.6 * clampedScale

  // Convert normalized coords (0-1) to viewBox coords (0-100)
  // NOT mirrored - hand points same direction as yours
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
            opacity={0.8}
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
            opacity={isFingertip ? 1 : 0.7}
          />
        )
      })}

      {/* Glow effect for fingertips */}
      {FINGERTIPS.map((idx) => {
        const p = points[idx]
        return (
          <circle
            key={`glow-${idx}`}
            cx={p.x}
            cy={p.y}
            r={fingertipRadius * 2}
            fill={color}
            opacity={0.2}
          />
        )
      })}
    </g>
  )
}

interface LaserBeamProps {
  ray: PinchRay
  color: string
}

function LaserBeam({ ray, color }: LaserBeamProps) {
  // Origin in screen coords (NOT mirrored)
  const originX = ray.origin.x * 100
  const originY = ray.origin.y * 100

  // Calculate end point - laser extends toward center of graph (into the nexus)
  // Direction points from hand toward center of screen
  const centerX = 50
  const centerY = 50
  const laserLength = 150 // Extend beyond viewport

  // Direction toward center
  const toCenterX = centerX - originX
  const toCenterY = centerY - originY
  const dist = Math.sqrt(toCenterX * toCenterX + toCenterY * toCenterY)
  const normX = dist > 0 ? toCenterX / dist : 0
  const normY = dist > 0 ? toCenterY / dist : 0

  const endX = originX + normX * laserLength
  const endY = originY + normY * laserLength

  // Visual properties based on pinch strength
  const strokeWidth = 0.2 + ray.strength * 0.5
  const opacity = 0.3 + ray.strength * 0.5
  const glowRadius = 0.8 + ray.strength * 0.8

  return (
    <g>
      {/* Laser glow (wider, more transparent) */}
      <line
        x1={originX}
        y1={originY}
        x2={endX}
        y2={endY}
        stroke={color}
        strokeWidth={strokeWidth * 3}
        strokeLinecap="round"
        opacity={opacity * 0.3}
      />

      {/* Main laser beam */}
      <line
        x1={originX}
        y1={originY}
        x2={endX}
        y2={endY}
        stroke={color}
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        opacity={opacity}
        strokeDasharray={ray.isValid ? 'none' : '2 1'}
      />

      {/* Origin glow sphere */}
      <circle
        cx={originX}
        cy={originY}
        r={glowRadius * 2}
        fill={color}
        opacity={0.3}
      />
      <circle
        cx={originX}
        cy={originY}
        r={glowRadius}
        fill={color}
        opacity={ray.isValid ? 0.9 : 0.5}
      />

      {/* Pulsing ring when pinch is active */}
      {ray.isValid && (
        <circle
          cx={originX}
          cy={originY}
          r={glowRadius * 1.5}
          fill="none"
          stroke={color}
          strokeWidth={0.2}
          opacity={0.6}
          className="animate-ping"
        />
      )}
    </g>
  )
}

export default Hand2DOverlay
