/**
 * Hand Skeleton Overlay
 *
 * Renders 3D wireframe hands overlaid on the graph visualization.
 * Uses React Three Fiber to render hand landmarks as glowing lines.
 */

import { useMemo } from 'react'
import { Line } from '@react-three/drei'
import type { HandLandmarks, GestureState, PinchRay } from '../hooks/useHandGestures'

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

interface HandSkeletonProps {
  hand: HandLandmarks
  color: string
  opacity?: number
  scale?: number
}

// Convert MediaPipe normalized coords (0-1) to Three.js world coords
function landmarkToWorld(
  landmark: { x: number; y: number; z: number },
  scale: number = 10
): [number, number, number] {
  // MediaPipe: x=0-1 (left-right), y=0-1 (top-bottom), z=depth
  // Three.js: x=-5 to 5, y=-5 to 5, z=depth
  return [
    (landmark.x - 0.5) * scale,
    -(landmark.y - 0.5) * scale, // Flip Y
    -landmark.z * scale * 2, // Z comes toward camera
  ]
}

function HandSkeleton({ hand, color, opacity = 0.8, scale = 10 }: HandSkeletonProps) {
  const lines = useMemo(() => {
    return HAND_CONNECTIONS.map(([i, j], idx) => {
      const start = landmarkToWorld(hand.landmarks[i], scale)
      const end = landmarkToWorld(hand.landmarks[j], scale)
      return { start, end, key: idx }
    })
  }, [hand.landmarks, scale])

  const jointPositions = useMemo(() => {
    return hand.landmarks.map((lm, idx) => ({
      position: landmarkToWorld(lm, scale),
      key: idx,
      // Fingertips get larger spheres
      isFingertip: [4, 8, 12, 16, 20].includes(idx),
    }))
  }, [hand.landmarks, scale])

  return (
    <group>
      {/* Skeleton lines */}
      {lines.map(({ start, end, key }) => (
        <Line
          key={key}
          points={[start, end]}
          color={color}
          lineWidth={3}
          transparent
          opacity={opacity}
        />
      ))}

      {/* Joint spheres */}
      {jointPositions.map(({ position, key, isFingertip }) => (
        <mesh key={key} position={position}>
          <sphereGeometry args={[isFingertip ? 0.15 : 0.08, 8, 8]} />
          <meshBasicMaterial
            color={color}
            transparent
            opacity={isFingertip ? opacity : opacity * 0.6}
          />
        </mesh>
      ))}
    </group>
  )
}

// Convert pinch ray origin (normalized 0-1 coords) to Three.js world coords
function pinchRayToWorld(
  ray: PinchRay,
  scale: number = 10
): { origin: [number, number, number]; end: [number, number, number] } {
  // Origin in 3D space
  const origin: [number, number, number] = [
    (ray.origin.x - 0.5) * scale,
    -(ray.origin.y - 0.5) * scale, // Flip Y
    -ray.origin.z * scale * 2,
  ]

  // Ray extends in the direction, scaled by ray length
  const rayLength = 100 // How far the laser extends
  const end: [number, number, number] = [
    origin[0] + ray.direction.x * rayLength,
    origin[1] - ray.direction.y * rayLength, // Flip Y for direction too
    origin[2] - ray.direction.z * rayLength,
  ]

  return { origin, end }
}

interface PinchRayBeamProps {
  ray: PinchRay
  color: string
  scale?: number
}

function PinchRayBeam({ ray, color, scale = 10 }: PinchRayBeamProps) {
  const { origin, end } = useMemo(() => pinchRayToWorld(ray, scale), [ray, scale])

  // Calculate visual properties based on pinch strength
  const lineWidth = 1 + ray.strength * 3 // Thicker when pinching harder
  const opacity = 0.3 + ray.strength * 0.5 // More visible when pinching

  // Glow sphere size at origin
  const sphereSize = 0.1 + ray.strength * 0.15

  return (
    <group>
      {/* Main laser beam */}
      <Line
        points={[origin, end]}
        color={color}
        lineWidth={lineWidth}
        transparent
        opacity={opacity}
        dashed={!ray.isValid} // Dashed when not fully pinched
        dashSize={ray.isValid ? 0 : 0.5}
        gapSize={ray.isValid ? 0 : 0.3}
      />

      {/* Origin glow sphere (where thumb meets index) */}
      <mesh position={origin}>
        <sphereGeometry args={[sphereSize, 16, 16]} />
        <meshBasicMaterial
          color={color}
          transparent
          opacity={ray.isValid ? 0.9 : 0.5}
        />
      </mesh>

      {/* Secondary glow ring when pinch is active */}
      {ray.isValid && (
        <mesh position={origin} rotation={[Math.PI / 2, 0, 0]}>
          <ringGeometry args={[sphereSize * 1.2, sphereSize * 1.5, 32]} />
          <meshBasicMaterial color={color} transparent opacity={0.4} />
        </mesh>
      )}
    </group>
  )
}

interface GestureIndicatorProps {
  gestureState: GestureState
}

function GestureIndicator({ gestureState }: GestureIndicatorProps) {
  const { handsDetected, zoomDelta, leftPinchRay, rightPinchRay } = gestureState

  return (
    <group>
      {/* Left hand pinch ray - cyan */}
      {leftPinchRay && leftPinchRay.strength > 0.3 && (
        <PinchRayBeam ray={leftPinchRay} color="#4ecdc4" />
      )}

      {/* Right hand pinch ray - magenta */}
      {rightPinchRay && rightPinchRay.strength > 0.3 && (
        <PinchRayBeam ray={rightPinchRay} color="#f72585" />
      )}

      {/* Two-hand zoom indicator */}
      {handsDetected === 2 && Math.abs(zoomDelta) > 0.01 && (
        <mesh position={[0, 4, 0]}>
          <torusGeometry args={[0.5 + Math.abs(zoomDelta) * 5, 0.1, 8, 32]} />
          <meshBasicMaterial
            color={zoomDelta > 0 ? '#4ecdc4' : '#ff6b6b'}
            transparent
            opacity={0.6}
          />
        </mesh>
      )}
    </group>
  )
}

interface HandSkeletonOverlayProps {
  gestureState: GestureState
  enabled?: boolean
}

export function HandSkeletonOverlay({ gestureState, enabled = true }: HandSkeletonOverlayProps) {
  if (!enabled || !gestureState.isTracking) return null

  return (
    <group>
      {/* Left hand - cyan */}
      {gestureState.leftHand && (
        <HandSkeleton
          hand={gestureState.leftHand}
          color="#4ecdc4"
          opacity={0.7}
        />
      )}

      {/* Right hand - magenta */}
      {gestureState.rightHand && (
        <HandSkeleton
          hand={gestureState.rightHand}
          color="#f72585"
          opacity={0.7}
        />
      )}

      {/* Gesture feedback */}
      <GestureIndicator gestureState={gestureState} />
    </group>
  )
}

export default HandSkeletonOverlay
