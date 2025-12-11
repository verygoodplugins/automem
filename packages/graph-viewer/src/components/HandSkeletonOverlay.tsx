/**
 * Hand Skeleton Overlay
 *
 * Renders 3D wireframe hands overlaid on the graph visualization.
 * Uses React Three Fiber to render hand landmarks as glowing lines.
 */

import { useMemo } from 'react'
import { Line } from '@react-three/drei'
import type { HandLandmarks, GestureState } from '../hooks/useHandGestures'

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

interface GestureIndicatorProps {
  gestureState: GestureState
}

function GestureIndicator({ gestureState }: GestureIndicatorProps) {
  const { handsDetected, zoomDelta, pointDirection } = gestureState

  // Pointing ray
  if (pointDirection && handsDetected === 1) {
    const rayStart: [number, number, number] = [
      (pointDirection.x - 0.5) * 10,
      (pointDirection.y - 0.5) * 10,
      0,
    ]
    const rayEnd: [number, number, number] = [
      (pointDirection.x - 0.5) * 10,
      (pointDirection.y - 0.5) * 10,
      -50, // Ray extends into the scene
    ]

    return (
      <group>
        <Line
          points={[rayStart, rayEnd]}
          color="#4ecdc4"
          lineWidth={2}
          transparent
          opacity={0.5}
          dashed
          dashSize={0.5}
          gapSize={0.3}
        />
        {/* Pointer dot */}
        <mesh position={rayStart}>
          <sphereGeometry args={[0.1, 16, 16]} />
          <meshBasicMaterial color="#4ecdc4" />
        </mesh>
      </group>
    )
  }

  // Two-hand zoom indicator
  if (handsDetected === 2 && Math.abs(zoomDelta) > 0.01) {
    const zoomColor = zoomDelta > 0 ? '#4ecdc4' : '#ff6b6b'
    return (
      <mesh position={[0, 4, 0]}>
        <torusGeometry args={[0.5 + Math.abs(zoomDelta) * 5, 0.1, 8, 32]} />
        <meshBasicMaterial color={zoomColor} transparent opacity={0.6} />
      </mesh>
    )
  }

  return null
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
