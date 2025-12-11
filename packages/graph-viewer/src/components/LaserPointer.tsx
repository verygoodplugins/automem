/**
 * Laser Pointer Component
 *
 * Renders a beautiful, accurate laser beam from the hand to the target.
 * Features:
 * - Gradient beam with glow effects
 * - Hit indicator when pointing at a node
 * - Ripple effect on activation
 * - Arm model visualization (optional debug)
 */

import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Line } from '@react-three/drei'
import * as THREE from 'three'
import type { StableRay, NodeHit, Vec3 } from '../hooks/useStablePointerRay'

interface LaserPointerProps {
  ray: StableRay
  hit: NodeHit | null
  color: string
  showArmModel?: boolean
}

export function LaserPointer({ ray, hit, color, showArmModel = false }: LaserPointerProps) {
  const glowRef = useRef<THREE.Mesh>(null)
  const hitGlowRef = useRef<THREE.Mesh>(null)
  const rippleRef = useRef<THREE.Mesh>(null)

  // Laser beam points
  const beamPoints = useMemo(() => {
    const start: [number, number, number] = [
      ray.origin.x,
      ray.origin.y,
      ray.origin.z,
    ]

    // End point: either hit point or extend ray into distance
    const maxDistance = 200
    const distance = hit ? hit.distance : maxDistance
    const end: [number, number, number] = [
      ray.origin.x + ray.direction.x * distance,
      ray.origin.y + ray.direction.y * distance,
      ray.origin.z + ray.direction.z * distance,
    ]

    return { start, end, distance }
  }, [ray, hit])

  // Animate glow effects
  useFrame((state) => {
    const time = state.clock.elapsedTime

    // Pulse the origin glow
    if (glowRef.current) {
      const pulse = 1 + Math.sin(time * 4) * 0.15
      const baseScale = 0.15 + ray.pinchStrength * 0.1
      glowRef.current.scale.setScalar(baseScale * pulse)
    }

    // Pulse the hit indicator
    if (hitGlowRef.current && hit) {
      const pulse = 1 + Math.sin(time * 6) * 0.2
      hitGlowRef.current.scale.setScalar(0.8 * pulse)
    }

    // Rotate ripple effect
    if (rippleRef.current && ray.isActive) {
      rippleRef.current.rotation.z = time * 2
    }
  })

  // Visual properties based on state
  const intensity = ray.isActive ? 1 : 0.4 + ray.pinchStrength * 0.4
  const lineWidth = ray.isActive ? 3 : 1.5 + ray.pinchStrength
  const glowOpacity = 0.3 + intensity * 0.4

  return (
    <group>
      {/* Main beam - inner bright core */}
      <Line
        points={[beamPoints.start, beamPoints.end]}
        color={ray.isActive ? '#ffffff' : color}
        lineWidth={lineWidth}
        transparent
        opacity={intensity}
      />

      {/* Outer glow beam */}
      <Line
        points={[beamPoints.start, beamPoints.end]}
        color={color}
        lineWidth={lineWidth * 2.5}
        transparent
        opacity={glowOpacity * 0.3}
      />

      {/* Origin glow sphere */}
      <mesh ref={glowRef} position={beamPoints.start}>
        <sphereGeometry args={[1, 16, 16]} />
        <meshBasicMaterial
          color={ray.isActive ? '#ffffff' : color}
          transparent
          opacity={ray.isActive ? 0.9 : 0.6}
        />
      </mesh>

      {/* Active ripple at origin */}
      {ray.isActive && (
        <mesh ref={rippleRef} position={beamPoints.start}>
          <ringGeometry args={[0.2, 0.3, 32]} />
          <meshBasicMaterial color={color} transparent opacity={0.4} side={THREE.DoubleSide} />
        </mesh>
      )}

      {/* Hit indicator */}
      {hit && (
        <group position={[hit.point.x, hit.point.y, hit.point.z]}>
          {/* Inner hit point */}
          <mesh>
            <sphereGeometry args={[0.3, 16, 16]} />
            <meshBasicMaterial
              color={ray.isActive ? '#ffffff' : color}
              transparent
              opacity={0.9}
            />
          </mesh>

          {/* Outer glow */}
          <mesh ref={hitGlowRef}>
            <sphereGeometry args={[0.5, 16, 16]} />
            <meshBasicMaterial color={color} transparent opacity={0.3} />
          </mesh>

          {/* Expanding rings when active */}
          {ray.isActive && (
            <>
              <ExpandingRing color={color} delay={0} />
              <ExpandingRing color={color} delay={0.3} />
              <ExpandingRing color={color} delay={0.6} />
            </>
          )}
        </group>
      )}

      {/* Debug: Arm model visualization */}
      {showArmModel && (
        <ArmModelDebug armPose={ray.armPose} color={color} />
      )}
    </group>
  )
}

/**
 * Expanding ring animation at hit point
 */
function ExpandingRing({ color, delay }: { color: string; delay: number }) {
  const ringRef = useRef<THREE.Mesh>(null)

  useFrame((state) => {
    if (!ringRef.current) return

    const time = (state.clock.elapsedTime + delay) % 1
    const scale = 0.5 + time * 2
    const opacity = (1 - time) * 0.4

    ringRef.current.scale.setScalar(scale)
    const mat = ringRef.current.material as THREE.MeshBasicMaterial
    mat.opacity = opacity
  })

  return (
    <mesh ref={ringRef}>
      <ringGeometry args={[0.8, 1, 32]} />
      <meshBasicMaterial color={color} transparent opacity={0.4} side={THREE.DoubleSide} />
    </mesh>
  )
}

/**
 * Debug visualization of the estimated arm model
 */
function ArmModelDebug({ armPose, color }: { armPose: StableRay['armPose']; color: string }) {
  const { shoulder, elbow, wrist, pinchPoint } = armPose

  return (
    <group>
      {/* Shoulder */}
      <mesh position={[shoulder.x, shoulder.y, shoulder.z]}>
        <sphereGeometry args={[0.15, 8, 8]} />
        <meshBasicMaterial color="#ff6b6b" transparent opacity={0.5} />
      </mesh>

      {/* Upper arm (shoulder → elbow) */}
      <Line
        points={[
          [shoulder.x, shoulder.y, shoulder.z],
          [elbow.x, elbow.y, elbow.z],
        ]}
        color="#ff6b6b"
        lineWidth={2}
        transparent
        opacity={0.4}
        dashed
        dashSize={0.1}
        gapSize={0.05}
      />

      {/* Elbow */}
      <mesh position={[elbow.x, elbow.y, elbow.z]}>
        <sphereGeometry args={[0.12, 8, 8]} />
        <meshBasicMaterial color="#fbbf24" transparent opacity={0.5} />
      </mesh>

      {/* Forearm (elbow → wrist) */}
      <Line
        points={[
          [elbow.x, elbow.y, elbow.z],
          [wrist.x, wrist.y, wrist.z],
        ]}
        color="#fbbf24"
        lineWidth={2}
        transparent
        opacity={0.4}
        dashed
        dashSize={0.1}
        gapSize={0.05}
      />

      {/* Wrist */}
      <mesh position={[wrist.x, wrist.y, wrist.z]}>
        <sphereGeometry args={[0.1, 8, 8]} />
        <meshBasicMaterial color="#4ade80" transparent opacity={0.5} />
      </mesh>

      {/* Hand (wrist → pinch point) */}
      <Line
        points={[
          [wrist.x, wrist.y, wrist.z],
          [pinchPoint.x, pinchPoint.y, pinchPoint.z],
        ]}
        color={color}
        lineWidth={2}
        transparent
        opacity={0.6}
      />

      {/* Pinch point */}
      <mesh position={[pinchPoint.x, pinchPoint.y, pinchPoint.z]}>
        <sphereGeometry args={[0.08, 8, 8]} />
        <meshBasicMaterial color={color} transparent opacity={0.8} />
      </mesh>
    </group>
  )
}

export default LaserPointer
