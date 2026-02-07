import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import type { Cluster } from '../hooks/useClusterDetection'

interface ClusterBoundariesProps {
  clusters: Cluster[]
  visible: boolean
  opacity?: number
}

// Generate points on a sphere surface for dotted effect
function generateSpherePoints(radius: number, count: number): Float32Array {
  const positions = new Float32Array(count * 3)

  // Fibonacci sphere distribution for even spacing
  const goldenRatio = (1 + Math.sqrt(5)) / 2
  const angleIncrement = Math.PI * 2 * goldenRatio

  for (let i = 0; i < count; i++) {
    const t = i / count
    const inclination = Math.acos(1 - 2 * t)
    const azimuth = angleIncrement * i

    const x = Math.sin(inclination) * Math.cos(azimuth) * radius
    const y = Math.sin(inclination) * Math.sin(azimuth) * radius
    const z = Math.cos(inclination) * radius

    positions[i * 3] = x
    positions[i * 3 + 1] = y
    positions[i * 3 + 2] = z
  }

  return positions
}

/**
 * A single cluster boundary sphere made of points
 */
function ClusterBoundary({
  cluster,
  opacity = 0.3,
}: {
  cluster: Cluster
  opacity?: number
}) {
  const pointsRef = useRef<THREE.Points>(null)

  // Number of points scales with radius
  const pointCount = Math.max(100, Math.floor(cluster.radius * 8))

  const positions = useMemo(() => {
    return generateSpherePoints(cluster.radius, pointCount)
  }, [cluster.radius, pointCount])

  // Gentle rotation for visual interest
  useFrame((_, delta) => {
    if (pointsRef.current) {
      pointsRef.current.rotation.y += delta * 0.05
      pointsRef.current.rotation.x += delta * 0.02
    }
  })

  // Parse color to THREE.Color
  const color = useMemo(() => {
    return new THREE.Color(cluster.color)
  }, [cluster.color])

  return (
    <points
      ref={pointsRef}
      position={[cluster.centroid.x, cluster.centroid.y, cluster.centroid.z]}
    >
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={pointCount}
          array={positions}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        color={color}
        size={1.5}
        transparent
        opacity={opacity}
        sizeAttenuation
        depthWrite={false}
      />
    </points>
  )
}

/**
 * Renders dotted sphere boundaries around detected clusters
 */
export function ClusterBoundaries({
  clusters,
  visible,
  opacity = 0.3,
}: ClusterBoundariesProps) {
  if (!visible || clusters.length === 0) {
    return null
  }

  return (
    <group>
      {clusters.map((cluster) => (
        <ClusterBoundary
          key={cluster.id}
          cluster={cluster}
          opacity={opacity}
        />
      ))}
    </group>
  )
}
