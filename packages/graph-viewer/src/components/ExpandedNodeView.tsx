/**
 * Expanded Node View
 *
 * When a user laser-points at a node and activates it (pinch),
 * the node expands with a beautiful bloom effect revealing:
 * - Full content
 * - Metadata (tags, importance, timestamps)
 * - Connected nodes as orbiting satellites
 *
 * Animation sequence:
 * 1. Initial pulse/ripple from hit point
 * 2. Node scales up with elastic easing
 * 3. Content fades in with stagger
 * 4. Connections bloom outward
 * 5. Orbiting satellites settle into position
 */

import { useRef, useMemo, useEffect, useState } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { Text, Billboard, Html } from '@react-three/drei'
import * as THREE from 'three'
import type { GraphNode, GraphEdge, SimulationNode } from '../lib/types'

// Animation timing
const EXPAND_DURATION = 0.6 // seconds
const CONTENT_DELAY = 0.2
const CONNECTION_DELAY = 0.35
const SATELLITE_DELAY = 0.5

// Easing functions
const easeOutElastic = (t: number): number => {
  const c4 = (2 * Math.PI) / 3
  return t === 0 ? 0 : t === 1 ? 1 :
    Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * c4) + 1
}

const easeOutCubic = (t: number): number => 1 - Math.pow(1 - t, 3)

const easeOutBack = (t: number): number => {
  const c1 = 1.70158
  const c3 = c1 + 1
  return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2)
}

interface ExpandedNodeViewProps {
  node: SimulationNode
  connectedNodes: SimulationNode[]
  edges: GraphEdge[]
  hitPoint: { x: number; y: number; z: number }
  onClose: () => void
  isExpanding: boolean
}

export function ExpandedNodeView({
  node,
  connectedNodes,
  edges,
  hitPoint,
  onClose,
  isExpanding,
}: ExpandedNodeViewProps) {
  const groupRef = useRef<THREE.Group>(null)
  const rippleRef = useRef<THREE.Mesh>(null)
  const glowRef = useRef<THREE.Mesh>(null)
  const { camera } = useThree()

  // Animation state
  const [animationProgress, setAnimationProgress] = useState(0)
  const animationStartRef = useRef<number>(0)
  const isAnimatingRef = useRef(false)

  // Start animation when expanding
  useEffect(() => {
    if (isExpanding) {
      animationStartRef.current = performance.now() / 1000
      isAnimatingRef.current = true
      setAnimationProgress(0)
    }
  }, [isExpanding])

  // Animation frame
  useFrame((state) => {
    if (!isAnimatingRef.current || !groupRef.current) return

    const elapsed = state.clock.elapsedTime - animationStartRef.current
    const t = Math.min(1, elapsed / EXPAND_DURATION)
    setAnimationProgress(t)

    // Scale animation with elastic easing
    const scale = easeOutElastic(t) * 1.5 + 0.5
    groupRef.current.scale.setScalar(scale)

    // Ripple animation
    if (rippleRef.current) {
      const rippleT = Math.min(1, elapsed / 0.4)
      const rippleScale = easeOutCubic(rippleT) * 3
      rippleRef.current.scale.setScalar(rippleScale)
      const rippleMat = rippleRef.current.material as THREE.MeshBasicMaterial
      rippleMat.opacity = (1 - rippleT) * 0.5
    }

    // Glow pulsing
    if (glowRef.current) {
      const pulse = 1 + Math.sin(state.clock.elapsedTime * 3) * 0.1
      glowRef.current.scale.setScalar(2.5 * pulse)
    }

    // Stop animation when complete
    if (t >= 1) {
      isAnimatingRef.current = false
    }
  })

  // Calculate satellite positions (orbiting connected nodes)
  const satellitePositions = useMemo(() => {
    const count = connectedNodes.length
    if (count === 0) return []

    const radius = 15 // Orbit radius
    return connectedNodes.map((_, i) => {
      const angle = (i / count) * Math.PI * 2
      return {
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius * 0.6, // Elliptical orbit
        z: Math.sin(angle) * radius * 0.3,
      }
    })
  }, [connectedNodes.length])

  // Get type color
  const typeColor = useMemo(() => {
    const typeColors: Record<string, string> = {
      Decision: '#f59e0b',
      Pattern: '#10b981',
      Insight: '#8b5cf6',
      Preference: '#ec4899',
      Context: '#3b82f6',
      Style: '#06b6d4',
      Habit: '#f97316',
    }
    return typeColors[node.type] || '#6b7280'
  }, [node.type])

  // Content visibility based on animation progress
  const contentT = Math.max(0, (animationProgress - CONTENT_DELAY) / (1 - CONTENT_DELAY))
  const connectionT = Math.max(0, (animationProgress - CONNECTION_DELAY) / (1 - CONNECTION_DELAY))
  const satelliteT = Math.max(0, (animationProgress - SATELLITE_DELAY) / (1 - SATELLITE_DELAY))

  return (
    <group
      ref={groupRef}
      position={[node.x ?? 0, node.y ?? 0, node.z ?? 0]}
    >
      {/* Initial ripple from hit point */}
      <mesh ref={rippleRef} position={[hitPoint.x - (node.x ?? 0), hitPoint.y - (node.y ?? 0), hitPoint.z - (node.z ?? 0)]}>
        <ringGeometry args={[0.8, 1, 32]} />
        <meshBasicMaterial color="#ffffff" transparent opacity={0.5} side={THREE.DoubleSide} />
      </mesh>

      {/* Glow sphere */}
      <mesh ref={glowRef}>
        <sphereGeometry args={[node.radius * 1.2, 32, 32]} />
        <meshBasicMaterial color={typeColor} transparent opacity={0.15} />
      </mesh>

      {/* Main node sphere */}
      <mesh>
        <sphereGeometry args={[node.radius, 32, 32]} />
        <meshStandardMaterial
          color={node.color}
          emissive={typeColor}
          emissiveIntensity={0.3}
          roughness={0.2}
          metalness={0.1}
        />
      </mesh>

      {/* Content card - using HTML for rich text */}
      <Billboard follow={true} lockX={false} lockY={false} lockZ={false}>
        <group position={[0, node.radius * 2 + 3, 0]}>
          <Html
            center
            style={{
              opacity: easeOutCubic(contentT),
              transform: `scale(${0.8 + easeOutBack(contentT) * 0.2})`,
              transition: 'none',
              pointerEvents: animationProgress > 0.8 ? 'auto' : 'none',
            }}
          >
            <ContentCard
              node={node}
              onClose={onClose}
              typeColor={typeColor}
            />
          </Html>
        </group>
      </Billboard>

      {/* Connection lines to satellites */}
      {connectedNodes.map((connNode, i) => {
        const satPos = satellitePositions[i]
        if (!satPos) return null

        const lineProgress = easeOutCubic(connectionT)
        const endX = satPos.x * lineProgress
        const endY = satPos.y * lineProgress
        const endZ = satPos.z * lineProgress

        return (
          <group key={connNode.id}>
            {/* Connection line */}
            <line>
              <bufferGeometry>
                <bufferAttribute
                  attach="attributes-position"
                  args={[new Float32Array([0, 0, 0, endX, endY, endZ]), 3]}
                />
              </bufferGeometry>
              <lineBasicMaterial
                color={typeColor}
                transparent
                opacity={connectionT * 0.4}
              />
            </line>
          </group>
        )
      })}

      {/* Satellite nodes */}
      {connectedNodes.map((connNode, i) => {
        const satPos = satellitePositions[i]
        if (!satPos) return null

        const satProgress = easeOutElastic(Math.max(0, satelliteT - i * 0.05))
        const satScale = satProgress * 0.6

        // Get edge info for this connection
        const edge = edges.find(
          e => (e.source === node.id && e.target === connNode.id) ||
               (e.target === node.id && e.source === connNode.id)
        )

        return (
          <group
            key={connNode.id}
            position={[
              satPos.x * satProgress,
              satPos.y * satProgress,
              satPos.z * satProgress,
            ]}
            scale={satScale}
          >
            {/* Satellite sphere */}
            <mesh>
              <sphereGeometry args={[connNode.radius * 0.7, 16, 16]} />
              <meshStandardMaterial
                color={connNode.color}
                transparent
                opacity={satProgress}
              />
            </mesh>

            {/* Satellite label */}
            <Billboard>
              <Text
                position={[0, connNode.radius + 1.5, 0]}
                fontSize={1.8}
                color="#f1f5f9"
                anchorX="center"
                anchorY="bottom"
                outlineWidth={0.08}
                outlineColor="#000000"
                fillOpacity={satProgress}
              >
                {connNode.content.slice(0, 25)}...
              </Text>
              {edge && (
                <Text
                  position={[0, -connNode.radius - 0.5, 0]}
                  fontSize={1.2}
                  color="#94a3b8"
                  anchorX="center"
                  anchorY="top"
                  fillOpacity={satProgress * 0.8}
                >
                  {edge.type}
                </Text>
              )}
            </Billboard>
          </group>
        )
      })}

      {/* Floating particles for ambiance */}
      <FloatingParticles
        count={20}
        radius={12}
        color={typeColor}
        progress={animationProgress}
      />
    </group>
  )
}

/**
 * Content Card - HTML overlay with node details
 */
interface ContentCardProps {
  node: SimulationNode
  onClose: () => void
  typeColor: string
}

function ContentCard({ node, onClose, typeColor }: ContentCardProps) {
  return (
    <div
      className="pointer-events-auto"
      style={{
        background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%)',
        borderRadius: '16px',
        padding: '20px',
        minWidth: '320px',
        maxWidth: '400px',
        boxShadow: `0 0 40px ${typeColor}40, 0 8px 32px rgba(0, 0, 0, 0.5)`,
        border: `1px solid ${typeColor}60`,
        backdropFilter: 'blur(12px)',
        fontFamily: 'system-ui, -apple-system, sans-serif',
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
        <span
          style={{
            background: `linear-gradient(135deg, ${typeColor} 0%, ${typeColor}80 100%)`,
            color: 'white',
            padding: '4px 12px',
            borderRadius: '20px',
            fontSize: '12px',
            fontWeight: 600,
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
          }}
        >
          {node.type}
        </span>
        <button
          onClick={onClose}
          style={{
            background: 'rgba(255, 255, 255, 0.1)',
            border: 'none',
            borderRadius: '50%',
            width: '28px',
            height: '28px',
            cursor: 'pointer',
            color: '#94a3b8',
            fontSize: '16px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'all 0.2s',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = 'rgba(255, 255, 255, 0.2)'
            e.currentTarget.style.color = '#f1f5f9'
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)'
            e.currentTarget.style.color = '#94a3b8'
          }}
        >
          ✕
        </button>
      </div>

      {/* Content */}
      <p style={{
        color: '#f1f5f9',
        fontSize: '14px',
        lineHeight: 1.6,
        margin: '0 0 16px 0',
      }}>
        {node.content}
      </p>

      {/* Tags */}
      {node.tags.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginBottom: '16px' }}>
          {node.tags.slice(0, 6).map((tag, i) => (
            <span
              key={i}
              style={{
                background: 'rgba(255, 255, 255, 0.08)',
                color: '#94a3b8',
                padding: '3px 10px',
                borderRadius: '12px',
                fontSize: '11px',
              }}
            >
              #{tag}
            </span>
          ))}
          {node.tags.length > 6 && (
            <span style={{ color: '#64748b', fontSize: '11px', padding: '3px 0' }}>
              +{node.tags.length - 6} more
            </span>
          )}
        </div>
      )}

      {/* Metadata */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '8px',
        padding: '12px',
        background: 'rgba(0, 0, 0, 0.2)',
        borderRadius: '8px',
        fontSize: '12px',
      }}>
        <div>
          <span style={{ color: '#64748b' }}>Importance</span>
          <div style={{
            color: '#f1f5f9',
            marginTop: '2px',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}>
            <div style={{
              width: '60px',
              height: '4px',
              background: 'rgba(255,255,255,0.1)',
              borderRadius: '2px',
              overflow: 'hidden',
            }}>
              <div style={{
                width: `${node.importance * 100}%`,
                height: '100%',
                background: `linear-gradient(90deg, ${typeColor} 0%, ${typeColor}80 100%)`,
                borderRadius: '2px',
              }} />
            </div>
            {(node.importance * 100).toFixed(0)}%
          </div>
        </div>
        <div>
          <span style={{ color: '#64748b' }}>Created</span>
          <div style={{ color: '#f1f5f9', marginTop: '2px' }}>
            {new Date(node.timestamp).toLocaleDateString()}
          </div>
        </div>
      </div>
    </div>
  )
}

/**
 * Floating particles for visual ambiance
 */
interface FloatingParticlesProps {
  count: number
  radius: number
  color: string
  progress: number
}

function FloatingParticles({ count, radius, color, progress }: FloatingParticlesProps) {
  const particlesRef = useRef<THREE.Points>(null)

  const particles = useMemo(() => {
    const positions = new Float32Array(count * 3)
    const sizes = new Float32Array(count)
    const phases = new Float32Array(count)

    for (let i = 0; i < count; i++) {
      const theta = Math.random() * Math.PI * 2
      const phi = Math.acos(2 * Math.random() - 1)
      const r = radius * (0.5 + Math.random() * 0.5)

      positions[i * 3] = r * Math.sin(phi) * Math.cos(theta)
      positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta)
      positions[i * 3 + 2] = r * Math.cos(phi)

      sizes[i] = 0.1 + Math.random() * 0.2
      phases[i] = Math.random() * Math.PI * 2
    }

    return { positions, sizes, phases }
  }, [count, radius])

  useFrame((state) => {
    if (!particlesRef.current) return

    const positions = particlesRef.current.geometry.attributes.position.array as Float32Array
    const time = state.clock.elapsedTime

    for (let i = 0; i < count; i++) {
      const phase = particles.phases[i]
      const drift = Math.sin(time + phase) * 0.5

      // Orbit slowly
      const angle = time * 0.2 + phase
      const r = radius * (0.5 + Math.sin(time * 0.5 + phase) * 0.2)

      positions[i * 3] = r * Math.cos(angle) * Math.sin(phase)
      positions[i * 3 + 1] = r * Math.sin(angle) * Math.cos(phase) + drift
      positions[i * 3 + 2] = r * Math.cos(angle) * Math.cos(phase)
    }

    particlesRef.current.geometry.attributes.position.needsUpdate = true
  })

  return (
    <points ref={particlesRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[particles.positions, 3]}
        />
      </bufferGeometry>
      <pointsMaterial
        color={color}
        size={0.3}
        transparent
        opacity={progress * 0.6}
        sizeAttenuation
      />
    </points>
  )
}

export default ExpandedNodeView
