/**
 * Stable Pointer Ray Hook
 *
 * Implements Meta Quest-style pointer ray with:
 * - Estimated arm model (shoulder → elbow → wrist → pinch)
 * - Virtual pivot point behind wrist for stability
 * - One Euro Filter for velocity-adaptive smoothing
 * - Ray-sphere intersection for node hit detection
 *
 * The key insight: humans point with their forearm, not their hand.
 * Small hand tremors cause huge angular changes if you pivot at the wrist.
 * By estimating the elbow and placing the pivot further back, we reduce jitter.
 */

import { useRef, useCallback, useMemo } from 'react'
import { PointerRayFilter, type PointerRay } from '../lib/OneEuroFilter'
import type { NormalizedLandmarkList } from '@mediapipe/hands'

// MediaPipe landmark indices
const WRIST = 0
const THUMB_TIP = 4
const THUMB_CMC = 1 // Base of thumb
const INDEX_TIP = 8
const INDEX_MCP = 5 // Knuckle
const MIDDLE_MCP = 9
const RING_MCP = 13
const PINKY_MCP = 17

export interface Vec3 {
  x: number
  y: number
  z: number
}

export interface StableRay extends PointerRay {
  /** Pinch strength 0-1 */
  pinchStrength: number
  /** Is ray valid for interaction (pinch > threshold) */
  isActive: boolean
  /** Confidence in the ray direction */
  confidence: number
  /** The pinch point (thumb-index midpoint) in normalized coords */
  pinchPoint: Vec3
  /** Screen intersection point (where laser hits the screen plane) */
  screenHit: { x: number; y: number } | null
  /** Estimated arm pose for visualization */
  armPose: ArmPose
}

export interface ArmPose {
  shoulder: Vec3
  elbow: Vec3
  wrist: Vec3
  pinchPoint: Vec3
}

export interface NodeHit {
  nodeId: string
  distance: number
  point: Vec3
}

interface UseStablePointerRayOptions {
  /** Handedness - affects arm model */
  handedness: 'left' | 'right'
  /** Pinch threshold to activate ray (0-1) */
  pinchThreshold?: number
  /** Release threshold (should be lower than pinch for hysteresis) */
  releaseThreshold?: number
  /** How far behind wrist to place virtual pivot (normalized units) */
  pivotDistance?: number
}

// Vector math utilities
function sub(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z }
}

function add(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z }
}

function scale(v: Vec3, s: number): Vec3 {
  return { x: v.x * s, y: v.y * s, z: v.z * s }
}

function length(v: Vec3): number {
  return Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
}

function normalize(v: Vec3): Vec3 {
  const len = length(v)
  return len > 0 ? scale(v, 1 / len) : { x: 0, y: 0, z: -1 }
}

function lerp3(a: Vec3, b: Vec3, t: number): Vec3 {
  return {
    x: a.x + (b.x - a.x) * t,
    y: a.y + (b.y - a.y) * t,
    z: a.z + (b.z - a.z) * t,
  }
}

function distance(a: Vec3, b: Vec3): number {
  return length(sub(a, b))
}

/**
 * Estimate pinch strength from thumb-index distance
 * Returns 0 (open) to 1 (fully pinched)
 */
function calculatePinchStrength(landmarks: NormalizedLandmarkList): number {
  const thumbTip = landmarks[THUMB_TIP]
  const indexTip = landmarks[INDEX_TIP]
  const dist = distance(
    { x: thumbTip.x, y: thumbTip.y, z: thumbTip.z || 0 },
    { x: indexTip.x, y: indexTip.y, z: indexTip.z || 0 }
  )
  // Typical range: 0.02 (pinched) to 0.15 (open)
  return Math.max(0, Math.min(1, 1 - (dist - 0.02) / 0.13))
}

/**
 * Estimate the arm pose from hand landmarks
 * Since we can only see the hand, we infer shoulder/elbow positions
 */
function estimateArmPose(landmarks: NormalizedLandmarkList, handedness: 'left' | 'right'): ArmPose {
  const wristLm = landmarks[WRIST]
  const thumbTip = landmarks[THUMB_TIP]
  const indexTip = landmarks[INDEX_TIP]

  const wrist: Vec3 = {
    x: wristLm.x,
    y: wristLm.y,
    z: wristLm.z || 0,
  }

  const pinchPoint: Vec3 = {
    x: (thumbTip.x + indexTip.x) / 2,
    y: (thumbTip.y + indexTip.y) / 2,
    z: ((thumbTip.z || 0) + (indexTip.z || 0)) / 2,
  }

  // Calculate palm orientation from MCP joints
  const indexMcp = landmarks[INDEX_MCP]
  const pinkyMcp = landmarks[PINKY_MCP]
  const palmWidth = distance(
    { x: indexMcp.x, y: indexMcp.y, z: indexMcp.z || 0 },
    { x: pinkyMcp.x, y: pinkyMcp.y, z: pinkyMcp.z || 0 }
  )

  // Estimate shoulder position - fixed relative to screen
  // Shoulder is off-screen, on the same side as the hand
  const isRight = handedness === 'right'
  const shoulder: Vec3 = {
    x: isRight ? 1.4 : -0.4, // Off screen
    y: 1.5, // Below screen
    z: 0.6, // Further from camera
  }

  // Estimate elbow using anatomical constraints
  // Elbow is roughly 35% of the way from shoulder to wrist
  // with some lateral offset (natural arm bend)
  const shoulderToWrist = sub(wrist, shoulder)
  const forearmRatio = 0.35
  const lateralOffset = isRight ? 0.12 : -0.12 // Natural arm bend outward

  // Hand depth affects elbow estimation
  // Hand closer to camera = elbow more bent (closer to body)
  const depthFactor = Math.max(0, 0.5 - (wrist.z || 0))

  const elbow: Vec3 = {
    x: shoulder.x + shoulderToWrist.x * forearmRatio + lateralOffset * (1 + depthFactor),
    y: shoulder.y + shoulderToWrist.y * forearmRatio - 0.08,
    z: shoulder.z + shoulderToWrist.z * forearmRatio + depthFactor * 0.1,
  }

  return { shoulder, elbow, wrist, pinchPoint }
}

/**
 * Calculate stable pointer ray using the arm model
 */
function calculateStableRay(
  armPose: ArmPose,
  pivotDistance: number
): PointerRay {
  const { elbow, wrist, pinchPoint } = armPose

  // Forearm direction (elbow → wrist)
  const forearmDir = normalize(sub(wrist, elbow))

  // Virtual pivot: offset behind wrist along forearm
  // This is the key to stability - small hand movements
  // cause smaller angular changes when pivoting from further back
  const pivot = sub(wrist, scale(forearmDir, pivotDistance))

  // Ray direction: from virtual pivot through pinch point
  const direction = normalize(sub(pinchPoint, pivot))

  return {
    origin: pivot,
    direction,
  }
}

/**
 * Calculate where the ray intersects the screen plane (z=0)
 */
function calculateScreenHit(ray: PointerRay): { x: number; y: number } | null {
  const { origin, direction } = ray

  // Avoid division by zero
  if (Math.abs(direction.z) < 0.0001) return null

  // t = -origin.z / direction.z (intersection with z=0 plane)
  const t = -origin.z / direction.z

  // Only forward intersections
  if (t < 0) return null

  return {
    x: origin.x + direction.x * t,
    y: origin.y + direction.y * t,
  }
}

/**
 * Estimate confidence in the ray direction
 * Based on hand visibility and pose stability
 */
function calculateConfidence(landmarks: NormalizedLandmarkList): number {
  // Check visibility of key landmarks
  const wrist = landmarks[WRIST]
  const thumbTip = landmarks[THUMB_TIP]
  const indexTip = landmarks[INDEX_TIP]

  // Visibility is 0-1 if present, otherwise assume low
  const wristVis = (wrist as any).visibility ?? 0.5
  const thumbVis = (thumbTip as any).visibility ?? 0.5
  const indexVis = (indexTip as any).visibility ?? 0.5

  // Check if hand is in reasonable pose (not twisted weirdly)
  const indexMcp = landmarks[INDEX_MCP]
  const middleMcp = landmarks[MIDDLE_MCP]

  // Palm should face camera - MCPs should be above wrist
  const palmFacing = wrist.y > indexMcp.y && wrist.y > middleMcp.y

  const baseConfidence = (wristVis + thumbVis + indexVis) / 3
  const poseBonus = palmFacing ? 0.2 : 0

  return Math.min(1, baseConfidence + poseBonus)
}

export function useStablePointerRay(options: UseStablePointerRayOptions) {
  const {
    handedness,
    pinchThreshold = 0.6,
    releaseThreshold = 0.35,
    pivotDistance = 0.12,
  } = options

  // Filter for smoothing
  const rayFilterRef = useRef<PointerRayFilter>(new PointerRayFilter())

  // Track previous active state for hysteresis
  const wasActiveRef = useRef(false)

  // Process hand landmarks and return stable ray
  const processLandmarks = useCallback(
    (landmarks: NormalizedLandmarkList | null, timestamp: number): StableRay | null => {
      if (!landmarks) {
        rayFilterRef.current.reset()
        wasActiveRef.current = false
        return null
      }

      // Calculate pinch strength
      const pinchStrength = calculatePinchStrength(landmarks)

      // Hysteresis: different thresholds for activating vs deactivating
      const threshold = wasActiveRef.current ? releaseThreshold : pinchThreshold
      const isActive = pinchStrength >= threshold
      wasActiveRef.current = isActive

      // Estimate arm pose
      const armPose = estimateArmPose(landmarks, handedness)

      // Calculate raw ray
      const rawRay = calculateStableRay(armPose, pivotDistance)

      // Apply One Euro Filter for stability
      const filteredRay = rayFilterRef.current.filter(rawRay, timestamp)

      // Calculate screen intersection
      const screenHit = calculateScreenHit(filteredRay)

      // Estimate confidence
      const confidence = calculateConfidence(landmarks)

      return {
        ...filteredRay,
        pinchStrength,
        isActive,
        confidence,
        pinchPoint: armPose.pinchPoint,
        screenHit,
        armPose,
      }
    },
    [handedness, pinchThreshold, releaseThreshold, pivotDistance]
  )

  // Reset filter state
  const reset = useCallback(() => {
    rayFilterRef.current.reset()
    wasActiveRef.current = false
  }, [])

  return { processLandmarks, reset }
}

/**
 * Ray-Sphere Intersection
 *
 * Tests if a ray intersects a sphere and returns hit info.
 * Used for detecting when the laser points at a node.
 */
export function rayIntersectSphere(
  ray: PointerRay,
  sphereCenter: Vec3,
  sphereRadius: number
): { hit: boolean; distance: number; point: Vec3 } | null {
  const oc = sub(ray.origin, sphereCenter)

  const a = ray.direction.x * ray.direction.x +
            ray.direction.y * ray.direction.y +
            ray.direction.z * ray.direction.z
  const b = 2 * (oc.x * ray.direction.x + oc.y * ray.direction.y + oc.z * ray.direction.z)
  const c = oc.x * oc.x + oc.y * oc.y + oc.z * oc.z - sphereRadius * sphereRadius

  const discriminant = b * b - 4 * a * c

  if (discriminant < 0) return null

  // Find nearest intersection
  const t = (-b - Math.sqrt(discriminant)) / (2 * a)

  if (t < 0) return null

  const point: Vec3 = {
    x: ray.origin.x + ray.direction.x * t,
    y: ray.origin.y + ray.direction.y * t,
    z: ray.origin.z + ray.direction.z * t,
  }

  return { hit: true, distance: t, point }
}

/**
 * Test ray against multiple nodes and return closest hit
 */
export interface NodeSphere {
  id: string
  x: number
  y: number
  z: number
  radius: number
}

export function findNodeHit(
  ray: PointerRay,
  nodes: NodeSphere[],
  maxDistance: number = 1000
): NodeHit | null {
  let closestHit: NodeHit | null = null

  for (const node of nodes) {
    const result = rayIntersectSphere(
      ray,
      { x: node.x, y: node.y, z: node.z },
      node.radius
    )

    if (result && result.distance < maxDistance) {
      if (!closestHit || result.distance < closestHit.distance) {
        closestHit = {
          nodeId: node.id,
          distance: result.distance,
          point: result.point,
        }
      }
    }
  }

  return closestHit
}

export default useStablePointerRay
