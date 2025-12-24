/**
 * useHandCursor - Simplified Hand Cursor Hook
 *
 * Treats hand tracking like a 3D touchpad:
 * - Index fingertip position on screen = cursor position
 * - Pinch (thumb+index close) = click
 * - No complex arm models, no acquisition delays, no lock states
 *
 * Coordinate System:
 * - Output is in NDC (Normalized Device Coordinates): x,y in range [-1, 1]
 * - X: -1 = left edge, +1 = right edge
 * - Y: -1 = bottom edge, +1 = top edge
 * - Origin (0,0) = screen center
 */

import { useState, useEffect, useRef } from 'react'
import type { GestureState, HandLandmarks } from './useHandGestures'

// Pinch detection thresholds (with hysteresis to prevent flickering)
const PINCH_DOWN_THRESHOLD = 0.70  // Requires intentional pinch to activate
const PINCH_UP_THRESHOLD = 0.45   // Lower threshold to release (hysteresis)

// Screen edge margin - positions within this margin are considered off-screen
const EDGE_MARGIN = 0.05  // 5% from edge

// Landmark indices
const INDEX_FINGERTIP = 8
const THUMB_TIP = 4

/**
 * Pinch state machine phases:
 * - idle: No pinch detected
 * - down: Just started pinching this frame (rising edge)
 * - held: Continuing to pinch
 * - up: Just released pinch this frame (falling edge)
 */
export type PinchPhase = 'idle' | 'down' | 'held' | 'up'

export interface HandCursorState {
  /** Whether the cursor is active (hand detected and visible) */
  isActive: boolean

  /** Screen position in NDC [-1, 1], null if no hand */
  screenPosition: { x: number; y: number } | null

  /** Current pinch state machine phase */
  pinchState: PinchPhase

  /** Raw pinch strength 0-1 */
  pinchStrength: number

  /** Which hand is controlling the cursor */
  activeHand: 'left' | 'right' | null

  /** Whether the cursor is near the screen edge */
  isOffScreen: boolean

  /** Depth value for visual feedback (0 = close/faint, 1 = far/bright) */
  normalizedDepth: number
}

interface UseHandCursorOptions {
  /** Enable/disable cursor tracking */
  enabled?: boolean
  /** Prefer left or right hand when both are present */
  preferredHand?: 'left' | 'right'
}

/**
 * Calculate pinch strength from thumb-index distance
 */
function calculatePinchStrength(hand: HandLandmarks): number {
  const thumb = hand.landmarks[THUMB_TIP]
  const index = hand.landmarks[INDEX_FINGERTIP]

  if (!thumb || !index) return 0

  // Distance in normalized coordinates (0-1 range each)
  const dx = thumb.x - index.x
  const dy = thumb.y - index.y
  const dz = (thumb.z || 0) - (index.z || 0)
  const distance = Math.sqrt(dx * dx + dy * dy + dz * dz)

  // Map distance to pinch strength
  // Pinched: ~0.02-0.04, Extended: ~0.15-0.25
  const minDist = 0.03  // Fully pinched
  const maxDist = 0.15  // Fully open
  const strength = 1 - Math.max(0, Math.min(1, (distance - minDist) / (maxDist - minDist)))

  return strength
}

/**
 * Convert hand landmark position to NDC screen coordinates
 */
function landmarkToNDC(hand: HandLandmarks): { x: number; y: number } {
  const fingertip = hand.landmarks[INDEX_FINGERTIP]
  if (!fingertip) return { x: 0, y: 0 }

  // Mirror X for selfie camera (webcam shows mirrored view)
  const mirroredX = 1 - fingertip.x

  // Convert to NDC: 0→-1, 0.5→0, 1→+1
  // Flip Y so that top of screen = +1, bottom = -1
  return {
    x: mirroredX * 2 - 1,
    y: (1 - fingertip.y) * 2 - 1,
  }
}

/**
 * Check if position is near screen edge (considered off-screen for UI purposes)
 */
function isNearEdge(ndc: { x: number; y: number }): boolean {
  const margin = EDGE_MARGIN * 2  // Convert to NDC range (-1 to 1)
  return (
    Math.abs(ndc.x) > 1 - margin ||
    Math.abs(ndc.y) > 1 - margin
  )
}

/**
 * Calculate normalized depth for visual feedback
 * Returns 0-1 where 0 = close to camera, 1 = far from camera
 */
function calculateNormalizedDepth(hand: HandLandmarks): number {
  const wristZ = hand.landmarks[0]?.z || 0

  // Detect if Z is in meters (LiDAR: 0.3-3.0m) or normalized (MediaPipe: -0.5 to +0.3)
  const isMeters = Math.abs(wristZ) > 0.5

  if (isMeters) {
    // LiDAR in meters: ~0.3m (close) to ~1.2m (far)
    return Math.max(0, Math.min(1, (wristZ - 0.3) / 0.9))
  } else {
    // MediaPipe: positive Z = closer, negative Z = farther
    // Range typically +0.15 (close) to -0.25 (far)
    return Math.max(0, Math.min(1, (0.15 - wristZ) / 0.4))
  }
}

/**
 * useHandCursor - Simplified hand cursor tracking
 *
 * @param gestureState - Current gesture state from useHandGestures
 * @param options - Configuration options
 * @returns HandCursorState with cursor position and pinch state
 */
export function useHandCursor(
  gestureState: GestureState,
  options: UseHandCursorOptions = {}
): HandCursorState {
  const { enabled = true, preferredHand = 'right' } = options

  // Track previous pinch state for state machine transitions
  const prevPinchStrengthRef = useRef(0)
  const pinchPhaseRef = useRef<PinchPhase>('idle')

  // Current cursor state
  const [cursorState, setCursorState] = useState<HandCursorState>({
    isActive: false,
    screenPosition: null,
    pinchState: 'idle',
    pinchStrength: 0,
    activeHand: null,
    isOffScreen: false,
    normalizedDepth: 0.5,
  })

  // Update cursor state based on gesture state
  useEffect(() => {
    if (!enabled || !gestureState.isTracking) {
      if (cursorState.isActive) {
        setCursorState({
          isActive: false,
          screenPosition: null,
          pinchState: 'idle',
          pinchStrength: 0,
          activeHand: null,
          isOffScreen: false,
          normalizedDepth: 0.5,
        })
        pinchPhaseRef.current = 'idle'
      }
      return
    }

    // Select active hand (prefer right, fall back to left, or use what's available)
    let activeHand: HandLandmarks | null = null
    let activeHandSide: 'left' | 'right' | null = null

    if (preferredHand === 'right') {
      if (gestureState.rightHand) {
        activeHand = gestureState.rightHand
        activeHandSide = 'right'
      } else if (gestureState.leftHand) {
        activeHand = gestureState.leftHand
        activeHandSide = 'left'
      }
    } else {
      if (gestureState.leftHand) {
        activeHand = gestureState.leftHand
        activeHandSide = 'left'
      } else if (gestureState.rightHand) {
        activeHand = gestureState.rightHand
        activeHandSide = 'right'
      }
    }

    if (!activeHand) {
      setCursorState({
        isActive: false,
        screenPosition: null,
        pinchState: 'idle',
        pinchStrength: 0,
        activeHand: null,
        isOffScreen: false,
        normalizedDepth: 0.5,
      })
      pinchPhaseRef.current = 'idle'
      return
    }

    // Calculate cursor position and pinch strength
    const screenPosition = landmarkToNDC(activeHand)
    const pinchStrength = calculatePinchStrength(activeHand)
    const isOffScreen = isNearEdge(screenPosition)
    const normalizedDepth = calculateNormalizedDepth(activeHand)

    // Pinch state machine with hysteresis
    const prevPhase = pinchPhaseRef.current
    let newPhase: PinchPhase

    switch (prevPhase) {
      case 'idle':
        // Transition to 'down' when pinch crosses threshold
        newPhase = pinchStrength >= PINCH_DOWN_THRESHOLD ? 'down' : 'idle'
        break
      case 'down':
        // Always transition to 'held' next frame (down is a single-frame event)
        newPhase = 'held'
        break
      case 'held':
        // Transition to 'up' when pinch drops below release threshold
        newPhase = pinchStrength < PINCH_UP_THRESHOLD ? 'up' : 'held'
        break
      case 'up':
        // Always transition to 'idle' next frame (up is a single-frame event)
        newPhase = 'idle'
        break
      default:
        newPhase = 'idle'
    }

    pinchPhaseRef.current = newPhase
    prevPinchStrengthRef.current = pinchStrength

    setCursorState({
      isActive: true,
      screenPosition,
      pinchState: newPhase,
      pinchStrength,
      activeHand: activeHandSide,
      isOffScreen,
      normalizedDepth,
    })
  }, [enabled, gestureState, preferredHand, cursorState.isActive])

  return cursorState
}

export default useHandCursor
