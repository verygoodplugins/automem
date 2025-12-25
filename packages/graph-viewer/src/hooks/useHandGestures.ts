/**
 * Hand Gesture Recognition Hook
 *
 * Uses MediaPipe Hands to detect hand gestures for controlling the 3D memory graph.
 * Supports Meta Quest-style gestures:
 * - Two-hand spread/pinch for zoom
 * - Two-hand rotation for orbit
 * - Single-hand point for hover
 * - Pinch to select
 */

import { useEffect, useRef, useState, useCallback } from 'react'
import { Hands, Results, NormalizedLandmarkList } from '@mediapipe/hands'
import { Camera } from '@mediapipe/camera_utils'

// Landmark indices
const WRIST = 0
const THUMB_TIP = 4
const INDEX_TIP = 8
const MIDDLE_TIP = 12
const RING_TIP = 16
const PINKY_TIP = 20
const INDEX_MCP = 5 // Base of index finger

export interface HandLandmarks {
  landmarks: NormalizedLandmarkList
  worldLandmarks: NormalizedLandmarkList
  handedness: 'Left' | 'Right'
}

export interface PinchRay {
  // Origin point (midpoint between thumb and index tips)
  origin: { x: number; y: number; z: number }
  // Direction vector (from wrist toward pinch point)
  direction: { x: number; y: number; z: number }
  // Is the ray valid for interaction?
  isValid: boolean
  // Current pinch strength (0-1)
  strength: number
}

export interface GestureState {
  // Are we tracking hands?
  isTracking: boolean
  handsDetected: number

  // Raw hand data
  leftHand: HandLandmarks | null
  rightHand: HandLandmarks | null

  // Computed gestures
  twoHandDistance: number // Distance between wrists (0-1 normalized)
  twoHandRotation: number // Angle in radians
  twoHandCenter: { x: number; y: number } // Center point between hands

  // Single hand gestures
  pointingHand: 'left' | 'right' | null
  pointDirection: { x: number; y: number } | null // Normalized screen coords
  pinchStrength: number // 0-1, how pinched is the pointing hand
  grabStrength: number // 0-1, how closed is the fist

  // Direct pinch selection point (midpoint between thumb tip and index tip)
  // Used for "pick the berry" selection - position this over a node and pinch
  pinchPoint: { x: number; y: number } | null // Normalized screen coords (0-1)

  // Pinch ray for laser pointer (Meta Quest style) - DEPRECATED, use pinchPoint instead
  leftPinchRay: PinchRay | null
  rightPinchRay: PinchRay | null
  activePinchRay: PinchRay | null // The one currently being used for interaction

  // Derived control signals
  zoomDelta: number // Positive = zoom in, negative = zoom out
  rotateDelta: number // Rotation change since last frame
  panDelta: { x: number; y: number } // Pan movement
}

interface UseHandGesturesOptions {
  enabled?: boolean
  smoothing?: number // 0-1, higher = smoother but laggier
  onGestureChange?: (state: GestureState) => void
}

const DEFAULT_STATE: GestureState = {
  isTracking: false,
  handsDetected: 0,
  leftHand: null,
  rightHand: null,
  twoHandDistance: 0.5,
  twoHandRotation: 0,
  twoHandCenter: { x: 0.5, y: 0.5 },
  pointingHand: null,
  pointDirection: null,
  pinchStrength: 0,
  grabStrength: 0,
  pinchPoint: null,
  leftPinchRay: null,
  rightPinchRay: null,
  activePinchRay: null,
  zoomDelta: 0,
  rotateDelta: 0,
  panDelta: { x: 0, y: 0 },
}

// Utility functions
function distance(a: { x: number; y: number; z?: number }, b: { x: number; y: number; z?: number }): number {
  const dx = a.x - b.x
  const dy = a.y - b.y
  const dz = (a.z || 0) - (b.z || 0)
  return Math.sqrt(dx * dx + dy * dy + dz * dz)
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

function lerpPoint(a: { x: number; y: number }, b: { x: number; y: number }, t: number): { x: number; y: number } {
  return { x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t) }
}

function normalizeAngle(angle: number): number {
  while (angle > Math.PI) angle -= 2 * Math.PI
  while (angle < -Math.PI) angle += 2 * Math.PI
  return angle
}

// Calculate pinch strength (0 = open, 1 = pinched)
function calculatePinchStrength(landmarks: NormalizedLandmarkList): number {
  const thumbTip = landmarks[THUMB_TIP]
  const indexTip = landmarks[INDEX_TIP]
  const dist = distance(thumbTip, indexTip)
  // Typical range: 0.02 (pinched) to 0.15 (open)
  return 1 - Math.min(1, Math.max(0, (dist - 0.02) / 0.13))
}

// Calculate grab strength (0 = open hand, 1 = fist)
function calculateGrabStrength(landmarks: NormalizedLandmarkList): number {
  const wrist = landmarks[WRIST]
  const fingertips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP].map(i => landmarks[i])
  const avgDist = fingertips.reduce((sum, tip) => sum + distance(tip, wrist), 0) / 5
  // Typical range: 0.08 (fist) to 0.25 (open)
  return 1 - Math.min(1, Math.max(0, (avgDist - 0.08) / 0.17))
}

// Check if hand is pointing (index extended, others curled)
function isPointing(landmarks: NormalizedLandmarkList): boolean {
  const indexExtended = landmarks[INDEX_TIP].y < landmarks[INDEX_MCP].y - 0.05
  const middleCurled = landmarks[MIDDLE_TIP].y > landmarks[12].y // MIDDLE_MCP
  const ringCurled = landmarks[RING_TIP].y > landmarks[16].y // RING_MCP
  const pinkyCurled = landmarks[PINKY_TIP].y > landmarks[20].y // PINKY_MCP
  return indexExtended && (middleCurled || ringCurled || pinkyCurled)
}

// Get pointing direction from index finger
function getPointDirection(landmarks: NormalizedLandmarkList): { x: number; y: number } {
  const indexTip = landmarks[INDEX_TIP]
  // Invert Y because screen coords are flipped
  return { x: indexTip.x, y: 1 - indexTip.y }
}

// Calculate pinch ray (Meta Quest style - ray from pinch midpoint)
function calculatePinchRay(landmarks: NormalizedLandmarkList): PinchRay {
  const thumbTip = landmarks[THUMB_TIP]
  const indexTip = landmarks[INDEX_TIP]
  const wrist = landmarks[WRIST]

  // Origin is the midpoint between thumb and index tips (Meta Quest PointerPose)
  const origin = {
    x: (thumbTip.x + indexTip.x) / 2,
    y: (thumbTip.y + indexTip.y) / 2,
    z: ((thumbTip.z || 0) + (indexTip.z || 0)) / 2,
  }

  // Direction vector from wrist toward the pinch point
  const rawDir = {
    x: origin.x - wrist.x,
    y: origin.y - wrist.y,
    z: (origin.z || 0) - (wrist.z || 0),
  }

  // Normalize the direction vector
  const length = Math.sqrt(rawDir.x * rawDir.x + rawDir.y * rawDir.y + rawDir.z * rawDir.z)
  const direction = length > 0
    ? { x: rawDir.x / length, y: rawDir.y / length, z: rawDir.z / length }
    : { x: 0, y: 0, z: -1 } // Default pointing into screen

  // Calculate pinch strength for this hand
  const pinchDist = distance(thumbTip, indexTip)
  const strength = 1 - Math.min(1, Math.max(0, (pinchDist - 0.02) / 0.13))

  // Ray is valid when pinch strength is above threshold
  const isValid = strength > 0.5

  return { origin, direction, isValid, strength }
}

export function useHandGestures(options: UseHandGesturesOptions = {}) {
  const { enabled = true, smoothing = 0.3, onGestureChange } = options

  const [gestureState, setGestureState] = useState<GestureState>(DEFAULT_STATE)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const handsRef = useRef<Hands | null>(null)
  const cameraRef = useRef<Camera | null>(null)
  const prevStateRef = useRef<GestureState>(DEFAULT_STATE)
  const isInitializedRef = useRef(false)
  const isCleaningUpRef = useRef(false) // Prevent send() after close()

  // Process MediaPipe results
  const onResults = useCallback((results: Results) => {
    const prev = prevStateRef.current
    const newState: GestureState = { ...DEFAULT_STATE }

    newState.isTracking = true
    newState.handsDetected = results.multiHandLandmarks?.length || 0

    if (results.multiHandLandmarks && results.multiHandedness) {
      // Sort hands into left/right
      for (let i = 0; i < results.multiHandLandmarks.length; i++) {
        const landmarks = results.multiHandLandmarks[i]
        const worldLandmarks = results.multiHandWorldLandmarks?.[i] || landmarks
        const handedness = results.multiHandedness[i].label as 'Left' | 'Right'

        const handData: HandLandmarks = {
          landmarks,
          worldLandmarks,
          // MediaPipe returns mirrored handedness, so flip it
          handedness: handedness === 'Left' ? 'Right' : 'Left',
        }

        if (handData.handedness === 'Left') {
          newState.leftHand = handData
          // Compute left pinch ray
          newState.leftPinchRay = calculatePinchRay(landmarks)
        } else {
          newState.rightHand = handData
          // Compute right pinch ray
          newState.rightPinchRay = calculatePinchRay(landmarks)
        }
      }

      // Determine active pinch ray (prefer right hand, use strongest pinch)
      if (newState.rightPinchRay && newState.rightPinchRay.isValid) {
        newState.activePinchRay = newState.rightPinchRay
      } else if (newState.leftPinchRay && newState.leftPinchRay.isValid) {
        newState.activePinchRay = newState.leftPinchRay
      }

      // Two-hand gestures
      if (newState.leftHand && newState.rightHand) {
        const leftWrist = newState.leftHand.landmarks[WRIST]
        const rightWrist = newState.rightHand.landmarks[WRIST]

        // Distance between hands (normalized 0-1)
        const rawDistance = distance(leftWrist, rightWrist)
        newState.twoHandDistance = lerp(prev.twoHandDistance, rawDistance, 1 - smoothing)

        // Rotation angle
        const rawRotation = Math.atan2(
          rightWrist.y - leftWrist.y,
          rightWrist.x - leftWrist.x
        )
        newState.twoHandRotation = lerp(prev.twoHandRotation, rawRotation, 1 - smoothing)

        // Center point
        const rawCenter = {
          x: (leftWrist.x + rightWrist.x) / 2,
          y: (leftWrist.y + rightWrist.y) / 2,
        }
        newState.twoHandCenter = lerpPoint(prev.twoHandCenter, rawCenter, 1 - smoothing)

        // Zoom delta (positive = spread apart = zoom in)
        newState.zoomDelta = (newState.twoHandDistance - prev.twoHandDistance) * 5

        // Rotation delta
        newState.rotateDelta = normalizeAngle(newState.twoHandRotation - prev.twoHandRotation)

        // Pan delta
        newState.panDelta = {
          x: (newState.twoHandCenter.x - prev.twoHandCenter.x) * 2,
          y: (newState.twoHandCenter.y - prev.twoHandCenter.y) * 2,
        }
      }

      // Single-hand gestures (prefer right hand for pointing)
      // Only do single-hand gestures if we don't have both hands
      if (newState.handsDetected === 1) {
        const pointingHandData = newState.rightHand || newState.leftHand
        if (pointingHandData) {
          const landmarks = pointingHandData.landmarks

          if (isPointing(landmarks)) {
            newState.pointingHand = pointingHandData.handedness === 'Right' ? 'right' : 'left'
            newState.pointDirection = getPointDirection(landmarks)
          }

          newState.pinchStrength = lerp(
            prev.pinchStrength,
            calculatePinchStrength(landmarks),
            1 - smoothing
          )
          newState.grabStrength = lerp(
            prev.grabStrength,
            calculateGrabStrength(landmarks),
            1 - smoothing
          )
        }
      }
    }

    prevStateRef.current = newState
    setGestureState(newState)
    onGestureChange?.(newState)
  }, [smoothing, onGestureChange])

  // Initialize MediaPipe
  useEffect(() => {
    if (!enabled || isInitializedRef.current) return

    isCleaningUpRef.current = false

    const initializeHands = async () => {
      // Create video element for camera
      const video = document.createElement('video')
      video.setAttribute('playsinline', '')
      video.style.display = 'none'
      document.body.appendChild(video)
      videoRef.current = video

      // Initialize MediaPipe Hands
      const hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
      })

      hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.5,
      })

      hands.onResults(onResults)
      handsRef.current = hands

      // Initialize camera
      const camera = new Camera(video, {
        onFrame: async () => {
          // Guard against calling send() after close()
          if (isCleaningUpRef.current) return
          if (handsRef.current && videoRef.current) {
            try {
              await handsRef.current.send({ image: videoRef.current })
            } catch (e) {
              // Ignore errors during cleanup (BindingError from deleted WASM object)
              if (!isCleaningUpRef.current) {
                console.warn('MediaPipe send error:', e)
              }
            }
          }
        },
        width: 640,
        height: 480,
      })

      cameraRef.current = camera
      await camera.start()
      isInitializedRef.current = true
    }

    initializeHands().catch(console.error)

    return () => {
      isCleaningUpRef.current = true
      cameraRef.current?.stop()
      cameraRef.current = null
      handsRef.current?.close()
      handsRef.current = null
      if (videoRef.current) {
        videoRef.current.remove()
        videoRef.current = null
      }
      isInitializedRef.current = false
    }
  }, [enabled, onResults])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cameraRef.current?.stop()
      handsRef.current?.close()
      if (videoRef.current) {
        videoRef.current.remove()
      }
    }
  }, [])

  return {
    gestureState,
    isEnabled: enabled && isInitializedRef.current,
    // Expose video ref for potential overlay rendering
    videoElement: videoRef.current,
  }
}

export default useHandGestures
