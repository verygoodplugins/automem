/**
 * iPhone Hand Tracking Hook
 *
 * Receives hand landmark data from iPhone via WebSocket.
 * Converts to the same format as MediaPipe for seamless integration.
 *
 * Key advantage: Real LiDAR depth values instead of estimated depth.
 */

import { useEffect, useRef, useState, useCallback } from 'react'
import type { GestureState, HandLandmarks, PinchRay } from './useHandGestures'
import type { NormalizedLandmarkList } from '@mediapipe/hands'

// iPhone landmark names to MediaPipe indices
const LANDMARK_MAP: Record<string, number> = {
  'VNHLKJWRIST': 0,
  'VNHLKJTHUMBCMC': 1,
  'VNHLKJTHUMBMP': 2,
  'VNHLKJTHUMBIP': 3,
  'VNHLKJTHUMBTIP': 4,
  'VNHLKJINDEXMCP': 5,
  'VNHLKJINDEXPIP': 6,
  'VNHLKJINDEXDIP': 7,
  'VNHLKJINDEXTIP': 8,
  'VNHLKJMIDDLEMCP': 9,
  'VNHLKJMIDDLEPIP': 10,
  'VNHLKJMIDDLEDIP': 11,
  'VNHLKJMIDDLETIP': 12,
  'VNHLKJRINGMCP': 13,
  'VNHLKJRINGPIP': 14,
  'VNHLKJRINGDIP': 15,
  'VNHLKJRINGTIP': 16,
  'VNHLKJLITTLEMCP': 17,
  'VNHLKJLITTLEPIP': 18,
  'VNHLKJLITTLEDIP': 19,
  'VNHLKJLITTLETIP': 20,
}

interface IPhoneLandmark {
  x: number
  y: number
  z: number
}

interface IPhoneHandPose {
  handedness: 'left' | 'right'
  landmarks: Record<string, IPhoneLandmark>
  confidence: number
  timestamp: number
  hasLiDARDepth: boolean
}

interface IPhoneMessage {
  type: string
  hands: IPhoneHandPose[]
  frameTimestamp: number
}

interface UseIPhoneHandTrackingOptions {
  /** WebSocket URL to connect to (e.g., ws://192.168.1.100:8765) */
  serverUrl?: string
  /** Enable/disable the connection */
  enabled?: boolean
  /** Callback when gesture state updates */
  onGestureChange?: (state: GestureState) => void
}

// Smoothing for stability
const SMOOTHING = 0.3

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

function distance3D(a: IPhoneLandmark, b: IPhoneLandmark): number {
  const dx = a.x - b.x
  const dy = a.y - b.y
  const dz = a.z - b.z
  return Math.sqrt(dx * dx + dy * dy + dz * dz)
}

// Calculate pinch strength from iPhone landmarks
function calculatePinchStrength(landmarks: Record<string, IPhoneLandmark>): number {
  const thumbTip = landmarks['VNHLKJTHUMBTIP']
  const indexTip = landmarks['VNHLKJINDEXTIP']
  if (!thumbTip || !indexTip) return 0

  const dist = distance3D(thumbTip, indexTip)
  // iPhone normalized coords: ~0.02 pinched, ~0.15 open
  return Math.max(0, Math.min(1, 1 - (dist - 0.02) / 0.13))
}

// Calculate pinch ray from iPhone landmarks (with REAL depth!)
function calculatePinchRay(landmarks: Record<string, IPhoneLandmark>, hasLiDAR: boolean): PinchRay {
  const thumbTip = landmarks['VNHLKJTHUMBTIP']
  const indexTip = landmarks['VNHLKJINDEXTIP']
  const wrist = landmarks['VNHLKJWRIST']

  if (!thumbTip || !indexTip || !wrist) {
    return { origin: { x: 0.5, y: 0.5, z: 0 }, direction: { x: 0, y: 0, z: -1 }, isValid: false, strength: 0 }
  }

  // Origin: midpoint between thumb and index tips
  const origin = {
    x: (thumbTip.x + indexTip.x) / 2,
    y: (thumbTip.y + indexTip.y) / 2,
    // Use REAL depth from LiDAR if available!
    z: hasLiDAR ? (thumbTip.z + indexTip.z) / 2 : 0,
  }

  // Direction: from wrist through pinch point
  const rawDir = {
    x: origin.x - wrist.x,
    y: origin.y - wrist.y,
    z: hasLiDAR ? (origin.z - wrist.z) : -0.5, // Default forward if no LiDAR
  }

  const length = Math.sqrt(rawDir.x * rawDir.x + rawDir.y * rawDir.y + rawDir.z * rawDir.z)
  const direction = length > 0
    ? { x: rawDir.x / length, y: rawDir.y / length, z: rawDir.z / length }
    : { x: 0, y: 0, z: -1 }

  const strength = calculatePinchStrength(landmarks)
  const isValid = strength > 0.5

  return { origin, direction, isValid, strength }
}

// Convert iPhone landmarks to MediaPipe-compatible format
function convertToMediaPipeLandmarks(landmarks: Record<string, IPhoneLandmark>): NormalizedLandmarkList {
  const result: NormalizedLandmarkList = []

  // Initialize all 21 landmarks with defaults
  for (let i = 0; i < 21; i++) {
    result.push({ x: 0.5, y: 0.5, z: 0, visibility: 0 })
  }

  // Map iPhone landmarks to MediaPipe indices
  for (const [name, idx] of Object.entries(LANDMARK_MAP)) {
    const lm = landmarks[name]
    if (lm) {
      result[idx] = {
        x: lm.x,
        y: lm.y,
        z: lm.z,
        visibility: 1,
      }
    }
  }

  return result
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
  leftPinchRay: null,
  rightPinchRay: null,
  activePinchRay: null,
  zoomDelta: 0,
  rotateDelta: 0,
  panDelta: { x: 0, y: 0 },
}

export function useIPhoneHandTracking(options: UseIPhoneHandTrackingOptions = {}) {
  const {
    serverUrl = 'ws://localhost:8765',
    enabled = true,
    onGestureChange
  } = options

  const [gestureState, setGestureState] = useState<GestureState>(DEFAULT_STATE)
  const [isConnected, setIsConnected] = useState(false)
  const [fps, setFps] = useState(0)
  const [hasLiDAR, setHasLiDAR] = useState(false)

  const wsRef = useRef<WebSocket | null>(null)
  const prevStateRef = useRef<GestureState>(DEFAULT_STATE)
  const frameCountRef = useRef(0)
  const lastFpsTimeRef = useRef(Date.now())
  const reconnectTimeoutRef = useRef<number>()

  // Process incoming hand data
  const processMessage = useCallback((data: IPhoneMessage) => {
    frameCountRef.current++
    const now = Date.now()
    if (now - lastFpsTimeRef.current >= 1000) {
      setFps(frameCountRef.current)
      frameCountRef.current = 0
      lastFpsTimeRef.current = now
    }

    const prev = prevStateRef.current
    const newState: GestureState = { ...DEFAULT_STATE, isTracking: true }

    newState.handsDetected = data.hands.length

    if (data.hands.length > 0) {
      setHasLiDAR(data.hands[0].hasLiDARDepth)
    }

    // Process each hand
    for (const hand of data.hands) {
      const landmarks = convertToMediaPipeLandmarks(hand.landmarks)
      const handData: HandLandmarks = {
        landmarks,
        worldLandmarks: landmarks,
        handedness: hand.handedness === 'left' ? 'Left' : 'Right',
      }

      const pinchRay = calculatePinchRay(hand.landmarks, hand.hasLiDARDepth)

      if (hand.handedness === 'left') {
        newState.leftHand = handData
        newState.leftPinchRay = pinchRay
      } else {
        newState.rightHand = handData
        newState.rightPinchRay = pinchRay
      }
    }

    // Determine active pinch ray
    if (newState.rightPinchRay?.isValid) {
      newState.activePinchRay = newState.rightPinchRay
    } else if (newState.leftPinchRay?.isValid) {
      newState.activePinchRay = newState.leftPinchRay
    }

    // Two-hand calculations
    if (newState.leftHand && newState.rightHand) {
      const leftWrist = newState.leftHand.landmarks[0]
      const rightWrist = newState.rightHand.landmarks[0]

      // Distance between hands
      const dx = leftWrist.x - rightWrist.x
      const dy = leftWrist.y - rightWrist.y
      const rawDistance = Math.sqrt(dx * dx + dy * dy)
      newState.twoHandDistance = lerp(prev.twoHandDistance, rawDistance, 1 - SMOOTHING)

      // Rotation
      const rawRotation = Math.atan2(rightWrist.y - leftWrist.y, rightWrist.x - leftWrist.x)
      newState.twoHandRotation = lerp(prev.twoHandRotation, rawRotation, 1 - SMOOTHING)

      // Center
      newState.twoHandCenter = {
        x: lerp(prev.twoHandCenter.x, (leftWrist.x + rightWrist.x) / 2, 1 - SMOOTHING),
        y: lerp(prev.twoHandCenter.y, (leftWrist.y + rightWrist.y) / 2, 1 - SMOOTHING),
      }

      // Deltas
      newState.zoomDelta = (newState.twoHandDistance - prev.twoHandDistance) * 5
      newState.rotateDelta = newState.twoHandRotation - prev.twoHandRotation
    }

    // Pinch strength (smoothed)
    const primaryHand = newState.rightHand || newState.leftHand
    if (primaryHand) {
      const strength = calculatePinchStrength(
        Object.fromEntries(
          Object.entries(LANDMARK_MAP).map(([name, idx]) => [
            name,
            { x: primaryHand.landmarks[idx].x, y: primaryHand.landmarks[idx].y, z: primaryHand.landmarks[idx].z || 0 }
          ])
        )
      )
      newState.pinchStrength = lerp(prev.pinchStrength, strength, 1 - SMOOTHING)
    }

    prevStateRef.current = newState
    setGestureState(newState)
    onGestureChange?.(newState)
  }, [onGestureChange])

  // WebSocket connection
  useEffect(() => {
    if (!enabled) {
      wsRef.current?.close()
      setIsConnected(false)
      return
    }

    const connect = () => {
      try {
        const ws = new WebSocket(serverUrl)
        wsRef.current = ws

        ws.onopen = () => {
          console.log('📱 Connected to iPhone hand tracking')
          setIsConnected(true)
        }

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data) as IPhoneMessage
            if (data.type === 'hand_tracking') {
              processMessage(data)
            }
          } catch (e) {
            console.error('Parse error:', e)
          }
        }

        ws.onclose = () => {
          console.log('📱 Disconnected from iPhone')
          setIsConnected(false)
          setGestureState(DEFAULT_STATE)

          // Reconnect after delay
          if (enabled) {
            reconnectTimeoutRef.current = window.setTimeout(connect, 2000)
          }
        }

        ws.onerror = (err) => {
          console.error('WebSocket error:', err)
        }
      } catch (e) {
        console.error('Connection error:', e)
        if (enabled) {
          reconnectTimeoutRef.current = window.setTimeout(connect, 2000)
        }
      }
    }

    connect()

    return () => {
      clearTimeout(reconnectTimeoutRef.current)
      wsRef.current?.close()
    }
  }, [enabled, serverUrl, processMessage])

  return {
    gestureState,
    isConnected,
    fps,
    hasLiDAR,
    isEnabled: enabled && isConnected,
  }
}

export default useIPhoneHandTracking
