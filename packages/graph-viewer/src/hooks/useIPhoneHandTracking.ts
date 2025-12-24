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
// Vision framework uses abbreviated keys: VNHLK + finger letter + joint
// W=wrist, T=thumb, I=index, M=middle, R=ring, P=pinky (little)
const LANDMARK_MAP: Record<string, number> = {
  // Wrist
  'VNHLKWRI': 0,
  // Thumb (T)
  'VNHLKTCMC': 1,
  'VNHLKTMP': 2,
  'VNHLKTIP': 3,
  'VNHLKTTIP': 4,
  // Index (I)
  'VNHLKIMCP': 5,
  'VNHLKIPIP': 6,
  'VNHLKIDIP': 7,
  'VNHLKITIP': 8,
  // Middle (M)
  'VNHLKMMCP': 9,
  'VNHLKMPIP': 10,
  'VNHLKMDIP': 11,
  'VNHLKMTIP': 12,
  // Ring (R)
  'VNHLKRMCP': 13,
  'VNHLKRPIP': 14,
  'VNHLKRDIP': 15,
  'VNHLKRTIP': 16,
  // Little/Pinky (P)
  'VNHLKPMCP': 17,
  'VNHLKPPIP': 18,
  'VNHLKPDIP': 19,
  'VNHLKPTIP': 20,
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
  phonePort?: number
  webPort?: number
  phoneConnected?: boolean
  ips?: string[]
  lastHandFrameAt?: number | null
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
  const thumbTip = landmarks['VNHLKTTIP']
  const indexTip = landmarks['VNHLKITIP']
  if (!thumbTip || !indexTip) return 0

  const dist = distance3D(thumbTip, indexTip)
  // iPhone normalized coords: ~0.02 pinched, ~0.15 open
  return Math.max(0, Math.min(1, 1 - (dist - 0.02) / 0.13))
}

// Closed fist strength from fingertip-to-wrist distances
function calculateGrabStrength(landmarks: Record<string, IPhoneLandmark>): number {
  const wrist = landmarks['VNHLKWRI']
  if (!wrist) return 0

  const tips = ['VNHLKTTIP', 'VNHLKITIP', 'VNHLKMTIP', 'VNHLKRTIP', 'VNHLKPTIP']
    .map((k) => landmarks[k])
    .filter(Boolean) as IPhoneLandmark[]
  if (tips.length < 3) return 0

  const avg = tips.reduce((sum, t) => sum + distance3D(t, wrist), 0) / tips.length
  // Typical range (rough): ~0.08 fist .. ~0.25 open
  return Math.max(0, Math.min(1, 1 - (avg - 0.08) / 0.17))
}

// Calculate pinch ray from iPhone landmarks (with normalized depth)
function calculatePinchRay(landmarks: Record<string, IPhoneLandmark>, hasLiDAR: boolean): PinchRay {
  const thumbTip = landmarks['VNHLKTTIP']
  const indexTip = landmarks['VNHLKITIP']
  const wrist = landmarks['VNHLKWRI']

  if (!thumbTip || !indexTip || !wrist) {
    return { origin: { x: 0.5, y: 0.5, z: 0 }, direction: { x: 0, y: 0, z: -1 }, isValid: false, strength: 0 }
  }

  // Normalize depths for consistent scaling
  const thumbZ = normalizeLiDARDepth(thumbTip.z, hasLiDAR)
  const indexZ = normalizeLiDARDepth(indexTip.z, hasLiDAR)
  const wristZ = normalizeLiDARDepth(wrist.z, hasLiDAR)

  // Origin: midpoint between thumb and index tips
  const origin = {
    x: (thumbTip.x + indexTip.x) / 2,
    y: (thumbTip.y + indexTip.y) / 2,
    z: (thumbZ + indexZ) / 2,
  }

  // Direction: from wrist through pinch point
  const rawDir = {
    x: origin.x - wrist.x,
    y: origin.y - wrist.y,
    z: origin.z - wristZ || -0.1, // Small default forward if no difference
  }

  const length = Math.sqrt(rawDir.x * rawDir.x + rawDir.y * rawDir.y + rawDir.z * rawDir.z)
  const direction = length > 0
    ? { x: rawDir.x / length, y: rawDir.y / length, z: rawDir.z / length }
    : { x: 0, y: 0, z: -1 }

  const strength = calculatePinchStrength(landmarks)
  const isValid = strength > 0.5

  return { origin, direction, isValid, strength }
}

// Normalize LiDAR depth (meters) to MediaPipe-like relative depth
// LiDAR: 0.3m (close) to 3.0m (far) -> MediaPipe-like: 0.1 to -0.3
// Reference: ~1.0m is "neutral" -> 0
function normalizeLiDARDepth(depthMeters: number, hasLiDAR: boolean): number {
  if (!hasLiDAR || depthMeters === 0) return 0

  // Invert and scale: closer = positive, farther = negative
  // At 1.0m -> 0, at 0.5m -> 0.1, at 2.0m -> -0.2
  const normalized = (1.0 - depthMeters) * 0.2
  return Math.max(-0.5, Math.min(0.5, normalized))
}

// Convert iPhone landmarks to MediaPipe-compatible format
function convertToMediaPipeLandmarks(landmarks: Record<string, IPhoneLandmark>, hasLiDAR: boolean = false): NormalizedLandmarkList {
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
        // Normalize LiDAR depth to MediaPipe-like values
        z: normalizeLiDARDepth(lm.z, hasLiDAR),
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
    // Default to the local bridge's web client endpoint
    // (iPhone connects to :8765; browser/web-app should connect to :8766/ws)
    serverUrl = 'ws://localhost:8766/ws',
    enabled = true,
    onGestureChange
  } = options

  const [gestureState, setGestureState] = useState<GestureState>(DEFAULT_STATE)
  const [isConnected, setIsConnected] = useState(false)
  const [fps, setFps] = useState(0)
  const [hasLiDAR, setHasLiDAR] = useState(false)
  const [phoneConnected, setPhoneConnected] = useState(false)
  const [bridgeIps, setBridgeIps] = useState<string[]>([])
  const [phonePort, setPhonePort] = useState<number | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const prevStateRef = useRef<GestureState>(DEFAULT_STATE)
  const frameCountRef = useRef(0)
  const lastFpsTimeRef = useRef(Date.now())
  const reconnectTimeoutRef = useRef<number>()
  const messageCountRef = useRef(0)
  const hasLoggedLandmarksRef = useRef(false)

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
      // Debug: log the actual landmark keys from iPhone (once per connection)
      if (!hasLoggedLandmarksRef.current && Object.keys(hand.landmarks).length > 0) {
        hasLoggedLandmarksRef.current = true
        console.log('📍 iPhone landmark keys:', Object.keys(hand.landmarks))
        console.log('📍 Expected keys:', Object.keys(LANDMARK_MAP))
        const sampleEntry = Object.entries(hand.landmarks)[0]
        console.log('📍 Sample landmark:', sampleEntry?.[0], '→', sampleEntry?.[1])
      }

      const landmarks = convertToMediaPipeLandmarks(hand.landmarks, hand.hasLiDARDepth)
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
      const reconstructed = Object.fromEntries(
        Object.entries(LANDMARK_MAP).map(([name, idx]) => [
          name,
          { x: primaryHand.landmarks[idx].x, y: primaryHand.landmarks[idx].y, z: primaryHand.landmarks[idx].z || 0 },
        ])
      ) as Record<string, IPhoneLandmark>

      const pinch = calculatePinchStrength(reconstructed)
      const grab = calculateGrabStrength(reconstructed)

      newState.pinchStrength = lerp(prev.pinchStrength, pinch, 1 - SMOOTHING)
      newState.grabStrength = lerp(prev.grabStrength, grab, 1 - SMOOTHING)
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
          console.log('📱 Connected to iPhone hand tracking bridge')
          setIsConnected(true)
          messageCountRef.current = 0
          hasLoggedLandmarksRef.current = false
        }

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data) as IPhoneMessage
            messageCountRef.current++

            // Log first few messages and then periodically
            if (messageCountRef.current <= 3 || messageCountRef.current % 100 === 0) {
              console.log(`📨 Message #${messageCountRef.current}:`, data.type, data.hands?.length || 0, 'hands')
            }

            if (data.type === 'hand_tracking') {
              processMessage(data)
            } else if (data.type === 'bridge_status') {
              console.log('📡 Bridge status:', data)
              if (typeof data.phoneConnected === 'boolean') setPhoneConnected(data.phoneConnected)
              if (Array.isArray(data.ips)) setBridgeIps(data.ips)
              if (typeof data.phonePort === 'number') setPhonePort(data.phonePort)
            } else {
              // Debug: log unexpected message types
              console.log('📨 Unknown message type:', data.type, data)
            }
          } catch (e) {
            console.error('Parse error:', e, event.data)
          }
        }

        ws.onclose = () => {
          console.log('📱 Disconnected from iPhone')
          setIsConnected(false)
          setGestureState(DEFAULT_STATE)
          setPhoneConnected(false)

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
    phoneConnected,
    bridgeIps,
    phonePort,
    isEnabled: enabled && isConnected,
  }
}

export default useIPhoneHandTracking
