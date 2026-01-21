/**
 * Hand Grab + Pinch Control (no explicit locking)
 *
 * Goal: Make gestures feel direct.
 * - No acquire/lock workflow; either hand can grab/pinch immediately.
 * - Closed fist ("grab") manipulates the cloud via displacement deltas.
 * - Pinch drives direct selection (hover + click via hysteresis).
 *
 * Works with either MediaPipe or iPhone-fed landmarks because it only needs GestureState.
 */

import { useRef } from 'react'
import type { GestureState } from './useHandGestures'

type HandSide = 'left' | 'right'

export interface HandLockMetrics {
  /** 0..1: how open/spread the hand is */
  spread: number
  /** -1..1: palm facing camera confidence-ish (1 = facing camera) */
  palmFacing: number
  /** 0..1: pointing pose score (index extended, others curled) */
  point: number
  /** 0..1: pinch strength (thumb-index) */
  pinch: number
  /** Thumb-index midpoint in normalized screen coords (0..1) */
  pinchPoint: { x: number; y: number }
  /** 0..1: fist/grab strength (1 = closed fist) */
  grab: number
  /** depth signal (meters for iPhone LiDAR when available, otherwise MediaPipe-relative) */
  depth: number
  /** 0..1 heuristic confidence */
  confidence: number
}

export type HandLockState =
  | { mode: 'idle'; metrics: HandLockMetrics | null }
  | { mode: 'candidate'; hand: HandSide; metrics: HandLockMetrics; frames: number }
  | {
      mode: 'locked'
      hand: HandSide
      metrics: HandLockMetrics
      lockedAtMs: number
      /** pose at lock time */
      neutral: { x: number; y: number; depth: number }
      /** are we currently in grab mode */
      grabbed: boolean
      /** pose at grab start */
      grabAnchor?: { x: number; y: number; depth: number }
      /** when we last saw a usable hand */
      lastSeenMs: number
      /** is pinch currently activated (for selection) */
      pinchActivated: boolean
      /** frames that acquire pose has been held (for clear selection gesture) */
      clearHoldFrames: number
    }

/** Per-hand lock state for dual-hand tracking */
export type SingleHandLockState =
  | { mode: 'idle' }
  | { mode: 'candidate'; frames: number }
  | {
      mode: 'locked'
      lockedAtMs: number
      neutral: { x: number; y: number; depth: number }
      grabbed: boolean
      grabAnchor?: { x: number; y: number; depth: number }
      lastSeenMs: number
      pinchActivated: boolean
      clearHoldFrames: number
    }

export interface CloudControlDeltas {
  /** zoom velocity (positive -> zoom in, negative -> zoom out) */
  zoom: number
  /** Displacement-based pan: how much to offset from grab start position */
  panX: number
  panY: number
  panZ: number
  /** Is this the first frame of a grab? (used to capture initial world position) */
  grabStart: boolean
}

const DEFAULT_CONFIDENCE = 0.7

// Tunables (these matter a lot for UX)
const SPREAD_THRESHOLD = 0.78
const PALM_FACING_THRESHOLD = 0.72

const GRAB_ON_THRESHOLD = 0.72
const GRAB_OFF_THRESHOLD = 0.45

// Pinch thresholds for direct selection ("pick the berry")
const PINCH_ON_THRESHOLD = 0.85
const PINCH_OFF_THRESHOLD = 0.65

// Pinch mode (hover) thresholds: allow hover while "half pinched"
const PINCH_MODE_ON_THRESHOLD = 0.35
const PINCH_MODE_OFF_THRESHOLD = 0.25

// Two-hand navigation: both hands pinching
const BIMANUAL_PINCH_ON_THRESHOLD = 0.55
const BIMANUAL_PINCH_OFF_THRESHOLD = 0.35
const BIMANUAL_PINCH_GRACE_MS = 220

// Clear selection: hold open palm for ~0.5 seconds
const CLEAR_FRAMES_REQUIRED = 30

// Control sensitivity
const DEPTH_DEADZONE = 0.01

function clamp(v: number, min: number, max: number) {
  return Math.max(min, Math.min(max, v))
}

function length2(dx: number, dy: number) {
  return Math.sqrt(dx * dx + dy * dy)
}

function safeDiv(a: number, b: number, fallback = 0) {
  return b !== 0 ? a / b : fallback
}

function isLikelyMetersZ(z: unknown): z is number {
  return typeof z === 'number' && Number.isFinite(z) && z > 0.1 && z < 8
}

function depthTowardCameraScore(wristZ: number, tipZ: number, isMeters: boolean) {
  // Positive when tip is closer to camera than wrist.
  // iPhone meters: smaller = closer. MediaPipe-like normalized: more negative = closer.
  const delta = wristZ - tipZ
  const deadzone = isMeters ? 0.015 : 0.01
  const fullScale = isMeters ? 0.08 : 0.06
  return clamp(safeDiv(delta - deadzone, fullScale), 0, 1)
}

function fingerExtensionScore(
  wrist: { x: number; y: number },
  mcp: { x: number; y: number },
  tip: { x: number; y: number }
) {
  // Extension proxy: tip should be noticeably farther from wrist than MCP when finger is extended.
  const dTip = length2(tip.x - wrist.x, tip.y - wrist.y)
  const dMcp = length2(mcp.x - wrist.x, mcp.y - wrist.y)
  return clamp(safeDiv(dTip - dMcp - 0.02, 0.10), 0, 1)
}

/**
 * Compute simple metrics from landmarks (MediaPipe-style normalized 0..1)
 * Works for both sources because iPhone data is mapped into GestureState landmarks.
 */
function computeMetrics(state: GestureState, hand: HandSide): HandLockMetrics | null {
  const handData = hand === 'right' ? state.rightHand : state.leftHand
  if (!handData) return null

  const lm = handData.landmarks
  const wm = handData.worldLandmarks || lm
  // Required joints
  const wrist = lm[0]
  const indexMcp = lm[5]
  const middleMcp = lm[9]
  const ringMcp = lm[13]
  const pinkyMcp = lm[17]

  // Fingertips
  const thumbTip = lm[4]
  const indexTip = lm[8]
  const middleTip = lm[12]
  const ringTip = lm[16]
  const pinkyTip = lm[20]

  // Spread: average fingertip distance from palm center proxy (middle MCP)
  const palmCx = middleMcp.x
  const palmCy = middleMcp.y
  const d1 = length2(indexTip.x - palmCx, indexTip.y - palmCy)
  const d2 = length2(middleTip.x - palmCx, middleTip.y - palmCy)
  const d3 = length2(ringTip.x - palmCx, ringTip.y - palmCy)
  const d4 = length2(pinkyTip.x - palmCx, pinkyTip.y - palmCy)
  const avg = (d1 + d2 + d3 + d4) / 4
  // Normalize: typical spread-ish values ~0.08..0.22 depending on distance/FOV
  const spread = clamp(safeDiv(avg - 0.06, 0.16), 0, 1)

  // Palm facing heuristic:
  // In image space, if wrist is "below" MCPs, palm likely faces camera.
  // (This is crude but works for the acquisition gesture.)
  const palmFacing = clamp(safeDiv((wrist.y - (indexMcp.y + middleMcp.y) / 2) - 0.02, 0.12), 0, 1) * 2 - 1

  // Pinch (thumb-index)
  const pinchRay = hand === 'right' ? state.rightPinchRay : state.leftPinchRay
  const pinchDist = length2(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y)
  const pinch2d = clamp(1 - safeDiv(pinchDist - 0.02, 0.13), 0, 1)
  const pinch = clamp((pinchRay?.strength ?? pinch2d) as number, 0, 1)
  const pinchPoint = {
    x: (thumbTip.x + indexTip.x) / 2,
    y: (thumbTip.y + indexTip.y) / 2,
  }

  // Pointing pose score:
  // index extended while the other 3 fingers are relatively curled.
  // NOTE: We don't require palm facing camera - natural pointing works from any angle
  const idxExt = fingerExtensionScore(wrist, indexMcp, indexTip)
  const midExt = fingerExtensionScore(wrist, middleMcp, middleTip)
  const ringExt = fingerExtensionScore(wrist, ringMcp, ringTip)
  const pinkyExt = fingerExtensionScore(wrist, pinkyMcp, pinkyTip)
  const others = clamp((midExt + ringExt + pinkyExt) / 3, 0, 1)
  // Point score: index extended (idxExt high) AND other fingers curled (others low)
  // If index is extended more than others, it's pointing
  const pointRaw = clamp((idxExt - others) * 2, 0, 1)
  const point2d = idxExt > 0.5 && others < 0.5 ? pointRaw : 0

  // Depth-based pointing:
  // When "pointing at the screen" the silhouette can still look fist-like in 2D.
  // LiDAR gives a strong signal: index tip moves toward the camera while the other fingers stay back/curled.
  const wristWz = (wm[0]?.z ?? wrist.z ?? 0) as number
  const indexTipWz = (wm[8]?.z ?? indexTip.z ?? 0) as number
  const middleTipWz = (wm[12]?.z ?? middleTip.z ?? 0) as number
  const ringTipWz = (wm[16]?.z ?? ringTip.z ?? 0) as number
  const pinkyTipWz = (wm[20]?.z ?? pinkyTip.z ?? 0) as number

  const isMeters = isLikelyMetersZ(wristWz)
  const idxToward = depthTowardCameraScore(wristWz, indexTipWz, isMeters)
  const midToward = depthTowardCameraScore(wristWz, middleTipWz, isMeters)
  const ringToward = depthTowardCameraScore(wristWz, ringTipWz, isMeters)
  const pinkyToward = depthTowardCameraScore(wristWz, pinkyTipWz, isMeters)
  const othersToward = clamp((midToward + ringToward + pinkyToward) / 3, 0, 1)
  const pointDepth = clamp(idxToward * (1 - othersToward * 0.9), 0, 1)

  const point = clamp(Math.max(point2d, pointDepth), 0, 1)

  // Grab: closed fist = ALL fingers curled including index
  // Exclude index from grab calculation to distinguish from pointing
  // Prefer per-hand heuristic. (state.grabStrength is only reliable in the single-hand path.)
  const dw1 = length2(indexTip.x - wrist.x, indexTip.y - wrist.y)
  const dw2 = length2(middleTip.x - wrist.x, middleTip.y - wrist.y)
  const dw3 = length2(ringTip.x - wrist.x, ringTip.y - wrist.y)
  const dw4 = length2(pinkyTip.x - wrist.x, pinkyTip.y - wrist.y)
  const avgDw = (dw1 + dw2 + dw3 + dw4) / 4
  const grab2d = clamp(1 - safeDiv(avgDw - 0.08, 0.17), 0, 1)
  let grab =
    state.handsDetected === 1 && typeof state.grabStrength === 'number' && state.grabStrength > 0
      ? clamp(state.grabStrength, 0, 1)
      : grab2d

  // Mutual exclusion: if pointing, suppress grab
  if (point > 0.55) grab = 0

  // Depth: prefer pinch ray origin z when present (iPhone LiDAR mapped into landmarks z)
  const depth = (pinchRay?.origin.z ?? wrist.z ?? 0) as number

  // Confidence: use landmark visibility if present; else assume ok
  const vis = (wrist as any).visibility
  const confidence = typeof vis === 'number' ? clamp(vis, 0, 1) : DEFAULT_CONFIDENCE

  return { spread, palmFacing, point, pinch, pinchPoint, grab, depth, confidence }
}

function isAcquirePose(m: HandLockMetrics) {
  // Additional gates: prevent accidental acquire from "pointing at screen" or semi-closed poses.
  return (
    m.spread >= SPREAD_THRESHOLD &&
    m.palmFacing >= PALM_FACING_THRESHOLD &&
    m.grab <= 0.25 &&
    m.pinch <= 0.25 &&
    m.point <= 0.25 &&
    m.confidence >= 0.6
  )
}

export interface HandLockResult {
  lock: HandLockState
  deltas: CloudControlDeltas
  /** True when user holds acquire pose to clear selection */
  clearRequested: boolean
  /** True when both hands are pinching (for bimanual manipulation) */
  bimanualPinch: boolean
  /** True when both hands are locked (acquired) */
  bothHandsLocked: boolean
  /** Metrics for left hand (for two-hand gestures) */
  leftMetrics: HandLockMetrics | null
  /** Metrics for right hand (for two-hand gestures) */
  rightMetrics: HandLockMetrics | null
  /** Per-hand lock states for visual feedback */
  leftLock: SingleHandLockState
  rightLock: SingleHandLockState
}

export function useHandLockAndGrab(state: GestureState, enabled: boolean): HandLockResult {
  const lockRef = useRef<HandLockState>({ mode: 'idle', metrics: null })
  const bimanualPinchRef = useRef(false)
  const bimanualLastGoodMsRef = useRef(0)
  const grabRef = useRef<{ left: boolean; right: boolean }>({ left: false, right: false })
  const pinchModeRef = useRef<{ left: boolean; right: boolean }>({ left: false, right: false })
  const pinchActivatedRef = useRef<{ left: boolean; right: boolean }>({ left: false, right: false })
  const clearHoldFramesRef = useRef(0)

  // Per-hand lock states for dual-hand tracking
  const leftLockRef = useRef<SingleHandLockState>({ mode: 'idle' })
  const rightLockRef = useRef<SingleHandLockState>({ mode: 'idle' })

  const nowMs = performance.now()

  const right = enabled ? computeMetrics(state, 'right') : null
  const left = enabled ? computeMetrics(state, 'left') : null

  // Bimanual pinch: both hands pinching, with hysteresis + short grace to tolerate brief signal drops.
  let bimanualPinch = false
  if (enabled && left && right) {
    const bothOn = left.pinch >= BIMANUAL_PINCH_ON_THRESHOLD && right.pinch >= BIMANUAL_PINCH_ON_THRESHOLD
    const bothOff = left.pinch >= BIMANUAL_PINCH_OFF_THRESHOLD && right.pinch >= BIMANUAL_PINCH_OFF_THRESHOLD

    if (!bimanualPinchRef.current) {
      if (bothOn) {
        bimanualPinch = true
        bimanualLastGoodMsRef.current = nowMs
      }
    } else {
      if (bothOff) {
        bimanualPinch = true
        bimanualLastGoodMsRef.current = nowMs
      } else if (nowMs - bimanualLastGoodMsRef.current <= BIMANUAL_PINCH_GRACE_MS) {
        bimanualPinch = true
      }
    }
  } else {
    bimanualLastGoodMsRef.current = 0
  }

  bimanualPinchRef.current = bimanualPinch

  const noDeltas: CloudControlDeltas = { zoom: 0, panX: 0, panY: 0, panZ: 0, grabStart: false }

  // Per-hand lock states are tracked for visual feedback
  // (Note: The actual lock system uses the simpler per-hand hysteresis below)
  const bothHandsLocked = leftLockRef.current.mode === 'locked' && rightLockRef.current.mode === 'locked'

  if (!enabled) {
    lockRef.current = { mode: 'idle', metrics: null }
    grabRef.current = { left: false, right: false }
    pinchModeRef.current = { left: false, right: false }
    pinchActivatedRef.current = { left: false, right: false }
    clearHoldFramesRef.current = 0
    leftLockRef.current = { mode: 'idle' }
    rightLockRef.current = { mode: 'idle' }
    return { lock: lockRef.current, deltas: noDeltas, clearRequested: false, bimanualPinch, bothHandsLocked: false, leftMetrics: left, rightMetrics: right, leftLock: leftLockRef.current, rightLock: rightLockRef.current }
  }

  // --- Per-hand hysteresis state ---
  const nextGrabLeft =
    !!left &&
    (grabRef.current.left ? left.grab >= GRAB_OFF_THRESHOLD : left.grab >= GRAB_ON_THRESHOLD)
  const nextGrabRight =
    !!right &&
    (grabRef.current.right ? right.grab >= GRAB_OFF_THRESHOLD : right.grab >= GRAB_ON_THRESHOLD)
  grabRef.current = { left: nextGrabLeft, right: nextGrabRight }

  const nextPinchModeLeft =
    !bimanualPinch &&
    !!left &&
    !nextGrabLeft &&
    (pinchModeRef.current.left ? left.pinch >= PINCH_MODE_OFF_THRESHOLD : left.pinch >= PINCH_MODE_ON_THRESHOLD)
  const nextPinchModeRight =
    !bimanualPinch &&
    !!right &&
    !nextGrabRight &&
    (pinchModeRef.current.right ? right.pinch >= PINCH_MODE_OFF_THRESHOLD : right.pinch >= PINCH_MODE_ON_THRESHOLD)
  pinchModeRef.current = { left: nextPinchModeLeft, right: nextPinchModeRight }

  const nextPinchActivatedLeft =
    !bimanualPinch &&
    !!left &&
    !nextGrabLeft &&
    (pinchActivatedRef.current.left ? left.pinch >= PINCH_OFF_THRESHOLD : left.pinch >= PINCH_ON_THRESHOLD)
  const nextPinchActivatedRight =
    !bimanualPinch &&
    !!right &&
    !nextGrabRight &&
    (pinchActivatedRef.current.right ? right.pinch >= PINCH_OFF_THRESHOLD : right.pinch >= PINCH_ON_THRESHOLD)
  pinchActivatedRef.current = { left: nextPinchActivatedLeft, right: nextPinchActivatedRight }

  // Choose active hand for single-hand interactions (grab > pinch > present)
  const choosePreferredHand = (a: HandSide, b: HandSide) => {
    // Prefer the hand with stronger intent signal; break ties to right for stability.
    const aMetrics = a === 'left' ? left : right
    const bMetrics = b === 'left' ? left : right
    if (!aMetrics) return b
    if (!bMetrics) return a

    const aGrab = a === 'left' ? nextGrabLeft : nextGrabRight
    const bGrab = b === 'left' ? nextGrabLeft : nextGrabRight
    if (aGrab !== bGrab) return aGrab ? a : b

    const aPinchMode = a === 'left' ? nextPinchModeLeft : nextPinchModeRight
    const bPinchMode = b === 'left' ? nextPinchModeLeft : nextPinchModeRight
    if (aPinchMode !== bPinchMode) return aPinchMode ? a : b

    const aScore = Math.max(aMetrics.grab, aMetrics.pinch)
    const bScore = Math.max(bMetrics.grab, bMetrics.pinch)
    if (Math.abs(aScore - bScore) > 0.05) return aScore > bScore ? a : b

    return 'right'
  }

  let activeHand: HandSide | null = null
  if (!bimanualPinch && (nextGrabLeft || nextGrabRight)) {
    activeHand = nextGrabLeft && nextGrabRight ? choosePreferredHand('left', 'right') : nextGrabLeft ? 'left' : 'right'
  } else if (!bimanualPinch && (nextPinchModeLeft || nextPinchModeRight)) {
    activeHand = nextPinchModeLeft && nextPinchModeRight ? choosePreferredHand('left', 'right') : nextPinchModeLeft ? 'left' : 'right'
  } else {
    activeHand = right ? 'right' : left ? 'left' : null
  }

  // Clear selection: hold acquire pose (open palm + spread + palm facing) for ~0.5s
  const anyAcquirePose = (!!left && isAcquirePose(left)) || (!!right && isAcquirePose(right))
  clearHoldFramesRef.current = anyAcquirePose ? clearHoldFramesRef.current + 1 : 0
  const clearRequested = clearHoldFramesRef.current >= CLEAR_FRAMES_REQUIRED

  // Update per-hand lock states for visual feedback based on pinch mode
  leftLockRef.current = nextPinchModeLeft || nextGrabLeft ? { mode: 'locked', lockedAtMs: nowMs, neutral: { x: 0, y: 0, depth: 0 }, grabbed: nextGrabLeft, pinchActivated: nextPinchActivatedLeft, lastSeenMs: nowMs, clearHoldFrames: 0 } : left ? { mode: 'candidate', frames: 1 } : { mode: 'idle' }
  rightLockRef.current = nextPinchModeRight || nextGrabRight ? { mode: 'locked', lockedAtMs: nowMs, neutral: { x: 0, y: 0, depth: 0 }, grabbed: nextGrabRight, pinchActivated: nextPinchActivatedRight, lastSeenMs: nowMs, clearHoldFrames: 0 } : right ? { mode: 'candidate', frames: 1 } : { mode: 'idle' }

  if (!activeHand) {
    lockRef.current = { mode: 'idle', metrics: null }
    return { lock: lockRef.current, deltas: noDeltas, clearRequested, bimanualPinch, bothHandsLocked, leftMetrics: left, rightMetrics: right, leftLock: leftLockRef.current, rightLock: rightLockRef.current }
  }

  const activeMetrics = activeHand === 'left' ? left : right
  const activeHandData = activeHand === 'left' ? state.leftHand : state.rightHand

  if (!activeMetrics || !activeHandData) {
    lockRef.current = { mode: 'idle', metrics: activeMetrics ?? null }
    return { lock: lockRef.current, deltas: noDeltas, clearRequested, bimanualPinch, bothHandsLocked, leftMetrics: left, rightMetrics: right, leftLock: leftLockRef.current, rightLock: rightLockRef.current }
  }

  const wrist = activeHandData.landmarks[0]
  const x = wrist?.x ?? 0.5
  const y = wrist?.y ?? 0.5

  const grabbed = activeHand === 'left' ? nextGrabLeft : nextGrabRight
  const pinchMode = activeHand === 'left' ? nextPinchModeLeft : nextPinchModeRight
  const pinchActivated = grabbed ? false : activeHand === 'left' ? nextPinchActivatedLeft : nextPinchActivatedRight

  // Expose a single "active" state for consumers (GraphCanvas / overlays).
  // We reuse the historical `mode: 'locked'` variant to avoid widespread type churn.
  const prev = lockRef.current
  const wasGrabbed = prev.mode === 'locked' && prev.hand === activeHand ? prev.grabbed : false
  const isActive = grabbed || pinchMode

  if (!isActive) {
    lockRef.current = { mode: 'idle', metrics: activeMetrics }
    return { lock: lockRef.current, deltas: noDeltas, clearRequested, bimanualPinch, bothHandsLocked, leftMetrics: left, rightMetrics: right, leftLock: leftLockRef.current, rightLock: rightLockRef.current }
  }

  const deltas: CloudControlDeltas = { ...noDeltas }
  let grabAnchor: { x: number; y: number; depth: number } | undefined

  if (grabbed) {
    const isFirstGrabFrame = !wasGrabbed
    const prevAnchor =
      prev.mode === 'locked' && prev.hand === activeHand ? prev.grabAnchor : undefined
    grabAnchor = isFirstGrabFrame ? { x, y, depth: activeMetrics.depth } : prevAnchor ?? { x, y, depth: activeMetrics.depth }
    deltas.grabStart = isFirstGrabFrame

    // Calculate displacement from anchor (how far hand moved since grab started)
    const dx = x - grabAnchor.x
    const dy = y - grabAnchor.y
    const dz = activeMetrics.depth - grabAnchor.depth

    // PAN the world: displacement-based, not velocity
    const PAN_GAIN = 300
    deltas.panX = dx * PAN_GAIN
    deltas.panY = dy * PAN_GAIN

    // Depth -> Z translation
    const DEPTH_PAN_GAIN = 250
    deltas.panZ = dz * DEPTH_PAN_GAIN

    // Gentle zoom based on depth
    if (Math.abs(dz) > DEPTH_DEADZONE) {
      deltas.zoom = dz * 0.5
    }
  }

  const locked: HandLockState = {
    mode: 'locked',
    hand: activeHand,
    metrics: activeMetrics,
    lockedAtMs: prev.mode === 'locked' && prev.hand === activeHand ? prev.lockedAtMs : nowMs,
    neutral:
      prev.mode === 'locked' && prev.hand === activeHand
        ? prev.neutral
        : { x, y, depth: activeMetrics.depth },
    grabbed,
    grabAnchor: grabbed ? grabAnchor : undefined,
    lastSeenMs: nowMs,
    pinchActivated,
    clearHoldFrames: clearHoldFramesRef.current,
  }

  lockRef.current = locked

  return {
    lock: lockRef.current,
    deltas,
    clearRequested,
    bimanualPinch,
    bothHandsLocked,
    leftMetrics: left,
    rightMetrics: right,
    leftLock: leftLockRef.current,
    rightLock: rightLockRef.current,
  }
}
