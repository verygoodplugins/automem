/**
 * Hand Lock + Grab State Machine
 *
 * Goal: Make gestures intentional.
 * - Hand is ignored until user presents an "open palm + spread fingers" pose (acquire).
 * - Once acquired, we maintain a lock for a short time even through partial landmark loss.
 * - In locked state, a closed fist ("grab") manipulates the cloud:
 *   - Pull toward body => zoom in (exponential response)
 *   - Push toward screen => zoom out (exponential response)
 *   - Move fist around screen => rotate cloud
 *
 * Works with either MediaPipe or iPhone-fed landmarks because it only needs GestureState.
 */

import { useMemo, useRef } from 'react'
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
const ACQUIRE_FRAMES_REQUIRED = 8
const LOCK_PERSIST_MS = 2000  // 2 seconds before unlocking when hand leaves frame

// Make acquisition VERY intentional: open palm + spread fingers + palm facing camera.
const SPREAD_THRESHOLD = 0.78
const PALM_FACING_THRESHOLD = 0.72

const GRAB_ON_THRESHOLD = 0.72
const GRAB_OFF_THRESHOLD = 0.45

// Pinch thresholds for direct selection ("pick the berry")
const PINCH_ON_THRESHOLD = 0.85
const PINCH_OFF_THRESHOLD = 0.65

// Two-hand navigation: both hands pinching
const BIMANUAL_PINCH_ON_THRESHOLD = 0.75
const BIMANUAL_PINCH_OFF_THRESHOLD = 0.55

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

  // Per-hand lock states for dual-hand tracking
  const leftLockRef = useRef<SingleHandLockState>({ mode: 'idle' })
  const rightLockRef = useRef<SingleHandLockState>({ mode: 'idle' })

  const nowMs = performance.now()

  const right = enabled ? computeMetrics(state, 'right') : null
  const left = enabled ? computeMetrics(state, 'left') : null

  // Bimanual pinch: both hands are pinching simultaneously (with hysteresis)
  const bimanualPinch =
    enabled && left && right
      ? bimanualPinchRef.current
          ? left.pinch >= BIMANUAL_PINCH_OFF_THRESHOLD && right.pinch >= BIMANUAL_PINCH_OFF_THRESHOLD
          : left.pinch >= BIMANUAL_PINCH_ON_THRESHOLD && right.pinch >= BIMANUAL_PINCH_ON_THRESHOLD
      : false
  bimanualPinchRef.current = bimanualPinch

  // Per-hand lock state machine (independent acquisition for both hands)
  const updateSingleHandLock = (
    metrics: HandLockMetrics | null,
    handData: { landmarks: { x: number; y: number; z?: number }[] } | null,
    prevLock: SingleHandLockState
  ): SingleHandLockState => {
    if (!enabled || !metrics || !handData) {
      // Hand not seen - check if we should persist lock briefly
      if (prevLock.mode === 'locked') {
        if (nowMs - prevLock.lastSeenMs <= LOCK_PERSIST_MS) {
          return prevLock
        }
      }
      return { mode: 'idle' }
    }

    if (prevLock.mode === 'locked') {
      // Update last seen time
      return { ...prevLock, lastSeenMs: nowMs }
    }

    if (prevLock.mode === 'candidate') {
      if (isAcquirePose(metrics)) {
        const frames = prevLock.frames + 1
        if (frames >= ACQUIRE_FRAMES_REQUIRED) {
          // Lock this hand!
          const wrist = handData.landmarks[0]
          return {
            mode: 'locked',
            lockedAtMs: nowMs,
            neutral: { x: wrist?.x ?? 0.5, y: wrist?.y ?? 0.5, depth: metrics.depth },
            grabbed: false,
            pinchActivated: false,
            lastSeenMs: nowMs,
            clearHoldFrames: 0,
          }
        }
        return { mode: 'candidate', frames }
      }
      // Lost acquire pose
      return { mode: 'idle' }
    }

    // Idle mode
    if (isAcquirePose(metrics)) {
      return { mode: 'candidate', frames: 1 }
    }
    return { mode: 'idle' }
  }

  // Update per-hand lock states
  leftLockRef.current = updateSingleHandLock(left, state.leftHand, leftLockRef.current)
  rightLockRef.current = updateSingleHandLock(right, state.rightHand, rightLockRef.current)

  const bothHandsLocked = leftLockRef.current.mode === 'locked' && rightLockRef.current.mode === 'locked'

  // Choose a hand to consider when not locked:
  // - Prefer a hand currently holding the acquire pose
  // - Otherwise prefer right if present, else left
  const rightAcquire = !!right && isAcquirePose(right)
  const leftAcquire = !!left && isAcquirePose(left)
  const chosenHand: HandSide | null =
    rightAcquire ? 'right' : leftAcquire ? 'left' : right ? 'right' : left ? 'left' : null
  const metrics = chosenHand === 'right' ? right : chosenHand === 'left' ? left : null

  const next = useMemo((): { lock: HandLockState; deltas: CloudControlDeltas; clearRequested: boolean } => {
    const noDeltas = { zoom: 0, panX: 0, panY: 0, panZ: 0, grabStart: false }

    if (!enabled) {
      lockRef.current = { mode: 'idle', metrics: null }
      return { lock: lockRef.current, deltas: noDeltas, clearRequested: false }
    }

    const prev = lockRef.current

    // Locked mode: ONLY the locked hand can drive state; never switch to the other hand implicitly.
    if (prev.mode === 'locked') {
      const lockedMetrics = prev.hand === 'right' ? right : left
      const handData = prev.hand === 'right' ? state.rightHand : state.leftHand

      // Locked hand not currently seen
      if (!lockedMetrics || !handData) {
        // Persist lock briefly, but don't let another hand keep it alive.
        if (nowMs - prev.lastSeenMs <= LOCK_PERSIST_MS) {
          const persisted: HandLockState = { ...prev, metrics: prev.metrics }
          lockRef.current = persisted
          return { lock: persisted, deltas: { zoom: 0, panX: 0, panY: 0, panZ: 0, grabStart: false }, clearRequested: false }
        }
        lockRef.current = { mode: 'idle', metrics: null }
        return { lock: lockRef.current, deltas: { zoom: 0, panX: 0, panY: 0, panZ: 0, grabStart: false }, clearRequested: false }
      }

      const wrist = handData.landmarks[0]
      const x = wrist?.x ?? prev.neutral.x
      const y = wrist?.y ?? prev.neutral.y

      // Grab hysteresis
      const grabbed =
        prev.grabbed
          ? lockedMetrics.grab >= GRAB_OFF_THRESHOLD
          : lockedMetrics.grab >= GRAB_ON_THRESHOLD

      // Pinch hysteresis for selection ("pick the berry")
      // Only activate when NOT grabbing (grab takes priority)
      const pinchActivated = grabbed
        ? false
        : prev.pinchActivated
          ? lockedMetrics.pinch >= PINCH_OFF_THRESHOLD
          : lockedMetrics.pinch >= PINCH_ON_THRESHOLD

      // Clear selection: hold acquire pose for ~0.5 seconds while locked
      // Only track when NOT grabbing and NOT pinching (intentional open palm)
      const isHoldingAcquirePose = isAcquirePose(lockedMetrics)
      const clearHoldFrames =
        !grabbed && !pinchActivated && isHoldingAcquirePose
          ? (prev.clearHoldFrames ?? 0) + 1
          : 0
      const clearRequested = clearHoldFrames >= CLEAR_FRAMES_REQUIRED

      const lock: HandLockState = {
        ...prev,
        metrics: lockedMetrics,
        grabbed,
        pinchActivated,
        lastSeenMs: nowMs,
        clearHoldFrames,
      }

      const deltas: CloudControlDeltas = { zoom: 0, panX: 0, panY: 0, panZ: 0, grabStart: false }

      if (grabbed) {
        const isFirstGrabFrame = !prev.grabbed

        if (isFirstGrabFrame) {
          // First frame of grab - set anchor and signal to capture world position
          lock.grabAnchor = { x, y, depth: lockedMetrics.depth }
          deltas.grabStart = true
          lockRef.current = lock
          return { lock, deltas, clearRequested: false }
        }

        const anchor = prev.grabAnchor ?? { x, y, depth: lockedMetrics.depth }

        // Calculate displacement from anchor (how far hand moved since grab started)
        const dx = x - anchor.x  // hand moved right in screen space (0-1 normalized)
        const dy = y - anchor.y  // hand moved down in screen space
        const dz = lockedMetrics.depth - anchor.depth  // hand moved toward/away from camera

        // PAN the world: displacement-based, not velocity
        // Scale: moving hand across half the screen (~0.5) should move graph ~150 world units
        // That's a reasonable "arm's reach" mapping
        const PAN_GAIN = 300  // world units per full screen unit of hand movement

        deltas.panX = dx * PAN_GAIN  // drag right = world moves right
        deltas.panY = dy * PAN_GAIN   // drag down = world moves down (Y is flipped in screen coords)

        // Depth -> Z translation
        // Moving hand ~0.2 depth units should translate maybe 50-100 world units
        const DEPTH_PAN_GAIN = 250
        deltas.panZ = dz * DEPTH_PAN_GAIN

        // Also apply zoom based on depth (optional, can remove if too much)
        if (Math.abs(dz) > DEPTH_DEADZONE) {
          deltas.zoom = dz * 0.5  // gentle zoom
        }
      } else {
        lock.grabAnchor = undefined
      }

      lockRef.current = lock
      return { lock, deltas, clearRequested }
    }

    // No hand seen
    if (!chosenHand || !metrics) {
      lockRef.current = { mode: 'idle', metrics: null }
      return { lock: lockRef.current, deltas: { zoom: 0, panX: 0, panY: 0, panZ: 0, grabStart: false }, clearRequested: false }
    }

    // Hand seen: update FSM
    if (prev.mode === 'idle') {
      if (isAcquirePose(metrics)) {
        const candidate: HandLockState = { mode: 'candidate', hand: chosenHand, metrics, frames: 1 }
        lockRef.current = candidate
        return { lock: candidate, deltas: { zoom: 0, panX: 0, panY: 0, panZ: 0, grabStart: false }, clearRequested: false }
      }
      const idle: HandLockState = { mode: 'idle', metrics }
      lockRef.current = idle
      return { lock: idle, deltas: { zoom: 0, panX: 0, panY: 0, panZ: 0, grabStart: false }, clearRequested: false }
    }

    if (prev.mode === 'candidate') {
      const candidateHand = prev.hand
      const candidateMetrics = candidateHand === 'right' ? right : left
      if (candidateMetrics && isAcquirePose(candidateMetrics)) {
        const frames = prev.frames + 1
        if (frames >= ACQUIRE_FRAMES_REQUIRED) {
          // lock!
          const handData = candidateHand === 'right' ? state.rightHand : state.leftHand
          const wrist = handData?.landmarks[0]
          const locked: HandLockState = {
            mode: 'locked',
            hand: candidateHand,
            metrics: candidateMetrics,
            lockedAtMs: nowMs,
            neutral: { x: wrist?.x ?? 0.5, y: wrist?.y ?? 0.5, depth: candidateMetrics.depth },
            grabbed: false,
            pinchActivated: false,
            lastSeenMs: nowMs,
            clearHoldFrames: 0,
          }
          lockRef.current = locked
          return { lock: locked, deltas: { zoom: 0, panX: 0, panY: 0, panZ: 0, grabStart: false }, clearRequested: false }
        }
        const candidate: HandLockState = { mode: 'candidate', hand: candidateHand, metrics: candidateMetrics, frames }
        lockRef.current = candidate
        return { lock: candidate, deltas: { zoom: 0, panX: 0, panY: 0, panZ: 0, grabStart: false }, clearRequested: false }
      }
      // lost candidate
      const idle: HandLockState = { mode: 'idle', metrics }
      lockRef.current = idle
      return { lock: idle, deltas: { zoom: 0, panX: 0, panY: 0, panZ: 0, grabStart: false }, clearRequested: false }
    }

    lockRef.current = { mode: 'idle', metrics }
    return { lock: lockRef.current, deltas: { zoom: 0, panX: 0, panY: 0, panZ: 0, grabStart: false }, clearRequested: false }
  }, [
    enabled,
    chosenHand,
    // metrics is a new object each render; depend on its fields instead
    metrics?.spread,
    metrics?.palmFacing,
    metrics?.point,
    metrics?.pinch,
    metrics?.grab,
    metrics?.depth,
    metrics?.confidence,
    nowMs,
    state.leftHand,
    state.rightHand,
    state.handsDetected,
    state.grabStrength,
    state.leftPinchRay,
    state.rightPinchRay,
    right?.spread,
    right?.palmFacing,
    right?.point,
    right?.pinch,
    right?.grab,
    right?.depth,
    right?.confidence,
    left?.spread,
    left?.palmFacing,
    left?.point,
    left?.pinch,
    left?.grab,
    left?.depth,
    left?.confidence,
  ])

  return {
    ...next,
    bimanualPinch,
    bothHandsLocked,
    leftMetrics: left,
    rightMetrics: right,
    leftLock: leftLockRef.current,
    rightLock: rightLockRef.current,
  }
}
