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
  /** 0..1: fist/grab strength (1 = closed fist) */
  grab: number
  /** depth signal (meters for iPhone LiDAR when available, otherwise MediaPipe-relative) */
  depth: number
  /** 0..1 heuristic confidence */
  confidence: number
}

export type HandLockState =
  | { mode: 'idle'; metrics: HandLockMetrics | null }
  | { mode: 'candidate'; metrics: HandLockMetrics; frames: number }
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
    }

export interface CloudControlDeltas {
  /** zoom velocity (positive -> zoom in, negative -> zoom out) */
  zoom: number
  /** rotation deltas (radians) */
  rotateX: number
  rotateY: number
}

const DEFAULT_CONFIDENCE = 0.7

// Tunables (these matter a lot for UX)
const ACQUIRE_FRAMES_REQUIRED = 4
const LOCK_PERSIST_MS = 450

const SPREAD_THRESHOLD = 0.65
const PALM_FACING_THRESHOLD = 0.55

const GRAB_ON_THRESHOLD = 0.72
const GRAB_OFF_THRESHOLD = 0.45

// Control sensitivity
const ROTATE_GAIN = 1.8
const DEPTH_GAIN = 2.6 // exponential factor
const DEPTH_DEADZONE = 0.01
const ROT_DEADZONE = 0.003

function clamp(v: number, min: number, max: number) {
  return Math.max(min, Math.min(max, v))
}

function length2(dx: number, dy: number) {
  return Math.sqrt(dx * dx + dy * dy)
}

function safeDiv(a: number, b: number, fallback = 0) {
  return b !== 0 ? a / b : fallback
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
  const pinchDist = length2(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y)
  const pinch = clamp(1 - safeDiv(pinchDist - 0.02, 0.13), 0, 1)

  // Pointing pose score:
  // index extended while the other 3 fingers are relatively curled.
  const idxExt = fingerExtensionScore(wrist, indexMcp, indexTip)
  const midExt = fingerExtensionScore(wrist, middleMcp, middleTip)
  const ringExt = fingerExtensionScore(wrist, ringMcp, ringTip)
  const pinkyExt = fingerExtensionScore(wrist, pinkyMcp, pinkyTip)
  const others = clamp((midExt + ringExt + pinkyExt) / 3, 0, 1)
  const palmForward = clamp((palmFacing + 1) / 2, 0, 1)
  const point = clamp(idxExt * (1 - others) * palmForward, 0, 1)

  // Grab: use state.grabStrength only if it's meaningful (computed by source);
  // otherwise approximate from fingertip distances to wrist (closed fist => smaller)
  const hasGrabStrength =
    typeof state.grabStrength === 'number' &&
    state.handsDetected >= 1 &&
    state.grabStrength > 0 // avoid default 0 from sources that don't compute it
  let grab = hasGrabStrength ? clamp(state.grabStrength, 0, 1) : 0
  if (!hasGrabStrength) {
    const dw1 = length2(indexTip.x - wrist.x, indexTip.y - wrist.y)
    const dw2 = length2(middleTip.x - wrist.x, middleTip.y - wrist.y)
    const dw3 = length2(ringTip.x - wrist.x, ringTip.y - wrist.y)
    const dw4 = length2(pinkyTip.x - wrist.x, pinkyTip.y - wrist.y)
    const avgDw = (dw1 + dw2 + dw3 + dw4) / 4
    // Typical: ~0.10 fist .. ~0.25 open
    grab = clamp(1 - safeDiv(avgDw - 0.10, 0.15), 0, 1)
  }

  // Depth: prefer pinch ray origin z when present (iPhone LiDAR mapped into landmarks z)
  const pinchRay = hand === 'right' ? state.rightPinchRay : state.leftPinchRay
  const depth = (pinchRay?.origin.z ?? wrist.z ?? 0) as number

  // Confidence: use landmark visibility if present; else assume ok
  const vis = (wrist as any).visibility
  const confidence = typeof vis === 'number' ? clamp(vis, 0, 1) : DEFAULT_CONFIDENCE

  return { spread, palmFacing, point, pinch, grab, depth, confidence }
}

function isAcquirePose(m: HandLockMetrics) {
  return m.spread >= SPREAD_THRESHOLD && m.palmFacing >= PALM_FACING_THRESHOLD && m.confidence >= 0.4
}

function expResponse(delta: number, gain: number) {
  const s = Math.sign(delta)
  const a = Math.abs(delta)
  return s * (Math.exp(a * gain) - 1)
}

export function useHandLockAndGrab(state: GestureState, enabled: boolean) {
  const lockRef = useRef<HandLockState>({ mode: 'idle', metrics: null })

  const nowMs = performance.now()

  const right = enabled ? computeMetrics(state, 'right') : null
  const left = enabled ? computeMetrics(state, 'left') : null

  // For now, single-hand only: prefer right if present, else left.
  const chosenHand: HandSide | null = right ? 'right' : left ? 'left' : null
  const metrics = chosenHand === 'right' ? right : chosenHand === 'left' ? left : null

  const next = useMemo((): { lock: HandLockState; deltas: CloudControlDeltas } => {
    if (!enabled) {
      lockRef.current = { mode: 'idle', metrics: null }
      return { lock: lockRef.current, deltas: { zoom: 0, rotateX: 0, rotateY: 0 } }
    }

    const prev = lockRef.current

    // No hand seen
    if (!chosenHand || !metrics) {
      if (prev.mode === 'locked') {
        // persist lock briefly
        if (nowMs - prev.lastSeenMs <= LOCK_PERSIST_MS) {
          const persisted: HandLockState = { ...prev, metrics: prev.metrics }
          lockRef.current = persisted
          return { lock: persisted, deltas: { zoom: 0, rotateX: 0, rotateY: 0 } }
        }
      }
      lockRef.current = { mode: 'idle', metrics: null }
      return { lock: lockRef.current, deltas: { zoom: 0, rotateX: 0, rotateY: 0 } }
    }

    // Hand seen: update FSM
    if (prev.mode === 'idle') {
      if (isAcquirePose(metrics)) {
        const candidate: HandLockState = { mode: 'candidate', metrics, frames: 1 }
        lockRef.current = candidate
        return { lock: candidate, deltas: { zoom: 0, rotateX: 0, rotateY: 0 } }
      }
      const idle: HandLockState = { mode: 'idle', metrics }
      lockRef.current = idle
      return { lock: idle, deltas: { zoom: 0, rotateX: 0, rotateY: 0 } }
    }

    if (prev.mode === 'candidate') {
      if (isAcquirePose(metrics)) {
        const frames = prev.frames + 1
        if (frames >= ACQUIRE_FRAMES_REQUIRED) {
          // lock!
          const handData = chosenHand === 'right' ? state.rightHand : state.leftHand
          const wrist = handData?.landmarks[0]
          const locked: HandLockState = {
            mode: 'locked',
            hand: chosenHand,
            metrics,
            lockedAtMs: nowMs,
            neutral: { x: wrist?.x ?? 0.5, y: wrist?.y ?? 0.5, depth: metrics.depth },
            grabbed: false,
            lastSeenMs: nowMs,
          }
          lockRef.current = locked
          return { lock: locked, deltas: { zoom: 0, rotateX: 0, rotateY: 0 } }
        }
        const candidate: HandLockState = { mode: 'candidate', metrics, frames }
        lockRef.current = candidate
        return { lock: candidate, deltas: { zoom: 0, rotateX: 0, rotateY: 0 } }
      }
      // lost candidate
      const idle: HandLockState = { mode: 'idle', metrics }
      lockRef.current = idle
      return { lock: idle, deltas: { zoom: 0, rotateX: 0, rotateY: 0 } }
    }

    // locked
    if (prev.mode === 'locked') {
      const handData = prev.hand === 'right' ? state.rightHand : state.leftHand
      const wrist = handData?.landmarks[0]
      const x = wrist?.x ?? prev.neutral.x
      const y = wrist?.y ?? prev.neutral.y

      // Grab hysteresis
      const grabbed =
        prev.grabbed ? metrics.grab >= GRAB_OFF_THRESHOLD : metrics.grab >= GRAB_ON_THRESHOLD

      const lock: HandLockState = {
        ...prev,
        metrics,
        grabbed,
        lastSeenMs: nowMs,
      }

      let deltas: CloudControlDeltas = { zoom: 0, rotateX: 0, rotateY: 0 }

      if (grabbed) {
        const anchor = prev.grabAnchor ?? { x, y, depth: metrics.depth }
        // On first grab frame, set anchor
        if (!prev.grabbed) {
          lock.grabAnchor = anchor
          lockRef.current = lock
          return { lock, deltas }
        }

        // Depth -> zoom (exponential)
        // User mental model: pull fist toward body (farther from camera) zooms IN; push toward screen/camera zooms OUT.
        const dz = metrics.depth - anchor.depth
        if (Math.abs(dz) > DEPTH_DEADZONE) {
          deltas.zoom = expResponse(dz, DEPTH_GAIN)
        }

        // Position -> rotation
        const dx = x - anchor.x
        const dy = y - anchor.y
        if (Math.abs(dx) > ROT_DEADZONE || Math.abs(dy) > ROT_DEADZONE) {
          deltas.rotateY = dx * ROTATE_GAIN
          deltas.rotateX = -dy * ROTATE_GAIN
        }
      } else {
        lock.grabAnchor = undefined
      }

      lockRef.current = lock
      return { lock, deltas }
    }

    lockRef.current = { mode: 'idle', metrics }
    return { lock: lockRef.current, deltas: { zoom: 0, rotateX: 0, rotateY: 0 } }
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
  ])

  return next
}
