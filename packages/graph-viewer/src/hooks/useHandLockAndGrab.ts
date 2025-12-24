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
  /** Displacement-based pan: how much to offset from grab start position */
  panX: number
  panY: number
  panZ: number
  /** Is this the first frame of a grab? (used to capture initial world position) */
  grabStart: boolean
}

const DEFAULT_CONFIDENCE = 0.7

// Tunables (these matter a lot for UX)
const ACQUIRE_FRAMES_REQUIRED = 4
const LOCK_PERSIST_MS = 2000  // 2 seconds before unlocking when hand leaves frame

const SPREAD_THRESHOLD = 0.65
const PALM_FACING_THRESHOLD = 0.55

const GRAB_ON_THRESHOLD = 0.72
const GRAB_OFF_THRESHOLD = 0.45

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
  // NOTE: We don't require palm facing camera - natural pointing works from any angle
  const idxExt = fingerExtensionScore(wrist, indexMcp, indexTip)
  const midExt = fingerExtensionScore(wrist, middleMcp, middleTip)
  const ringExt = fingerExtensionScore(wrist, ringMcp, ringTip)
  const pinkyExt = fingerExtensionScore(wrist, pinkyMcp, pinkyTip)
  const others = clamp((midExt + ringExt + pinkyExt) / 3, 0, 1)
  // Point score: index extended (idxExt high) AND other fingers curled (others low)
  // If index is extended more than others, it's pointing
  const pointRaw = clamp((idxExt - others) * 2, 0, 1)
  const point = idxExt > 0.5 && others < 0.5 ? pointRaw : 0

  // Grab: closed fist = ALL fingers curled including index
  // Exclude index from grab calculation to distinguish from pointing
  const hasGrabStrength =
    typeof state.grabStrength === 'number' &&
    state.handsDetected >= 1 &&
    state.grabStrength > 0 // avoid default 0 from sources that don't compute it
  let grab = hasGrabStrength ? clamp(state.grabStrength, 0, 1) : 0
  if (!hasGrabStrength) {
    // For grab, require ALL fingers to be curled (including index)
    const dw1 = length2(indexTip.x - wrist.x, indexTip.y - wrist.y)
    const dw2 = length2(middleTip.x - wrist.x, middleTip.y - wrist.y)
    const dw3 = length2(ringTip.x - wrist.x, ringTip.y - wrist.y)
    const dw4 = length2(pinkyTip.x - wrist.x, pinkyTip.y - wrist.y)
    const avgDw = (dw1 + dw2 + dw3 + dw4) / 4
    // All fingers must be close to wrist
    const allCurled = dw1 < 0.15 && dw2 < 0.15 && dw3 < 0.15 && dw4 < 0.15
    grab = allCurled ? clamp(1 - safeDiv(avgDw - 0.08, 0.07), 0, 1) : 0
  }

  // Mutual exclusion: if pointing, suppress grab
  if (point > 0.5) {
    grab = 0
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
      return { lock: lockRef.current, deltas: { zoom: 0, panX: 0, panY: 0, panZ: 0, grabStart: false } }
    }

    const prev = lockRef.current

    // No hand seen
    if (!chosenHand || !metrics) {
      if (prev.mode === 'locked') {
        // persist lock briefly
        if (nowMs - prev.lastSeenMs <= LOCK_PERSIST_MS) {
          const persisted: HandLockState = { ...prev, metrics: prev.metrics }
          lockRef.current = persisted
          return { lock: persisted, deltas: { zoom: 0, panX: 0, panY: 0, panZ: 0, grabStart: false } }
        }
      }
      lockRef.current = { mode: 'idle', metrics: null }
      return { lock: lockRef.current, deltas: { zoom: 0, panX: 0, panY: 0, panZ: 0, grabStart: false } }
    }

    // Hand seen: update FSM
    if (prev.mode === 'idle') {
      if (isAcquirePose(metrics)) {
        const candidate: HandLockState = { mode: 'candidate', metrics, frames: 1 }
        lockRef.current = candidate
        return { lock: candidate, deltas: { zoom: 0, panX: 0, panY: 0, panZ: 0, grabStart: false } }
      }
      const idle: HandLockState = { mode: 'idle', metrics }
      lockRef.current = idle
      return { lock: idle, deltas: { zoom: 0, panX: 0, panY: 0, panZ: 0, grabStart: false } }
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
          return { lock: locked, deltas: { zoom: 0, panX: 0, panY: 0, panZ: 0, grabStart: false } }
        }
        const candidate: HandLockState = { mode: 'candidate', metrics, frames }
        lockRef.current = candidate
        return { lock: candidate, deltas: { zoom: 0, panX: 0, panY: 0, panZ: 0, grabStart: false } }
      }
      // lost candidate
      const idle: HandLockState = { mode: 'idle', metrics }
      lockRef.current = idle
      return { lock: idle, deltas: { zoom: 0, panX: 0, panY: 0, panZ: 0, grabStart: false } }
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

      let deltas: CloudControlDeltas = { zoom: 0, panX: 0, panY: 0, panZ: 0, grabStart: false }

      if (grabbed) {
        const isFirstGrabFrame = !prev.grabbed

        if (isFirstGrabFrame) {
          // First frame of grab - set anchor and signal to capture world position
          lock.grabAnchor = { x, y, depth: metrics.depth }
          deltas.grabStart = true
          lockRef.current = lock
          return { lock, deltas }
        }

        const anchor = prev.grabAnchor ?? { x, y, depth: metrics.depth }

        // Calculate displacement from anchor (how far hand moved since grab started)
        const dx = x - anchor.x  // hand moved right in screen space (0-1 normalized)
        const dy = y - anchor.y  // hand moved down in screen space
        const dz = metrics.depth - anchor.depth  // hand moved toward/away from camera

        // PAN the world: displacement-based, not velocity
        // Scale: moving hand across half the screen (~0.5) should move graph ~150 world units
        // That's a reasonable "arm's reach" mapping
        const PAN_GAIN = 300  // world units per full screen unit of hand movement

        deltas.panX = -dx * PAN_GAIN  // drag right = world moves right (opposite sign because we're moving world)
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
      return { lock, deltas }
    }

    lockRef.current = { mode: 'idle', metrics }
    return { lock: lockRef.current, deltas: { zoom: 0, panX: 0, panY: 0, panZ: 0, grabStart: false } }
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
