/**
 * Hand 2D Overlay
 *
 * Renders hands as a 2D SVG overlay with:
 * - Ghost 3D hand effect (translucent, glowing Master Hand style)
 * - Smoothing/interpolation (ghost persists briefly when hand disappears)
 * - Depth-aware scaling ("reach through screen" paradigm)
 *
 * SIMPLIFIED: No lasers, no center target, just visual hand feedback.
 */

import { useState, useEffect, useRef } from 'react'
import type { GestureState } from '../hooks/useHandGestures'
import type { HandLockState } from '../hooks/useHandLockAndGrab'

// Fingertip indices
const FINGERTIPS = [4, 8, 12, 16, 20]

// Knuckle indices (base of fingers)
const KNUCKLES = [5, 9, 13, 17]

// Smoothing configuration
const SMOOTHING_FACTOR = 0.2 // Lower = smoother but laggier
const GHOST_FADE_DURATION = 500 // ms to fade out ghost hand
const GHOST_PERSIST_DURATION = 300 // ms to keep ghost before fading

interface SmoothedHand {
  landmarks: { x: number; y: number; z: number }[]
  lastSeen: number
  isGhost: boolean
  opacity: number
}

interface Hand2DOverlayProps {
  gestureState: GestureState
  enabled?: boolean
  lock?: HandLockState
}

export function Hand2DOverlay({
  gestureState,
  enabled = true,
  lock,
}: Hand2DOverlayProps) {
  // Track smoothed hand positions with ghost effect
  const [leftSmoothed, setLeftSmoothed] = useState<SmoothedHand | null>(null)
  const [rightSmoothed, setRightSmoothed] = useState<SmoothedHand | null>(null)
  const animationRef = useRef<number>()

  // Smoothing and ghost effect
  useEffect(() => {
    if (!enabled) return

    const now = Date.now()

    // Process left hand
    if (gestureState.leftHand) {
      setLeftSmoothed(prev => {
        const hand = gestureState.leftHand!
        const newLandmarks = hand.landmarks.map((lm, i) => {
          // Prefer world Z (meters for iPhone LiDAR) when available; fall back to normalized z.
          const zTarget = (hand.worldLandmarks?.[i]?.z ?? (lm.z || 0)) as number
          const prevLm = prev?.landmarks[i]
          if (prevLm && !prev.isGhost) {
            // Interpolate toward new position
            return {
              x: prevLm.x + (lm.x - prevLm.x) * SMOOTHING_FACTOR,
              y: prevLm.y + (lm.y - prevLm.y) * SMOOTHING_FACTOR,
              z: prevLm.z + (zTarget - prevLm.z) * SMOOTHING_FACTOR,
            }
          }
          return { x: lm.x, y: lm.y, z: zTarget }
        })
        return { landmarks: newLandmarks, lastSeen: now, isGhost: false, opacity: 1 }
      })
    } else if (leftSmoothed && !leftSmoothed.isGhost) {
      // Hand disappeared - start ghost mode
      setLeftSmoothed(prev => prev ? { ...prev, isGhost: true, lastSeen: now } : null)
    }

    // Process right hand
    if (gestureState.rightHand) {
      setRightSmoothed(prev => {
        const hand = gestureState.rightHand!
        const newLandmarks = hand.landmarks.map((lm, i) => {
          // Prefer world Z (meters for iPhone LiDAR) when available; fall back to normalized z.
          const zTarget = (hand.worldLandmarks?.[i]?.z ?? (lm.z || 0)) as number
          const prevLm = prev?.landmarks[i]
          if (prevLm && !prev.isGhost) {
            return {
              x: prevLm.x + (lm.x - prevLm.x) * SMOOTHING_FACTOR,
              y: prevLm.y + (lm.y - prevLm.y) * SMOOTHING_FACTOR,
              z: prevLm.z + (zTarget - prevLm.z) * SMOOTHING_FACTOR,
            }
          }
          return { x: lm.x, y: lm.y, z: zTarget }
        })
        return { landmarks: newLandmarks, lastSeen: now, isGhost: false, opacity: 1 }
      })
    } else if (rightSmoothed && !rightSmoothed.isGhost) {
      setRightSmoothed(prev => prev ? { ...prev, isGhost: true, lastSeen: now } : null)
    }
  }, [gestureState, enabled])

  // Ghost fade animation
  useEffect(() => {
    const animate = () => {
      const now = Date.now()

      // Fade left ghost
      if (leftSmoothed?.isGhost) {
        const elapsed = now - leftSmoothed.lastSeen
        if (elapsed > GHOST_PERSIST_DURATION) {
          const fadeProgress = (elapsed - GHOST_PERSIST_DURATION) / GHOST_FADE_DURATION
          if (fadeProgress >= 1) {
            setLeftSmoothed(null)
          } else {
            setLeftSmoothed(prev => prev ? { ...prev, opacity: 1 - fadeProgress } : null)
          }
        }
      }

      // Fade right ghost
      if (rightSmoothed?.isGhost) {
        const elapsed = now - rightSmoothed.lastSeen
        if (elapsed > GHOST_PERSIST_DURATION) {
          const fadeProgress = (elapsed - GHOST_PERSIST_DURATION) / GHOST_FADE_DURATION
          if (fadeProgress >= 1) {
            setRightSmoothed(null)
          } else {
            setRightSmoothed(prev => prev ? { ...prev, opacity: 1 - fadeProgress } : null)
          }
        }
      }

      animationRef.current = requestAnimationFrame(animate)
    }

    animationRef.current = requestAnimationFrame(animate)
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
    }
  }, [leftSmoothed?.isGhost, rightSmoothed?.isGhost])

  if (!enabled || !gestureState.isTracking) return null

  // Visibility gating:
  // - Non-acquired hands should be *extremely* faint to avoid the "desk hand blur" problem.
  // - Locked hand is bright.
  const lockMode = lock?.mode ?? 'idle'
  const lockedHand =
    lock?.mode === 'locked'
      ? lock.hand
      : lock?.mode === 'candidate'
        ? lock.hand
        : null
  const baseOpacity =
    lockMode === 'locked' ? 0.85 :
    lockMode === 'candidate' ? 0.25 :
    0.06

  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden">
      <svg
        className="w-full h-full"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
      >
        {/* Define filters and gradients */}
        <defs>
          {/* Ghost glow filter */}
          <filter id="ghost-glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="0.8" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>

          {/* Grip glow filter */}
          <filter id="grip-glow" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="1.5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>

          {/* Gradient for 3D depth effect - cyan */}
          <radialGradient id="hand-gradient-cyan" cx="50%" cy="30%" r="70%">
            <stop offset="0%" stopColor="#4ecdc4" stopOpacity="0.6" />
            <stop offset="100%" stopColor="#2a7a75" stopOpacity="0.2" />
          </radialGradient>

          {/* Gradient for 3D depth effect - magenta */}
          <radialGradient id="hand-gradient-magenta" cx="50%" cy="30%" r="70%">
            <stop offset="0%" stopColor="#f72585" stopOpacity="0.6" />
            <stop offset="100%" stopColor="#a01850" stopOpacity="0.2" />
          </radialGradient>
        </defs>

        {/* Left hand - cyan ghost */}
        {leftSmoothed && (
          <g opacity={leftSmoothed.opacity} filter="url(#ghost-glow)">
            <GhostHand
              landmarks={leftSmoothed.landmarks}
              color="#4ecdc4"
              gradientId="hand-gradient-cyan"
              isGhost={leftSmoothed.isGhost}
              opacityMultiplier={baseOpacity * (lockedHand && lockedHand !== 'left' ? 0.08 : 1)}
            />
          </g>
        )}

        {/* Right hand - magenta ghost */}
        {rightSmoothed && (
          <g opacity={rightSmoothed.opacity} filter="url(#ghost-glow)">
            <GhostHand
              landmarks={rightSmoothed.landmarks}
              color="#f72585"
              gradientId="hand-gradient-magenta"
              isGhost={rightSmoothed.isGhost}
              opacityMultiplier={baseOpacity * (lockedHand && lockedHand !== 'right' ? 0.08 : 1)}
            />
          </g>
        )}
      </svg>
    </div>
  )
}

interface GhostHandProps {
  landmarks: { x: number; y: number; z: number }[]
  color: string
  gradientId: string
  isGhost?: boolean
  opacityMultiplier?: number
}

type Point2 = { x: number; y: number }

/**
 * MasterHand - Smash Bros Master Hand / Crazy Hand style
 * Volumetric filled shapes with soft gradients and ambient occlusion
 *
 * Z-AXIS: "Reach Through Screen" Paradigm
 * - Hand moves TOWARD camera → Virtual hand goes INTO the scene (smaller, recedes)
 * - Hand moves AWAY from camera → Virtual hand comes OUT of the scene (larger, approaches)
 * This creates the feeling of reaching through a portal into the 3D world.
 */
function GhostHand({
  landmarks,
  color: _color,
  gradientId: _gradientId,
  isGhost = false,
  opacityMultiplier = 1,
}: GhostHandProps) {
  const wristZ = landmarks[0].z || 0

  // Detect if Z is in meters (LiDAR) vs normalized (MediaPipe-style).
  // With iPhone LiDAR we use worldLandmarks.z in meters, which is positive (e.g. 0.3..2.0).
  const isMeters = wristZ >= 0.1 && wristZ <= 8.0

  // Z-AXIS INVERSION: "Reach Through Portal" Paradigm
  // Physical hand closer to camera → Virtual hand appears SMALLER (receding into scene)
  // Physical hand farther from camera → Virtual hand appears LARGER (coming out of scene)
  //
  // This creates the illusion that you're reaching THROUGH the screen INTO the 3D world.
  // Convention for all sources (MediaPipe, iPhone LiDAR):
  //   negative Z = closer to camera
  //   positive Z = farther from camera

  let scaleFactor = 1.0
  let depthOpacity = 1.0  // Additional opacity based on depth

  if (isMeters) {
    // LiDAR in meters (positive): ~0.25m (very close) .. ~1.10m (arm's length / comfy)
    // Portal mapping: close -> small, far -> large (strong enough to overcome perspective scaling)
    const t = Math.max(0, Math.min(1, (wristZ - 0.25) / 0.85))
    scaleFactor = 0.25 + t * 1.55        // 0.25 (close) .. 1.80 (far)
    depthOpacity = 0.55 + t * 0.45       // 0.55 (close) .. 1.00 (far)
  } else {
    // MediaPipe normalized: positive Z = FARTHER from camera, negative Z = CLOSER
    // Typical range: -0.25 (close) to +0.15 (far)
    // "Reach through screen": Close → small/faint, Far → large/bright
    const t = Math.max(0, Math.min(1, (wristZ + 0.25) / 0.35))
    scaleFactor = 0.25 + t * 1.55        // 0.25 (close) .. 1.80 (far)
    depthOpacity = 0.55 + t * 0.45       // 0.55 (close) .. 1.00 (far)
  }

  // Apply the depth opacity to the overall opacity multiplier
  const effectiveOpacityMultiplier = opacityMultiplier * depthOpacity

  const clampedScale = Math.max(0.3, Math.min(2.0, scaleFactor))

  // "Reach Through Portal" X-axis: Keep the natural webcam mirror
  // This makes the virtual hand appear ON THE OTHER SIDE of the screen
  // When you point at the camera, the virtual finger points INTO the 3D scene
  const toSvg = (lm: { x: number; y: number }) => ({
    x: lm.x * 100,
    y: lm.y * 100,
  })

  // IMPORTANT: apply portal scaling to the landmark positions (not just stroke widths),
  // otherwise the overlay will behave like a normal camera feed (closer = bigger).
  const rawPoints = landmarks.map(toSvg)
  const cx = rawPoints[0]?.x ?? 50 // scale around wrist so position stays intuitive
  const cy = rawPoints[0]?.y ?? 50

  // Extra depth warp per-landmark: makes "pointing at the screen" read as pointing *into* the scene.
  // (Index tip physically closer than wrist should appear *deeper* / smaller in the portal model.)
  const warpGain = isMeters ? 2.2 : 6.0
  const clampRange = (v: number, min: number, max: number) => Math.max(min, Math.min(max, v))

  const points = rawPoints.map((p, i) => {
    const zi = landmarks[i]?.z ?? wristZ
    const dz = zi - wristZ // meters or normalized depending on source
    const pointScale = clampRange(1 + dz * warpGain, 0.65, 1.45)
    const s = clampedScale * pointScale
    return {
      x: cx + (p.x - cx) * s,
      y: cy + (p.y - cy) * s,
    }
  })
  const gloveOpacity = (isGhost ? 0.5 : 0.85) * effectiveOpacityMultiplier

  // Finger width based on scale - fatter fingers for Master Hand look
  const fingerWidth = 1.8 * clampedScale

  // Unique ID for this hand's gradients/filters
  const handId = Math.round(points[0].x * 10)

  // Helper: get perpendicular offset for finger width
  const getPerpendicular = (p1: Point2, p2: Point2, width: number) => {
    const dx = p2.x - p1.x
    const dy = p2.y - p1.y
    const len = Math.sqrt(dx * dx + dy * dy) || 1
    return { x: -dy / len * width, y: dx / len * width }
  }

  // Create filled finger shape (capsule/sausage shape)
  const createFingerShape = (indices: number[], width: number) => {
    const pts = indices.map(i => points[i])
    if (pts.length < 2) return ''

    // Build outline going down one side and back up the other
    const leftSide: string[] = []
    const rightSide: string[] = []

    for (let i = 0; i < pts.length - 1; i++) {
      const perp = getPerpendicular(pts[i], pts[i + 1], width)
      leftSide.push(`${pts[i].x + perp.x},${pts[i].y + perp.y}`)
      rightSide.unshift(`${pts[i].x - perp.x},${pts[i].y - perp.y}`)
    }

    // Add rounded tip
    const lastPt = pts[pts.length - 1]
    const prevPt = pts[pts.length - 2]
    const tipPerp = getPerpendicular(prevPt, lastPt, width)

    // Rounded end cap using arc
    const tipLeft = `${lastPt.x + tipPerp.x},${lastPt.y + tipPerp.y}`
    const tipRight = `${lastPt.x - tipPerp.x},${lastPt.y - tipPerp.y}`

    return `M ${leftSide[0]} L ${leftSide.join(' L ')} L ${tipLeft} A ${width} ${width} 0 0 1 ${tipRight} L ${rightSide.join(' L ')} Z`;
  }

  // Create palm shape connecting all finger bases
  const createPalmShape = () => {
    // Palm outline: wrist -> thumb base -> around finger bases -> back to wrist
    const wrist = points[0]
    const thumbBase = points[1]
    const indexBase = points[5]
    const middleBase = points[9]
    const ringBase = points[13]
    const pinkyBase = points[17]

    // Offset points outward for palm width
    const palmWidth = fingerWidth * 1.2

    return `
      M ${wrist.x} ${wrist.y + palmWidth}
      Q ${thumbBase.x - palmWidth} ${thumbBase.y} ${thumbBase.x} ${thumbBase.y - palmWidth * 0.5}
      L ${indexBase.x - palmWidth * 0.3} ${indexBase.y - palmWidth * 0.5}
      Q ${(indexBase.x + middleBase.x) / 2} ${Math.min(indexBase.y, middleBase.y) - palmWidth * 0.8}
        ${middleBase.x} ${middleBase.y - palmWidth * 0.5}
      Q ${(middleBase.x + ringBase.x) / 2} ${Math.min(middleBase.y, ringBase.y) - palmWidth * 0.6}
        ${ringBase.x} ${ringBase.y - palmWidth * 0.5}
      Q ${(ringBase.x + pinkyBase.x) / 2} ${Math.min(ringBase.y, pinkyBase.y) - palmWidth * 0.5}
        ${pinkyBase.x + palmWidth * 0.3} ${pinkyBase.y - palmWidth * 0.3}
      L ${pinkyBase.x + palmWidth * 0.5} ${pinkyBase.y + palmWidth * 0.5}
      Q ${wrist.x + palmWidth * 1.5} ${(wrist.y + pinkyBase.y) / 2}
        ${wrist.x} ${wrist.y + palmWidth}
      Z
    `;
  }

  // Finger definitions: [landmark indices]
  const fingers = [
    { indices: [1, 2, 3, 4], width: fingerWidth * 0.9 },      // Thumb (slightly thinner)
    { indices: [5, 6, 7, 8], width: fingerWidth },            // Index
    { indices: [9, 10, 11, 12], width: fingerWidth * 1.05 },  // Middle (slightly thicker)
    { indices: [13, 14, 15, 16], width: fingerWidth * 0.95 }, // Ring
    { indices: [17, 18, 19, 20], width: fingerWidth * 0.85 }, // Pinky (thinnest)
  ]

  return (
    <g opacity={gloveOpacity}>
      {/* Definitions */}
      <defs>
        {/* Main hand gradient - white to soft lavender */}
        <radialGradient id={`hand-fill-${handId}`} cx="30%" cy="25%" r="80%">
          <stop offset="0%" stopColor="#ffffff" />
          <stop offset="40%" stopColor="#f5f5fa" />
          <stop offset="70%" stopColor="#e8e8f0" />
          <stop offset="100%" stopColor="#d8d8e5" />
        </radialGradient>

        {/* Ambient occlusion gradient for creases */}
        <radialGradient id={`ao-gradient-${handId}`} cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#000000" stopOpacity="0" />
          <stop offset="70%" stopColor="#000000" stopOpacity="0.1" />
          <stop offset="100%" stopColor="#000000" stopOpacity="0.25" />
        </radialGradient>

        {/* Rim light gradient */}
        <linearGradient id={`rim-light-${handId}`} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#ffffff" stopOpacity="0.8" />
          <stop offset="50%" stopColor="#ffffff" stopOpacity="0.1" />
          <stop offset="100%" stopColor="#ffffff" stopOpacity="0.4" />
        </linearGradient>

        {/* Soft blur for glow effect */}
        <filter id={`hand-glow-${handId}`} x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur in="SourceGraphic" stdDeviation="0.5" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>

        {/* Drop shadow */}
        <filter id={`hand-shadow-${handId}`} x="-50%" y="-50%" width="200%" height="200%">
          <feDropShadow dx="0.3" dy="0.5" stdDeviation="0.8" floodColor="#000000" floodOpacity="0.3" />
        </filter>
      </defs>

      {/* Shadow layer */}
      <g filter={`url(#hand-shadow-${handId})`}>
        {/* Palm base shape */}
        <path d={createPalmShape()} fill={`url(#hand-fill-${handId})`} />

        {/* Fingers - rendered back to front for proper overlapping */}
        {[...fingers].reverse().map((finger, idx) => (
          <path
            key={`finger-base-${idx}`}
            d={createFingerShape(finger.indices, finger.width)}
            fill={`url(#hand-fill-${handId})`}
          />
        ))}
      </g>

      {/* Ambient occlusion in creases (between fingers) */}
      {[5, 9, 13].map((baseIdx, idx) => {
        const p1 = points[baseIdx]
        const p2 = points[baseIdx + 4]
        return (
          <ellipse
            key={`ao-${idx}`}
            cx={(p1.x + p2.x) / 2}
            cy={(p1.y + p2.y) / 2 - fingerWidth * 0.3}
            rx={fingerWidth * 0.6}
            ry={fingerWidth * 0.4}
            fill={`url(#ao-gradient-${handId})`}
            opacity={0.5}
          />
        )
      })}

      {/* Knuckle definition shadows */}
      {KNUCKLES.map((idx) => {
        const p = points[idx]
        return (
          <circle
            key={`knuckle-shadow-${idx}`}
            cx={p.x}
            cy={p.y + fingerWidth * 0.2}
            r={fingerWidth * 0.5}
            fill="#000000"
            opacity={0.08}
          />
        )
      })}

      {/* Highlight layer - rim lighting effect */}
      <g filter={`url(#hand-glow-${handId})`}>
        {/* Palm highlight */}
        <path
          d={createPalmShape()}
          fill="none"
          stroke={`url(#rim-light-${handId})`}
          strokeWidth={0.3 * clampedScale}
        />

        {/* Finger highlights */}
        {fingers.map((finger, idx) => (
          <path
            key={`finger-highlight-${idx}`}
            d={createFingerShape(finger.indices, finger.width * 0.7)}
            fill="none"
            stroke="#ffffff"
            strokeWidth={0.2 * clampedScale}
            strokeOpacity={0.4}
          />
        ))}

        {/* Fingertip highlights - small specular dots */}
        {FINGERTIPS.map((idx) => {
          const p = points[idx]
          return (
            <circle
              key={`tip-highlight-${idx}`}
              cx={p.x - fingerWidth * 0.15}
              cy={p.y - fingerWidth * 0.15}
              r={fingerWidth * 0.2}
              fill="#ffffff"
              opacity={0.7}
            />
          )
        })}
      </g>

      {/* Wrist cuff */}
      <ellipse
        cx={points[0].x}
        cy={points[0].y + fingerWidth * 0.5}
        rx={fingerWidth * 1.5}
        ry={fingerWidth * 0.8}
        fill="none"
        stroke="#ffffff"
        strokeWidth={fingerWidth * 0.4}
        opacity={0.3}
      />
    </g>
  )
}

export default Hand2DOverlay
