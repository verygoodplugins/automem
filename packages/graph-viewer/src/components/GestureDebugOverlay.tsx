/**
 * Gesture Debug Overlay
 *
 * Shows all hand tracking data in real-time for debugging:
 * - Raw landmark positions
 * - Computed gesture values
 * - Hand detection confidence
 * - FPS counter
 */

import { useEffect, useRef, useState } from 'react'
import type { GestureState } from '../hooks/useHandGestures'

interface GestureDebugOverlayProps {
  gestureState: GestureState
  visible: boolean
}

export function GestureDebugOverlay({ gestureState, visible }: GestureDebugOverlayProps) {
  const [fps, setFps] = useState(0)
  const frameCountRef = useRef(0)
  const lastTimeRef = useRef(performance.now())

  // FPS counter
  useEffect(() => {
    frameCountRef.current++
    const now = performance.now()
    if (now - lastTimeRef.current >= 1000) {
      setFps(frameCountRef.current)
      frameCountRef.current = 0
      lastTimeRef.current = now
    }
  }, [gestureState])

  if (!visible) return null

  const {
    isTracking,
    handsDetected,
    leftHand,
    rightHand,
    twoHandDistance,
    twoHandRotation,
    twoHandCenter,
    pointingHand,
    pointDirection,
    pinchStrength,
    grabStrength,
    leftPinchRay,
    rightPinchRay,
    activePinchRay,
    zoomDelta,
    rotateDelta,
    panDelta,
  } = gestureState

  // Format number for display
  const fmt = (n: number, decimals = 3) => n.toFixed(decimals)
  const fmtMeters = (n: number) => `${fmt(n, 2)}m`

  // Get landmark name
  const landmarkName = (i: number) => {
    const names = [
      'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
      'INDEX_MCP', 'INDEX_PIP', 'INDEX_DIP', 'INDEX_TIP',
      'MIDDLE_MCP', 'MIDDLE_PIP', 'MIDDLE_DIP', 'MIDDLE_TIP',
      'RING_MCP', 'RING_PIP', 'RING_DIP', 'RING_TIP',
      'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP',
    ]
    return names[i] || `L${i}`
  }

  return (
    <div className="fixed top-16 left-4 z-50 font-mono text-xs bg-black/80 text-green-400 p-4 rounded-lg border border-green-500/30 max-h-[calc(100vh-100px)] overflow-y-auto w-80">
      {/* Header */}
      <div className="flex items-center justify-between mb-3 pb-2 border-b border-green-500/30">
        <span className="text-green-300 font-bold">GESTURE DEBUG</span>
        <span className={`px-2 py-0.5 rounded ${fps >= 25 ? 'bg-green-500/20' : fps >= 15 ? 'bg-yellow-500/20' : 'bg-red-500/20'}`}>
          {fps} FPS
        </span>
      </div>

      {/* Tracking Status */}
      <div className="mb-3">
        <div className="text-green-300 mb-1">Status</div>
        <div className="grid grid-cols-2 gap-1 text-[10px]">
          <div>Tracking: <span className={isTracking ? 'text-green-400' : 'text-red-400'}>{isTracking ? 'YES' : 'NO'}</span></div>
          <div>Hands: <span className="text-cyan-400">{handsDetected}</span></div>
        </div>
      </div>

      {/* Two-Hand Gestures */}
      <div className="mb-3">
        <div className="text-green-300 mb-1">Two-Hand Gestures</div>
        <div className="grid grid-cols-1 gap-0.5 text-[10px]">
          <div className="flex justify-between">
            <span>Distance:</span>
            <span className="text-yellow-400">{fmt(twoHandDistance)}</span>
            <div className="w-20 h-2 bg-gray-700 rounded overflow-hidden">
              <div
                className="h-full bg-yellow-400 transition-all duration-75"
                style={{ width: `${Math.min(100, twoHandDistance * 100)}%` }}
              />
            </div>
          </div>
          <div className="flex justify-between">
            <span>Rotation:</span>
            <span className="text-purple-400">{fmt(twoHandRotation * 180 / Math.PI)}°</span>
          </div>
          <div className="flex justify-between">
            <span>Center:</span>
            <span className="text-blue-400">({fmt(twoHandCenter.x, 2)}, {fmt(twoHandCenter.y, 2)})</span>
          </div>
          <div className="flex justify-between">
            <span>Zoom Δ:</span>
            <span className={zoomDelta > 0 ? 'text-green-400' : zoomDelta < 0 ? 'text-red-400' : 'text-gray-400'}>
              {zoomDelta > 0 ? '+' : ''}{fmt(zoomDelta, 4)}
            </span>
          </div>
          <div className="flex justify-between">
            <span>Rotate Δ:</span>
            <span className="text-purple-400">{fmt(rotateDelta * 180 / Math.PI, 2)}°</span>
          </div>
          <div className="flex justify-between">
            <span>Pan Δ:</span>
            <span className="text-blue-400">({fmt(panDelta.x, 3)}, {fmt(panDelta.y, 3)})</span>
          </div>
        </div>
      </div>

      {/* Single-Hand Gestures */}
      <div className="mb-3">
        <div className="text-green-300 mb-1">Single-Hand Gestures</div>
        <div className="grid grid-cols-1 gap-0.5 text-[10px]">
          <div className="flex justify-between">
            <span>Pointing:</span>
            <span className={pointingHand ? 'text-cyan-400' : 'text-gray-500'}>
              {pointingHand || 'NONE'}
            </span>
          </div>
          {pointDirection && (
            <div className="flex justify-between">
              <span>Point Dir:</span>
              <span className="text-cyan-400">({fmt(pointDirection.x, 2)}, {fmt(pointDirection.y, 2)})</span>
            </div>
          )}
          <div className="flex justify-between items-center">
            <span>Pinch:</span>
            <span className="text-orange-400">{fmt(pinchStrength, 2)}</span>
            <div className="w-20 h-2 bg-gray-700 rounded overflow-hidden">
              <div
                className={`h-full transition-all duration-75 ${pinchStrength > 0.7 ? 'bg-orange-500' : 'bg-orange-400/50'}`}
                style={{ width: `${pinchStrength * 100}%` }}
              />
            </div>
          </div>
          <div className="flex justify-between items-center">
            <span>Grab:</span>
            <span className="text-red-400">{fmt(grabStrength, 2)}</span>
            <div className="w-20 h-2 bg-gray-700 rounded overflow-hidden">
              <div
                className={`h-full transition-all duration-75 ${grabStrength > 0.7 ? 'bg-red-500' : 'bg-red-400/50'}`}
                style={{ width: `${grabStrength * 100}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Pinch Ray (Laser Pointer) */}
      <div className="mb-3">
        <div className="text-green-300 mb-1">Pinch Ray (Laser)</div>
        <div className="grid grid-cols-1 gap-0.5 text-[10px]">
          <div className="flex justify-between">
            <span>Active:</span>
            <span className={activePinchRay?.isValid ? 'text-green-400' : 'text-gray-500'}>
              {activePinchRay?.isValid ? 'YES' : 'NO'}
            </span>
          </div>
          {leftPinchRay && (
            <>
              <div className="flex justify-between items-center">
                <span>L Pinch:</span>
                <span className={leftPinchRay.isValid ? 'text-cyan-400' : 'text-gray-500'}>
                  {fmt(leftPinchRay.strength, 2)}
                </span>
                <div className="w-16 h-2 bg-gray-700 rounded overflow-hidden">
                  <div
                    className={`h-full transition-all duration-75 ${leftPinchRay.isValid ? 'bg-cyan-500' : 'bg-cyan-400/30'}`}
                    style={{ width: `${leftPinchRay.strength * 100}%` }}
                  />
                </div>
              </div>
              <div className="flex justify-between">
                <span>L Origin:</span>
                <span className="text-cyan-400 text-[9px]">
                  ({fmt(leftPinchRay.origin.x, 2)}, {fmt(leftPinchRay.origin.y, 2)}, {fmt(leftPinchRay.origin.z, 2)})
                </span>
              </div>
            </>
          )}
          {rightPinchRay && (
            <>
              <div className="flex justify-between items-center">
                <span>R Pinch:</span>
                <span className={rightPinchRay.isValid ? 'text-pink-400' : 'text-gray-500'}>
                  {fmt(rightPinchRay.strength, 2)}
                </span>
                <div className="w-16 h-2 bg-gray-700 rounded overflow-hidden">
                  <div
                    className={`h-full transition-all duration-75 ${rightPinchRay.isValid ? 'bg-pink-500' : 'bg-pink-400/30'}`}
                    style={{ width: `${rightPinchRay.strength * 100}%` }}
                  />
                </div>
              </div>
              <div className="flex justify-between">
                <span>R Origin:</span>
                <span className="text-pink-400 text-[9px]">
                  ({fmt(rightPinchRay.origin.x, 2)}, {fmt(rightPinchRay.origin.y, 2)}, {fmt(rightPinchRay.origin.z, 2)})
                </span>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Left Hand Landmarks */}
      {leftHand && (
        <div className="mb-3">
          <div className="text-cyan-300 mb-1">Left Hand (21 landmarks)</div>
          <div className="grid grid-cols-1 gap-0.5 text-[9px] max-h-32 overflow-y-auto">
            {[0, 4, 8, 12, 16, 20].map((i) => {
              const lm = leftHand.landmarks[i]
              const wm = leftHand.worldLandmarks?.[i]
              const worldZ = (wm?.z ?? 0) as number
              return (
                <div key={i} className="flex justify-between">
                  <span className="text-gray-400">{landmarkName(i)}:</span>
                  <span className="text-cyan-400">
                    ({fmt(lm.x, 2)}, {fmt(lm.y, 2)}, {fmt((lm.z || 0) as number, 2)}
                    {worldZ > 0 ? ` | ${fmtMeters(worldZ)}` : ''})
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Right Hand Landmarks */}
      {rightHand && (
        <div className="mb-3">
          <div className="text-pink-300 mb-1">Right Hand (21 landmarks)</div>
          <div className="grid grid-cols-1 gap-0.5 text-[9px] max-h-32 overflow-y-auto">
            {[0, 4, 8, 12, 16, 20].map((i) => {
              const lm = rightHand.landmarks[i]
              const wm = rightHand.worldLandmarks?.[i]
              const worldZ = (wm?.z ?? 0) as number
              return (
                <div key={i} className="flex justify-between">
                  <span className="text-gray-400">{landmarkName(i)}:</span>
                  <span className="text-pink-400">
                    ({fmt(lm.x, 2)}, {fmt(lm.y, 2)}, {fmt((lm.z || 0) as number, 2)}
                    {worldZ > 0 ? ` | ${fmtMeters(worldZ)}` : ''})
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Instructions */}
      <div className="mt-3 pt-2 border-t border-green-500/30 text-[9px] text-gray-400">
        <div>TIP: Landmarks x,y are 0-1 normalized</div>
        <div>TIP: z is depth relative to wrist</div>
      </div>
    </div>
  )
}

export default GestureDebugOverlay
