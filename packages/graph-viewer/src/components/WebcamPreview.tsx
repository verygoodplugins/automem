/**
 * Webcam Preview Component
 *
 * Shows a small video preview with hand skeleton overlay in the corner.
 * Similar to the attractors demo - provides visual feedback about what
 * the hand tracking system is detecting.
 *
 * Features:
 * - Live webcam feed in a compact corner view
 * - Hand skeleton drawn on canvas overlay
 * - Smooth animations and glassy styling
 * - Detection status indicator
 */

import { useEffect, useRef } from 'react'
import type { GestureState } from '../hooks/useHandGestures'

// MediaPipe hand landmark connections for drawing skeleton
const HAND_CONNECTIONS = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index finger
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle finger
  [5, 9], [9, 10], [10, 11], [11, 12],
  // Ring finger
  [9, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [13, 17], [17, 18], [18, 19], [19, 20],
  // Palm
  [0, 17],
]

// Fingertip indices for larger dots
const FINGERTIPS = [4, 8, 12, 16, 20]

interface WebcamPreviewProps {
  videoElement: HTMLVideoElement | null
  gestureState: GestureState
  visible: boolean
  position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left'
  size?: 'small' | 'medium' | 'large'
}

export function WebcamPreview({
  videoElement,
  gestureState,
  visible,
  position = 'top-right',
  size = 'medium',
}: WebcamPreviewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  // Size configurations
  const sizeConfig = {
    small: { width: 160, height: 120 },
    medium: { width: 240, height: 180 },
    large: { width: 320, height: 240 },
  }

  const { width, height } = sizeConfig[size]

  // Position configurations
  const positionClasses = {
    'top-right': 'top-20 right-4',
    'top-left': 'top-20 left-4',
    'bottom-right': 'bottom-4 right-4',
    'bottom-left': 'bottom-4 left-4',
  }

  // Draw video frame and hand skeleton
  useEffect(() => {
    if (!visible || !videoElement || !canvasRef.current) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let animationId: number

    const draw = () => {
      // Clear canvas
      ctx.clearRect(0, 0, width, height)

      // Draw video frame (unmirrored to match main 3D hand overlay)
      ctx.drawImage(videoElement, 0, 0, width, height)

      // Draw hand skeletons
      const drawHand = (
        landmarks: { x: number; y: number; z?: number }[] | undefined,
        color: string,
        glowColor: string
      ) => {
        if (!landmarks || landmarks.length === 0) return

        // Draw connections (skeleton lines)
        ctx.strokeStyle = color
        ctx.lineWidth = 2
        ctx.lineCap = 'round'

        // Add glow effect
        ctx.shadowColor = glowColor
        ctx.shadowBlur = 8

        for (const [start, end] of HAND_CONNECTIONS) {
          const p1 = landmarks[start]
          const p2 = landmarks[end]
          if (!p1 || !p2) continue

          // Use raw coordinates to match main 3D hand overlay
          const x1 = p1.x * width
          const y1 = p1.y * height
          const x2 = p2.x * width
          const y2 = p2.y * height

          ctx.beginPath()
          ctx.moveTo(x1, y1)
          ctx.lineTo(x2, y2)
          ctx.stroke()
        }

        // Draw landmark points
        for (let i = 0; i < landmarks.length; i++) {
          const lm = landmarks[i]
          const x = lm.x * width
          const y = lm.y * height

          // Larger dots for fingertips and wrist
          const radius = FINGERTIPS.includes(i) ? 5 : i === 0 ? 6 : 3

          ctx.beginPath()
          ctx.arc(x, y, radius, 0, Math.PI * 2)
          ctx.fillStyle = FINGERTIPS.includes(i) ? '#ffffff' : color
          ctx.fill()

          // Outline for visibility
          ctx.strokeStyle = 'rgba(0,0,0,0.5)'
          ctx.lineWidth = 1
          ctx.shadowBlur = 0
          ctx.stroke()
          ctx.shadowBlur = 8
        }
      }

      // Draw left hand (cyan)
      if (gestureState.leftHand) {
        drawHand(
          gestureState.leftHand.landmarks as { x: number; y: number; z?: number }[],
          '#4ecdc4',
          'rgba(78, 205, 196, 0.6)'
        )
      }

      // Draw right hand (magenta/pink)
      if (gestureState.rightHand) {
        drawHand(
          gestureState.rightHand.landmarks as { x: number; y: number; z?: number }[],
          '#f72585',
          'rgba(247, 37, 133, 0.6)'
        )
      }

      // Reset shadow
      ctx.shadowBlur = 0

      animationId = requestAnimationFrame(draw)
    }

    draw()

    return () => {
      cancelAnimationFrame(animationId)
    }
  }, [visible, videoElement, gestureState, width, height])

  if (!visible) return null

  return (
    <div
      ref={containerRef}
      className={`fixed ${positionClasses[position]} z-50 pointer-events-auto`}
    >
      <div className="glass border border-white/10 rounded-xl overflow-hidden shadow-2xl">
        {/* Header bar */}
        <div className="flex items-center justify-between px-3 py-1.5 bg-black/30 border-b border-white/5">
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${
                gestureState.handsDetected > 0
                  ? 'bg-emerald-400 animate-pulse'
                  : 'bg-slate-500'
              }`}
            />
            <span className="text-[10px] text-slate-300 font-medium">
              {gestureState.handsDetected > 0
                ? `${gestureState.handsDetected} hand${gestureState.handsDetected > 1 ? 's' : ''}`
                : 'No hands'}
            </span>
          </div>
          <span className="text-[9px] text-slate-500">WEBCAM</span>
        </div>

        {/* Video/Canvas container */}
        <div className="relative" style={{ width, height }}>
          {/* Video is rendered via canvas for overlay */}
          <canvas
            ref={canvasRef}
            width={width}
            height={height}
            className="w-full h-full"
          />

          {/* No hands detected overlay */}
          {gestureState.isTracking && gestureState.handsDetected === 0 && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/40">
              <div className="text-center">
                <div className="text-slate-400 text-xs">Show your hands</div>
                <div className="text-slate-500 text-[10px] mt-1">to the camera</div>
              </div>
            </div>
          )}

          {/* Not tracking overlay */}
          {!gestureState.isTracking && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/60">
              <div className="text-center">
                <div className="text-slate-400 text-xs">Initializing...</div>
                <div className="text-slate-500 text-[10px] mt-1">camera access</div>
              </div>
            </div>
          )}
        </div>

        {/* Footer with gesture hints */}
        <div className="px-3 py-1.5 bg-black/30 border-t border-white/5">
          <div className="flex items-center justify-between text-[9px] text-slate-500">
            <span>
              {gestureState.leftHand && (
                <span className="text-cyan-400 mr-2">L</span>
              )}
              {gestureState.rightHand && (
                <span className="text-pink-400">R</span>
              )}
              {!gestureState.leftHand && !gestureState.rightHand && (
                <span>-</span>
              )}
            </span>
            <span>
              {gestureState.pinchStrength > 0.5 && (
                <span className="text-orange-400">PINCH</span>
              )}
              {gestureState.grabStrength > 0.7 && (
                <span className="text-red-400 ml-1">GRAB</span>
              )}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default WebcamPreview
