import type { HandLockState } from '../hooks/useHandLockAndGrab'

interface HandControlOverlayProps {
  enabled: boolean
  lock: HandLockState
  source: 'mediapipe' | 'iphone'
}

export function HandControlOverlay({ enabled, lock, source }: HandControlOverlayProps) {
  if (!enabled) return null

  const badge =
    lock.mode === 'locked'
      ? lock.grabbed
        ? { text: 'GRABBED', color: 'bg-emerald-500/20 text-emerald-200 border-emerald-400/30' }
        : { text: 'LOCKED', color: 'bg-cyan-500/20 text-cyan-200 border-cyan-400/30' }
      : lock.mode === 'candidate'
        ? { text: `ACQUIRING (${lock.frames})`, color: 'bg-yellow-500/20 text-yellow-200 border-yellow-400/30' }
        : { text: 'IDLE', color: 'bg-slate-500/20 text-slate-200 border-slate-400/30' }

  const m = lock.mode === 'idle' ? lock.metrics : lock.metrics

  return (
    <div className="absolute left-4 bottom-4 z-50 pointer-events-none">
      <div className="glass border border-white/10 rounded-xl px-4 py-3 text-xs text-slate-200 space-y-2 w-[280px]">
        <div className="flex items-center justify-between">
          <span className="text-slate-400">Hand Control</span>
          <span className={`px-2 py-1 rounded-md border ${badge.color}`}>{badge.text}</span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-slate-400">Source</span>
          <span className="font-medium">{source === 'iphone' ? 'iPhone (LiDAR)' : 'Webcam (MediaPipe)'}</span>
        </div>

        {m && (
          <div className="grid grid-cols-2 gap-x-3 gap-y-1">
            <div className="flex justify-between">
              <span className="text-slate-400">spread</span>
              <span>{m.spread.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">palm</span>
              <span>{m.palmFacing.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">point</span>
              <span>{m.point.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">pinch</span>
              <span>{m.pinch.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">grab</span>
              <span>{m.grab.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">depth</span>
              <span>{m.depth.toFixed(3)}</span>
            </div>
          </div>
        )}

        <div className="pt-1 text-[11px] text-slate-400 leading-snug">
          <div>
            <span className="text-slate-300">Acquire:</span> raise open palm + spread fingers
          </div>
          <div>
            <span className="text-slate-300">Navigate:</span> make fist; pull/push to zoom; move to rotate
          </div>
          <div>
            <span className="text-slate-300">Select:</span> point (index out) + pinch thumb/index to click
          </div>
        </div>
      </div>
    </div>
  )
}

export default HandControlOverlay
