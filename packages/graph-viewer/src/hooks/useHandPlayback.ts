/**
 * Hand Playback Hook
 *
 * Replays recorded hand tracking data for automated testing.
 * Bypasses the WebSocket connection and injects frames directly.
 *
 * Usage:
 * 1. Load a recording from localStorage or file
 * 2. Use the returned gestureState in place of real hand tracking
 * 3. Control playback with play/pause/seek
 *
 * For Chrome automation:
 * - Expose window.__handPlayback for external control
 * - Listen to console logs for gesture events
 */

import { useState, useRef, useCallback, useEffect } from 'react'
import type { GestureState, HandLandmarks } from './useHandGestures'
import type { NormalizedLandmarkList } from '@mediapipe/hands'
import type { HandRecording, RecordedFrame } from './useHandRecording'

export interface PlaybackState {
  isPlaying: boolean
  isPaused: boolean
  currentTime: number // ms
  duration: number // ms
  currentFrame: number
  totalFrames: number
  speed: number // 1.0 = normal, 0.5 = half, 2.0 = double
  isLooped: boolean
  recordingName: string
}

export interface UseHandPlaybackOptions {
  /** Callback when gesture state changes */
  onGestureChange?: (state: GestureState) => void
  /** Callback when playback ends */
  onPlaybackEnd?: () => void
  /** Log gesture events to console for automation */
  logEvents?: boolean
  /** Expose window.__handPlayback for external control */
  exposeGlobal?: boolean
}

// Default empty gesture state
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
  pinchPoint: null,
  leftPinchRay: null,
  rightPinchRay: null,
  activePinchRay: null,
  zoomDelta: 0,
  rotateDelta: 0,
  panDelta: { x: 0, y: 0 },
}

export function useHandPlayback(options: UseHandPlaybackOptions = {}) {
  const { onGestureChange, onPlaybackEnd, logEvents = true, exposeGlobal = true } = options

  const [state, setState] = useState<PlaybackState>({
    isPlaying: false,
    isPaused: false,
    currentTime: 0,
    duration: 0,
    currentFrame: 0,
    totalFrames: 0,
    speed: 1.0,
    isLooped: false,
    recordingName: '',
  })

  const [gestureState, setGestureState] = useState<GestureState>(DEFAULT_STATE)

  const recordingRef = useRef<HandRecording | null>(null)
  const animationFrameRef = useRef<number>(0)
  const playbackStartRef = useRef<number>(0)
  const pausedAtRef = useRef<number>(0)
  const prevGestureRef = useRef<Partial<GestureState>>({})

  // Convert recorded frame to gesture state
  const frameToGestureState = useCallback((frame: RecordedFrame): GestureState => {
    const convertLandmarks = (landmarks: { x: number; y: number; z: number; visibility?: number }[] | null): NormalizedLandmarkList | null => {
      if (!landmarks) return null
      return landmarks.map((lm) => ({
        x: lm.x,
        y: lm.y,
        z: lm.z,
        visibility: lm.visibility ?? 1,
      }))
    }

    const leftLandmarks = convertLandmarks(frame.leftLandmarks)
    const rightLandmarks = convertLandmarks(frame.rightLandmarks)
    const leftWorldLandmarks = convertLandmarks(frame.leftWorldLandmarks)
    const rightWorldLandmarks = convertLandmarks(frame.rightWorldLandmarks)

    const leftHand: HandLandmarks | null = leftLandmarks
      ? { landmarks: leftLandmarks, worldLandmarks: leftWorldLandmarks || leftLandmarks, handedness: 'Left' }
      : null

    const rightHand: HandLandmarks | null = rightLandmarks
      ? { landmarks: rightLandmarks, worldLandmarks: rightWorldLandmarks || rightLandmarks, handedness: 'Right' }
      : null

    return {
      ...DEFAULT_STATE,
      isTracking: frame.handsDetected > 0,
      handsDetected: frame.handsDetected,
      leftHand,
      rightHand,
      pinchStrength: frame.metrics.pinchStrength,
      grabStrength: frame.metrics.grabStrength,
      // Restore other fields from recorded gesture state
      pointingHand: frame.gestureState.pointingHand ?? null,
      pointDirection: frame.gestureState.pointDirection ?? null,
      twoHandDistance: frame.gestureState.twoHandDistance ?? 0.5,
      twoHandRotation: frame.gestureState.twoHandRotation ?? 0,
      twoHandCenter: frame.gestureState.twoHandCenter ?? { x: 0.5, y: 0.5 },
    }
  }, [])

  // Find frame at given time
  const getFrameAtTime = useCallback((time: number): RecordedFrame | null => {
    const recording = recordingRef.current
    if (!recording || recording.frames.length === 0) return null

    // Binary search for efficiency
    const frames = recording.frames
    let left = 0
    let right = frames.length - 1

    while (left < right) {
      const mid = Math.floor((left + right) / 2)
      if (frames[mid].timestamp < time) {
        left = mid + 1
      } else {
        right = mid
      }
    }

    // Return closest frame
    if (left > 0 && time - frames[left - 1].timestamp < frames[left].timestamp - time) {
      return frames[left - 1]
    }
    return frames[left]
  }, [])

  // Log gesture events for automation
  const logGestureEvent = useCallback(
    (newState: GestureState) => {
      if (!logEvents) return

      const prev = prevGestureRef.current

      // Detect state changes
      if (newState.pinchStrength > 0.85 && (prev.pinchStrength ?? 0) <= 0.85) {
        console.log('[GESTURE] PINCH_START', { strength: newState.pinchStrength })
      }
      if (newState.pinchStrength < 0.65 && (prev.pinchStrength ?? 0) >= 0.65) {
        console.log('[GESTURE] PINCH_END', { strength: newState.pinchStrength })
      }
      if (newState.grabStrength > 0.72 && (prev.grabStrength ?? 0) <= 0.72) {
        console.log('[GESTURE] GRAB_START', { strength: newState.grabStrength })
      }
      if (newState.grabStrength < 0.45 && (prev.grabStrength ?? 0) >= 0.45) {
        console.log('[GESTURE] GRAB_END', { strength: newState.grabStrength })
      }
      if (newState.pointingHand && !prev.pointingHand) {
        console.log('[GESTURE] POINT_START', { hand: newState.pointingHand, direction: newState.pointDirection })
      }
      if (!newState.pointingHand && prev.pointingHand) {
        console.log('[GESTURE] POINT_END')
      }

      prevGestureRef.current = {
        pinchStrength: newState.pinchStrength,
        grabStrength: newState.grabStrength,
        pointingHand: newState.pointingHand,
      }
    },
    [logEvents]
  )

  // Playback loop
  const tick = useCallback(() => {
    if (!state.isPlaying || state.isPaused) return

    const recording = recordingRef.current
    if (!recording) return

    const now = performance.now()
    const elapsed = (now - playbackStartRef.current) * state.speed
    let currentTime = elapsed

    // Handle looping
    if (currentTime >= recording.metadata.duration) {
      if (state.isLooped) {
        playbackStartRef.current = now
        currentTime = 0
      } else {
        // End playback
        setState((prev) => ({
          ...prev,
          isPlaying: false,
          currentTime: recording.metadata.duration,
          currentFrame: recording.frames.length - 1,
        }))
        setGestureState(DEFAULT_STATE)
        onPlaybackEnd?.()
        console.log('[PLAYBACK] END')
        return
      }
    }

    // Get frame at current time
    const frame = getFrameAtTime(currentTime)
    if (frame) {
      const newGestureState = frameToGestureState(frame)
      setGestureState(newGestureState)
      onGestureChange?.(newGestureState)
      logGestureEvent(newGestureState)

      const frameIndex = recording.frames.indexOf(frame)
      setState((prev) => ({
        ...prev,
        currentTime,
        currentFrame: frameIndex,
      }))
    }

    animationFrameRef.current = requestAnimationFrame(tick)
  }, [state.isPlaying, state.isPaused, state.speed, state.isLooped, getFrameAtTime, frameToGestureState, onGestureChange, onPlaybackEnd, logGestureEvent])

  // Start playback
  const play = useCallback(() => {
    const recording = recordingRef.current
    if (!recording) {
      console.warn('[PLAYBACK] No recording loaded')
      return
    }

    if (state.isPaused) {
      // Resume from pause
      playbackStartRef.current = performance.now() - pausedAtRef.current / state.speed
    } else {
      // Start fresh
      playbackStartRef.current = performance.now()
    }

    setState((prev) => ({
      ...prev,
      isPlaying: true,
      isPaused: false,
    }))

    console.log('[PLAYBACK] START', recording.metadata.name)
    animationFrameRef.current = requestAnimationFrame(tick)
  }, [state.isPaused, state.speed, tick])

  // Pause playback
  const pause = useCallback(() => {
    cancelAnimationFrame(animationFrameRef.current)
    pausedAtRef.current = state.currentTime

    setState((prev) => ({
      ...prev,
      isPaused: true,
    }))

    console.log('[PLAYBACK] PAUSE at', state.currentTime)
  }, [state.currentTime])

  // Stop playback
  const stop = useCallback(() => {
    cancelAnimationFrame(animationFrameRef.current)

    setState((prev) => ({
      ...prev,
      isPlaying: false,
      isPaused: false,
      currentTime: 0,
      currentFrame: 0,
    }))

    setGestureState(DEFAULT_STATE)
    console.log('[PLAYBACK] STOP')
  }, [])

  // Seek to time
  const seek = useCallback(
    (time: number) => {
      const recording = recordingRef.current
      if (!recording) return

      const clampedTime = Math.max(0, Math.min(time, recording.metadata.duration))
      const frame = getFrameAtTime(clampedTime)

      if (frame) {
        const newGestureState = frameToGestureState(frame)
        setGestureState(newGestureState)
        onGestureChange?.(newGestureState)

        const frameIndex = recording.frames.indexOf(frame)
        setState((prev) => ({
          ...prev,
          currentTime: clampedTime,
          currentFrame: frameIndex,
        }))

        if (state.isPlaying && !state.isPaused) {
          playbackStartRef.current = performance.now() - clampedTime / state.speed
        } else {
          pausedAtRef.current = clampedTime
        }
      }

      console.log('[PLAYBACK] SEEK to', clampedTime)
    },
    [getFrameAtTime, frameToGestureState, onGestureChange, state.isPlaying, state.isPaused, state.speed]
  )

  // Load recording
  const loadRecording = useCallback((recording: HandRecording) => {
    cancelAnimationFrame(animationFrameRef.current)
    recordingRef.current = recording
    prevGestureRef.current = {}

    setState({
      isPlaying: false,
      isPaused: false,
      currentTime: 0,
      duration: recording.metadata.duration,
      currentFrame: 0,
      totalFrames: recording.frames.length,
      speed: 1.0,
      isLooped: false,
      recordingName: recording.metadata.name,
    })

    setGestureState(DEFAULT_STATE)
    console.log('[PLAYBACK] LOADED', recording.metadata.name, recording.frames.length, 'frames')
  }, [])

  // Set playback speed
  const setSpeed = useCallback((speed: number) => {
    setState((prev) => ({
      ...prev,
      speed: Math.max(0.1, Math.min(5.0, speed)),
    }))
  }, [])

  // Set loop mode
  const setLooped = useCallback((looped: boolean) => {
    setState((prev) => ({ ...prev, isLooped: looped }))
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cancelAnimationFrame(animationFrameRef.current)
    }
  }, [])

  // Effect to continue playback loop
  useEffect(() => {
    if (state.isPlaying && !state.isPaused) {
      animationFrameRef.current = requestAnimationFrame(tick)
    }
    return () => {
      cancelAnimationFrame(animationFrameRef.current)
    }
  }, [state.isPlaying, state.isPaused, tick])

  // Expose global interface for Chrome automation
  useEffect(() => {
    if (exposeGlobal) {
      const api = {
        loadRecording,
        play,
        pause,
        stop,
        seek,
        setSpeed,
        setLooped,
        getState: () => state,
        getGestureState: () => gestureState,
        getRecording: () => recordingRef.current,
      }
      ;(window as unknown as Record<string, unknown>).__handPlayback = api
      console.log('[PLAYBACK] Exposed window.__handPlayback for automation')
    }

    return () => {
      if (exposeGlobal) {
        delete (window as unknown as Record<string, unknown>).__handPlayback
      }
    }
  }, [exposeGlobal, loadRecording, play, pause, stop, seek, setSpeed, setLooped, state, gestureState])

  return {
    // Current gesture state (use this instead of real hand tracking)
    gestureState,

    // Playback state
    isPlaying: state.isPlaying,
    isPaused: state.isPaused,
    currentTime: state.currentTime,
    duration: state.duration,
    currentFrame: state.currentFrame,
    totalFrames: state.totalFrames,
    speed: state.speed,
    isLooped: state.isLooped,
    recordingName: state.recordingName,

    // Controls
    loadRecording,
    play,
    pause,
    stop,
    seek,
    setSpeed,
    setLooped,

    // Computed
    progress: state.duration > 0 ? state.currentTime / state.duration : 0,
    hasRecording: recordingRef.current !== null,
  }
}

export default useHandPlayback
