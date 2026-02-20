/**
 * Hand Recording Hook
 *
 * Records hand tracking data for playback and automated testing.
 * Captures raw landmarks, computed metrics, and gesture state.
 *
 * Usage:
 * 1. Press 'R' to start/stop recording
 * 2. Recordings are saved to localStorage and can be downloaded as JSON
 * 3. Use useHandPlayback to replay recordings without iPhone
 */

import { useState, useRef, useCallback, useEffect } from 'react'
import type { GestureState } from './useHandGestures'

// Recording data structures
export interface RecordedLandmark {
  x: number
  y: number
  z: number
  visibility?: number
}

export interface RecordedMetrics {
  pinchStrength: number
  grabStrength: number
  spreadAmount?: number
  palmFacing?: number
  pointScore?: number
}

export interface RecordedFrame {
  timestamp: number // ms from recording start
  leftLandmarks: RecordedLandmark[] | null
  rightLandmarks: RecordedLandmark[] | null
  leftWorldLandmarks: RecordedLandmark[] | null
  rightWorldLandmarks: RecordedLandmark[] | null
  metrics: RecordedMetrics
  hasLiDARDepth: boolean
  handsDetected: number
  // Full gesture state for validation
  gestureState: Partial<GestureState>
}

export interface HandRecording {
  metadata: {
    id: string
    name: string
    description: string
    recordedAt: string
    duration: number // ms
    frameCount: number
    avgFps: number
  }
  frames: RecordedFrame[]
}

export interface UseHandRecordingOptions {
  /** Maximum recording duration in ms. Default: 60000 (1 minute) */
  maxDuration?: number
  /** Keyboard key to toggle recording. Default: 'r' */
  toggleKey?: string
  /** Auto-download JSON when recording stops. Default: false */
  autoDownload?: boolean
}

interface RecordingState {
  isRecording: boolean
  isPaused: boolean
  duration: number
  frameCount: number
  recordingName: string
}

export function useHandRecording(options: UseHandRecordingOptions = {}) {
  const { maxDuration = 60000, toggleKey = 'r', autoDownload = false } = options

  const [state, setState] = useState<RecordingState>({
    isRecording: false,
    isPaused: false,
    duration: 0,
    frameCount: 0,
    recordingName: '',
  })

  const framesRef = useRef<RecordedFrame[]>([])
  const startTimeRef = useRef<number>(0)
  const recordingIdRef = useRef<string>('')

  // Start a new recording
  const startRecording = useCallback((name?: string) => {
    const id = `rec_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    recordingIdRef.current = id
    framesRef.current = []
    startTimeRef.current = Date.now()

    setState({
      isRecording: true,
      isPaused: false,
      duration: 0,
      frameCount: 0,
      recordingName: name || `Recording ${new Date().toLocaleTimeString()}`,
    })

    console.log('🔴 Recording started:', id)
  }, [])

  // Stop recording and return the data
  const stopRecording = useCallback((): HandRecording | null => {
    if (!state.isRecording) return null

    const endTime = Date.now()
    const duration = endTime - startTimeRef.current
    const frames = framesRef.current
    const avgFps = frames.length > 0 ? (frames.length / duration) * 1000 : 0

    const recording: HandRecording = {
      metadata: {
        id: recordingIdRef.current,
        name: state.recordingName,
        description: '',
        recordedAt: new Date(startTimeRef.current).toISOString(),
        duration,
        frameCount: frames.length,
        avgFps: Math.round(avgFps * 10) / 10,
      },
      frames,
    }

    // Save to localStorage
    const key = `hand_recording_${recordingIdRef.current}`
    try {
      localStorage.setItem(key, JSON.stringify(recording))
      console.log('💾 Recording saved to localStorage:', key)
    } catch (e) {
      console.warn('Failed to save to localStorage:', e)
    }

    setState({
      isRecording: false,
      isPaused: false,
      duration: 0,
      frameCount: 0,
      recordingName: '',
    })

    console.log('⏹️ Recording stopped:', frames.length, 'frames,', duration, 'ms')

    // Auto-download if enabled
    if (autoDownload) {
      downloadRecording(recording)
    }

    return recording
  }, [state.isRecording, state.recordingName, autoDownload])

  // Record a single frame
  const recordFrame = useCallback(
    (gestureState: GestureState, hasLiDARDepth: boolean = false) => {
      if (!state.isRecording || state.isPaused) return

      const now = Date.now()
      const elapsed = now - startTimeRef.current

      // Check max duration
      if (elapsed > maxDuration) {
        stopRecording()
        return
      }

      const frame: RecordedFrame = {
        timestamp: elapsed,
        leftLandmarks: gestureState.leftHand?.landmarks.map((lm) => ({
          x: lm.x,
          y: lm.y,
          z: lm.z ?? 0,
          visibility: lm.visibility,
        })) ?? null,
        rightLandmarks: gestureState.rightHand?.landmarks.map((lm) => ({
          x: lm.x,
          y: lm.y,
          z: lm.z ?? 0,
          visibility: lm.visibility,
        })) ?? null,
        leftWorldLandmarks: gestureState.leftHand?.worldLandmarks?.map((lm) => ({
          x: lm.x,
          y: lm.y,
          z: lm.z ?? 0,
          visibility: lm.visibility,
        })) ?? null,
        rightWorldLandmarks: gestureState.rightHand?.worldLandmarks?.map((lm) => ({
          x: lm.x,
          y: lm.y,
          z: lm.z ?? 0,
          visibility: lm.visibility,
        })) ?? null,
        metrics: {
          pinchStrength: gestureState.pinchStrength,
          grabStrength: gestureState.grabStrength,
        },
        hasLiDARDepth,
        handsDetected: gestureState.handsDetected,
        // Include relevant gesture state fields
        gestureState: {
          isTracking: gestureState.isTracking,
          handsDetected: gestureState.handsDetected,
          pointingHand: gestureState.pointingHand,
          pointDirection: gestureState.pointDirection,
          pinchStrength: gestureState.pinchStrength,
          grabStrength: gestureState.grabStrength,
          twoHandDistance: gestureState.twoHandDistance,
          twoHandRotation: gestureState.twoHandRotation,
          twoHandCenter: gestureState.twoHandCenter,
        },
      }

      framesRef.current.push(frame)

      // Update state periodically (every 10 frames to reduce re-renders)
      if (framesRef.current.length % 10 === 0) {
        setState((prev) => ({
          ...prev,
          duration: elapsed,
          frameCount: framesRef.current.length,
        }))
      }
    },
    [state.isRecording, state.isPaused, maxDuration, stopRecording]
  )

  // Toggle recording on/off
  const toggleRecording = useCallback(() => {
    if (state.isRecording) {
      return stopRecording()
    } else {
      startRecording()
      return null
    }
  }, [state.isRecording, startRecording, stopRecording])

  // Pause/resume
  const pauseRecording = useCallback(() => {
    if (state.isRecording) {
      setState((prev) => ({ ...prev, isPaused: true }))
    }
  }, [state.isRecording])

  const resumeRecording = useCallback(() => {
    if (state.isRecording && state.isPaused) {
      setState((prev) => ({ ...prev, isPaused: false }))
    }
  }, [state.isRecording, state.isPaused])

  // Keyboard shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return
      }

      if (e.key.toLowerCase() === toggleKey.toLowerCase() && !e.metaKey && !e.ctrlKey) {
        e.preventDefault()
        toggleRecording()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [toggleKey, toggleRecording])

  return {
    // State
    isRecording: state.isRecording,
    isPaused: state.isPaused,
    duration: state.duration,
    frameCount: state.frameCount,
    recordingName: state.recordingName,

    // Actions
    startRecording,
    stopRecording,
    toggleRecording,
    pauseRecording,
    resumeRecording,
    recordFrame,

    // Utilities
    setRecordingName: (name: string) =>
      setState((prev) => ({ ...prev, recordingName: name })),
  }
}

// Utility functions

export function downloadRecording(recording: HandRecording): void {
  const json = JSON.stringify(recording, null, 2)
  const blob = new Blob([json], { type: 'application/json' })
  const url = URL.createObjectURL(blob)

  const a = document.createElement('a')
  a.href = url
  a.download = `${recording.metadata.name.replace(/[^a-z0-9]/gi, '_')}_${recording.metadata.id}.json`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)

  console.log('📥 Downloaded:', a.download)
}

export function loadRecordingFromFile(file: File): Promise<HandRecording> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      try {
        const recording = JSON.parse(e.target?.result as string) as HandRecording
        resolve(recording)
      } catch (err) {
        reject(new Error('Invalid recording file'))
      }
    }
    reader.onerror = () => reject(new Error('Failed to read file'))
    reader.readAsText(file)
  })
}

export function listSavedRecordings(): { key: string; metadata: HandRecording['metadata'] }[] {
  const recordings: { key: string; metadata: HandRecording['metadata'] }[] = []

  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i)
    if (key?.startsWith('hand_recording_')) {
      try {
        const data = JSON.parse(localStorage.getItem(key) || '{}') as HandRecording
        if (data.metadata) {
          recordings.push({ key, metadata: data.metadata })
        }
      } catch {
        // Ignore invalid entries
      }
    }
  }

  // Sort by date, newest first
  return recordings.sort(
    (a, b) => new Date(b.metadata.recordedAt).getTime() - new Date(a.metadata.recordedAt).getTime()
  )
}

export function loadRecordingFromStorage(key: string): HandRecording | null {
  try {
    const data = localStorage.getItem(key)
    return data ? (JSON.parse(data) as HandRecording) : null
  } catch {
    return null
  }
}

export function deleteRecordingFromStorage(key: string): void {
  localStorage.removeItem(key)
}

export default useHandRecording
