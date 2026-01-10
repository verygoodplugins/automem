/**
 * useTimeTravel - Navigate memories chronologically
 *
 * Provides time-based filtering and playback controls to watch
 * the memory graph evolve over time.
 */

import { useState, useCallback, useMemo, useRef, useEffect } from 'react'
import type { GraphNode } from '../lib/types'

export interface TimeTravelState {
  isActive: boolean
  isPlaying: boolean
  currentTime: number // Unix timestamp in ms
  playbackSpeed: number // 0.5, 1, 2, 4
  minTime: number
  maxTime: number
}

interface UseTimeTravelOptions {
  nodes: GraphNode[]
  enabled?: boolean
}

interface UseTimeTravelReturn {
  // State
  state: TimeTravelState
  isActive: boolean
  isPlaying: boolean
  currentTime: number
  currentDate: Date
  playbackSpeed: number
  minTime: number
  maxTime: number
  progress: number // 0-1 progress through timeline

  // Filtered data
  visibleNodes: Set<string>
  visibleCount: number
  totalCount: number

  // Actions
  activate: () => void
  deactivate: () => void
  toggleActive: () => void
  play: () => void
  pause: () => void
  togglePlay: () => void
  setTime: (time: number) => void
  setProgress: (progress: number) => void
  stepForward: () => void
  stepBackward: () => void
  setSpeed: (speed: number) => void
  cycleSpeed: () => void
  goToStart: () => void
  goToEnd: () => void
}

const SPEEDS = [0.5, 1, 2, 4]
const DEFAULT_STEP_MS = 24 * 60 * 60 * 1000 // 1 day

export function useTimeTravel({ nodes, enabled: _enabled = true }: UseTimeTravelOptions): UseTimeTravelReturn {
  // Calculate time bounds from nodes
  const { minTime, maxTime } = useMemo(() => {
    if (nodes.length === 0) {
      const now = Date.now()
      return { minTime: now - 30 * 24 * 60 * 60 * 1000, maxTime: now }
    }

    let min = Infinity
    let max = -Infinity

    nodes.forEach((node) => {
      const time = new Date(node.timestamp).getTime()
      if (!isNaN(time)) {
        if (time < min) min = time
        if (time > max) max = time
      }
    })

    // If no valid timestamps, use last 30 days
    if (min === Infinity) {
      const now = Date.now()
      return { minTime: now - 30 * 24 * 60 * 60 * 1000, maxTime: now }
    }

    return { minTime: min, maxTime: max }
  }, [nodes])

  const [state, setState] = useState<TimeTravelState>({
    isActive: false,
    isPlaying: false,
    currentTime: maxTime,
    playbackSpeed: 1,
    minTime,
    maxTime,
  })

  // Update bounds when nodes change
  useEffect(() => {
    setState((prev) => ({
      ...prev,
      minTime,
      maxTime,
      currentTime: prev.isActive ? prev.currentTime : maxTime,
    }))
  }, [minTime, maxTime])

  // Animation frame ref for playback
  const animationRef = useRef<number | null>(null)
  const lastFrameTimeRef = useRef<number>(0)

  // Playback loop
  useEffect(() => {
    if (!state.isActive || !state.isPlaying) {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
        animationRef.current = null
      }
      return
    }

    const animate = (frameTime: number) => {
      const deltaMs = frameTime - lastFrameTimeRef.current
      lastFrameTimeRef.current = frameTime

      // Skip if too much time passed (tab was hidden)
      if (deltaMs > 500) {
        animationRef.current = requestAnimationFrame(animate)
        return
      }

      // Calculate time advancement
      // At 1x speed, advance 1 day per second of real time
      const timeAdvance = (deltaMs / 1000) * DEFAULT_STEP_MS * state.playbackSpeed

      setState((prev) => {
        const newTime = prev.currentTime + timeAdvance

        // Stop at end
        if (newTime >= prev.maxTime) {
          return { ...prev, currentTime: prev.maxTime, isPlaying: false }
        }

        return { ...prev, currentTime: newTime }
      })

      animationRef.current = requestAnimationFrame(animate)
    }

    lastFrameTimeRef.current = performance.now()
    animationRef.current = requestAnimationFrame(animate)

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [state.isActive, state.isPlaying, state.playbackSpeed])

  // Calculate visible nodes
  const visibleNodes = useMemo(() => {
    if (!state.isActive) {
      return new Set(nodes.map((n) => n.id))
    }

    const visible = new Set<string>()
    nodes.forEach((node) => {
      const nodeTime = new Date(node.timestamp).getTime()
      if (!isNaN(nodeTime) && nodeTime <= state.currentTime) {
        visible.add(node.id)
      }
    })

    return visible
  }, [nodes, state.isActive, state.currentTime])

  // Actions
  const activate = useCallback(() => {
    setState((prev) => ({
      ...prev,
      isActive: true,
      currentTime: prev.minTime, // Start from beginning
    }))
  }, [])

  const deactivate = useCallback(() => {
    setState((prev) => ({
      ...prev,
      isActive: false,
      isPlaying: false,
      currentTime: prev.maxTime,
    }))
  }, [])

  const toggleActive = useCallback(() => {
    setState((prev) => {
      if (prev.isActive) {
        return { ...prev, isActive: false, isPlaying: false, currentTime: prev.maxTime }
      }
      return { ...prev, isActive: true, currentTime: prev.minTime }
    })
  }, [])

  const play = useCallback(() => {
    setState((prev) => ({ ...prev, isPlaying: true }))
  }, [])

  const pause = useCallback(() => {
    setState((prev) => ({ ...prev, isPlaying: false }))
  }, [])

  const togglePlay = useCallback(() => {
    setState((prev) => ({ ...prev, isPlaying: !prev.isPlaying }))
  }, [])

  const setTime = useCallback((time: number) => {
    setState((prev) => ({
      ...prev,
      currentTime: Math.max(prev.minTime, Math.min(prev.maxTime, time)),
    }))
  }, [])

  const setProgress = useCallback((progress: number) => {
    setState((prev) => {
      const range = prev.maxTime - prev.minTime
      const time = prev.minTime + range * Math.max(0, Math.min(1, progress))
      return { ...prev, currentTime: time }
    })
  }, [])

  const stepForward = useCallback(() => {
    setState((prev) => ({
      ...prev,
      currentTime: Math.min(prev.maxTime, prev.currentTime + DEFAULT_STEP_MS),
    }))
  }, [])

  const stepBackward = useCallback(() => {
    setState((prev) => ({
      ...prev,
      currentTime: Math.max(prev.minTime, prev.currentTime - DEFAULT_STEP_MS),
    }))
  }, [])

  const setSpeed = useCallback((speed: number) => {
    setState((prev) => ({ ...prev, playbackSpeed: speed }))
  }, [])

  const cycleSpeed = useCallback(() => {
    setState((prev) => {
      const currentIndex = SPEEDS.indexOf(prev.playbackSpeed)
      const nextIndex = (currentIndex + 1) % SPEEDS.length
      return { ...prev, playbackSpeed: SPEEDS[nextIndex] }
    })
  }, [])

  const goToStart = useCallback(() => {
    setState((prev) => ({ ...prev, currentTime: prev.minTime }))
  }, [])

  const goToEnd = useCallback(() => {
    setState((prev) => ({ ...prev, currentTime: prev.maxTime }))
  }, [])

  // Calculate progress
  const progress = useMemo(() => {
    const range = state.maxTime - state.minTime
    if (range === 0) return 1
    return (state.currentTime - state.minTime) / range
  }, [state.currentTime, state.minTime, state.maxTime])

  return {
    // State
    state,
    isActive: state.isActive,
    isPlaying: state.isPlaying,
    currentTime: state.currentTime,
    currentDate: new Date(state.currentTime),
    playbackSpeed: state.playbackSpeed,
    minTime: state.minTime,
    maxTime: state.maxTime,
    progress,

    // Filtered data
    visibleNodes,
    visibleCount: visibleNodes.size,
    totalCount: nodes.length,

    // Actions
    activate,
    deactivate,
    toggleActive,
    play,
    pause,
    togglePlay,
    setTime,
    setProgress,
    stepForward,
    stepBackward,
    setSpeed,
    cycleSpeed,
    goToStart,
    goToEnd,
  }
}
