/**
 * useSoundEffects - Hook for sound effects throughout the app
 *
 * Provides easy access to sound manager and reactive settings state.
 */

import { useState, useCallback, useEffect } from 'react'
import { soundManager, type SoundSettings, type SoundType } from '../lib/sounds'

export interface UseSoundEffectsReturn {
  // Settings
  settings: SoundSettings
  setMasterVolume: (volume: number) => void
  setEnabled: (enabled: boolean) => void
  toggleSound: (type: SoundType, enabled: boolean) => void

  // Play sounds
  playSelect: (importance?: number) => void
  playHover: () => void
  playZoomIn: () => void
  playZoomOut: () => void
  playSearch: () => void
  playBookmark: () => void
  playDelete: () => void
  playError: () => void
  playSuccess: () => void
  playPathFound: () => void
  playTimeTravel: () => void
  playLasso: () => void
}

export function useSoundEffects(): UseSoundEffectsReturn {
  const [settings, setSettings] = useState<SoundSettings>(() => soundManager.getSettings())

  // Sync settings from localStorage on mount
  useEffect(() => {
    setSettings(soundManager.getSettings())
  }, [])

  const setMasterVolume = useCallback((volume: number) => {
    soundManager.setMasterVolume(volume)
    setSettings(soundManager.getSettings())
  }, [])

  const setEnabled = useCallback((enabled: boolean) => {
    soundManager.setEnabled(enabled)
    setSettings(soundManager.getSettings())
  }, [])

  const toggleSound = useCallback((type: SoundType, enabled: boolean) => {
    soundManager.toggleSound(type, enabled)
    setSettings(soundManager.getSettings())
  }, [])

  // Sound playback functions
  const playSelect = useCallback((importance?: number) => {
    soundManager.playSelect(importance)
  }, [])

  const playHover = useCallback(() => {
    soundManager.playHover()
  }, [])

  const playZoomIn = useCallback(() => {
    soundManager.playZoom(true)
  }, [])

  const playZoomOut = useCallback(() => {
    soundManager.playZoom(false)
  }, [])

  const playSearch = useCallback(() => {
    soundManager.playSearch()
  }, [])

  const playBookmark = useCallback(() => {
    soundManager.playBookmark()
  }, [])

  const playDelete = useCallback(() => {
    soundManager.playDelete()
  }, [])

  const playError = useCallback(() => {
    soundManager.playError()
  }, [])

  const playSuccess = useCallback(() => {
    soundManager.playSuccess()
  }, [])

  const playPathFound = useCallback(() => {
    soundManager.playPathFound()
  }, [])

  const playTimeTravel = useCallback(() => {
    soundManager.playTimeTravel()
  }, [])

  const playLasso = useCallback(() => {
    soundManager.playLasso()
  }, [])

  return {
    settings,
    setMasterVolume,
    setEnabled,
    toggleSound,
    playSelect,
    playHover,
    playZoomIn,
    playZoomOut,
    playSearch,
    playBookmark,
    playDelete,
    playError,
    playSuccess,
    playPathFound,
    playTimeTravel,
    playLasso,
  }
}
