/**
 * Sound Design System - Procedural audio for the graph viewer
 *
 * Uses Web Audio API to generate all sounds procedurally.
 * No external audio files needed - everything is synthesized.
 *
 * Sound palette:
 * - Node select: Soft crystalline ping (pitch varies with importance)
 * - Node hover: Whisper-quiet high tone
 * - Zoom: Gentle whoosh (pitch direction matches zoom direction)
 * - Search: Soft keyboard click
 * - Bookmark save: Camera shutter sound
 * - Delete: Low thud with decay
 * - Error: Gentle dissonant tone
 * - Success: Ascending chime
 */

type SoundType =
  | 'select'
  | 'hover'
  | 'zoomIn'
  | 'zoomOut'
  | 'search'
  | 'bookmark'
  | 'delete'
  | 'error'
  | 'success'
  | 'pathFound'
  | 'timeTravel'
  | 'lasso'

interface SoundSettings {
  masterVolume: number // 0-1
  enabled: boolean
  individualSounds: Record<SoundType, boolean>
}

const DEFAULT_SETTINGS: SoundSettings = {
  masterVolume: 0.3,
  enabled: false, // Default off per spec
  individualSounds: {
    select: true,
    hover: true,
    zoomIn: true,
    zoomOut: true,
    search: true,
    bookmark: true,
    delete: true,
    error: true,
    success: true,
    pathFound: true,
    timeTravel: true,
    lasso: true,
  },
}

class SoundManager {
  private audioContext: AudioContext | null = null
  private settings: SoundSettings = DEFAULT_SETTINGS
  private lastHoverTime = 0
  private hoverThrottle = 50 // ms between hover sounds

  constructor() {
    // Load settings from localStorage
    this.loadSettings()
  }

  private getContext(): AudioContext | null {
    if (!this.audioContext) {
      try {
        this.audioContext = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)()
      } catch {
        console.warn('Web Audio API not supported')
        return null
      }
    }

    // Resume if suspended (browser autoplay policy)
    if (this.audioContext.state === 'suspended') {
      this.audioContext.resume()
    }

    return this.audioContext
  }

  private loadSettings() {
    try {
      const stored = localStorage.getItem('graph-viewer-sound-settings')
      if (stored) {
        this.settings = { ...DEFAULT_SETTINGS, ...JSON.parse(stored) }
      }
    } catch {
      // Use defaults
    }
  }

  saveSettings() {
    try {
      localStorage.setItem('graph-viewer-sound-settings', JSON.stringify(this.settings))
    } catch {
      // Ignore storage errors
    }
  }

  getSettings(): SoundSettings {
    return { ...this.settings }
  }

  setMasterVolume(volume: number) {
    this.settings.masterVolume = Math.max(0, Math.min(1, volume))
    this.saveSettings()
  }

  setEnabled(enabled: boolean) {
    this.settings.enabled = enabled
    this.saveSettings()
  }

  toggleSound(type: SoundType, enabled: boolean) {
    this.settings.individualSounds[type] = enabled
    this.saveSettings()
  }

  private canPlay(type: SoundType): boolean {
    return this.settings.enabled && this.settings.individualSounds[type]
  }

  private getVolume(): number {
    return this.settings.masterVolume * 0.5 // Overall quieter
  }

  /**
   * Node selection - crystalline ping
   * @param importance 0-1, affects pitch (higher = higher pitch)
   */
  playSelect(importance = 0.5) {
    if (!this.canPlay('select')) return

    const ctx = this.getContext()
    if (!ctx) return

    const now = ctx.currentTime
    const volume = this.getVolume() * 0.4

    // Base frequency varies with importance (400Hz - 800Hz)
    const baseFreq = 400 + importance * 400

    // Create oscillator for main tone
    const osc = ctx.createOscillator()
    osc.type = 'sine'
    osc.frequency.setValueAtTime(baseFreq, now)
    osc.frequency.exponentialRampToValueAtTime(baseFreq * 1.5, now + 0.1)

    // Add harmonic
    const osc2 = ctx.createOscillator()
    osc2.type = 'sine'
    osc2.frequency.setValueAtTime(baseFreq * 2, now)
    osc2.frequency.exponentialRampToValueAtTime(baseFreq * 3, now + 0.08)

    // Gain envelope
    const gain = ctx.createGain()
    gain.gain.setValueAtTime(0, now)
    gain.gain.linearRampToValueAtTime(volume, now + 0.01)
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.3)

    const gain2 = ctx.createGain()
    gain2.gain.setValueAtTime(0, now)
    gain2.gain.linearRampToValueAtTime(volume * 0.3, now + 0.01)
    gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.2)

    osc.connect(gain).connect(ctx.destination)
    osc2.connect(gain2).connect(ctx.destination)

    osc.start(now)
    osc.stop(now + 0.35)
    osc2.start(now)
    osc2.stop(now + 0.25)
  }

  /**
   * Node hover - whisper-quiet high tone
   */
  playHover() {
    if (!this.canPlay('hover')) return

    // Throttle hover sounds
    const now = Date.now()
    if (now - this.lastHoverTime < this.hoverThrottle) return
    this.lastHoverTime = now

    const ctx = this.getContext()
    if (!ctx) return

    const time = ctx.currentTime
    const volume = this.getVolume() * 0.08 // Very quiet

    const osc = ctx.createOscillator()
    osc.type = 'sine'
    osc.frequency.setValueAtTime(1200, time)

    const gain = ctx.createGain()
    gain.gain.setValueAtTime(0, time)
    gain.gain.linearRampToValueAtTime(volume, time + 0.02)
    gain.gain.exponentialRampToValueAtTime(0.001, time + 0.08)

    osc.connect(gain).connect(ctx.destination)
    osc.start(time)
    osc.stop(time + 0.1)
  }

  /**
   * Zoom - gentle whoosh
   * @param zoomIn true for zoom in, false for zoom out
   */
  playZoom(zoomIn: boolean) {
    if (!this.canPlay(zoomIn ? 'zoomIn' : 'zoomOut')) return

    const ctx = this.getContext()
    if (!ctx) return

    const now = ctx.currentTime
    const volume = this.getVolume() * 0.15

    // Noise-based whoosh using filtered noise
    const bufferSize = ctx.sampleRate * 0.15
    const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate)
    const data = buffer.getChannelData(0)

    // Generate noise
    for (let i = 0; i < bufferSize; i++) {
      data[i] = Math.random() * 2 - 1
    }

    const source = ctx.createBufferSource()
    source.buffer = buffer

    // Bandpass filter for tonal quality
    const filter = ctx.createBiquadFilter()
    filter.type = 'bandpass'
    filter.Q.value = 2

    // Frequency sweep direction based on zoom
    if (zoomIn) {
      filter.frequency.setValueAtTime(200, now)
      filter.frequency.exponentialRampToValueAtTime(800, now + 0.15)
    } else {
      filter.frequency.setValueAtTime(800, now)
      filter.frequency.exponentialRampToValueAtTime(200, now + 0.15)
    }

    const gain = ctx.createGain()
    gain.gain.setValueAtTime(0, now)
    gain.gain.linearRampToValueAtTime(volume, now + 0.02)
    gain.gain.linearRampToValueAtTime(0, now + 0.15)

    source.connect(filter).connect(gain).connect(ctx.destination)
    source.start(now)
  }

  /**
   * Search keystroke - soft click
   */
  playSearch() {
    if (!this.canPlay('search')) return

    const ctx = this.getContext()
    if (!ctx) return

    const now = ctx.currentTime
    const volume = this.getVolume() * 0.1

    // Short noise burst
    const bufferSize = ctx.sampleRate * 0.02
    const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate)
    const data = buffer.getChannelData(0)

    for (let i = 0; i < bufferSize; i++) {
      // Decaying envelope baked in
      const env = 1 - i / bufferSize
      data[i] = (Math.random() * 2 - 1) * env * env
    }

    const source = ctx.createBufferSource()
    source.buffer = buffer

    const filter = ctx.createBiquadFilter()
    filter.type = 'highpass'
    filter.frequency.value = 2000

    const gain = ctx.createGain()
    gain.gain.value = volume

    source.connect(filter).connect(gain).connect(ctx.destination)
    source.start(now)
  }

  /**
   * Bookmark save - camera shutter
   */
  playBookmark() {
    if (!this.canPlay('bookmark')) return

    const ctx = this.getContext()
    if (!ctx) return

    const now = ctx.currentTime
    const volume = this.getVolume() * 0.2

    // First click
    const click1Size = ctx.sampleRate * 0.015
    const click1 = ctx.createBuffer(1, click1Size, ctx.sampleRate)
    const click1Data = click1.getChannelData(0)
    for (let i = 0; i < click1Size; i++) {
      const env = Math.exp(-i / (ctx.sampleRate * 0.003))
      click1Data[i] = (Math.random() * 2 - 1) * env
    }

    // Second click (slightly delayed)
    const click2Size = ctx.sampleRate * 0.012
    const click2 = ctx.createBuffer(1, click2Size, ctx.sampleRate)
    const click2Data = click2.getChannelData(0)
    for (let i = 0; i < click2Size; i++) {
      const env = Math.exp(-i / (ctx.sampleRate * 0.004))
      click2Data[i] = (Math.random() * 2 - 1) * env
    }

    const source1 = ctx.createBufferSource()
    source1.buffer = click1
    const source2 = ctx.createBufferSource()
    source2.buffer = click2

    const gain = ctx.createGain()
    gain.gain.value = volume

    source1.connect(gain).connect(ctx.destination)
    source2.connect(gain)

    source1.start(now)
    source2.start(now + 0.05)
  }

  /**
   * Delete - low thud
   */
  playDelete() {
    if (!this.canPlay('delete')) return

    const ctx = this.getContext()
    if (!ctx) return

    const now = ctx.currentTime
    const volume = this.getVolume() * 0.3

    const osc = ctx.createOscillator()
    osc.type = 'sine'
    osc.frequency.setValueAtTime(80, now)
    osc.frequency.exponentialRampToValueAtTime(40, now + 0.15)

    const gain = ctx.createGain()
    gain.gain.setValueAtTime(volume, now)
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.2)

    // Add subtle noise for "thump" texture
    const noiseSize = ctx.sampleRate * 0.05
    const noise = ctx.createBuffer(1, noiseSize, ctx.sampleRate)
    const noiseData = noise.getChannelData(0)
    for (let i = 0; i < noiseSize; i++) {
      const env = Math.exp(-i / (ctx.sampleRate * 0.01))
      noiseData[i] = (Math.random() * 2 - 1) * env
    }
    const noiseSource = ctx.createBufferSource()
    noiseSource.buffer = noise

    const lowpass = ctx.createBiquadFilter()
    lowpass.type = 'lowpass'
    lowpass.frequency.value = 200

    const noiseGain = ctx.createGain()
    noiseGain.gain.value = volume * 0.5

    osc.connect(gain).connect(ctx.destination)
    noiseSource.connect(lowpass).connect(noiseGain).connect(ctx.destination)

    osc.start(now)
    osc.stop(now + 0.25)
    noiseSource.start(now)
  }

  /**
   * Error - gentle dissonant tone
   */
  playError() {
    if (!this.canPlay('error')) return

    const ctx = this.getContext()
    if (!ctx) return

    const now = ctx.currentTime
    const volume = this.getVolume() * 0.2

    // Slightly dissonant interval
    const osc1 = ctx.createOscillator()
    osc1.type = 'sine'
    osc1.frequency.value = 280

    const osc2 = ctx.createOscillator()
    osc2.type = 'sine'
    osc2.frequency.value = 340 // Minor second above

    const gain = ctx.createGain()
    gain.gain.setValueAtTime(0, now)
    gain.gain.linearRampToValueAtTime(volume, now + 0.02)
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.4)

    osc1.connect(gain).connect(ctx.destination)
    osc2.connect(gain)

    osc1.start(now)
    osc1.stop(now + 0.45)
    osc2.start(now)
    osc2.stop(now + 0.45)
  }

  /**
   * Success - ascending chime
   */
  playSuccess() {
    if (!this.canPlay('success')) return

    const ctx = this.getContext()
    if (!ctx) return

    const now = ctx.currentTime
    const volume = this.getVolume() * 0.25

    // Ascending arpeggio
    const notes = [523.25, 659.25, 783.99] // C5, E5, G5
    notes.forEach((freq, i) => {
      const osc = ctx.createOscillator()
      osc.type = 'sine'
      osc.frequency.value = freq

      const gain = ctx.createGain()
      const startTime = now + i * 0.08
      gain.gain.setValueAtTime(0, startTime)
      gain.gain.linearRampToValueAtTime(volume * (1 - i * 0.2), startTime + 0.02)
      gain.gain.exponentialRampToValueAtTime(0.001, startTime + 0.3)

      osc.connect(gain).connect(ctx.destination)
      osc.start(startTime)
      osc.stop(startTime + 0.35)
    })
  }

  /**
   * Path found - magical discovery sound
   */
  playPathFound() {
    if (!this.canPlay('pathFound')) return

    const ctx = this.getContext()
    if (!ctx) return

    const now = ctx.currentTime
    const volume = this.getVolume() * 0.2

    // Shimmering ascending tone
    const osc = ctx.createOscillator()
    osc.type = 'sine'
    osc.frequency.setValueAtTime(400, now)
    osc.frequency.exponentialRampToValueAtTime(800, now + 0.3)

    // Add subtle harmonics
    const osc2 = ctx.createOscillator()
    osc2.type = 'sine'
    osc2.frequency.setValueAtTime(800, now)
    osc2.frequency.exponentialRampToValueAtTime(1600, now + 0.3)

    const gain = ctx.createGain()
    gain.gain.setValueAtTime(0, now)
    gain.gain.linearRampToValueAtTime(volume, now + 0.05)
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.5)

    const gain2 = ctx.createGain()
    gain2.gain.setValueAtTime(0, now)
    gain2.gain.linearRampToValueAtTime(volume * 0.3, now + 0.05)
    gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.4)

    osc.connect(gain).connect(ctx.destination)
    osc2.connect(gain2).connect(ctx.destination)

    osc.start(now)
    osc.stop(now + 0.55)
    osc2.start(now)
    osc2.stop(now + 0.45)
  }

  /**
   * Time travel - whooshy temporal effect
   */
  playTimeTravel() {
    if (!this.canPlay('timeTravel')) return

    const ctx = this.getContext()
    if (!ctx) return

    const now = ctx.currentTime
    const volume = this.getVolume() * 0.15

    // Sweeping filter on noise
    const bufferSize = ctx.sampleRate * 0.4
    const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate)
    const data = buffer.getChannelData(0)

    for (let i = 0; i < bufferSize; i++) {
      data[i] = Math.random() * 2 - 1
    }

    const source = ctx.createBufferSource()
    source.buffer = buffer

    const filter = ctx.createBiquadFilter()
    filter.type = 'bandpass'
    filter.Q.value = 5
    filter.frequency.setValueAtTime(500, now)
    filter.frequency.exponentialRampToValueAtTime(2000, now + 0.2)
    filter.frequency.exponentialRampToValueAtTime(300, now + 0.4)

    const gain = ctx.createGain()
    gain.gain.setValueAtTime(0, now)
    gain.gain.linearRampToValueAtTime(volume, now + 0.05)
    gain.gain.linearRampToValueAtTime(0, now + 0.4)

    source.connect(filter).connect(gain).connect(ctx.destination)
    source.start(now)
  }

  /**
   * Lasso selection complete
   */
  playLasso() {
    if (!this.canPlay('lasso')) return

    const ctx = this.getContext()
    if (!ctx) return

    const now = ctx.currentTime
    const volume = this.getVolume() * 0.15

    // Quick ascending sweep
    const osc = ctx.createOscillator()
    osc.type = 'triangle'
    osc.frequency.setValueAtTime(300, now)
    osc.frequency.exponentialRampToValueAtTime(600, now + 0.1)

    const gain = ctx.createGain()
    gain.gain.setValueAtTime(0, now)
    gain.gain.linearRampToValueAtTime(volume, now + 0.02)
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.15)

    osc.connect(gain).connect(ctx.destination)
    osc.start(now)
    osc.stop(now + 0.2)
  }
}

// Singleton instance
export const soundManager = new SoundManager()

export type { SoundType, SoundSettings }
