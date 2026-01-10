/**
 * One Euro Filter - Adaptive Low-Pass Filter for Noisy Input
 *
 * The 1€ filter is the industry standard for VR/AR hand tracking smoothing.
 * It adapts its cutoff frequency based on movement speed:
 * - At low speeds: More aggressive smoothing (reduces jitter)
 * - At high speeds: Less smoothing (reduces lag)
 *
 * Reference: Casiez, Roussel, Vogel. "1€ Filter: A Simple Speed-based
 * Low-pass Filter for Noisy Input in Interactive Systems" (CHI 2012)
 *
 * Used by: Meta Quest, Microsoft HoloLens, Apple Vision Pro
 */

// Low-pass filter with exponential smoothing
class LowPassFilter {
  private y: number | null = null
  private a: number = 0

  constructor(alpha: number = 1.0) {
    this.a = alpha
  }

  setAlpha(alpha: number): void {
    this.a = Math.max(0, Math.min(1, alpha))
  }

  filter(value: number): number {
    if (this.y === null) {
      this.y = value
    } else {
      this.y = this.a * value + (1 - this.a) * this.y
    }
    return this.y
  }

  hasLastValue(): boolean {
    return this.y !== null
  }

  getLastValue(): number {
    return this.y ?? 0
  }

  reset(): void {
    this.y = null
  }
}

export interface OneEuroFilterConfig {
  /**
   * Minimum cutoff frequency (Hz). Controls jitter at low speeds.
   * Lower = more smoothing when still/slow. Default: 1.0
   * Recommended range: 0.5 - 3.0
   */
  minCutoff?: number

  /**
   * Speed coefficient (beta). Controls lag at high speeds.
   * Higher = less lag when moving fast. Default: 0.007
   * Recommended range: 0.001 - 0.1
   */
  beta?: number

  /**
   * Derivative cutoff frequency (Hz). Smooths the velocity estimation.
   * Higher = more responsive to speed changes. Default: 1.0
   * Recommended range: 0.5 - 2.0
   */
  dCutoff?: number

  /**
   * Expected sample rate in Hz. Used for timestamping.
   * Default: 60 (typical camera frame rate)
   */
  freq?: number
}

// Default configs optimized for different use cases
export const ONE_EURO_PRESETS = {
  // For landmark positions - prioritize stability
  landmark: { minCutoff: 0.8, beta: 0.005, dCutoff: 1.0 },

  // For pointer/ray direction - balance stability and responsiveness
  pointer: { minCutoff: 1.2, beta: 0.008, dCutoff: 1.0 },

  // For gesture strength values (pinch, grab) - quick response
  gesture: { minCutoff: 1.5, beta: 0.01, dCutoff: 1.2 },

  // High precision for selection - very stable
  precision: { minCutoff: 0.5, beta: 0.003, dCutoff: 0.8 },

  // Fast motion tracking - minimize lag
  fastMotion: { minCutoff: 2.0, beta: 0.02, dCutoff: 1.5 },
} as const

export class OneEuroFilter {
  private freq: number
  private minCutoff: number
  private beta: number
  private dCutoff: number

  private x: LowPassFilter
  private dx: LowPassFilter
  private lastTime: number | null = null

  constructor(config: OneEuroFilterConfig = {}) {
    this.freq = config.freq ?? 60
    this.minCutoff = config.minCutoff ?? 1.0
    this.beta = config.beta ?? 0.007
    this.dCutoff = config.dCutoff ?? 1.0

    this.x = new LowPassFilter(this.alpha(this.minCutoff))
    this.dx = new LowPassFilter(this.alpha(this.dCutoff))
  }

  private alpha(cutoff: number): number {
    const te = 1.0 / this.freq
    const tau = 1.0 / (2 * Math.PI * cutoff)
    return 1.0 / (1.0 + tau / te)
  }

  /**
   * Filter a value with automatic timestamping
   * @param value The noisy input value
   * @param timestamp Optional timestamp in milliseconds. If not provided, uses Date.now()
   * @returns Filtered (smoothed) value
   */
  filter(value: number, timestamp?: number): number {
    const t = timestamp ?? Date.now()

    // Calculate time delta and update frequency
    if (this.lastTime !== null) {
      const dt = (t - this.lastTime) / 1000 // Convert ms to seconds
      if (dt > 0 && dt < 1) {
        // Clamp to reasonable range
        this.freq = 1.0 / dt
      }
    }
    this.lastTime = t

    // Estimate derivative (velocity)
    const prevX = this.x.hasLastValue() ? this.x.getLastValue() : value
    const dx = this.freq * (value - prevX)

    // Filter the derivative
    this.dx.setAlpha(this.alpha(this.dCutoff))
    const edx = this.dx.filter(dx)

    // Adaptive cutoff based on velocity magnitude
    // Key insight: fc = minCutoff + beta * |velocity|
    const cutoff = this.minCutoff + this.beta * Math.abs(edx)

    // Filter the value with adaptive cutoff
    this.x.setAlpha(this.alpha(cutoff))
    return this.x.filter(value)
  }

  /**
   * Reset the filter state. Call when tracking is lost/reacquired.
   */
  reset(): void {
    this.x.reset()
    this.dx.reset()
    this.lastTime = null
  }

  /**
   * Update configuration parameters
   */
  setConfig(config: Partial<OneEuroFilterConfig>): void {
    if (config.minCutoff !== undefined) this.minCutoff = config.minCutoff
    if (config.beta !== undefined) this.beta = config.beta
    if (config.dCutoff !== undefined) this.dCutoff = config.dCutoff
    if (config.freq !== undefined) this.freq = config.freq
  }
}

/**
 * Filter for 2D points (x, y)
 */
export class OneEuroFilter2D {
  private xFilter: OneEuroFilter
  private yFilter: OneEuroFilter

  constructor(config: OneEuroFilterConfig = {}) {
    this.xFilter = new OneEuroFilter(config)
    this.yFilter = new OneEuroFilter(config)
  }

  filter(point: { x: number; y: number }, timestamp?: number): { x: number; y: number } {
    return {
      x: this.xFilter.filter(point.x, timestamp),
      y: this.yFilter.filter(point.y, timestamp),
    }
  }

  reset(): void {
    this.xFilter.reset()
    this.yFilter.reset()
  }

  setConfig(config: Partial<OneEuroFilterConfig>): void {
    this.xFilter.setConfig(config)
    this.yFilter.setConfig(config)
  }
}

/**
 * Filter for 3D points (x, y, z)
 */
export class OneEuroFilter3D {
  private xFilter: OneEuroFilter
  private yFilter: OneEuroFilter
  private zFilter: OneEuroFilter

  constructor(config: OneEuroFilterConfig = {}) {
    this.xFilter = new OneEuroFilter(config)
    this.yFilter = new OneEuroFilter(config)
    this.zFilter = new OneEuroFilter(config)
  }

  filter(point: { x: number; y: number; z: number }, timestamp?: number): { x: number; y: number; z: number } {
    return {
      x: this.xFilter.filter(point.x, timestamp),
      y: this.yFilter.filter(point.y, timestamp),
      z: this.zFilter.filter(point.z, timestamp),
    }
  }

  reset(): void {
    this.xFilter.reset()
    this.yFilter.reset()
    this.zFilter.reset()
  }

  setConfig(config: Partial<OneEuroFilterConfig>): void {
    this.xFilter.setConfig(config)
    this.yFilter.setConfig(config)
    this.zFilter.setConfig(config)
  }
}

/**
 * Filter for all 21 hand landmarks at once
 */
export class HandLandmarkFilter {
  private filters: OneEuroFilter3D[]

  constructor(config: OneEuroFilterConfig = ONE_EURO_PRESETS.landmark) {
    this.filters = Array.from({ length: 21 }, () => new OneEuroFilter3D(config))
  }

  filter(
    landmarks: Array<{ x: number; y: number; z: number; visibility?: number }>,
    timestamp?: number
  ): Array<{ x: number; y: number; z: number; visibility?: number }> {
    return landmarks.map((lm, i) => ({
      ...this.filters[i].filter(lm, timestamp),
      visibility: lm.visibility,
    }))
  }

  reset(): void {
    this.filters.forEach((f) => f.reset())
  }

  setConfig(config: Partial<OneEuroFilterConfig>): void {
    this.filters.forEach((f) => f.setConfig(config))
  }
}

export default OneEuroFilter
