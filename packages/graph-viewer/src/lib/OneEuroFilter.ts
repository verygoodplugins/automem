/**
 * One Euro Filter - Velocity-Adaptive Low-Pass Filter
 *
 * The gold standard for jitter reduction in tracking systems.
 * Used by Meta Quest, Apple Vision Pro, and most AR/VR tracking systems.
 *
 * Key insight: Use low smoothing when moving fast (responsive),
 * high smoothing when moving slow (stable/jitter-free).
 *
 * Paper: "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems"
 * Géry Casiez, Nicolas Roussel, Daniel Vogel - CHI 2012
 */

export interface OneEuroFilterConfig {
  /** Minimum cutoff frequency in Hz. Lower = more smoothing when slow. Default: 1.0 */
  minCutoff: number
  /** Speed coefficient. Higher = more responsive when moving fast. Default: 0.007 */
  beta: number
  /** Derivative cutoff frequency in Hz. Default: 1.0 */
  dCutoff: number
}

const DEFAULT_CONFIG: OneEuroFilterConfig = {
  minCutoff: 1.0,
  beta: 0.007,
  dCutoff: 1.0,
}

export class OneEuroFilter {
  private config: OneEuroFilterConfig
  private xPrev: number | null = null
  private dxPrev: number = 0
  private tPrev: number | null = null

  constructor(config: Partial<OneEuroFilterConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config }
  }

  /**
   * Filter a single value
   * @param x - Current value
   * @param t - Current timestamp in seconds
   * @returns Filtered value
   */
  filter(x: number, t: number): number {
    if (this.xPrev === null || this.tPrev === null) {
      this.xPrev = x
      this.dxPrev = 0
      this.tPrev = t
      return x
    }

    const dt = t - this.tPrev
    if (dt <= 0) return this.xPrev

    // Estimate derivative (velocity)
    const dx = (x - this.xPrev) / dt

    // Filter the derivative
    const alphaDx = this.computeAlpha(this.config.dCutoff, dt)
    const dxFiltered = this.lowPass(dx, this.dxPrev, alphaDx)
    this.dxPrev = dxFiltered

    // Compute adaptive cutoff based on velocity
    // Fast movement = high cutoff = responsive
    // Slow movement = low cutoff = stable
    const cutoff = this.config.minCutoff + this.config.beta * Math.abs(dxFiltered)

    // Filter the position with adaptive cutoff
    const alpha = this.computeAlpha(cutoff, dt)
    const xFiltered = this.lowPass(x, this.xPrev, alpha)

    this.xPrev = xFiltered
    this.tPrev = t

    return xFiltered
  }

  /**
   * Reset the filter state
   */
  reset(): void {
    this.xPrev = null
    this.dxPrev = 0
    this.tPrev = null
  }

  private computeAlpha(cutoff: number, dt: number): number {
    const tau = 1.0 / (2 * Math.PI * cutoff)
    return 1.0 / (1.0 + tau / dt)
  }

  private lowPass(x: number, xPrev: number, alpha: number): number {
    return xPrev + alpha * (x - xPrev)
  }
}

/**
 * 3D Point filter - applies One Euro Filter to each component
 */
export class OneEuroFilter3D {
  private filterX: OneEuroFilter
  private filterY: OneEuroFilter
  private filterZ: OneEuroFilter

  constructor(config: Partial<OneEuroFilterConfig> = {}) {
    this.filterX = new OneEuroFilter(config)
    this.filterY = new OneEuroFilter(config)
    this.filterZ = new OneEuroFilter(config)
  }

  filter(point: { x: number; y: number; z: number }, t: number): { x: number; y: number; z: number } {
    return {
      x: this.filterX.filter(point.x, t),
      y: this.filterY.filter(point.y, t),
      z: this.filterZ.filter(point.z, t),
    }
  }

  reset(): void {
    this.filterX.reset()
    this.filterY.reset()
    this.filterZ.reset()
  }
}

/**
 * Pointer Ray filter - stabilizes both origin and direction
 */
export interface PointerRay {
  origin: { x: number; y: number; z: number }
  direction: { x: number; y: number; z: number }
}

export class PointerRayFilter {
  private originFilter: OneEuroFilter3D
  private directionFilter: OneEuroFilter3D

  constructor(config: Partial<OneEuroFilterConfig> = {}) {
    // Origin needs more stability (user doesn't consciously control it)
    this.originFilter = new OneEuroFilter3D({
      minCutoff: 0.5,
      beta: 0.3,
      dCutoff: 1.0,
      ...config,
    })

    // Direction can be more responsive (user actively aims)
    this.directionFilter = new OneEuroFilter3D({
      minCutoff: 1.5,
      beta: 0.8,
      dCutoff: 1.0,
      ...config,
    })
  }

  filter(ray: PointerRay, t: number): PointerRay {
    const origin = this.originFilter.filter(ray.origin, t)
    const rawDirection = this.directionFilter.filter(ray.direction, t)

    // Re-normalize direction after filtering
    const len = Math.sqrt(
      rawDirection.x * rawDirection.x +
      rawDirection.y * rawDirection.y +
      rawDirection.z * rawDirection.z
    )

    const direction = len > 0
      ? { x: rawDirection.x / len, y: rawDirection.y / len, z: rawDirection.z / len }
      : { x: 0, y: 0, z: -1 }

    return { origin, direction }
  }

  reset(): void {
    this.originFilter.reset()
    this.directionFilter.reset()
  }
}
