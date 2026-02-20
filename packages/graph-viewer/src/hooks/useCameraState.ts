/**
 * useCameraState - Track and control camera position
 *
 * Provides camera position for mini-map and navigation
 */

import { useRef, useCallback } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'

export interface CameraState {
  position: { x: number; y: number; z: number }
  zoom: number
  target: { x: number; y: number; z: number }
}

interface UseCameraStateOptions {
  onCameraChange?: (state: CameraState) => void
  updateInterval?: number // ms between updates
}

export function useCameraState({ onCameraChange, updateInterval = 100 }: UseCameraStateOptions = {}) {
  const { camera } = useThree()
  const lastUpdateRef = useRef(0)
  const lastStateRef = useRef<CameraState>({
    position: { x: 0, y: 0, z: 100 },
    zoom: 1,
    target: { x: 0, y: 0, z: 0 },
  })

  useFrame(() => {
    if (!onCameraChange) return

    const now = performance.now()
    if (now - lastUpdateRef.current < updateInterval) return
    lastUpdateRef.current = now

    const pos = camera.position
    const state: CameraState = {
      position: { x: pos.x, y: pos.y, z: pos.z },
      zoom: camera instanceof THREE.PerspectiveCamera
        ? 100 / pos.distanceTo(new THREE.Vector3(0, 0, 0))
        : 1,
      target: { x: 0, y: 0, z: 0 }, // OrbitControls target would go here
    }

    // Only update if position changed significantly
    const lastPos = lastStateRef.current.position
    const dist = Math.sqrt(
      Math.pow(pos.x - lastPos.x, 2) +
      Math.pow(pos.y - lastPos.y, 2) +
      Math.pow(pos.z - lastPos.z, 2)
    )

    if (dist > 0.5 || Math.abs(state.zoom - lastStateRef.current.zoom) > 0.01) {
      lastStateRef.current = state
      onCameraChange(state)
    }
  })

  return lastStateRef.current
}

/**
 * Camera navigation helper - smoothly animate to a position
 */
export function useCameraNavigation() {
  const { camera, controls } = useThree()
  const animationRef = useRef<number | null>(null)

  const navigateTo = useCallback((targetX: number, targetY: number, duration = 500) => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current)
    }

    const startPos = camera.position.clone()
    const targetPos = new THREE.Vector3(targetX, targetY, startPos.z)
    const startTime = performance.now()

    const animate = () => {
      const elapsed = performance.now() - startTime
      const progress = Math.min(elapsed / duration, 1)

      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3)

      camera.position.lerpVectors(startPos, targetPos, eased)

      // Update OrbitControls target if available
      if (controls && 'target' in controls) {
        const orbitControls = controls as { target: THREE.Vector3 }
        orbitControls.target.set(targetX, targetY, 0)
      }

      if (progress < 1) {
        animationRef.current = requestAnimationFrame(animate)
      } else {
        animationRef.current = null
      }
    }

    animationRef.current = requestAnimationFrame(animate)
  }, [camera, controls])

  return { navigateTo }
}
