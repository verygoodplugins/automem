/**
 * Hand Interaction Hook
 *
 * Combines gesture tracking with stable pointer ray and hit detection.
 * Provides a complete interaction system for the memory graph:
 *
 * - Accurate laser pointing with arm model
 * - Node hit detection
 * - Pinch-to-select with hysteresis
 * - Pull/push gestures for Z manipulation
 * - Two-hand rotation
 *
 * This is the main entry point for hand-based graph interaction.
 */

import { useRef, useCallback, useEffect, useState } from 'react'
import { useStablePointerRay, findNodeHit, type StableRay, type NodeHit, type NodeSphere } from './useStablePointerRay'
import type { GestureState, HandLandmarks } from './useHandGestures'
import type { SimulationNode } from '../lib/types'

export interface InteractionState {
  /** Left hand ray (if tracking) */
  leftRay: StableRay | null
  /** Right hand ray (if tracking) */
  rightRay: StableRay | null
  /** Currently hovered node (ray intersects) */
  hoveredNode: NodeHit | null
  /** Selected node (pinch activated on hover) */
  selectedNodeId: string | null
  /** Is a node being dragged */
  isDragging: boolean
  /** Drag delta for Z manipulation */
  dragDeltaZ: number
  /** Two-hand rotation delta */
  rotationDelta: { x: number; y: number }
  /** Two-hand zoom delta */
  zoomDelta: number
}

interface UseHandInteractionOptions {
  /** Nodes to test for hit detection */
  nodes: SimulationNode[]
  /** Callback when node selection changes */
  onNodeSelect?: (nodeId: string | null) => void
  /** Callback when node hover changes */
  onNodeHover?: (nodeId: string | null) => void
}

// Sensitivity settings
const DRAG_Z_SENSITIVITY = 80
const ROTATION_SENSITIVITY = 2
const ZOOM_SENSITIVITY = 150

export function useHandInteraction({ nodes, onNodeSelect, onNodeHover }: UseHandInteractionOptions) {
  // Stable pointer ray processors for each hand
  const leftRayProcessor = useStablePointerRay({ handedness: 'left' })
  const rightRayProcessor = useStablePointerRay({ handedness: 'right' })

  // State
  const [interactionState, setInteractionState] = useState<InteractionState>({
    leftRay: null,
    rightRay: null,
    hoveredNode: null,
    selectedNodeId: null,
    isDragging: false,
    dragDeltaZ: 0,
    rotationDelta: { x: 0, y: 0 },
    zoomDelta: 0,
  })

  // Previous state for delta calculations
  const prevStateRef = useRef<{
    leftZ: number | null
    rightZ: number | null
    selectedNodeId: string | null
    twoHandDistance: number | null
    twoHandRotation: number | null
  }>({
    leftZ: null,
    rightZ: null,
    selectedNodeId: null,
    twoHandDistance: null,
    twoHandRotation: null,
  })

  // Convert SimulationNodes to NodeSpheres for hit testing
  const nodeSpheres: NodeSphere[] = nodes.map(n => ({
    id: n.id,
    x: n.x ?? 0,
    y: n.y ?? 0,
    z: n.z ?? 0,
    radius: n.radius * 1.5, // Slightly larger hit area
  }))

  // Process gesture state and update interaction state
  const processGestures = useCallback((gestureState: GestureState) => {
    const timestamp = performance.now() / 1000
    const prev = prevStateRef.current

    // Process hand landmarks through stable ray pipeline
    const leftRay = gestureState.leftHand
      ? leftRayProcessor.processLandmarks(gestureState.leftHand.landmarks, timestamp)
      : null

    const rightRay = gestureState.rightHand
      ? rightRayProcessor.processLandmarks(gestureState.rightHand.landmarks, timestamp)
      : null

    // Determine primary ray (prefer right hand)
    const primaryRay = rightRay?.confidence > (leftRay?.confidence ?? 0) ? rightRay : leftRay

    // Find node hit
    let hoveredNode: NodeHit | null = null
    if (primaryRay) {
      // Convert normalized ray to world coordinates for hit testing
      // The ray is in normalized 0-1 space, nodes are in world space (-100 to 100 etc)
      const worldRay = {
        origin: {
          x: (primaryRay.origin.x - 0.5) * 200,
          y: -(primaryRay.origin.y - 0.5) * 200,
          z: -primaryRay.origin.z * 200,
        },
        direction: {
          x: primaryRay.direction.x,
          y: -primaryRay.direction.y, // Flip Y for world coords
          z: -primaryRay.direction.z,
        },
      }
      hoveredNode = findNodeHit(worldRay, nodeSpheres, 500)
    }

    // Handle selection (pinch on hover)
    let selectedNodeId = prev.selectedNodeId
    let isDragging = false
    let dragDeltaZ = 0

    if (primaryRay?.isActive) {
      if (hoveredNode && !prev.selectedNodeId) {
        // Start selection
        selectedNodeId = hoveredNode.nodeId
        onNodeSelect?.(selectedNodeId)
      }

      if (selectedNodeId) {
        isDragging = true

        // Calculate Z drag from hand movement
        const currentZ = primaryRay.origin.z
        const prevZ = primaryRay === leftRay ? prev.leftZ : prev.rightZ

        if (prevZ !== null) {
          // Negative Z movement (hand toward camera) = push node away
          dragDeltaZ = (currentZ - prevZ) * DRAG_Z_SENSITIVITY
        }
      }
    } else {
      // Released - clear selection
      if (prev.selectedNodeId) {
        selectedNodeId = null
        onNodeSelect?.(null)
      }
    }

    // Update hover callback
    if (hoveredNode?.nodeId !== prev.selectedNodeId) {
      // Don't update hover while dragging the same node
      const hoverId = hoveredNode?.nodeId ?? null
      const prevHoverId = interactionState.hoveredNode?.nodeId ?? null
      if (hoverId !== prevHoverId && !isDragging) {
        onNodeHover?.(hoverId)
      }
    }

    // Two-hand gestures
    let rotationDelta = { x: 0, y: 0 }
    let zoomDelta = 0

    if (leftRay?.isActive && rightRay?.isActive) {
      // Both hands pinching - two-hand mode

      // Calculate distance between pinch points
      const dx = rightRay.pinchPoint.x - leftRay.pinchPoint.x
      const dy = rightRay.pinchPoint.y - leftRay.pinchPoint.y
      const distance = Math.sqrt(dx * dx + dy * dy)

      // Calculate rotation angle
      const rotation = Math.atan2(dy, dx)

      if (prev.twoHandDistance !== null && prev.twoHandRotation !== null) {
        // Zoom from distance change
        zoomDelta = (distance - prev.twoHandDistance) * ZOOM_SENSITIVITY

        // Rotation from angle change
        let rotDelta = rotation - prev.twoHandRotation
        // Normalize to -PI to PI
        while (rotDelta > Math.PI) rotDelta -= Math.PI * 2
        while (rotDelta < -Math.PI) rotDelta += Math.PI * 2

        rotationDelta = {
          x: rotDelta * ROTATION_SENSITIVITY,
          y: 0, // Y rotation from individual hand movements, handled above
        }
      }

      prev.twoHandDistance = distance
      prev.twoHandRotation = rotation
    } else {
      prev.twoHandDistance = null
      prev.twoHandRotation = null
    }

    // Update previous state
    prev.leftZ = leftRay?.origin.z ?? null
    prev.rightZ = rightRay?.origin.z ?? null
    prev.selectedNodeId = selectedNodeId

    // Update interaction state
    setInteractionState({
      leftRay,
      rightRay,
      hoveredNode,
      selectedNodeId,
      isDragging,
      dragDeltaZ,
      rotationDelta,
      zoomDelta,
    })
  }, [nodeSpheres, leftRayProcessor, rightRayProcessor, onNodeSelect, onNodeHover])

  // Reset
  const reset = useCallback(() => {
    leftRayProcessor.reset()
    rightRayProcessor.reset()
    prevStateRef.current = {
      leftZ: null,
      rightZ: null,
      selectedNodeId: null,
      twoHandDistance: null,
      twoHandRotation: null,
    }
    setInteractionState({
      leftRay: null,
      rightRay: null,
      hoveredNode: null,
      selectedNodeId: null,
      isDragging: false,
      dragDeltaZ: 0,
      rotationDelta: { x: 0, y: 0 },
      zoomDelta: 0,
    })
  }, [leftRayProcessor, rightRayProcessor])

  return {
    interactionState,
    processGestures,
    reset,
  }
}

export default useHandInteraction
