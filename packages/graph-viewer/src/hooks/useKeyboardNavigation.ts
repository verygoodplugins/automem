import { useEffect, useCallback, useRef } from 'react'
import type { GraphNode, SimulationNode } from '../lib/types'

interface UseKeyboardNavigationOptions {
  nodes: SimulationNode[]
  selectedNode: GraphNode | null
  onNodeSelect: (node: GraphNode | null) => void
  onReheat?: () => void
  onResetView?: () => void
  onToggleSettings?: () => void
  onToggleLabels?: () => void
  enabled?: boolean
}

interface KeyboardShortcuts {
  [key: string]: {
    description: string
    action: () => void
  }
}

/**
 * Keyboard navigation for graph viewer
 * Provides Obsidian-style keyboard shortcuts
 */
export function useKeyboardNavigation({
  nodes,
  selectedNode,
  onNodeSelect,
  onReheat,
  onResetView,
  onToggleSettings,
  onToggleLabels,
  enabled = true,
}: UseKeyboardNavigationOptions) {
  const nodesRef = useRef(nodes)
  nodesRef.current = nodes

  const selectedRef = useRef(selectedNode)
  selectedRef.current = selectedNode

  // Find nearest node in a direction from the selected node
  const findNodeInDirection = useCallback(
    (direction: 'up' | 'down' | 'left' | 'right' | 'forward' | 'backward') => {
      const current = selectedRef.current
      const allNodes = nodesRef.current

      if (!current || allNodes.length === 0) {
        // If no selection, select first node
        return allNodes[0] || null
      }

      const currentNode = allNodes.find((n) => n.id === current.id)
      if (!currentNode) return null

      const cx = currentNode.x ?? 0
      const cy = currentNode.y ?? 0
      const cz = currentNode.z ?? 0

      // Filter candidates based on direction
      const candidates = allNodes.filter((n) => {
        if (n.id === current.id) return false

        const nx = n.x ?? 0
        const ny = n.y ?? 0
        const nz = n.z ?? 0

        switch (direction) {
          case 'up':
            return ny > cy
          case 'down':
            return ny < cy
          case 'left':
            return nx < cx
          case 'right':
            return nx > cx
          case 'forward':
            return nz < cz
          case 'backward':
            return nz > cz
          default:
            return false
        }
      })

      if (candidates.length === 0) return null

      // Find nearest candidate
      let nearest = candidates[0]
      let minDist = Infinity

      for (const n of candidates) {
        const dx = (n.x ?? 0) - cx
        const dy = (n.y ?? 0) - cy
        const dz = (n.z ?? 0) - cz
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz)
        if (dist < minDist) {
          minDist = dist
          nearest = n
        }
      }

      return nearest
    },
    []
  )

  // Navigate to next/previous node in list order
  const navigateSequential = useCallback((direction: 'next' | 'previous') => {
    const allNodes = nodesRef.current
    const current = selectedRef.current

    if (allNodes.length === 0) return null

    if (!current) {
      return direction === 'next' ? allNodes[0] : allNodes[allNodes.length - 1]
    }

    const currentIndex = allNodes.findIndex((n) => n.id === current.id)
    if (currentIndex === -1) return allNodes[0]

    const nextIndex =
      direction === 'next'
        ? (currentIndex + 1) % allNodes.length
        : (currentIndex - 1 + allNodes.length) % allNodes.length

    return allNodes[nextIndex]
  }, [])

  // Keyboard event handler
  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (!enabled) return

      // Ignore if focus is in an input
      if (
        event.target instanceof HTMLInputElement ||
        event.target instanceof HTMLTextAreaElement
      ) {
        return
      }

      const shortcuts: KeyboardShortcuts = {
        // Navigation
        ArrowUp: {
          description: 'Navigate up',
          action: () => {
            const node = event.shiftKey
              ? findNodeInDirection('backward')
              : findNodeInDirection('up')
            if (node) onNodeSelect(node)
          },
        },
        ArrowDown: {
          description: 'Navigate down',
          action: () => {
            const node = event.shiftKey
              ? findNodeInDirection('forward')
              : findNodeInDirection('down')
            if (node) onNodeSelect(node)
          },
        },
        ArrowLeft: {
          description: 'Navigate left',
          action: () => {
            const node = findNodeInDirection('left')
            if (node) onNodeSelect(node)
          },
        },
        ArrowRight: {
          description: 'Navigate right',
          action: () => {
            const node = findNodeInDirection('right')
            if (node) onNodeSelect(node)
          },
        },
        Tab: {
          description: 'Next/previous node',
          action: () => {
            event.preventDefault()
            const node = event.shiftKey
              ? navigateSequential('previous')
              : navigateSequential('next')
            if (node) onNodeSelect(node)
          },
        },

        // Selection
        Escape: {
          description: 'Deselect',
          action: () => {
            onNodeSelect(null)
          },
        },

        // Actions
        r: {
          description: 'Reheat simulation',
          action: () => {
            if (!event.metaKey && !event.ctrlKey) {
              onReheat?.()
            }
          },
        },
        R: {
          description: 'Reset view',
          action: () => {
            if (event.shiftKey) {
              onResetView?.()
            }
          },
        },
        ',': {
          description: 'Toggle settings',
          action: () => {
            onToggleSettings?.()
          },
        },
        l: {
          description: 'Toggle labels',
          action: () => {
            if (!event.metaKey && !event.ctrlKey) {
              onToggleLabels?.()
            }
          },
        },

        // Help
        '?': {
          description: 'Show help',
          action: () => {
            // Could show a help modal in the future
            console.log('Keyboard shortcuts:')
            console.log('  Arrow keys: Navigate between nodes')
            console.log('  Shift+Arrow Up/Down: Navigate in Z axis')
            console.log('  Tab/Shift+Tab: Cycle through nodes')
            console.log('  Escape: Deselect')
            console.log('  R: Reheat simulation')
            console.log('  Shift+R: Reset view')
            console.log('  ,: Toggle settings')
            console.log('  L: Toggle labels')
          },
        },
      }

      const shortcut = shortcuts[event.key]
      if (shortcut) {
        shortcut.action()
      }
    },
    [enabled, findNodeInDirection, navigateSequential, onNodeSelect, onReheat, onResetView, onToggleSettings, onToggleLabels]
  )

  // Attach event listener
  useEffect(() => {
    if (!enabled) return

    window.addEventListener('keydown', handleKeyDown)
    return () => {
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [enabled, handleKeyDown])

  // Return shortcuts info for help display
  return {
    shortcuts: [
      { key: '↑↓←→', description: 'Navigate between nodes' },
      { key: 'Shift+↑↓', description: 'Navigate in Z axis' },
      { key: 'Tab', description: 'Next node' },
      { key: 'Shift+Tab', description: 'Previous node' },
      { key: 'Esc', description: 'Deselect' },
      { key: 'R', description: 'Reheat simulation' },
      { key: 'Shift+R', description: 'Reset view' },
      { key: ',', description: 'Toggle settings' },
      { key: 'L', description: 'Toggle labels' },
      { key: '?', description: 'Show help' },
    ],
  }
}
