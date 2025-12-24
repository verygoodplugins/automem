/**
 * RadialMenu - Quick actions menu that appears on right-click
 *
 * Features:
 * - 8 action items arranged in a circle
 * - Smooth expand animation from center
 * - Icons scale in sequentially
 * - Hover enlarges items
 * - Click outside or press Escape to close
 */

import { useEffect, useCallback, useState } from 'react'
import {
  Search,
  Sun,
  Route,
  Plus,
  Pencil,
  FileText,
  Copy,
  Trash2,
  X,
} from 'lucide-react'
import type { GraphNode } from '../lib/types'

interface RadialMenuItem {
  id: string
  icon: React.ReactNode
  label: string
  color: string
  action: () => void
  disabled?: boolean
}

interface RadialMenuProps {
  node: GraphNode
  position: { x: number; y: number }
  onClose: () => void
  onFindSimilar?: (node: GraphNode) => void
  onToggleFocus?: () => void
  onStartPath?: (nodeId: string) => void
  onAddToSelection?: (node: GraphNode) => void
  onEdit?: (node: GraphNode) => void
  onViewContent?: (node: GraphNode) => void
  onCopyId?: (nodeId: string) => void
  onDelete?: (node: GraphNode) => void
  focusModeEnabled?: boolean
}

export function RadialMenu({
  node,
  position,
  onClose,
  onFindSimilar,
  onToggleFocus,
  onStartPath,
  onAddToSelection,
  onEdit,
  onViewContent,
  onCopyId,
  onDelete,
  focusModeEnabled = false,
}: RadialMenuProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [hoveredItem, setHoveredItem] = useState<string | null>(null)

  // Animate open on mount
  useEffect(() => {
    const timer = setTimeout(() => setIsOpen(true), 10)
    return () => clearTimeout(timer)
  }, [])

  // Handle escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [onClose])

  // Handle click outside
  const handleBackdropClick = useCallback(
    (e: React.MouseEvent) => {
      if (e.target === e.currentTarget) {
        onClose()
      }
    },
    [onClose]
  )

  // Copy ID to clipboard
  const handleCopyId = useCallback(() => {
    navigator.clipboard.writeText(node.id)
    onCopyId?.(node.id)
    onClose()
  }, [node.id, onCopyId, onClose])

  // Menu items arranged in a circle (8 positions, starting from top)
  const menuItems: RadialMenuItem[] = [
    {
      id: 'find-similar',
      icon: <Search className="w-5 h-5" />,
      label: 'Find Similar',
      color: 'from-blue-500 to-cyan-500',
      action: () => {
        onFindSimilar?.(node)
        onClose()
      },
    },
    {
      id: 'focus',
      icon: <Sun className="w-5 h-5" />,
      label: focusModeEnabled ? 'Exit Focus' : 'Focus Mode',
      color: focusModeEnabled ? 'from-amber-500 to-yellow-500' : 'from-amber-400 to-orange-500',
      action: () => {
        onToggleFocus?.()
        onClose()
      },
    },
    {
      id: 'start-path',
      icon: <Route className="w-5 h-5" />,
      label: 'Find Path To...',
      color: 'from-cyan-500 to-teal-500',
      action: () => {
        onStartPath?.(node.id)
        onClose()
      },
    },
    {
      id: 'add-selection',
      icon: <Plus className="w-5 h-5" />,
      label: 'Add to Selection',
      color: 'from-green-500 to-emerald-500',
      action: () => {
        onAddToSelection?.(node)
        onClose()
      },
      disabled: true, // TODO: Implement multi-selection
    },
    {
      id: 'view-content',
      icon: <FileText className="w-5 h-5" />,
      label: 'View Content',
      color: 'from-purple-500 to-violet-500',
      action: () => {
        onViewContent?.(node)
        onClose()
      },
    },
    {
      id: 'edit',
      icon: <Pencil className="w-5 h-5" />,
      label: 'Edit Memory',
      color: 'from-indigo-500 to-blue-500',
      action: () => {
        onEdit?.(node)
        onClose()
      },
      disabled: true, // TODO: Implement edit
    },
    {
      id: 'copy-id',
      icon: <Copy className="w-5 h-5" />,
      label: 'Copy ID',
      color: 'from-slate-500 to-gray-500',
      action: handleCopyId,
    },
    {
      id: 'delete',
      icon: <Trash2 className="w-5 h-5" />,
      label: 'Delete',
      color: 'from-red-500 to-rose-500',
      action: () => {
        onDelete?.(node)
        onClose()
      },
      disabled: true, // TODO: Implement delete with confirmation
    },
  ]

  // Calculate position for each item (arranged in a circle)
  const radius = 80 // Distance from center
  const getItemPosition = (index: number, total: number) => {
    // Start from top (-90 degrees) and go clockwise
    const angle = ((index / total) * 360 - 90) * (Math.PI / 180)
    return {
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius,
    }
  }

  return (
    <div
      className="fixed inset-0 z-50"
      onClick={handleBackdropClick}
      style={{ cursor: 'default' }}
    >
      {/* Menu container positioned at click location */}
      <div
        className="absolute"
        style={{
          left: position.x,
          top: position.y,
          transform: 'translate(-50%, -50%)',
        }}
      >
        {/* Center button (close) */}
        <button
          onClick={onClose}
          className={`
            absolute z-10 w-12 h-12 -translate-x-1/2 -translate-y-1/2
            rounded-full bg-slate-800 border border-slate-600
            flex items-center justify-center
            transition-all duration-300 ease-out
            hover:bg-slate-700 hover:scale-110
            ${isOpen ? 'scale-100 opacity-100' : 'scale-0 opacity-0'}
          `}
          style={{ left: 0, top: 0 }}
        >
          <X className="w-5 h-5 text-slate-400" />
        </button>

        {/* Radial menu items */}
        {menuItems.map((item, index) => {
          const pos = getItemPosition(index, menuItems.length)
          const isHovered = hoveredItem === item.id
          const delay = index * 30 // Staggered animation

          return (
            <button
              key={item.id}
              onClick={item.disabled ? undefined : item.action}
              onMouseEnter={() => setHoveredItem(item.id)}
              onMouseLeave={() => setHoveredItem(null)}
              disabled={item.disabled}
              className={`
                absolute w-12 h-12 -translate-x-1/2 -translate-y-1/2
                rounded-full
                flex items-center justify-center
                transition-all ease-out
                ${item.disabled
                  ? 'opacity-30 cursor-not-allowed bg-slate-800'
                  : `bg-gradient-to-br ${item.color} shadow-lg cursor-pointer hover:shadow-xl`
                }
                ${isHovered && !item.disabled ? 'scale-125 z-20' : 'scale-100'}
              `}
              style={{
                left: isOpen ? pos.x : 0,
                top: isOpen ? pos.y : 0,
                opacity: isOpen ? 1 : 0,
                transitionDuration: '300ms',
                transitionDelay: isOpen ? `${delay}ms` : '0ms',
              }}
              title={item.label}
            >
              <div className="text-white">{item.icon}</div>
            </button>
          )
        })}

        {/* Tooltip for hovered item */}
        {hoveredItem && (
          <div
            className={`
              absolute left-1/2 -translate-x-1/2
              px-3 py-1.5 rounded-lg
              bg-slate-900/95 backdrop-blur-sm
              border border-slate-700/50
              text-sm text-white font-medium
              whitespace-nowrap
              pointer-events-none
              transition-all duration-150
              ${isOpen ? 'opacity-100' : 'opacity-0'}
            `}
            style={{ top: radius + 40 }}
          >
            {menuItems.find((i) => i.id === hoveredItem)?.label}
          </div>
        )}
      </div>
    </div>
  )
}
