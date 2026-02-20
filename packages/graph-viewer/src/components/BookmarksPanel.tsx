/**
 * BookmarksPanel - Save and restore camera positions
 *
 * Shows a collapsible list of saved bookmarks with:
 * - Click to navigate
 * - Delete bookmark
 * - Rename bookmark
 * - Quick access indicator (1-9)
 */

import { useState, useCallback } from 'react'
import type { Bookmark } from '../hooks/useBookmarks'

interface BookmarksPanelProps {
  bookmarks: Bookmark[]
  onNavigate: (bookmark: Bookmark) => void
  onDelete: (id: string) => void
  onRename: (id: string, name: string) => void
  onSaveBookmark: () => void
  visible?: boolean
}

export function BookmarksPanel({
  bookmarks,
  onNavigate,
  onDelete,
  onRename,
  onSaveBookmark,
  visible = true,
}: BookmarksPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editName, setEditName] = useState('')

  const handleStartEdit = useCallback((bookmark: Bookmark) => {
    setEditingId(bookmark.id)
    setEditName(bookmark.name)
  }, [])

  const handleSaveEdit = useCallback(() => {
    if (editingId && editName.trim()) {
      onRename(editingId, editName.trim())
    }
    setEditingId(null)
    setEditName('')
  }, [editingId, editName, onRename])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSaveEdit()
    } else if (e.key === 'Escape') {
      setEditingId(null)
      setEditName('')
    }
  }, [handleSaveEdit])

  if (!visible) return null

  return (
    <div className="absolute top-16 right-4 z-40">
      {/* Toggle button */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className={`
          flex items-center gap-2 px-3 py-2 rounded-lg
          bg-slate-800/90 backdrop-blur-sm border border-slate-700/50
          text-slate-300 hover:text-white hover:bg-slate-700/90
          transition-all duration-200
          ${isExpanded ? 'rounded-b-none border-b-0' : ''}
        `}
        title="Bookmarks (Cmd+B to save)"
      >
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
        </svg>
        <span className="text-sm font-medium">Bookmarks</span>
        {bookmarks.length > 0 && (
          <span className="bg-blue-500/30 text-blue-300 text-xs px-1.5 py-0.5 rounded-full">
            {bookmarks.length}
          </span>
        )}
        <svg
          className={`w-4 h-4 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Panel content */}
      {isExpanded && (
        <div
          className="
            bg-slate-800/95 backdrop-blur-sm border border-slate-700/50 border-t-0
            rounded-b-lg rounded-tl-lg overflow-hidden
            max-h-80 overflow-y-auto
            min-w-[240px]
          "
        >
          {/* Save button */}
          <div className="p-2 border-b border-slate-700/50">
            <button
              onClick={onSaveBookmark}
              className="
                w-full flex items-center justify-center gap-2
                px-3 py-2 rounded-md
                bg-blue-600/20 hover:bg-blue-600/30 border border-blue-500/30
                text-blue-300 hover:text-blue-200
                transition-colors text-sm
              "
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              Save Current View
              <span className="text-xs text-blue-400/60 ml-1">(Cmd+B)</span>
            </button>
          </div>

          {/* Bookmark list */}
          {bookmarks.length === 0 ? (
            <div className="p-4 text-center text-slate-500 text-sm">
              No bookmarks yet.
              <br />
              Press Cmd+B to save.
            </div>
          ) : (
            <div className="divide-y divide-slate-700/30">
              {bookmarks.map((bookmark, index) => (
                <div
                  key={bookmark.id}
                  className="group flex items-center gap-2 px-3 py-2 hover:bg-slate-700/30 transition-colors"
                >
                  {/* Quick access number (1-9) */}
                  {index < 9 && (
                    <span className="w-5 h-5 flex items-center justify-center bg-slate-700/50 rounded text-xs text-slate-400 font-mono">
                      {index + 1}
                    </span>
                  )}
                  {index >= 9 && <span className="w-5" />}

                  {/* Bookmark info */}
                  <div className="flex-1 min-w-0">
                    {editingId === bookmark.id ? (
                      <input
                        type="text"
                        value={editName}
                        onChange={(e) => setEditName(e.target.value)}
                        onBlur={handleSaveEdit}
                        onKeyDown={handleKeyDown}
                        autoFocus
                        className="
                          w-full bg-slate-900/50 border border-slate-600 rounded px-2 py-1
                          text-sm text-white focus:outline-none focus:border-blue-500
                        "
                      />
                    ) : (
                      <button
                        onClick={() => onNavigate(bookmark)}
                        className="
                          w-full text-left truncate text-sm text-slate-300 hover:text-white
                          transition-colors
                        "
                        title={`Navigate to ${bookmark.name}`}
                      >
                        {bookmark.name}
                      </button>
                    )}
                    <div className="text-xs text-slate-500 truncate">
                      z: {bookmark.zoom.toFixed(1)}x
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={() => handleStartEdit(bookmark)}
                      className="p-1 rounded hover:bg-slate-600/50 text-slate-400 hover:text-slate-200"
                      title="Rename"
                    >
                      <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                      </svg>
                    </button>
                    <button
                      onClick={() => onDelete(bookmark.id)}
                      className="p-1 rounded hover:bg-red-600/30 text-slate-400 hover:text-red-400"
                      title="Delete"
                    >
                      <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Footer hint */}
          {bookmarks.length > 0 && (
            <div className="p-2 border-t border-slate-700/50 text-xs text-slate-500 text-center">
              Press 1-9 to quick navigate
            </div>
          )}
        </div>
      )}
    </div>
  )
}
