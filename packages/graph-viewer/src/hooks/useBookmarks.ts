/**
 * useBookmarks - Save and restore camera positions
 *
 * Persists bookmarks to localStorage for cross-session access.
 * Each bookmark captures camera position, zoom, and optionally selected node.
 */

import { useState, useCallback, useEffect } from 'react'

export interface Bookmark {
  id: string
  name: string
  position: { x: number; y: number; z: number }
  zoom: number
  selectedNodeId?: string
  createdAt: string
  thumbnail?: string // Base64 encoded thumbnail (optional)
}

interface UseBookmarksOptions {
  storageKey?: string
  maxBookmarks?: number
}

const DEFAULT_STORAGE_KEY = 'graph-viewer-bookmarks'
const DEFAULT_MAX_BOOKMARKS = 20

export function useBookmarks({
  storageKey = DEFAULT_STORAGE_KEY,
  maxBookmarks = DEFAULT_MAX_BOOKMARKS,
}: UseBookmarksOptions = {}) {
  const [bookmarks, setBookmarks] = useState<Bookmark[]>([])

  // Load bookmarks from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(storageKey)
      if (stored) {
        const parsed = JSON.parse(stored)
        if (Array.isArray(parsed)) {
          setBookmarks(parsed)
        }
      }
    } catch (e) {
      console.warn('Failed to load bookmarks:', e)
    }
  }, [storageKey])

  // Save bookmarks to localStorage whenever they change
  useEffect(() => {
    try {
      localStorage.setItem(storageKey, JSON.stringify(bookmarks))
    } catch (e) {
      console.warn('Failed to save bookmarks:', e)
    }
  }, [bookmarks, storageKey])

  // Add a new bookmark
  const addBookmark = useCallback((
    position: { x: number; y: number; z: number },
    zoom: number,
    selectedNodeId?: string,
    name?: string,
    thumbnail?: string
  ) => {
    const newBookmark: Bookmark = {
      id: `bookmark-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      name: name || `Bookmark ${bookmarks.length + 1}`,
      position,
      zoom,
      selectedNodeId,
      createdAt: new Date().toISOString(),
      thumbnail,
    }

    setBookmarks(prev => {
      const updated = [newBookmark, ...prev]
      // Limit to max bookmarks
      return updated.slice(0, maxBookmarks)
    })

    return newBookmark
  }, [bookmarks.length, maxBookmarks])

  // Update an existing bookmark
  const updateBookmark = useCallback((id: string, updates: Partial<Omit<Bookmark, 'id' | 'createdAt'>>) => {
    setBookmarks(prev =>
      prev.map(b => b.id === id ? { ...b, ...updates } : b)
    )
  }, [])

  // Delete a bookmark
  const deleteBookmark = useCallback((id: string) => {
    setBookmarks(prev => prev.filter(b => b.id !== id))
  }, [])

  // Get bookmark by index (1-9 for quick access)
  const getBookmarkByIndex = useCallback((index: number): Bookmark | undefined => {
    // index is 1-based (keyboard shortcuts 1-9)
    return bookmarks[index - 1]
  }, [bookmarks])

  // Clear all bookmarks
  const clearAllBookmarks = useCallback(() => {
    setBookmarks([])
  }, [])

  // Reorder bookmarks (for drag-and-drop)
  const reorderBookmarks = useCallback((fromIndex: number, toIndex: number) => {
    setBookmarks(prev => {
      const updated = [...prev]
      const [moved] = updated.splice(fromIndex, 1)
      updated.splice(toIndex, 0, moved)
      return updated
    })
  }, [])

  return {
    bookmarks,
    addBookmark,
    updateBookmark,
    deleteBookmark,
    getBookmarkByIndex,
    clearAllBookmarks,
    reorderBookmarks,
  }
}
