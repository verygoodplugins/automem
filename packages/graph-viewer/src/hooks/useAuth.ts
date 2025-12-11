import { useState, useCallback, useEffect } from 'react'
import { isAuthenticated as checkAuth } from '../api/client'

const TOKEN_KEY = 'automem_token'
const SERVER_KEY = 'automem_server'

export function useAuth() {
  const [token, setTokenState] = useState<string | null>(() => {
    return localStorage.getItem(TOKEN_KEY)
  })
  const [serverUrl, setServerUrlState] = useState<string | null>(() => {
    return localStorage.getItem(SERVER_KEY)
  })

  const setToken = useCallback((newToken: string) => {
    localStorage.setItem(TOKEN_KEY, newToken)
    setTokenState(newToken)
  }, [])

  const setServerUrl = useCallback((url: string) => {
    localStorage.setItem(SERVER_KEY, url)
    setServerUrlState(url)
  }, [])

  const clearAuth = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(SERVER_KEY)
    setTokenState(null)
    setServerUrlState(null)
  }, [])

  // Sync with localStorage changes from other tabs
  useEffect(() => {
    const handleStorage = (e: StorageEvent) => {
      if (e.key === TOKEN_KEY) {
        setTokenState(e.newValue)
      }
      if (e.key === SERVER_KEY) {
        setServerUrlState(e.newValue)
      }
    }
    window.addEventListener('storage', handleStorage)
    return () => window.removeEventListener('storage', handleStorage)
  }, [])

  // Check authentication using client's method (supports hash tokens)
  const isAuthenticated = checkAuth()

  return {
    token,
    serverUrl,
    setToken,
    setServerUrl,
    clearAuth,
    isAuthenticated,
  }
}
