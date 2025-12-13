import type { GraphSnapshot, GraphNeighbors, GraphStats } from '../lib/types'

/**
 * Detect if running in embedded mode (served from /viewer/ on same origin).
 * In embedded mode, we use relative URLs and get token from URL hash.
 */
function isEmbeddedMode(): boolean {
  return window.location.pathname.startsWith('/viewer')
}

/**
 * Get token from URL hash (e.g., /viewer/#token=xxx).
 * This keeps the token client-side only, never sent to server in URL.
 */
function getTokenFromHash(): string | null {
  const hash = window.location.hash
  if (!hash) return null
  const params = new URLSearchParams(hash.slice(1))
  return params.get('token')
}

function getApiBase(): string {
  // Allow override via URL param for local dev against remote backend
  const urlParams = new URLSearchParams(window.location.search)
  const serverOverride = urlParams.get('server')
  if (serverOverride) {
    return serverOverride
  }

  if (isEmbeddedMode()) {
    // In embedded mode, use relative URL (same origin)
    return ''
  }
  return localStorage.getItem('automem_server') || 'http://localhost:8001'
}

function getTokenFromQuery(): string | null {
  const urlParams = new URLSearchParams(window.location.search)
  return urlParams.get('token')
}

function getToken(): string | null {
  // Priority: URL query param > URL hash > localStorage
  return getTokenFromQuery() || getTokenFromHash() || localStorage.getItem('automem_token')
}

function getAuthHeaders(): HeadersInit {
  const token = getToken()
  if (!token) {
    throw new Error('No API token configured')
  }
  return {
    'Content-Type': 'application/json',
    'X-API-Key': token,
  }
}

export function setServerConfig(serverUrl: string, token: string): void {
  localStorage.setItem('automem_server', serverUrl)
  localStorage.setItem('automem_token', token)
}

export function getServerConfig(): { serverUrl: string; token: string } | null {
  // Check URL params first (for local dev against remote backend)
  const urlParams = new URLSearchParams(window.location.search)
  const serverOverride = urlParams.get('server')
  const tokenOverride = urlParams.get('token')
  if (serverOverride && tokenOverride) {
    return { serverUrl: serverOverride, token: tokenOverride }
  }

  // In embedded mode, check for hash token
  if (isEmbeddedMode()) {
    const hashToken = getTokenFromHash()
    if (hashToken) {
      return { serverUrl: window.location.origin, token: hashToken }
    }
  }

  const serverUrl = localStorage.getItem('automem_server')
  const token = localStorage.getItem('automem_token')
  if (!serverUrl || !token) return null
  return { serverUrl, token }
}

export function isAuthenticated(): boolean {
  return !!getToken()
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text()
    let message = `API error: ${response.status}`
    try {
      const json = JSON.parse(text)
      message = json.description || json.error || message
    } catch {
      message = text || message
    }
    throw new Error(message)
  }
  return response.json()
}

export interface SnapshotParams {
  limit?: number
  minImportance?: number
  types?: string[]
  since?: string
}

export async function fetchGraphSnapshot(params: SnapshotParams = {}): Promise<GraphSnapshot> {
  const searchParams = new URLSearchParams()

  if (params.limit) searchParams.set('limit', String(params.limit))
  if (params.minImportance) searchParams.set('min_importance', String(params.minImportance))
  if (params.types?.length) searchParams.set('types', params.types.join(','))
  if (params.since) searchParams.set('since', params.since)

  const url = `${getApiBase()}/graph/snapshot?${searchParams}`
  const response = await fetch(url, { headers: getAuthHeaders() })
  return handleResponse<GraphSnapshot>(response)
}

export interface NeighborsParams {
  depth?: number
  includeSemantic?: boolean
  semanticLimit?: number
}

export async function fetchGraphNeighbors(
  memoryId: string,
  params: NeighborsParams = {}
): Promise<GraphNeighbors> {
  const searchParams = new URLSearchParams()

  if (params.depth) searchParams.set('depth', String(params.depth))
  if (params.includeSemantic !== undefined) {
    searchParams.set('include_semantic', String(params.includeSemantic))
  }
  if (params.semanticLimit) searchParams.set('semantic_limit', String(params.semanticLimit))

  const url = `${getApiBase()}/graph/neighbors/${memoryId}?${searchParams}`
  const response = await fetch(url, { headers: getAuthHeaders() })
  return handleResponse<GraphNeighbors>(response)
}

export async function fetchGraphStats(): Promise<GraphStats> {
  const response = await fetch(`${getApiBase()}/graph/stats`, { headers: getAuthHeaders() })
  return handleResponse<GraphStats>(response)
}

export async function updateMemory(
  memoryId: string,
  updates: { importance?: number; tags?: string[]; content?: string }
): Promise<void> {
  const response = await fetch(`${getApiBase()}/memory/${memoryId}`, {
    method: 'PATCH',
    headers: getAuthHeaders(),
    body: JSON.stringify(updates),
  })
  await handleResponse(response)
}

export async function deleteMemory(memoryId: string): Promise<void> {
  const response = await fetch(`${getApiBase()}/memory/${memoryId}`, {
    method: 'DELETE',
    headers: getAuthHeaders(),
  })
  await handleResponse(response)
}

export async function checkHealth(serverUrl?: string): Promise<{ status: string }> {
  const base = serverUrl || getApiBase()
  const response = await fetch(`${base}/health`)
  return handleResponse(response)
}
