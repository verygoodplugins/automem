import { useQuery } from '@tanstack/react-query'
import { fetchGraphSnapshot, fetchGraphNeighbors, fetchGraphStats, type SnapshotParams, type NeighborsParams } from '../api/client'

export function useGraphSnapshot(params: SnapshotParams & { enabled?: boolean } = {}) {
  const { enabled = true, ...queryParams } = params

  return useQuery({
    queryKey: ['graph', 'snapshot', queryParams],
    queryFn: () => fetchGraphSnapshot(queryParams),
    enabled,
  })
}

export function useGraphNeighbors(memoryId: string | null, params: NeighborsParams = {}) {
  return useQuery({
    queryKey: ['graph', 'neighbors', memoryId, params],
    queryFn: () => fetchGraphNeighbors(memoryId!, params),
    enabled: !!memoryId,
  })
}

export function useGraphStats(enabled = true) {
  return useQuery({
    queryKey: ['graph', 'stats'],
    queryFn: fetchGraphStats,
    enabled,
  })
}
