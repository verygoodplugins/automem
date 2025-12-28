import { useQuery } from '@tanstack/react-query'
import {
  fetchGraphSnapshot,
  fetchGraphNeighbors,
  fetchGraphStats,
  fetchProjectedGraph,
  type SnapshotParams,
  type NeighborsParams,
  type ProjectedParams,
} from '../api/client'

export function useGraphSnapshot(params: SnapshotParams & { enabled?: boolean } = {}) {
  const { enabled = true, ...queryParams } = params

  return useQuery({
    queryKey: ['graph', 'snapshot', queryParams],
    queryFn: () => fetchGraphSnapshot(queryParams),
    enabled,
  })
}

export function useProjectedGraph(params: ProjectedParams & { enabled?: boolean } = {}) {
  const { enabled = true, ...queryParams } = params

  return useQuery({
    queryKey: ['graph', 'projected', queryParams],
    queryFn: () => fetchProjectedGraph(queryParams),
    enabled,
    staleTime: 1000 * 60 * 5, // Cache UMAP projections for 5 mins (expensive to compute)
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
