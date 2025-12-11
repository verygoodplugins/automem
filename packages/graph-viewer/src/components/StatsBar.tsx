import { Database, GitBranch } from 'lucide-react'

interface StatsBarProps {
  stats?: {
    total_nodes: number
    total_edges: number
    returned_nodes: number
    returned_edges: number
    sampled: boolean
    sample_ratio: number
  }
  isLoading: boolean
}

export function StatsBar({ stats, isLoading }: StatsBarProps) {
  if (isLoading || !stats) {
    return (
      <div className="flex items-center gap-4 text-sm text-slate-500">
        <div className="flex items-center gap-1.5">
          <Database className="w-4 h-4" />
          <span>Loading...</span>
        </div>
      </div>
    )
  }

  return (
    <div className="flex items-center gap-4 text-sm">
      <div className="flex items-center gap-1.5 text-slate-400">
        <Database className="w-4 h-4 text-blue-400" />
        <span>
          <span className="text-slate-200">{stats.returned_nodes.toLocaleString()}</span>
          {stats.sampled && (
            <span className="text-slate-500">
              {' '}/ {stats.total_nodes.toLocaleString()}
            </span>
          )}
        </span>
      </div>
      <div className="flex items-center gap-1.5 text-slate-400">
        <GitBranch className="w-4 h-4 text-purple-400" />
        <span>
          <span className="text-slate-200">{stats.returned_edges.toLocaleString()}</span>
          {stats.sampled && (
            <span className="text-slate-500">
              {' '}/ {stats.total_edges.toLocaleString()}
            </span>
          )}
        </span>
      </div>
    </div>
  )
}
