import { useState, FormEvent } from 'react'
import { KeyRound, ArrowRight, AlertCircle, Server } from 'lucide-react'
import { checkHealth, setServerConfig } from '../api/client'

interface TokenPromptProps {
  onSubmit: (token: string) => void
}

export function TokenPrompt({ onSubmit }: TokenPromptProps) {
  const [serverUrl, setServerUrl] = useState('https://automem.up.railway.app')
  const [token, setToken] = useState('')
  const [isValidating, setIsValidating] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    if (!token.trim() || !serverUrl.trim()) return

    setIsValidating(true)
    setError(null)

    try {
      // Test connection to the server
      await checkHealth(serverUrl)

      // Store config and notify parent
      setServerConfig(serverUrl, token)
      onSubmit(token)
    } catch (err) {
      setError((err as Error).message || 'Connection failed')
    } finally {
      setIsValidating(false)
    }
  }

  return (
    <div className="h-screen w-screen bg-[#0a0a0f] flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="flex flex-col items-center mb-8">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center mb-4 shadow-lg shadow-blue-500/20">
            <span className="text-white font-bold text-2xl">AM</span>
          </div>
          <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            AutoMem Graph Viewer
          </h1>
          <p className="text-slate-400 mt-2 text-center">
            Explore your AI memory graph in 3D
          </p>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="glass rounded-xl p-6 space-y-4">
          {/* Server URL */}
          <div>
            <label className="block mb-2 text-sm font-medium text-slate-300">
              Server URL
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Server className="w-5 h-5 text-slate-500" />
              </div>
              <input
                type="url"
                value={serverUrl}
                onChange={(e) => setServerUrl(e.target.value)}
                placeholder="https://automem.up.railway.app"
                className="w-full pl-10 pr-4 py-3 bg-black/30 border border-white/10 rounded-lg focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none transition-colors text-slate-100 placeholder-slate-500"
              />
            </div>
            <p className="mt-1 text-xs text-slate-500">
              Your AutoMem server endpoint
            </p>
          </div>

          {/* API Token */}
          <div>
            <label className="block mb-2 text-sm font-medium text-slate-300">
              API Token
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <KeyRound className="w-5 h-5 text-slate-500" />
              </div>
              <input
                type="password"
                value={token}
                onChange={(e) => setToken(e.target.value)}
                placeholder="Enter your AutoMem API token"
                className="w-full pl-10 pr-4 py-3 bg-black/30 border border-white/10 rounded-lg focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none transition-colors text-slate-100 placeholder-slate-500"
                autoFocus
              />
            </div>
          </div>

          {error && (
            <div className="flex items-center gap-2 text-red-400 text-sm p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={!token.trim() || !serverUrl.trim() || isValidating}
            className="w-full flex items-center justify-center gap-2 py-3 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg font-medium transition-all"
          >
            {isValidating ? (
              <>
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Connecting...
              </>
            ) : (
              <>
                Connect
                <ArrowRight className="w-5 h-5" />
              </>
            )}
          </button>

          <p className="text-xs text-slate-500 text-center">
            Your credentials are stored locally and never sent to third parties.
          </p>
        </form>

        {/* Help text */}
        <p className="mt-6 text-sm text-slate-500 text-center">
          Don&apos;t have an AutoMem server?{' '}
          <a
            href="https://github.com/verygoodplugins/automem"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-400 hover:text-blue-300"
          >
            Deploy one now
          </a>
        </p>
      </div>
    </div>
  )
}
