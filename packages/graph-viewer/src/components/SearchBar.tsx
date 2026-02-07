import { useState, useEffect, useCallback } from 'react'
import { Search, X } from 'lucide-react'

interface SearchBarProps {
  value: string
  onChange: (value: string) => void
  className?: string
}

export function SearchBar({ value, onChange, className = '' }: SearchBarProps) {
  const [localValue, setLocalValue] = useState(value)

  // Debounce the onChange callback
  useEffect(() => {
    const timer = setTimeout(() => {
      onChange(localValue)
    }, 300)

    return () => clearTimeout(timer)
  }, [localValue, onChange])

  // Sync external value changes
  useEffect(() => {
    setLocalValue(value)
  }, [value])

  const handleClear = useCallback(() => {
    setLocalValue('')
    onChange('')
  }, [onChange])

  return (
    <div className={`relative ${className}`}>
      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
        <Search className="w-4 h-4 text-slate-500" />
      </div>
      <input
        type="text"
        value={localValue}
        onChange={(e) => setLocalValue(e.target.value)}
        placeholder="Search memories..."
        className="w-full pl-9 pr-9 py-2 bg-black/30 border border-white/10 rounded-lg focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/50 outline-none transition-colors text-sm text-slate-100 placeholder-slate-500"
      />
      {localValue && (
        <button
          onClick={handleClear}
          className="absolute inset-y-0 right-0 pr-3 flex items-center text-slate-500 hover:text-slate-300"
        >
          <X className="w-4 h-4" />
        </button>
      )}
    </div>
  )
}
