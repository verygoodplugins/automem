interface ToggleControlProps {
  label: string
  checked: boolean
  onChange: (checked: boolean) => void
  description?: string
}

export function ToggleControl({ label, checked, onChange, description }: ToggleControlProps) {
  return (
    <div className="flex items-center justify-between gap-3">
      <div className="flex-1 min-w-0">
        <label className="text-xs text-slate-400 block">{label}</label>
        {description && (
          <span className="text-xs text-slate-600 block mt-0.5">{description}</span>
        )}
      </div>
      <button
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`
          relative inline-flex h-5 w-9 flex-shrink-0 cursor-pointer rounded-full
          transition-colors duration-200 ease-in-out focus:outline-none
          ${checked ? 'bg-blue-500' : 'bg-white/20'}
        `}
      >
        <span
          className={`
            pointer-events-none inline-block h-4 w-4 transform rounded-full
            bg-white shadow ring-0 transition duration-200 ease-in-out
            ${checked ? 'translate-x-4' : 'translate-x-0.5'}
            mt-0.5
          `}
        />
      </button>
    </div>
  )
}
