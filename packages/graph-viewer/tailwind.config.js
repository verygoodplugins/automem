/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Memory type colors
        decision: '#3B82F6',
        pattern: '#10B981',
        preference: '#8B5CF6',
        style: '#EC4899',
        habit: '#F59E0B',
        insight: '#F97316',
        context: '#6B7280',
        memory: '#94A3B8',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 5px currentColor' },
          '100%': { boxShadow: '0 0 20px currentColor' },
        },
      },
    },
  },
  plugins: [],
}
