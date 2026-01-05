import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  // Use VITE_BASE for deployment, default to /viewer/static/ for local dev
  base: process.env.VITE_BASE || '/viewer/static/',
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      // IMPORTANT: Force a single Three.js instance.
      // @react-three/xr -> @pmndrs/xr currently pulls in @iwer/devui/@iwer/sem which depend on three@0.165.0,
      // which triggers: "WARNING: Multiple instances of Three.js being imported."
      // This alias ensures all imports resolve to the top-level three dependency.
      three: path.resolve(__dirname, './node_modules/three'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      // Proxy to Railway backend (or override via VITE_API_TARGET env var)
      '/graph': {
        target: process.env.VITE_API_TARGET || 'https://automem.up.railway.app',
        changeOrigin: true,
        secure: true,
      },
      '/recall': {
        target: process.env.VITE_API_TARGET || 'https://automem.up.railway.app',
        changeOrigin: true,
        secure: true,
      },
      '/memory': {
        target: process.env.VITE_API_TARGET || 'https://automem.up.railway.app',
        changeOrigin: true,
        secure: true,
      },
      '/health': {
        target: process.env.VITE_API_TARGET || 'https://automem.up.railway.app',
        changeOrigin: true,
        secure: true,
      },
    },
  },
})
