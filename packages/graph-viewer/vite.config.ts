import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  // Base path for embedded mode - assets served from /viewer/static/
  base: '/viewer/static/',
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
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
