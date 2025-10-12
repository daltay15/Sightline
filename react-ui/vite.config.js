import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/events': 'http://localhost:8080',
      '/stats': 'http://localhost:8080',
      '/thumb': 'http://localhost:8080',
      '/stream': 'http://localhost:8080',
      '/health': 'http://localhost:8080',
      '/perf': 'http://localhost:8080',
      '/debug': 'http://localhost:8080',
      '/scan-status': 'http://localhost:8080',
      '/db-health': 'http://localhost:8080',
      '/processing-status': 'http://localhost:8080',
      '/detection-status': 'http://localhost:8080',
      '/config': 'http://localhost:8080',
      '/backup': 'http://localhost:8080',
      '/cleanup': 'http://localhost:8080',
    }
  }
})
