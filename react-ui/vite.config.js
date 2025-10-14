import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 3000,
    proxy: {
      // Core API endpoints
      '/events': 'http://localhost:8080',
      '/cameras': 'http://localhost:8080',
      '/stats': 'http://localhost:8080',
      '/thumb': 'http://localhost:8080',
      '/stream': 'http://localhost:8080',
      '/detection': 'http://localhost:8080',
      '/update-video-metadata': 'http://localhost:8080',
      
      // System endpoints
      '/health': 'http://localhost:8080',
      '/perf': 'http://localhost:8080',
      '/debug': 'http://localhost:8080',
      '/scan-status': 'http://localhost:8080',
      '/db-health': 'http://localhost:8080',
      '/processing-status': 'http://localhost:8080',
      '/detection-status': 'http://localhost:8080',
      '/watcher-health': 'http://localhost:8080',
      '/system-metrics': 'http://localhost:8080',
      
      // GPU/Detection endpoints
      '/gpu': 'http://localhost:8080',
      '/detections': 'http://localhost:8080',
      '/test': 'http://localhost:8080',
      
      // Configuration endpoints
      '/api/config': 'http://localhost:8080',
      '/config': 'http://localhost:8080',
      
      // Backup endpoints
      '/api/backup': 'http://localhost:8080',
      '/backup': 'http://localhost:8080',
      
      // Utility endpoints
      '/cleanup': 'http://localhost:8080',
      '/ui': 'http://localhost:8080',
    }
  }
})
