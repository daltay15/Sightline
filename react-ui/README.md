# 🚀 React Frontend - Setup and Usage Guide

## Overview

This is the React-based frontend for the Security Camera UI, refactored from the original HTML pages while maintaining the same functionality and design.

## Technology Stack

- **React 18** - Modern React with hooks
- **Vite** - Fast build tool and dev server
- **React Router** - Client-side routing
- **Chart.js** - Data visualization for stats
- **CSS Modules** - Component-scoped styling

## Prerequisites

Before running the React frontend, ensure you have:

- **Node.js** v18 or higher
- **npm** v9 or higher
- The Go backend API server running on `http://localhost:8080`

## Installation

1. Navigate to the react-ui directory:
```bash
cd react-ui
```

2. Install dependencies:
```bash
npm install
```

## Running the Development Server

1. **Start the Go backend** (if not already running):
```bash
cd ../api
go build -o scui-api
./scui-api
```

The backend should now be running on `http://localhost:8080`

2. **Start the React dev server** (in a new terminal):
```bash
cd react-ui
npm run dev
```

The React app will start on `http://localhost:3000` by default.

3. **Open your browser** and navigate to:
```
http://localhost:3000
```

## Available Scripts

### `npm run dev`
Starts the development server with hot module replacement (HMR).
- Opens at `http://localhost:3000`
- Auto-reloads on code changes
- Proxies API requests to `http://localhost:8080`

### `npm run build`
Creates an optimized production build in the `dist/` folder.
```bash
npm run build
```

### `npm run preview`
Preview the production build locally before deploying.
```bash
npm run preview
```

### `npm run lint`
Runs ESLint to check for code quality issues.
```bash
npm run lint
```

## Project Structure

```
react-ui/
├── public/              # Static assets
├── src/
│   ├── components/      # Reusable React components
│   │   ├── Header.jsx   # Navigation header
│   │   └── Header.css
│   ├── pages/           # Page components (routes)
│   │   ├── Detections.jsx    # AI detections page
│   │   ├── Events.jsx         # Camera events page
│   │   ├── Stats.jsx          # Statistics page
│   │   ├── About.jsx          # About/info page
│   │   ├── Config.jsx         # Configuration page
│   │   ├── DebugStats.jsx     # Debug stats page
│   │   └── BackupStatus.jsx   # Backup status page
│   ├── utils/           # Utility functions
│   ├── App.jsx          # Main app component with routing
│   ├── main.jsx         # Entry point
│   └── index.css        # Global styles
├── package.json         # Dependencies and scripts
├── vite.config.js       # Vite configuration
└── README.md            # This file
```

## Features Implemented

### ✅ Core Pages
- **AI Detections** - View events with AI object detection
- **Events** - Browse all camera events with filters
- **Stats** - Analytics and charts with Chart.js
- **About** - System information and documentation
- **Debug Stats** - System health and performance metrics

### ✅ Key Features
- Real-time event loading with pagination
- Camera and date range filtering
- Detection type filtering (people, cars, etc.)
- Grid size customization (1-8 columns)
- Fullscreen image viewing
- Lazy loading for images
- Responsive design for mobile and desktop
- Dark theme matching original design

## API Endpoints Used

The React app communicates with these backend endpoints:

- `GET /events` - List all events
- `GET /events/correlated` - Events with detection data
- `GET /events/cameras` - List of available cameras
- `GET /stats/motion` - Motion detection statistics
- `GET /thumb/:id` - Thumbnail images
- `GET /stream/:id` - Video streaming
- `GET /db-health` - Database health status
- `GET /scan-status` - Scanning status
- `GET /perf` - Performance metrics

## Configuration

### Proxy Configuration
The Vite dev server proxies API requests to the Go backend. This is configured in `vite.config.js`:

```javascript
server: {
  port: 3000,
  proxy: {
    '/events': 'http://localhost:8080',
    '/stats': 'http://localhost:8080',
    '/thumb': 'http://localhost:8080',
    // ... other endpoints
  }
}
```

### Backend URL
If your backend runs on a different port, update the proxy configuration in `vite.config.js`.

## Production Deployment

### Building for Production

1. Build the React app:
```bash
npm run build
```

This creates optimized files in the `dist/` folder.

2. The `dist/` folder contains:
- `index.html` - Entry point
- `assets/` - Minified JS and CSS
- Other static assets

### Deployment Options

#### Option 1: Serve with Go Backend
Copy the built files to the Go backend's static directory:
```bash
# Build the React app
npm run build

# Copy to Go backend (if needed)
# The Go backend can serve these files directly
```

#### Option 2: Separate Static Server
Serve the `dist/` folder with any static file server:
```bash
# Using Python
cd dist
python -m http.server 3000

# Using Node serve package
npx serve dist -p 3000

# Using nginx (configure nginx.conf to point to dist/)
```

Make sure to configure API proxy or CORS on the backend if serving separately.

## Troubleshooting

### Port Already in Use
If port 3000 is busy, Vite will automatically try the next available port. You can also specify a different port:
```bash
npm run dev -- --port 3001
```

### API Connection Issues
If the React app can't connect to the backend:
1. Verify the Go backend is running on `http://localhost:8080`
2. Check the proxy configuration in `vite.config.js`
3. Check browser console for CORS errors

### Build Errors
If you encounter build errors:
1. Delete `node_modules` and `package-lock.json`
2. Run `npm install` again
3. Try `npm run build` again

### Missing Dependencies
If you see import errors:
```bash
npm install react-router-dom chart.js react-chartjs-2
```

## Differences from HTML Version

The React version maintains feature parity with the original HTML version but with improved:
- **Code Organization** - Component-based architecture
- **State Management** - React hooks for state
- **Routing** - Client-side routing with React Router
- **Developer Experience** - Hot module replacement, better debugging
- **Maintainability** - Easier to extend and modify

Some advanced features from the HTML version (like video playback controls) may be enhanced in future updates.

## Browser Support

The React app supports modern browsers:
- Chrome/Edge (last 2 versions)
- Firefox (last 2 versions)
- Safari (last 2 versions)

## Development Tips

### Hot Module Replacement (HMR)
Vite provides instant updates when you edit code. No need to refresh the browser manually.

### React DevTools
Install the React DevTools browser extension for better debugging:
- [Chrome Extension](https://chrome.google.com/webstore/detail/react-developer-tools/fmkadmapgofadopljbjfkapdkoienihi)
- [Firefox Extension](https://addons.mozilla.org/en-US/firefox/addon/react-devtools/)

### VS Code Extensions
Recommended extensions:
- ESLint
- Prettier
- ES7+ React/Redux/React-Native snippets

## Future Enhancements

Potential improvements for future versions:
- [ ] Video playback with controls
- [ ] Full configuration page implementation
- [ ] Backup management interface
- [ ] Real-time event notifications
- [ ] Advanced filtering and search
- [ ] Event tagging and organization
- [ ] Export functionality
- [ ] Multi-language support

## Support

For issues or questions:
1. Check the [main repository README](../README.md)
2. Review the [setup guide](../SETUP.md)
3. Check the browser console for errors
4. Verify backend logs for API issues

## License

Same license as the main project.
