# HTML to React Migration Guide

This document maps the original HTML pages to their React equivalents.

## Page Mapping

| Original HTML File | React Component | Route | Status |
|-------------------|-----------------|-------|--------|
| `api/ui/index.html` | Redirect to `/detections` | `/` | ✅ Complete |
| `api/ui/detections.html` | `src/pages/Detections.jsx` | `/detections` | ✅ Complete |
| `api/ui/events.html` | `src/pages/Events.jsx` | `/events` | ✅ Complete |
| `api/ui/stats.html` | `src/pages/Stats.jsx` | `/stats` | ✅ Complete |
| `api/ui/about.html` | `src/pages/About.jsx` | `/about` | ✅ Complete |
| `api/ui/debug-stats.html` | `src/pages/DebugStats.jsx` | `/debug-stats` | ✅ Complete |
| `api/ui/config.html` | `src/pages/Config.jsx` | `/config` | ⚠️ Placeholder |
| `api/ui/backup-status.html` | `src/pages/BackupStatus.jsx` | `/backup-status` | ⚠️ Placeholder |

## Component Architecture

### Shared Components

- **Header** (`src/components/Header.jsx`)
  - Navigation menu
  - Active route highlighting
  - Reused across all pages

### Page Components

#### Detections (`src/pages/Detections.jsx`)
**Features:**
- Camera filter dropdown
- Detection type filter (people, cars, etc.)
- Grid size selector (1-8 columns)
- Toggle between detections and all events
- Pagination with "Load More"
- Fullscreen image viewing
- Detection badges showing object type and confidence
- Lazy image loading

**API Endpoints Used:**
- `GET /events/cameras` - Get camera list
- `GET /events/correlated` - Get events with detections
- `GET /thumb/:id` - Get thumbnails

#### Events (`src/pages/Events.jsx`)
**Features:**
- Camera filter dropdown
- Date range picker (from/to)
- Date shortcuts (Today, Yesterday, Week, Month, All)
- Event grid with thumbnails
- Pagination with "Load More"
- Event metadata display

**API Endpoints Used:**
- `GET /events/cameras` - Get camera list
- `GET /events` - Get all events
- `GET /thumb/:id` - Get thumbnails

#### Stats (`src/pages/Stats.jsx`)
**Features:**
- Time range selector (24h, 7d, 30d)
- Summary statistics cards
- Timeline chart (Line chart)
- Camera distribution chart (Bar chart)
- Top cameras table
- Chart.js integration

**API Endpoints Used:**
- `GET /stats/motion` - Get motion statistics

#### About (`src/pages/About.jsx`)
**Features:**
- System information
- Feature list
- Technology stack
- API endpoints documentation
- Links to other admin pages

**API Endpoints Used:**
- None (static content)

#### Debug Stats (`src/pages/DebugStats.jsx`)
**Features:**
- Database health display
- System statistics
- Performance metrics
- Scan status
- JSON response viewer
- Refresh functionality

**API Endpoints Used:**
- `GET /db-health` - Database health
- `GET /stats` - System stats
- `GET /perf` - Performance metrics
- `GET /scan-status` - Scanning status

## Styling Approach

The React version preserves the original dark theme styling:

### Color Palette
- Background: `#0b0c10`
- Card background: `#111318`, `#1a1d26`
- Borders: `#2a2f3a`
- Primary blue: `#375dfb`
- Success green: `#66ff66`
- Text: `#e8e8e8`

### Component Styling
- Each page has its own CSS file (e.g., `Detections.css`)
- Global styles in `index.css`
- Header styles in `components/Header.css`
- CSS is component-scoped to prevent conflicts

## State Management

All state is managed using React hooks:

- `useState` - Component state (filters, events, loading states)
- `useEffect` - Side effects (API calls, subscriptions)
- `useCallback` - Memoized callbacks
- `useRef` - DOM references

No external state management library is used (Redux, MobX, etc.) to keep it simple.

## API Integration

### Proxy Configuration
In development, Vite proxies all API requests to the Go backend:

```javascript
// vite.config.js
server: {
  port: 3000,
  proxy: {
    '/events': 'http://localhost:8080',
    '/stats': 'http://localhost:8080',
    // ... other endpoints
  }
}
```

### Fetch API
All API calls use the native `fetch` API:

```javascript
const response = await fetch('/events?limit=100');
const data = await response.json();
```

### Error Handling
Each component handles its own API errors:
- Loading states during requests
- Error messages displayed to user
- Console logging for debugging

## Routing

React Router v7 is used for client-side routing:

```javascript
// src/App.jsx
<Routes>
  <Route path="/" element={<Navigate to="/detections" />} />
  <Route path="/detections" element={<Detections />} />
  <Route path="/events" element={<Events />} />
  // ... other routes
</Routes>
```

### Navigation
- Uses `<Link>` components for navigation
- Active route highlighting in header
- Browser back/forward buttons work correctly
- No page reloads on navigation

## Key Differences from HTML

### Improved
1. **Component Reusability** - Header component shared across pages
2. **State Management** - Cleaner with React hooks
3. **Hot Module Replacement** - Instant updates during development
4. **Code Organization** - Separated by concern (components, pages, styles)
5. **Type Safety Ready** - Can easily add TypeScript

### Simplified
1. **Video Playback** - Basic implementation (can be enhanced)
2. **Advanced Animations** - Some CSS animations simplified
3. **Mobile Menu** - Simplified mobile navigation

### Not Yet Implemented
1. **Configuration Page** - Full CRUD interface for config
2. **Backup Management** - Full backup interface
3. **Real-time Updates** - WebSocket integration
4. **Video Player Controls** - Play/pause/seek controls

## Development Workflow

### Adding a New Page

1. Create component file:
```bash
touch src/pages/NewPage.jsx
touch src/pages/NewPage.css
```

2. Create component:
```javascript
import Header from '../components/Header';
import './NewPage.css';

function NewPage() {
  return (
    <div className="new-page">
      <Header title="New Page" />
      <main>
        {/* Page content */}
      </main>
    </div>
  );
}

export default NewPage;
```

3. Add route:
```javascript
// src/App.jsx
import NewPage from './pages/NewPage';

<Route path="/new-page" element={<NewPage />} />
```

4. Add navigation link in Header if needed

### Adding a New API Endpoint

1. Update proxy config:
```javascript
// vite.config.js
proxy: {
  '/new-endpoint': 'http://localhost:8080',
}
```

2. Use in component:
```javascript
const response = await fetch('/new-endpoint');
const data = await response.json();
```

## Testing

Currently no automated tests are included, but the app can be manually tested:

1. Start backend: `cd api && ./scui-api`
2. Start React: `cd react-ui && npm run dev`
3. Open `http://localhost:3000`
4. Navigate through all pages
5. Test filters and controls
6. Check browser console for errors

Future: Add Jest + React Testing Library for unit tests.

## Deployment

### Development
```bash
npm run dev
```

### Production Build
```bash
npm run build
# Output in dist/ folder
```

### Serve Production Build
```bash
npm run preview
# or
cd dist && python -m http.server 3000
```

## Browser Compatibility

- Chrome/Edge: ✅ Latest 2 versions
- Firefox: ✅ Latest 2 versions
- Safari: ✅ Latest 2 versions
- Mobile browsers: ✅ iOS Safari, Chrome Mobile

## Performance Considerations

1. **Lazy Loading** - Images load on-demand with `loading="lazy"`
2. **Pagination** - Load events in batches
3. **Memoization** - Use `useCallback` for expensive functions
4. **Code Splitting** - Vite automatically splits code by route
5. **Compression** - Production build is minified and compressed

## Troubleshooting

### Common Issues

**Port already in use:**
```bash
npm run dev -- --port 3001
```

**API connection refused:**
- Ensure Go backend is running on port 8080
- Check proxy config in `vite.config.js`

**Build fails:**
```bash
rm -rf node_modules package-lock.json
npm install
npm run build
```

**White screen:**
- Check browser console for errors
- Verify all imports are correct
- Check React DevTools

## Resources

- [React Documentation](https://react.dev/)
- [Vite Documentation](https://vitejs.dev/)
- [React Router Documentation](https://reactrouter.com/)
- [Chart.js Documentation](https://www.chartjs.org/)
