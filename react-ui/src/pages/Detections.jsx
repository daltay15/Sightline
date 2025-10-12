import { useState, useEffect, useRef, useCallback } from 'react';
import Header from '../components/Header';
import './Detections.css';

function Detections() {
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [cameras, setCameras] = useState([]);
  const [filters, setFilters] = useState({
    camera: '',
    filter: '',
    gridSize: 3
  });
  const [showDetectionsOnly, setShowDetectionsOnly] = useState(true);
  const [paging, setPaging] = useState({
    limit: 100,
    offset: 0,
    total: 0
  });
  
  const gridRef = useRef(null);
  const [fullscreenImage, setFullscreenImage] = useState(null);

  // Format timestamp
  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp * 1000);
    return date.toLocaleString('en-US', {
      timeZone: 'America/Chicago',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: true
    }) + ' CT';
  };

  // Format file size
  const formatSize = (bytes) => {
    return (bytes / 1e6).toFixed(1) + ' MB';
  };

  // Load cameras
  useEffect(() => {
    const loadCameras = async () => {
      try {
        const res = await fetch('/events/cameras');
        if (res.ok) {
          const data = await res.json();
          setCameras(data.cameras || []);
        }
      } catch (e) {
        console.error('Failed to load cameras:', e);
      }
    };
    loadCameras();
  }, []);

  // Load events
  const loadEvents = useCallback(async (reset = false) => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      params.append('limit', paging.limit);
      params.append('offset', reset ? 0 : paging.offset);
      
      if (filters.camera) params.append('camera', filters.camera);
      if (filters.filter === 'person') params.append('detection_type', 'person');
      if (filters.filter === 'car') params.append('detection_type', 'car');
      if (filters.filter === 'truck') params.append('detection_type', 'truck');
      if (filters.filter === 'dog') params.append('detection_type', 'dog');
      if (filters.filter === 'high-confidence') params.append('min_confidence', '0.8');

      // Use correlated endpoint if showing detections only
      const endpoint = showDetectionsOnly ? '/events/correlated' : '/events';
      const res = await fetch(`${endpoint}?${params}`);
      
      if (!res.ok) throw new Error('HTTP ' + res.status);
      
      const data = await res.json();
      
      if (reset) {
        setEvents(data.events || []);
        setPaging(prev => ({ ...prev, offset: 0, total: data.total || 0 }));
      } else {
        setEvents(prev => [...prev, ...(data.events || [])]);
        setPaging(prev => ({ ...prev, offset: prev.offset + paging.limit, total: data.total || 0 }));
      }
    } catch (e) {
      console.error('Failed to load events:', e);
    } finally {
      setLoading(false);
    }
  }, [filters, paging.limit, paging.offset, showDetectionsOnly]);

  // Initial load
  useEffect(() => {
    loadEvents(true);
  }, [filters, showDetectionsOnly]); // eslint-disable-line react-hooks/exhaustive-deps

  // Handle filter changes
  const handleFilterChange = (key, value) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  // Handle apply button
  const handleApply = () => {
    loadEvents(true);
  };

  // Handle reset button
  const handleReset = () => {
    setFilters({ camera: '', filter: '', gridSize: 3 });
    setShowDetectionsOnly(true);
  };

  // Handle grid size change
  const handleGridSizeChange = (size) => {
    setFilters(prev => ({ ...prev, gridSize: parseInt(size) }));
  };

  // Toggle view between detections and all events
  const toggleView = () => {
    setShowDetectionsOnly(!showDetectionsOnly);
  };

  // Handle thumbnail click for fullscreen
  const handleThumbnailClick = (event) => {
    if (event.detection_path) {
      setFullscreenImage(`/thumb/${event.id}`);
    }
  };

  // Close fullscreen
  const closeFullscreen = () => {
    setFullscreenImage(null);
  };

  // Get detection badges
  const getDetectionBadges = (event) => {
    if (!event.detections || event.detections.length === 0) return null;
    
    const detectionTypes = {};
    event.detections.forEach(d => {
      if (!detectionTypes[d.detection_type]) {
        detectionTypes[d.detection_type] = {
          count: 0,
          maxConfidence: 0
        };
      }
      detectionTypes[d.detection_type].count++;
      detectionTypes[d.detection_type].maxConfidence = Math.max(
        detectionTypes[d.detection_type].maxConfidence,
        d.confidence
      );
    });

    return Object.entries(detectionTypes).map(([type, info]) => (
      <span key={type} className="detection-badge">
        {getEmoji(type)} {type} ({(info.maxConfidence * 100).toFixed(0)}%)
      </span>
    ));
  };

  // Get emoji for detection type
  const getEmoji = (type) => {
    const emojiMap = {
      person: '👤',
      car: '🚗',
      truck: '🚛',
      bicycle: '🚲',
      motorcycle: '🏍️',
      bird: '🐦',
      dog: '🐕',
      cat: '🐈'
    };
    return emojiMap[type] || '📷';
  };

  return (
    <div className="detections-page">
      <Header title="AI Detections" />
      
      <main>
        <div className="controls">
          <select 
            value={filters.camera} 
            onChange={(e) => handleFilterChange('camera', e.target.value)}
          >
            <option value="">All Cameras</option>
            {cameras.map(cam => (
              <option key={cam} value={cam}>{cam}</option>
            ))}
          </select>
          
          <select 
            value={filters.filter} 
            onChange={(e) => handleFilterChange('filter', e.target.value)}
          >
            <option value="">All Detections</option>
            <option value="person">👤 People Only</option>
            <option value="car">🚗 Cars Only</option>
            <option value="truck">🚛 Trucks Only</option>
            <option value="bicycle">🚲 Bicycles Only</option>
            <option value="motorcycle">🏍️ Motorcycles Only</option>
            <option value="bird">🐦 Birds Only</option>
            <option value="dog">🐕 Dogs Only</option>
            <option value="high-confidence">High Confidence</option>
          </select>
          
          <select 
            value={filters.gridSize} 
            onChange={(e) => handleGridSizeChange(e.target.value)}
          >
            <option value="1">1 per row</option>
            <option value="2">2 per row</option>
            <option value="3">3 per row</option>
            <option value="4">4 per row</option>
            <option value="5">5 per row</option>
            <option value="6">6 per row</option>
            <option value="7">7 per row</option>
            <option value="8">8 per row</option>
          </select>
          
          <button 
            className="primary" 
            onClick={toggleView}
            style={{ background: showDetectionsOnly ? '#44ff44' : '#375dfb' }}
          >
            {showDetectionsOnly ? '🖼️ Show All Events' : '🎯 Show Detections Only'}
          </button>
          
          <button className="primary" onClick={handleApply}>Apply</button>
          <button onClick={handleReset}>Reset</button>
          <button onClick={() => loadEvents(true)}>Refresh</button>
        </div>
        
        <div className="count">
          {loading ? 'Loading...' : `Showing ${events.length} of ${paging.total} events`}
        </div>
        
        <div 
          ref={gridRef}
          className={`grid grid-${filters.gridSize}`}
        >
          {events.map(event => (
            <div key={event.id} className="card">
              <div className="detection-badges">
                {getDetectionBadges(event)}
              </div>
              <img
                src={`/thumb/${event.id}`}
                alt={event.camera}
                className="thumb"
                onClick={() => handleThumbnailClick(event)}
                loading="lazy"
              />
              <div className="meta">
                <div>
                  <span className="camera">{event.camera}</span>
                  <span className="ts">{formatTimestamp(event.start_ts)}</span>
                </div>
                <div>
                  <span className="size">{formatSize(event.size_bytes)}</span>
                  {event.duration_ms > 0 && (
                    <span className="duration">{(event.duration_ms / 1000).toFixed(1)}s</span>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
        
        {events.length === 0 && !loading && (
          <div className="empty">No events found. Try adjusting your filters.</div>
        )}
        
        {events.length > 0 && events.length < paging.total && (
          <div className="load-more">
            <button className="primary" onClick={() => loadEvents(false)}>
              Load More
            </button>
          </div>
        )}
      </main>
      
      {fullscreenImage && (
        <div className="fullscreen" onClick={closeFullscreen}>
          <button className="fullscreen-close" onClick={closeFullscreen}>✕</button>
          <img src={fullscreenImage} alt="Fullscreen" />
        </div>
      )}
    </div>
  );
}

export default Detections;
