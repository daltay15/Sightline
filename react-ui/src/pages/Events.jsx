import { useState, useEffect, useCallback } from 'react';
import Header from '../components/Header';
import './Events.css';

function Events() {
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [cameras, setCameras] = useState([]);
  const [filters, setFilters] = useState({
    camera: '',
    from: '',
    to: ''
  });
  const [paging, setPaging] = useState({
    limit: 100,
    offset: 0,
    total: 0
  });

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
      if (filters.from) {
        const fromDate = new Date(filters.from);
        params.append('from', Math.floor(fromDate.getTime() / 1000));
      }
      if (filters.to) {
        const toDate = new Date(filters.to);
        toDate.setHours(23, 59, 59);
        params.append('to', Math.floor(toDate.getTime() / 1000));
      }

      const res = await fetch(`/events?${params}`);
      
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
  }, [filters, paging.limit, paging.offset]);

  // Initial load
  useEffect(() => {
    loadEvents(true);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

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
    setFilters({ camera: '', from: '', to: '' });
    loadEvents(true);
  };

  // Set date range shortcuts
  const setDateRange = (range) => {
    const today = new Date();
    const todayStr = today.toISOString().split('T')[0];
    
    switch(range) {
      case 'today':
        setFilters(prev => ({ ...prev, from: todayStr, to: todayStr }));
        break;
      case 'yesterday': {
        const yesterday = new Date(today);
        yesterday.setDate(yesterday.getDate() - 1);
        const yesterdayStr = yesterday.toISOString().split('T')[0];
        setFilters(prev => ({ ...prev, from: yesterdayStr, to: yesterdayStr }));
        break;
      }
      case 'week': {
        const weekStart = new Date(today);
        weekStart.setDate(today.getDate() - today.getDay());
        setFilters(prev => ({ ...prev, from: weekStart.toISOString().split('T')[0], to: todayStr }));
        break;
      }
      case 'month': {
        const monthStart = new Date(today.getFullYear(), today.getMonth(), 1);
        setFilters(prev => ({ ...prev, from: monthStart.toISOString().split('T')[0], to: todayStr }));
        break;
      }
      case 'all':
        setFilters(prev => ({ ...prev, from: '', to: '' }));
        break;
      default:
        break;
    }
  };

  return (
    <div className="events-page">
      <Header title="Camera Events" />
      
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
          
          <input
            type="date"
            value={filters.from}
            onChange={(e) => handleFilterChange('from', e.target.value)}
            placeholder="From"
          />
          
          <input
            type="date"
            value={filters.to}
            onChange={(e) => handleFilterChange('to', e.target.value)}
            placeholder="To"
          />
          
          <button className="primary" onClick={handleApply}>Apply</button>
          <button onClick={handleReset}>Reset</button>
          <button onClick={() => loadEvents(true)}>Refresh</button>
        </div>
        
        <div className="date-shortcuts">
          <button onClick={() => setDateRange('today')}>Today</button>
          <button onClick={() => setDateRange('yesterday')}>Yesterday</button>
          <button onClick={() => setDateRange('week')}>This Week</button>
          <button onClick={() => setDateRange('month')}>This Month</button>
          <button onClick={() => setDateRange('all')}>All Time</button>
        </div>
        
        <div className="count">
          {loading ? 'Loading...' : `Showing ${events.length} of ${paging.total} events`}
        </div>
        
        <div className="grid">
          {events.map(event => (
            <div key={event.id} className="card">
              <img
                src={`/thumb/${event.id}`}
                alt={event.camera}
                className="thumb"
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
    </div>
  );
}

export default Events;
