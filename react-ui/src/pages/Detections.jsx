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
    gridSize: 3,
    from: '',
    to: ''
  });
  const [showDetectionsOnly, setShowDetectionsOnly] = useState(true);
  const [paging, setPaging] = useState({
    limit: 100,
    offset: 0,
    total: 0
  });
  
  const gridRef = useRef(null);
  const [fullscreenImage, setFullscreenImage] = useState(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panX, setPanX] = useState(0);
  const [panY, setPanY] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const fullscreenImageRef = useRef(null);
  const lastUpdateTime = useRef(0);
  const videoRefs = useRef({});
  const [loadingMore, setLoadingMore] = useState(false);
  const [mediaTypes, setMediaTypes] = useState({}); // Track which media type is shown for each event
  const [lastEventCount, setLastEventCount] = useState(0);
  const [lastDetectionCount, setLastDetectionCount] = useState(0);
  const [autoRefreshInterval, setAutoRefreshInterval] = useState(null);
  const [currentlyPlayingVideo, setCurrentlyPlayingVideo] = useState(null); // Track currently playing video
  const [status, setStatus] = useState({ message: '', type: '', visible: false });

  // Format timestamp
  const formatTimestamp = (timestamp) => {
    if (!timestamp || isNaN(timestamp)) return 'Invalid Date CT';
    const date = new Date(timestamp * 1000);
    if (isNaN(date.getTime())) return 'Invalid Date CT';
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
    if (!bytes || isNaN(bytes)) return '0 MB';
    return (bytes / 1e6).toFixed(1) + ' MB';
  };

  // Load cameras
  useEffect(() => {
    const loadCameras = async () => {
      try {
        const res = await fetch('/cameras');
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
    if (reset) {
      setLoading(true);
    } else {
      setLoadingMore(true);
    }
    
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
      if (filters.from) {
        const fromDate = new Date(filters.from);
        params.append('from', Math.floor(fromDate.getTime() / 1000));
      }
      if (filters.to) {
        const toDate = new Date(filters.to);
        toDate.setHours(23, 59, 59);
        params.append('to', Math.floor(toDate.getTime() / 1000));
      }

      // Use correlated endpoint if showing detections only
      const endpoint = showDetectionsOnly ? '/events/correlated' : '/events';
      const res = await fetch(`${endpoint}?${params}`);
      
      if (!res.ok) throw new Error('HTTP ' + res.status);
      
      const data = await res.json();
      
      if (reset) {
        setEvents(data.items || data.events || []);
        setPaging(prev => ({ ...prev, offset: 0, total: data.total || 0 }));
      } else {
        setEvents(prev => [...prev, ...(data.items || data.events || [])]);
        setPaging(prev => ({ ...prev, offset: prev.offset + paging.limit, total: data.total || 0 }));
      }
    } catch (e) {
      console.error('Failed to load events:', e);
    } finally {
      setLoading(false);
      setLoadingMore(false);
    }
  }, [filters, paging.limit, paging.offset, showDetectionsOnly]);

  // Initial load
  useEffect(() => {
    loadEvents(true);
  }, [filters, showDetectionsOnly]); // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-refresh functionality (like HTML version)
  const fetchStats = async () => {
    try {
      const res = await fetch('/scan-status');
      if (res.ok) {
        return await res.json();
      }
    } catch (e) {
      console.error('Failed to fetch stats:', e);
    }
    return null;
  };

  const startAutoRefresh = useCallback(() => {
    if (autoRefreshInterval) return;
    
    const interval = setInterval(async () => {
      try {
        const stats = await fetchStats();
        if (stats) {
          const currentCount = stats.totalEvents || 0;
          const detectionCount = stats.detectionEvents || 0;
          const videoCount = stats.videoEvents || 0;
          const thumbnailsCount = stats.thumbnailsGenerated || 0;
          const eventsWithDetectionData = stats.eventsWithDetectionData || 0;
          
          if (currentCount > lastEventCount || detectionCount > lastDetectionCount) {
            console.log(`New events detected! ${currentCount} total, ${detectionCount} detections`);
            // Only refresh if we're on the first page and no filters applied
            if (paging.offset === 0 && !filters.camera && !filters.filter && !filters.from && !filters.to) {
              loadEvents(true);
            }
          }
          
          // Always show status notification
          if (currentCount === 0) {
            showStatus('Scanning for events...', 'loading');
          } else {
            const statusMessage = `${currentCount} events indexed (${videoCount} videos, ${thumbnailsCount} thumbnails, ${detectionCount} detections, ${eventsWithDetectionData} with detection data)`;
            console.log('Showing status:', statusMessage);
            showStatus(statusMessage, 'success');
          }
          
          setLastEventCount(currentCount);
          setLastDetectionCount(detectionCount);
        }
      } catch (e) {
        console.error('Auto-refresh error:', e);
      }
    }, 10000); // Check every 10 seconds like HTML version
    
    setAutoRefreshInterval(interval);
  }, [autoRefreshInterval, lastEventCount, lastDetectionCount, paging.offset, filters.camera, filters.filter, loadEvents]);

  const stopAutoRefresh = useCallback(() => {
    if (autoRefreshInterval) {
      clearInterval(autoRefreshInterval);
      setAutoRefreshInterval(null);
    }
  }, [autoRefreshInterval]);

  // Start auto-refresh when component mounts
  useEffect(() => {
    startAutoRefresh();
    return () => stopAutoRefresh();
  }, [startAutoRefresh, stopAutoRefresh]);


  // Stop all currently playing videos (like HTML version)
  const stopAllVideos = useCallback(() => {
    if (currentlyPlayingVideo) {
      currentlyPlayingVideo.pause();
      currentlyPlayingVideo.currentTime = 0;
      setCurrentlyPlayingVideo(null);
    }
    
    // Also stop any videos that might be playing in the DOM
    const allVideos = document.querySelectorAll('video');
    allVideos.forEach(video => {
      if (!video.paused) {
        video.pause();
        video.currentTime = 0;
      }
    });
  }, [currentlyPlayingVideo]);

  // Check video processing status (like HTML version)
  const updateVideoProcessingStatus = useCallback(async () => {
    const processingButtons = document.querySelectorAll('.media-btn[data-media-type="video"]');
    if (processingButtons.length === 0) return;

    const now = Math.floor(Date.now() / 1000);
    const checkedPaths = new Set();

    for (const btn of processingButtons) {
      const eventId = btn.dataset.eventId;
      const expectedVideoPath = btn.dataset.videoPath;
      
      if (eventId && expectedVideoPath && (btn.textContent === 'Processing...' || btn.textContent === 'Video Processing...')) {
        const checkKey = `${eventId}-${expectedVideoPath}`;
        
        if (!checkedPaths.has(checkKey)) {
          checkedPaths.add(checkKey);
          
          // Skip checking if the event is very recent (less than 1 minute old)
          const eventTime = parseInt(btn.dataset.eventTime || '0');
          if (eventTime > 0 && (now - eventTime) < 60) {
            console.log(`Skipping video check for very recent event ${eventId}`);
            continue;
          }
          
          console.log(`Checking video availability for event ${eventId}: ${expectedVideoPath}`);
          
          // Check if video is now available by testing the stream endpoint
          try {
            const response = await fetch(`/stream/${eventId}?path=${encodeURIComponent(expectedVideoPath)}`, { method: 'HEAD' });
            console.log(`Video check response for ${eventId}: ${response.status}`);
            
            if (response.ok) {
              btn.textContent = 'Video';
              btn.disabled = false;
              btn.style.opacity = '1';
              console.log(`Video now available for event ${eventId}`);
            } else if (response.status === 206) {
              // Partial content - video might still be processing
              console.log(`Video partially available for event ${eventId} (still processing)`);
            } else {
              console.log(`Video still processing for event ${eventId} (${response.status})`);
            }
          } catch (err) {
            if (err.message.includes('CONTENT_LENGTH_MISMATCH')) {
              console.log(`Video file incomplete for event ${eventId} (still processing)`);
            } else {
              console.log('Error checking video status:', err);
            }
          }
        }
      }
    }
  }, []);

  // Update video processing status every 15 seconds
  useEffect(() => {
    const interval = setInterval(updateVideoProcessingStatus, 15000);
    return () => clearInterval(interval);
  }, [updateVideoProcessingStatus]);

  // Update video metadata when new events are detected (like HTML version)
  const updateVideoMetadata = useCallback(async () => {
    try {
      const response = await fetch('/update-video-metadata', { method: 'POST' });
      if (response.ok) {
        const data = await response.json();
        if (data.updated > 0) {
          console.log(`Updated ${data.updated} video metadata entries`);
          // Refresh the page to get updated data
          loadEvents(true);
        }
      }
    } catch (err) {
      console.error('Error updating video metadata:', err);
    }
  }, [loadEvents]);

  // Trigger video metadata update when new events are detected
  useEffect(() => {
    if (lastEventCount > 0) {
      updateVideoMetadata();
    }
  }, [lastEventCount, updateVideoMetadata]);


  // Infinite scroll
  useEffect(() => {
    const handleScroll = () => {
      if (loadingMore || events.length >= paging.total) return;
      
      const { scrollTop, scrollHeight, clientHeight } = document.documentElement;
      if (scrollTop + clientHeight >= scrollHeight - 1000) {
        loadEvents(false);
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, [loadingMore, events.length, paging.total, loadEvents]);

  // Stop videos when scrolling (like HTML version)
  useEffect(() => {
    let scrollTimeout;
    const handleScrollStop = () => {
      // Stop videos when user starts scrolling
      if (currentlyPlayingVideo) {
        currentlyPlayingVideo.pause();
        currentlyPlayingVideo.currentTime = 0;
        setCurrentlyPlayingVideo(null);
      }
      
      // Clear any existing timeout
      clearTimeout(scrollTimeout);
      
      // Set a timeout to allow scrolling to settle
      scrollTimeout = setTimeout(() => {
        // Videos can resume after scrolling stops
      }, 100);
    };

    window.addEventListener('scroll', handleScrollStop, { passive: true });
    return () => {
      window.removeEventListener('scroll', handleScrollStop);
      clearTimeout(scrollTimeout);
    };
  }, [currentlyPlayingVideo]);

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
    setFilters({ camera: '', filter: '', gridSize: 3, from: '', to: '' });
    setShowDetectionsOnly(true);
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

  // Handle grid size change
  const handleGridSizeChange = (size) => {
    setFilters(prev => ({ ...prev, gridSize: parseInt(size) }));
  };

  // Toggle view between detections and all events
  const toggleView = () => {
    setShowDetectionsOnly(!showDetectionsOnly);
  };

  // Status notification functions
  const showStatus = (message, type = 'loading') => {
    setStatus({ message, type, visible: true });
  };

  const hideStatus = () => {
    setStatus({ message: '', type: '', visible: false });
  };


  // Close fullscreen
  const closeFullscreen = () => {
    setFullscreenImage(null);
    setZoomLevel(1);
    setPanX(0);
    setPanY(0);
  };

  // Get detection badges
  const getDetectionBadges = (event) => {
    // Check for detection data in various possible locations
    let detections = event.detections || [];
    
    // If no detections array, try to parse from detection_data JSON
    if (detections.length === 0 && event.detectionData) {
      try {
        const detectionData = JSON.parse(event.detectionData);
        if (detectionData.detections && Array.isArray(detectionData.detections)) {
          detections = detectionData.detections;
        } else if (Array.isArray(detectionData)) {
          detections = detectionData;
        }
      } catch (e) {
        console.log(`Failed to parse detection data for event ${event.id}:`, e);
      }
    }
    
    if (!detections || detections.length === 0) return null;
    
    const detectionTypes = {};
    detections.forEach(d => {
      const type = d.detection_type || d.type || 'unknown';
      const confidence = d.confidence || d.conf || 0;
      
      if (!detectionTypes[type]) {
        detectionTypes[type] = {
          count: 0,
          maxConfidence: 0
        };
      }
      detectionTypes[type].count++;
      detectionTypes[type].maxConfidence = Math.max(
        detectionTypes[type].maxConfidence,
        confidence
      );
    });

    return Object.entries(detectionTypes).map(([type, info]) => (
      <span key={type} className={`detection-badge ${type}`}>
        {getEmoji(type)}
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
    return emojiMap[type] || '🔍';
  };



  // Get detection count for badges
  const getDetectionCount = (event) => {
    let detections = event.detections || [];
    
    // If no detections array, try to parse from detection_data JSON
    if (detections.length === 0 && event.detectionData) {
      try {
        const detectionData = JSON.parse(event.detectionData);
        if (detectionData.detections && Array.isArray(detectionData.detections)) {
          detections = detectionData.detections;
        } else if (Array.isArray(detectionData)) {
          detections = detectionData;
        }
      } catch (e) {
        console.log(`Failed to parse detection data for event ${event.id}:`, e);
      }
    }
    
    if (!detections || detections.length === 0) return 0;
    return detections.length;
  };


  // Check if media type is available for an event (matching HTML version logic)
  const isMediaTypeAvailable = (event, mediaType) => {
    switch (mediaType) {
      case 'original':
        return true; // Always available
      case 'detection':
        // Only show detection button if there are detections (like HTML version)
        const detectionCount = getDetectionCount(event);
        return detectionCount > 0;
      case 'video':
        // Show video button for all files (like HTML version)
        return true;
      default:
        return false;
    }
  };

  const controls = (
    <>
      <select 
        value={filters.camera} 
        onChange={(e) => handleFilterChange('camera', e.target.value)}
      >
        <option value="">All Cameras</option>
        {cameras.map(cam => (
          <option key={cam.camera} value={cam.camera}>{cam.camera} ({cam.count})</option>
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
      
      <div className="date-shortcuts">
        <button onClick={() => setDateRange('today')}>Today</button>
        <button onClick={() => setDateRange('yesterday')}>Yesterday</button>
        <button onClick={() => setDateRange('week')}>This Week</button>
        <button onClick={() => setDateRange('month')}>This Month</button>
        <button onClick={() => setDateRange('all')}>All Time</button>
      </div>
      
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
    </>
  );

  return (
    <div className="detections-page">
      <Header 
        title="AI Detections" 
        controls={controls}
        count={loading ? 'Loading...' : `${events.length} / ${paging.total}`}
        status={status.visible ? { message: status.message, type: status.type } : null}
      />
      
      <main>
        
        <div 
          ref={gridRef}
          className={`grid grid-${filters.gridSize}`}
        >
          {events.map(event => {
            // Set default media type like HTML version: detection if available, otherwise original
            let hasDetections = (event.detections && event.detections.length > 0);
            if (!hasDetections && event.detectionData) {
              try {
                const detectionData = JSON.parse(event.detectionData);
                hasDetections = detectionData.detections && detectionData.detections.length > 0;
              } catch (e) {
                // Ignore parse errors
              }
            }
            const defaultMediaType = hasDetections ? 'detection' : 'original';
            const currentMediaType = mediaTypes[event.id] || defaultMediaType;
            
            
            // Exact same logic as HTML version
            let mediaUrl;
            if (currentMediaType === 'detection') {
              // Check if detection image is available using metadata (exact HTML logic)
              if (event.detectionPath) {
                mediaUrl = `/detection/${event.id}`;
              } else if (event.detectionData) {
                try {
                  const detectionData = JSON.parse(event.detectionData);
                  if (detectionData.annotated_image) {
                    mediaUrl = `/detection/${event.id}`;
                  } else {
                    mediaUrl = `/thumb/${event.id}`;
                  }
                } catch (err) {
                  mediaUrl = `/thumb/${event.id}`;
                }
              } else {
                mediaUrl = `/thumb/${event.id}`;
              }
            } else if (currentMediaType === 'video') {
              let videoPath = event.videoPath;
              if (!videoPath && event.path && event.path.toLowerCase().endsWith('.jpg')) {
                videoPath = event.path.replace(/\.jpg$/i, '.mp4');
              }
              mediaUrl = videoPath ? `/stream/${event.id}?path=${encodeURIComponent(videoPath)}` : `/thumb/${event.id}`;
            } else {
              mediaUrl = `/thumb/${event.id}`;
            }
            
            const detectionCount = getDetectionCount(event);
            
            
            return (
              <div key={event.id} className={`card ${detectionCount > 0 ? 'has-detections' : ''}`}>
                <div className="detection-badges">
                  {getDetectionBadges(event)}
                </div>
                <div className="media-container">
                  {currentMediaType === 'video' ? (
                      <video
                        key={`video-${event.id}`}
                        ref={(el) => { if (el) videoRefs.current[event.id] = el; }}
                        className="thumb"
                        poster={`/thumb/${event.id}`}
                        controls
                        preload="metadata"
                        src={mediaUrl}
                      onPlay={(e) => {
                        // Stop other videos when this one starts playing
                        const allVideos = document.querySelectorAll('video');
                        allVideos.forEach(video => {
                          if (video !== e.target && !video.paused) {
                            video.pause();
                          }
                        });
                        setCurrentlyPlayingVideo(e.target);
                      }}
                      onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        
                        console.log('Video clicked - currentlyPlayingVideo:', currentlyPlayingVideo === e.target, 'video.paused:', e.target.paused);
                        
                        // Check if this video is the currently tracked video
                        const isCurrentlyTracked = currentlyPlayingVideo === e.target;
                        
                        if (isCurrentlyTracked) {
                          // This is the currently tracked video - toggle play/pause
                          if (e.target.paused) {
                            console.log('Video is paused, resuming...');
                            e.target.play().then(() => {
                              console.log('Video resumed successfully');
                            }).catch((err) => {
                              console.error('Video resume failed:', err);
                            });
                          } else {
                            console.log('Video is playing, pausing...');
                            e.target.pause();
                            console.log('Video paused, state after pause:', e.target.paused);
                          }
                        } else {
                          console.log('Starting new video');
                          // Stop any other currently playing video first
                          const allVideos = document.querySelectorAll('video');
                          allVideos.forEach(v => {
                            if (v !== e.target && !v.paused) {
                              v.pause();
                            }
                          });
                          
                          // Set as currently playing video
                          setCurrentlyPlayingVideo(e.target);
                          
                          // Play this video
                          console.log('Attempting to play video:', e.target.src);
                          e.target.play().catch(err => {
                            console.log('Video autoplay failed:', err);
                            // If autoplay fails, try with muted
                            e.target.muted = true;
                            e.target.play().catch(err2 => {
                              console.log('Video autoplay failed even with muted:', err2);
                            });
                          });
                        }
                      }}
                      onDoubleClick={(e) => {
                        // Only handle double-click on desktop
                        if (window.innerWidth > 768) {
                          e.preventDefault();
                          e.stopPropagation();
                          
                          // Check if we're already in fullscreen
                          if (document.fullscreenElement || document.webkitFullscreenElement || 
                              document.mozFullScreenElement || document.msFullscreenElement) {
                            console.log('Video double-clicked, exiting fullscreen');
                            
                            // Exit fullscreen
                            if (document.exitFullscreen) {
                              document.exitFullscreen();
                            } else if (document.webkitExitFullscreen) {
                              document.webkitExitFullscreen();
                            } else if (document.mozCancelFullScreen) {
                              document.mozCancelFullScreen();
                            } else if (document.msExitFullscreen) {
                              document.msExitFullscreen();
                            }
                          } else {
                            console.log('Video double-clicked, entering fullscreen:', e.target.src);
                            
                            // Enter fullscreen
                            if (e.target.requestFullscreen) {
                              e.target.requestFullscreen();
                            } else if (e.target.webkitRequestFullscreen) {
                              e.target.webkitRequestFullscreen();
                            } else if (e.target.mozRequestFullScreen) {
                              e.target.mozRequestFullScreen();
                            } else if (e.target.msRequestFullscreen) {
                              e.target.msRequestFullscreen();
                            }
                          }
                        }
                      }}
                      onPause={(e) => {
                        // Clear when video pauses
                        if (currentlyPlayingVideo === e.target) {
                          setCurrentlyPlayingVideo(null);
                        }
                      }}
                      onEnded={() => {
                        // Clear currently playing video when it ends
                        setCurrentlyPlayingVideo(null);
                      }}
                        onError={(e) => {
                          console.log(`Failed to load video for event ${event.id}: ${mediaUrl}`);
                          // Show error placeholder for failed videos
                          e.target.style.display = 'none';
                          const errorDiv = document.createElement('div');
                          errorDiv.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="169" viewBox="0 0 300 169"><rect width="300" height="169" fill="#0f1116"/><text x="50%" y="50%" text-anchor="middle" dy=".3em" fill="#ff6666">Video Not Found</text></svg>';
                          errorDiv.style.width = '100%';
                          errorDiv.style.height = '169px';
                          errorDiv.style.display = 'flex';
                          errorDiv.style.alignItems = 'center';
                          errorDiv.style.justifyContent = 'center';
                          e.target.parentNode.appendChild(errorDiv);
                        }}
                      >
                        <source src={mediaUrl} type="video/mp4" />
                      </video>
                  ) : (
                    <img
                      src={mediaUrl}
                      alt={event.camera}
                      className="thumb"
                      onClick={() => setFullscreenImage(mediaUrl)}
                      loading="lazy"
                      onError={(e) => {
                        // Show error placeholder for failed images
                        e.target.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="300" height="169" viewBox="0 0 300 169"><rect width="300" height="169" fill="%230f1116"/><text x="50%" y="50%" text-anchor="middle" dy=".3em" fill="%23ff6666">File Not Found</text></svg>';
                      }}
                    />
                  )}
                    </div>
                <div className="meta">
                  <div>
                    <span className="camera">{event.camera || 'Unknown'}</span>
                    <span className="ts">{formatTimestamp(event.start_ts)}</span>
                    </div>
                  <div>
                    <span className="size">{formatSize(event.size_bytes)}</span>
                    {event.duration_ms > 0 && (
                      <span className="duration">{(event.duration_ms / 1000).toFixed(1)}s</span>
                  )}
                  </div>
                </div>
                  <div className="media-controls">
                    <button 
                      className={`media-btn ${currentMediaType === 'original' ? 'active' : ''}`}
                      onClick={() => {
                        stopAllVideos();
                        setMediaTypes(prev => ({ ...prev, [event.id]: 'original' }));
                      }}
                      disabled={!isMediaTypeAvailable(event, 'original')}
                    >
                      Original
                    </button>
                  {detectionCount > 0 && (
                    <button 
                      className={`media-btn ${currentMediaType === 'detection' ? 'active' : ''}`}
                      onClick={() => {
                        stopAllVideos();
                        setMediaTypes(prev => ({ ...prev, [event.id]: 'detection' }));
                      }}
                      disabled={!isMediaTypeAvailable(event, 'detection')}
                    >
                      Detection
                    </button>
                  )}
                    <button 
                      className={`media-btn ${currentMediaType === 'video' ? 'active' : ''}`}
                      onClick={() => {
                        stopAllVideos();
                        setMediaTypes(prev => ({ ...prev, [event.id]: 'video' }));
                        
                        // Auto-play the video when it becomes available (like HTML version)
                        setTimeout(() => {
                          // Use the ref to get the video element directly
                          const video = videoRefs.current[event.id];
                          if (video) {
                            console.log('Video button clicked, setting up autoplay for:', mediaUrl);
                            console.log('Video readyState:', video.readyState);
                            console.log('Video src:', video.src);
                            
                            // Try immediate play first
                            console.log('Attempting immediate play...');
                            video.play().then(() => {
                              console.log('Video started playing immediately');
                              setCurrentlyPlayingVideo(video);
                            }).catch((err) => {
                              console.log('Immediate play failed, setting up canplay listener:', err);
                              
                              // Set up autoplay when video is ready
                              const handleCanPlay = () => {
                                console.log('Video ready, auto-playing...');
                                video.play().then(() => {
                                  console.log('Video started playing successfully');
                                  setCurrentlyPlayingVideo(video);
                                }).catch((err) => {
                                  console.error('Video autoplay error:', err);
                                });
                                video.removeEventListener('canplay', handleCanPlay);
                              };
                              
                              video.addEventListener('canplay', handleCanPlay);
                              
                              // Also try on loadeddata
                              const handleLoadedData = () => {
                                console.log('Video loadeddata, trying to play...');
                                video.play().then(() => {
                                  console.log('Video started playing on loadeddata');
                                  setCurrentlyPlayingVideo(video);
                                }).catch((err) => {
                                  console.error('Video autoplay error on loadeddata:', err);
                                });
                                video.removeEventListener('loadeddata', handleLoadedData);
                              };
                              
                              video.addEventListener('loadeddata', handleLoadedData);
                            });
                          } else {
                            console.log('Video element not found for:', mediaUrl);
                          }
                        }, 100); // Small delay to allow React to render the video
                        
                        // Fallback: try to play after a longer delay
                        setTimeout(() => {
                          const video = videoRefs.current[event.id];
                          if (video && video.paused) {
                            console.log('Fallback: trying to play video after delay');
                            video.play().then(() => {
                              console.log('Fallback play successful');
                              setCurrentlyPlayingVideo(video);
                            }).catch((err) => {
                              console.log('Fallback play failed:', err);
                            });
                          }
                        }, 500);
                      }}
                      disabled={!isMediaTypeAvailable(event, 'video')}
                      data-media-type="video"
                      data-event-id={event.id}
                      data-video-path={event.videoPath || (event.path && event.path.toLowerCase().endsWith('.jpg') ? event.path.replace(/\.jpg$/i, '.mp4') : '')}
                      data-event-time={event.start_ts}
                    >
                      Video
                    </button>
                </div>
              </div>
            );
          })}
        </div>
        
        {events.length === 0 && !loading && (
          <div className="empty">No events found. Try adjusting your filters.</div>
        )}
        
        {loadingMore && (
          <div className="loading-more">
            <div className="spinner"></div>
            Loading more events...
          </div>
        )}
      </main>
      
      {fullscreenImage && (
        <div className="fullscreen" onClick={closeFullscreen}>
          <button className="fullscreen-close" onClick={closeFullscreen}>✕</button>
          {fullscreenImage.includes('.mp4') || fullscreenImage.includes('/stream/') ? (
            <video 
              ref={fullscreenImageRef}
              src={fullscreenImage} 
              className="fullscreen-image"
              controls
              autoPlay
              style={{ 
                transform: window.innerWidth <= 768 ? `scale(${zoomLevel}) translate(${panX}px, ${panY}px)` : 'none',
                transformOrigin: 'center center'
              }}
            onClick={(e) => {
              e.stopPropagation();
              // On desktop, single click exits fullscreen
              if (window.innerWidth > 768) {
                closeFullscreen();
              }
            }}
            onTouchStart={(e) => {
              // Only handle touch events on mobile devices
              if (window.innerWidth <= 768) {
                // Only prevent default for pinch zoom
                if (e.touches.length === 2) {
                  e.preventDefault();
                }
                
                if (e.touches.length === 1) {
                  // Single touch - prepare for potential tap to exit or pan
                  e.target.dataset.touchStartTime = Date.now();
                  e.target.dataset.touchStartX = e.touches[0].clientX;
                  e.target.dataset.touchStartY = e.touches[0].clientY;
                  e.target.dataset.panStartX = panX;
                  e.target.dataset.panStartY = panY;
                  e.target.dataset.isPanning = 'false';
                  setIsDragging(false);
                } else if (e.touches.length === 2) {
                  const touch1 = e.touches[0];
                  const touch2 = e.touches[1];
                  const distance = Math.sqrt(
                    Math.pow(touch2.clientX - touch1.clientX, 2) + 
                    Math.pow(touch2.clientY - touch1.clientY, 2)
                  );
                  e.target.dataset.initialDistance = distance;
                  e.target.dataset.initialScale = zoomLevel;
                }
              }
            }}
            onTouchMove={(e) => {
              // Only handle touch events on mobile devices
              if (window.innerWidth <= 768) {
                // Only prevent default if we're actually panning or zooming
                if (e.touches.length === 1 && zoomLevel > 1) {
                  const touchStartX = parseFloat(e.target.dataset.touchStartX || 0);
                  const touchStartY = parseFloat(e.target.dataset.touchStartY || 0);
                  const deltaX = e.touches[0].clientX - touchStartX;
                  const deltaY = e.touches[0].clientY - touchStartY;
                  
                  // Only prevent default if we're actually moving (panning)
                  if (Math.abs(deltaX) > 5 || Math.abs(deltaY) > 5) {
                    e.preventDefault();
                  }
                } else if (e.touches.length === 2) {
                  // Always prevent default for pinch zoom
                  e.preventDefault();
                }
                
                // Throttle updates to 60fps
                const now = Date.now();
                if (now - lastUpdateTime.current < 16) return;
                lastUpdateTime.current = now;
                
                if (e.touches.length === 1 && zoomLevel > 1) {
                  // Single touch pan when zoomed in
                  const touchStartX = parseFloat(e.target.dataset.touchStartX || 0);
                  const touchStartY = parseFloat(e.target.dataset.touchStartY || 0);
                  const panStartX = parseFloat(e.target.dataset.panStartX || 0);
                  const panStartY = parseFloat(e.target.dataset.panStartY || 0);
                  
                  const deltaX = e.touches[0].clientX - touchStartX;
                  const deltaY = e.touches[0].clientY - touchStartY;
                  
                  // Only start panning if moved more than 10px
                  if (Math.abs(deltaX) > 10 || Math.abs(deltaY) > 10) {
                    e.target.dataset.isPanning = 'true';
                    setIsDragging(true);
                  }
                  
                  if (e.target.dataset.isPanning === 'true') {
                    const newPanX = panStartX + deltaX;
                    const newPanY = panStartY + deltaY;
                    
                    // Limit pan to prevent image from going too far off screen
                    const maxPan = 200; // Adjust based on your needs
                    const clampedPanX = Math.max(-maxPan, Math.min(maxPan, newPanX));
                    const clampedPanY = Math.max(-maxPan, Math.min(maxPan, newPanY));
                    
                    // Direct DOM manipulation for smoother performance
                    if (fullscreenImageRef.current) {
                      fullscreenImageRef.current.style.transform = `scale(${zoomLevel}) translate(${clampedPanX}px, ${clampedPanY}px)`;
                    }
                    
                    // Update state for consistency
                    setPanX(clampedPanX);
                    setPanY(clampedPanY);
                  }
                } else if (e.touches.length === 2) {
                  const touch1 = e.touches[0];
                  const touch2 = e.touches[1];
                  const distance = Math.sqrt(
                    Math.pow(touch2.clientX - touch1.clientX, 2) + 
                    Math.pow(touch2.clientY - touch1.clientY, 2)
                  );
                  const initialDistance = parseFloat(e.target.dataset.initialDistance || 0);
                  const initialScale = parseFloat(e.target.dataset.initialScale || 1);
                  if (initialDistance > 0) {
                    const scale = Math.max(0.5, Math.min(3, (distance / initialDistance) * initialScale));
                    
                    // Direct DOM manipulation for smoother performance
                    if (fullscreenImageRef.current) {
                      fullscreenImageRef.current.style.transform = `scale(${scale}) translate(${panX}px, ${panY}px)`;
                    }
                    
                    // Update state for consistency
                    setZoomLevel(scale);
                  }
                }
              }
            }}
            onTouchEnd={(e) => {
              // Only handle touch events on mobile devices
              if (window.innerWidth <= 768) {
                if (e.touches.length === 0) {
                  if (zoomLevel < 0.8) {
                    setZoomLevel(1);
                    setPanX(0);
                    setPanY(0);
                  }
                  
                  // Check for single tap to exit fullscreen (only if not panning)
                  const touchStartTime = parseInt(e.target.dataset.touchStartTime || 0);
                  const touchStartX = parseFloat(e.target.dataset.touchStartX || 0);
                  const touchStartY = parseFloat(e.target.dataset.touchStartY || 0);
                  const touchEndTime = Date.now();
                  const touchDuration = touchEndTime - touchStartTime;
                  const wasPanning = e.target.dataset.isPanning === 'true';
                  
                  // If it was a quick tap (less than 300ms), didn't move much (less than 10px), and wasn't panning, exit fullscreen
                  if (touchDuration < 300 && e.changedTouches.length === 1 && !wasPanning) {
                    const touchEndX = e.changedTouches[0].clientX;
                    const touchEndY = e.changedTouches[0].clientY;
                    const distance = Math.sqrt(
                      Math.pow(touchEndX - touchStartX, 2) + 
                      Math.pow(touchEndY - touchStartY, 2)
                    );
                    
                    if (distance < 10) {
                      closeFullscreen();
                    }
                  }
                }
              }
            }}
          />
          ) : (
            <img 
              ref={fullscreenImageRef}
              src={fullscreenImage} 
              alt="Fullscreen" 
              className="fullscreen-image"
              style={{ 
                transform: window.innerWidth <= 768 ? `scale(${zoomLevel}) translate(${panX}px, ${panY}px)` : 'none',
                transformOrigin: 'center center'
              }}
              onClick={(e) => {
                e.stopPropagation();
                // On desktop, single click exits fullscreen
                if (window.innerWidth > 768) {
                  closeFullscreen();
                }
              }}
              onTouchStart={(e) => {
                // Only handle touch events on mobile devices
                if (window.innerWidth <= 768) {
                  // Only prevent default for pinch zoom
                  if (e.touches.length === 2) {
                    e.preventDefault();
                  }
                  
                  if (e.touches.length === 1) {
                    // Single touch - prepare for potential tap to exit or pan
                    e.target.dataset.touchStartTime = Date.now();
                    e.target.dataset.touchStartX = e.touches[0].clientX;
                    e.target.dataset.touchStartY = e.touches[0].clientY;
                    e.target.dataset.panStartX = panX;
                    e.target.dataset.panStartY = panY;
                    e.target.dataset.isPanning = 'false';
                    setIsDragging(false);
                  } else if (e.touches.length === 2) {
                    const touch1 = e.touches[0];
                    const touch2 = e.touches[1];
                    const distance = Math.sqrt(
                      Math.pow(touch2.clientX - touch1.clientX, 2) + 
                      Math.pow(touch2.clientY - touch1.clientY, 2)
                    );
                    e.target.dataset.initialDistance = distance;
                    e.target.dataset.initialScale = zoomLevel;
                  }
                }
              }}
              onTouchMove={(e) => {
                // Only handle touch events on mobile devices
                if (window.innerWidth <= 768) {
                  // Only prevent default if we're actually panning or zooming
                  if (e.touches.length === 1 && zoomLevel > 1) {
                    const touchStartX = parseFloat(e.target.dataset.touchStartX || 0);
                    const touchStartY = parseFloat(e.target.dataset.touchStartY || 0);
                    const deltaX = e.touches[0].clientX - touchStartX;
                    const deltaY = e.touches[0].clientY - touchStartY;
                    
                    // Only prevent default if we're actually moving (panning)
                    if (Math.abs(deltaX) > 5 || Math.abs(deltaY) > 5) {
                      e.preventDefault();
                    }
                  } else if (e.touches.length === 2) {
                    // Always prevent default for pinch zoom
                    e.preventDefault();
                  }
                  
                  // Throttle updates to 60fps
                  const now = Date.now();
                  if (now - lastUpdateTime.current < 16) return;
                  lastUpdateTime.current = now;
                  
                  if (e.touches.length === 1 && zoomLevel > 1) {
                    // Single touch pan when zoomed in
                    const touchStartX = parseFloat(e.target.dataset.touchStartX || 0);
                    const touchStartY = parseFloat(e.target.dataset.touchStartY || 0);
                    const panStartX = parseFloat(e.target.dataset.panStartX || 0);
                    const panStartY = parseFloat(e.target.dataset.panStartY || 0);
                    
                    const deltaX = e.touches[0].clientX - touchStartX;
                    const deltaY = e.touches[0].clientY - touchStartY;
                    
                    // Only start panning if moved more than 10px
                    if (Math.abs(deltaX) > 10 || Math.abs(deltaY) > 10) {
                      e.target.dataset.isPanning = 'true';
                      setIsDragging(true);
                    }
                    
                    if (e.target.dataset.isPanning === 'true') {
                      const newPanX = panStartX + deltaX;
                      const newPanY = panStartY + deltaY;
                      
                      // Limit pan to prevent image from going too far off screen
                      const maxPan = 200; // Adjust based on your needs
                      const clampedPanX = Math.max(-maxPan, Math.min(maxPan, newPanX));
                      const clampedPanY = Math.max(-maxPan, Math.min(maxPan, newPanY));
                      
                      // Direct DOM manipulation for smoother performance
                      if (fullscreenImageRef.current) {
                        fullscreenImageRef.current.style.transform = `scale(${zoomLevel}) translate(${clampedPanX}px, ${clampedPanY}px)`;
                      }
                      
                      // Update state for consistency
                      setPanX(clampedPanX);
                      setPanY(clampedPanY);
                    }
                  } else if (e.touches.length === 2) {
                    const touch1 = e.touches[0];
                    const touch2 = e.touches[1];
                    const distance = Math.sqrt(
                      Math.pow(touch2.clientX - touch1.clientX, 2) + 
                      Math.pow(touch2.clientY - touch1.clientY, 2)
                    );
                    const initialDistance = parseFloat(e.target.dataset.initialDistance || 0);
                    const initialScale = parseFloat(e.target.dataset.initialScale || 1);
                    if (initialDistance > 0) {
                      const scale = Math.max(0.5, Math.min(3, (distance / initialDistance) * initialScale));
                      
                      // Direct DOM manipulation for smoother performance
                      if (fullscreenImageRef.current) {
                        fullscreenImageRef.current.style.transform = `scale(${scale}) translate(${panX}px, ${panY}px)`;
                      }
                      
                      // Update state for consistency
                      setZoomLevel(scale);
                    }
                  }
                }
              }}
              onTouchEnd={(e) => {
                // Only handle touch events on mobile devices
                if (window.innerWidth <= 768) {
                  if (e.touches.length === 0) {
                    if (zoomLevel < 0.8) {
                      setZoomLevel(1);
                      setPanX(0);
                      setPanY(0);
                    }
                    
                    // Check for single tap to exit fullscreen (only if not panning)
                    const touchStartTime = parseInt(e.target.dataset.touchStartTime || 0);
                    const touchStartX = parseFloat(e.target.dataset.touchStartX || 0);
                    const touchStartY = parseFloat(e.target.dataset.touchStartY || 0);
                    const touchEndTime = Date.now();
                    const touchDuration = touchEndTime - touchStartTime;
                    const wasPanning = e.target.dataset.isPanning === 'true';
                    
                    // If it was a quick tap (less than 300ms), didn't move much (less than 10px), and wasn't panning, exit fullscreen
                    if (touchDuration < 300 && e.changedTouches.length === 1 && !wasPanning) {
                      const touchEndX = e.changedTouches[0].clientX;
                      const touchEndY = e.changedTouches[0].clientY;
                      const distance = Math.sqrt(
                        Math.pow(touchEndX - touchStartX, 2) + 
                        Math.pow(touchEndY - touchStartY, 2)
                      );
                      
                      if (distance < 10) {
                        closeFullscreen();
                      }
                    }
                  }
                }
              }}
            />
          )}
        </div>
      )}
    </div>
  );
}

export default Detections;
