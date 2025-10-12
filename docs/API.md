# Security Camera UI API Documentation

## Overview

The Security Camera UI API provides comprehensive endpoints for managing security camera footage, AI detection, and system monitoring.

## Base URL

```
http://localhost:8080
```

## Authentication

Currently, the API does not require authentication. For production deployments, consider implementing proper authentication mechanisms.

## Core Endpoints

### System Health

#### GET /health
Basic health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

#### GET /system-metrics
Comprehensive system metrics including CPU, RAM, disk usage, and Go runtime statistics.

**Response:**
```json
{
  "system_memory_total_mb": 16384.0,
  "system_memory_used_mb": 8192.0,
  "system_memory_percent": 50.0,
  "system_disk_total_mb": 1000000.0,
  "system_disk_used_mb": 500000.0,
  "system_disk_percent": 50.0,
  "system_cpu_percent": 25.5,
  "nas_disk_total_mb": 2000000.0,
  "nas_disk_used_mb": 1000000.0,
  "nas_disk_percent": 50.0,
  "go_memory_mb": 45.2,
  "go_goroutines": 12,
  "go_cpus": 8
}
```

### Database Operations

#### GET /stats
Basic system statistics including event counts and database information.

**Response:**
```json
{
  "total_events": 178091,
  "video_events": 62235,
  "thumbnails_generated": 178091,
  "thumbnail_coverage": 286.2,
  "database_size_mb": 125.5,
  "program_disk_usage_mb": 45.2
}
```

#### GET /db-health
Database health status and performance metrics.

**Response:**
```json
{
  "status": "Healthy",
  "connection_time_ms": 2.5,
  "query_performance": "Good",
  "database_size_mb": 125.5,
  "last_backup": "2024-01-01T12:00:00Z"
}
```

### File Processing

#### GET /processing-status
Current file processing queue status and performance metrics.

**Response:**
```json
{
  "queue_healthy": true,
  "pending_files": 0,
  "processing_files": 0,
  "completed_files": 144608,
  "recent_processed": 52,
  "processing_rate_per_hour": 45.2
}
```

#### GET /watcher-health
File watcher status and health metrics.

**Response:**
```json
{
  "status": "enabled",
  "watching_directories": 3,
  "events_processed": 144608,
  "errors_count": 0,
  "last_cleanup": "2024-01-01T12:00:00Z"
}
```

### Events and Detections

#### GET /events
Retrieve camera events with filtering and pagination.

**Query Parameters:**
- `camera`: Filter by camera name
- `from`: Start date (YYYY-MM-DD)
- `to`: End date (YYYY-MM-DD)
- `limit`: Number of results (default: 100)
- `offset`: Pagination offset (default: 0)

**Response:**
```json
{
  "events": [
    {
      "id": 123,
      "camera": "Camera_01",
      "path": "/path/to/file.jpg",
      "start_ts": 1704110400,
      "end_ts": 1704110460,
      "has_detections": true,
      "detection_count": 3
    }
  ],
  "total": 178091,
  "limit": 100,
  "offset": 0
}
```

#### GET /detections
Retrieve AI detections with filtering and pagination.

**Query Parameters:**
- `camera`: Filter by camera name
- `type`: Filter by object type (person, car, etc.)
- `confidence_min`: Minimum confidence threshold
- `from`: Start date (YYYY-MM-DD)
- `to`: End date (YYYY-MM-DD)
- `limit`: Number of results (default: 100)
- `offset`: Pagination offset (default: 0)

**Response:**
```json
{
  "detections": [
    {
      "id": 456,
      "event_id": 123,
      "camera": "Camera_01",
      "object_type": "person",
      "confidence": 0.85,
      "timestamp": 1704110400,
      "bbox": {
        "x": 100,
        "y": 150,
        "width": 50,
        "height": 120
      }
    }
  ],
  "total": 45000,
  "limit": 100,
  "offset": 0
}
```

### GPU Detection Service

#### GET /gpu/status
Get GPU detection service status and performance metrics.

**Response:**
```json
{
  "service_healthy": true,
  "gpu_available": true,
  "model_loaded": true,
  "processing_queue": 0,
  "total_processed": 15000,
  "average_processing_time_ms": 150.5
}
```

#### POST /gpu/detect/image
Detect objects in an image file.

**Request Body:**
```json
{
  "file_path": "/path/to/image.jpg",
  "file_type": "image",
  "options": {
    "confidence_threshold": 0.45,
    "classes": ["person", "car", "truck"]
  },
  "metadata": {
    "camera": "front_door",
    "timestamp": "2024-01-01T12:00:00Z"
  }
}
```

**Response:**
```json
{
  "request_id": "img_1704110400000",
  "file_path": "/path/to/image.jpg",
  "file_type": "image",
  "detections": [
    {
      "type": "person",
      "confidence": 0.85,
      "bbox": {
        "x": 100,
        "y": 150,
        "width": 50,
        "height": 120
      }
    }
  ],
  "processing_time_ms": 150.5,
  "success": true,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### POST /events/:id/detect
Detect objects in a specific event by ID.

**Response:**
```json
{
  "event_id": 123,
  "detections_added": 3,
  "processing_time_ms": 250.0,
  "success": true
}
```

### Performance Monitoring

#### GET /perf
Detailed performance metrics and system statistics.

**Response:**
```json
{
  "system_uptime_seconds": 86400,
  "requests_total": 15000,
  "requests_per_second": 0.17,
  "average_response_time_ms": 45.2,
  "database_queries_total": 50000,
  "database_queries_per_second": 0.58,
  "file_operations_total": 25000,
  "ai_detections_total": 15000,
  "ai_detections_per_hour": 625
}
```

#### GET /scan-status
Live scanning status and file processing information.

**Response:**
```json
{
  "totalEvents": 179152,
  "videoEvents": 62609,
  "thumbnailsGenerated": 179152,
  "detectionEvents": 53934,
  "eventsWithDetectionData": 62138,
  "recentEvents": [
    {
      "id": "180025",
      "camera": "DETECTION",
      "path": "/mnt/nas/pool/Cameras/GPU_Processing/completed/180023__House PTZ_01_20251012034343_det.jpg",
      "startTs": 1760258623
    }
  ],
  "scanning": false
}
```

#### GET /debug-detections
Comprehensive detection processing debugging information.

**Response:**
```json
{
  "totalEvents": 179152,
  "detectionEvents": 53934,
  "eventsWithDetectionData": 62138,
  "detectionsTable": 103975,
  "recentDetectionUpdates": 5,
  "detectionSamples": [
    {
      "id": "180025",
      "camera": "DETECTION",
      "path": "/mnt/nas/pool/Cameras/GPU_Processing/completed/180023__House PTZ_01_20251012034343_det.jpg",
      "startTs": 1760258623,
      "hasDetectionData": false
    }
  ],
  "eventsWithDetectionSamples": [
    {
      "id": "180017",
      "camera": "PTZ_01",
      "path": "/mnt/nas/pool/Cameras/House/2025/10/12/House PTZ_01_20251012033421.jpg",
      "startTs": 1760258061
    }
  ],
  "pendingDetectionSamples": [
    {
      "id": "180023",
      "camera": "PTZ_01",
      "path": "/mnt/nas/pool/Cameras/House/2025/10/12/House PTZ_01_20251012034343.jpg",
      "startTs": 1760258623,
      "createdAt": 1760258701,
      "hasDetectionEvent": true
    }
  ]
}
```

#### POST /test/process-detections
Manually trigger detection processing for testing purposes.

**Response:**
```json
{
  "message": "Detection processing completed",
  "eventsWithDetectionData": 62138
}
```

## Error Responses

All endpoints may return the following error responses:

### 400 Bad Request
```json
{
  "error": "Invalid request parameters",
  "details": "Missing required parameter: camera"
}
```

### 404 Not Found
```json
{
  "error": "Resource not found",
  "details": "Event with ID 123 not found"
}
```

### 500 Internal Server Error
```json
{
  "error": "Internal server error",
  "details": "Database connection failed"
}
```

### 503 Service Unavailable
```json
{
  "error": "Service unavailable",
  "details": "GPU detection service is not responding"
}
```

## Rate Limiting

Currently, no rate limiting is implemented. For production deployments, consider implementing appropriate rate limiting mechanisms.

## CORS

The API supports Cross-Origin Resource Sharing (CORS) for web interface access. All origins are currently allowed.

## WebSocket Support

Real-time updates are provided through the web interface using standard HTTP polling. WebSocket support may be added in future versions.

## Examples

### Get Recent Detections
```bash
curl "http://localhost:8080/detections?limit=10&type=person&confidence_min=0.8"
```

### Get Events for Specific Camera
```bash
curl "http://localhost:8080/events?camera=Camera_01&from=2024-01-01&to=2024-01-31"
```

### Check System Health
```bash
curl "http://localhost:8080/health"
```

### Get System Metrics
```bash
curl "http://localhost:8080/system-metrics"
```

## SDK Examples

### JavaScript/TypeScript
```javascript
// Get recent detections
const response = await fetch('/api/detections?limit=50&type=person');
const data = await response.json();
console.log(data.detections);

// Get system health
const health = await fetch('/api/health');
const status = await health.json();
console.log(status);
```

### Python
```python
import requests

# Get system metrics
response = requests.get('http://localhost:8080/system-metrics')
metrics = response.json()
print(f"CPU Usage: {metrics['system_cpu_percent']}%")

# Get detections
detections = requests.get('http://localhost:8080/detections?type=car').json()
print(f"Found {len(detections['detections'])} car detections")
```

### Go
```go
package main

import (
    "encoding/json"
    "fmt"
    "net/http"
)

type SystemMetrics struct {
    CPUPercent float64 `json:"system_cpu_percent"`
    MemoryUsed float64 `json:"system_memory_used_mb"`
}

func main() {
    resp, err := http.Get("http://localhost:8080/system-metrics")
    if err != nil {
        panic(err)
    }
    defer resp.Body.Close()
    
    var metrics SystemMetrics
    json.NewDecoder(resp.Body).Decode(&metrics)
    fmt.Printf("CPU: %.1f%%, Memory: %.1f MB\n", 
        metrics.CPUPercent, metrics.MemoryUsed)
}
```

## Changelog

### Version 1.0.0
- Initial API release
- Basic CRUD operations for events and detections
- System monitoring endpoints
- GPU detection service integration
- Comprehensive error handling
- Performance metrics and health checks
