# GPU Detection Service Integration

This document describes the integration between the Security Camera UI API and the GPU Detection Service.

## Overview

The Security Camera UI API now includes endpoints to interact with the GPU Detection Service, which provides YOLO-based object detection capabilities for images and videos.

## New Endpoints

### GPU Detection Service Status
- **GET** `/gpu/status` - Get GPU detection service status and performance metrics
- **GET** `/gpu/health` - Check if GPU detection service is healthy

### Object Detection
- **POST** `/gpu/detect/image` - Detect objects in an image
- **POST** `/gpu/detect/video` - Detect objects in a video
- **POST** `/events/:id/detect` - Detect objects in a specific event by ID

## Configuration

The GPU detection service URL can be configured using environment variables:

- `GPU_DETECTION_URL` - URL of the GPU detection service (default: `http://localhost:8000`)
- `GPU_DETECTION_TIMEOUT` - Request timeout in seconds (default: `30`)

## Request/Response Format

### Detection Request
```json
{
  "file_path": "/path/to/file.jpg",
  "file_type": "image",
  "options": {
    "confidence_threshold": 0.45,
    "classes": ["person", "car", "truck", "bus", "bicycle", "motorcycle"]
  },
  "metadata": {
    "camera": "front_door",
    "timestamp": "2024-01-01T12:00:00Z"
  }
}
```

### Detection Response
```json
{
  "request_id": "img_1704110400000",
  "file_path": "/path/to/file.jpg",
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
  "frame_count": null,
  "success": true,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Usage Examples

### Check GPU Service Status
```bash
curl http://localhost:8080/gpu/status
```

### Detect Objects in an Image
```bash
curl -X POST http://localhost:8080/gpu/detect/image \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/mnt/nas/pool/Cameras/House/front_door/2024/01/01/image.jpg",
    "file_type": "image",
    "options": {
      "confidence_threshold": 0.5,
      "classes": ["person", "car"]
    }
  }'
```

### Detect Objects in an Event
```bash
curl -X POST http://localhost:8080/events/123/detect
```

## Integration with Existing API

The GPU detection service integrates seamlessly with the existing Security Camera UI API:

1. **Event Detection**: Use `/events/:id/detect` to automatically detect objects in any recorded event
2. **File Type Detection**: The API automatically determines whether to use image or video detection based on file extension
3. **Default Classes**: Optimized for security camera scenarios (person, car, truck, bus, bicycle, motorcycle)
4. **Error Handling**: Graceful fallback when GPU service is unavailable

## Testing

Run the integration test to verify the GPU detection service is working:

```bash
cd api
go run test_gpu_integration.go
```

## Dependencies

- GPU Detection Service must be running on the configured URL
- The service should be accessible from the API server
- Sufficient GPU memory for YOLO model inference

## Error Handling

- If GPU service is unavailable, endpoints return HTTP 503 Service Unavailable
- File not found errors return HTTP 404 Not Found
- Invalid requests return HTTP 400 Bad Request
- Detection failures are logged and returned with error details
