# Docker Setup Guide

This project is now fully dockerized with both the Go API (scui) and Python detection service running in Docker containers.

## Quick Start

1. **Ensure you have the required files:**
   - `api/config.json` - Configuration for the Go API
   - `gpu-detection-python/test/config.json` - Configuration for the Python detection service
   - `gpu-detection-python/test/yolo11x.pt` - YOLO model file

2. **Create required directories:**
   ```bash
   mkdir -p data/gpu_processing data/cameras data/thumbs data/backups
   ```

3. **Update configuration files for Docker:**

   **`gpu-detection-python/test/config.json`** - Update the API endpoints:
   ```json
   {
     "gpu_config": {
       "api_endpoint": "http://scui-go:8080/ingest_detection",
       "telegram_endpoint": "http://scui-go:8080/telegram/send_detection",
       "base": "/mnt/nas/pool/Cameras/GPU_Processing"
     },
     "cpu_config": {
       "api_endpoint": "http://scui-go:8080/ingest_detection",
       "telegram_endpoint": "http://scui-go:8080/telegram/send_detection",
       "base": "/mnt/nas/pool/Cameras/GPU_Processing"
     }
   }
   ```

   **`api/config.json`** - Ensure paths point to mounted volumes:
   ```json
   {
     "camera_dir": "/mnt/nas/pool/Cameras/House",
     "db_path": "/app/data/events.db",
     "gpu_detection_url": "http://python-detection:8000"
   }
   ```

4. **Build and start the services:**
   ```bash
   docker-compose up -d
   ```

5. **View logs:**
   ```bash
   # All services
   docker-compose logs -f
   
   # Go API only
   docker-compose logs -f scui-go
   
   # Python detection only
   docker-compose logs -f python-detection
   ```

6. **Stop services:**
   ```bash
   docker-compose down
   ```

## Service Startup Order

The Docker Compose setup ensures:
1. **Python detection service** starts first and initializes
2. A health check confirms Python service is ready (30 second start period)
3. **Go API service** waits until Python is healthy, then waits an additional 5 seconds before starting
4. Total wait time: ~35 seconds after Python container starts

## Networking

- Both services are on the same Docker network (`scui-network`)
- Python service connects to Go API using service name: `http://scui-go:8080`
- Go API is exposed on host port `8080`

## Volumes

The following directories are mounted:

- `./data/gpu_processing` → `/mnt/nas/pool/Cameras/GPU_Processing` (shared between both services)
- `./data/cameras` → `/mnt/nas/pool/Cameras/House` (Go API camera directory)
- `./data` → `/app/data` (Go API database and thumbnails)
- `./api/config.json` → `/app/config.json` (Go API config)
- `./gpu-detection-python/test/config.json` → `/app/config.json` (Python config)
- `./gpu-detection-python/test/yolo11x.pt` → `/app/yolo11x.pt` (YOLO model)

## GPU Support

If you have a GPU available and want to use it for detection:

1. Install [nvidia-docker2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

2. Update `docker-compose.yml` to add GPU support:
   ```yaml
   python-detection:
     deploy:
       resources:
         reservations:
           devices:
             - driver: nvidia
               count: 1
               capabilities: [gpu]
   ```

## Troubleshooting

### Python service can't connect to Go API
- Check that both services are on the same network
- Verify the Go API is running: `docker-compose logs scui-go`
- Check the service name matches: `scui-go` (not `localhost`)

### Go API starts before Python is ready
- The `depends_on` with `condition: service_healthy` should prevent this
- Check Python health check: `docker-compose ps`
- Increase the `start_period` in the Python healthcheck if needed

### Missing directories
- Ensure all required directories exist before starting
- Check volume mount paths match your configuration

### Config not updating
- The Python config is mounted as read-write and will be updated by the entrypoint script
- Environment variables `API_ENDPOINT` and `TELEGRAM_ENDPOINT` can override config values

## Development

To rebuild after code changes:
```bash
docker-compose build
docker-compose up -d
```

To rebuild without cache:
```bash
docker-compose build --no-cache
docker-compose up -d
```

