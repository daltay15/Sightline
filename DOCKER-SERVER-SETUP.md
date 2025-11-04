# Docker Setup for Server

## Server Directory Structure

Your server at `/media/nas/cache1/scui` should have:
```
/media/nas/cache1/scui/
├── api/
│   ├── Dockerfile
│   ├── scui-api          (35MB binary)
│   ├── config.json
│   └── config.json.template
├── sightline_object_detection/
│   ├── Dockerfile
│   ├── config.json
│   ├── yolo11x.pt
│   └── ...
├── config.json           (optional, root level)
├── docker-compose.yml    (or docker-compose.server.yml)
└── data/
    ├── gpu_processing/
    └── cameras/
```

## Quick Setup

1. **Copy docker-compose.yml to your server:**
   ```bash
   # On your server, in /media/nas/cache1/scui
   # Copy the docker-compose.yml file here
   ```

2. **Or use the server-specific version:**
   ```bash
   # Rename docker-compose.server.yml to docker-compose.yml
   cp docker-compose.server.yml docker-compose.yml
   ```

3. **Ensure directories exist:**
   ```bash
   mkdir -p data/gpu_processing data/cameras data/thumbs data/backups
   ```

4. **Build and run:**
   ```bash
   docker-compose build
   docker-compose up -d
   ```

5. **Check logs:**
   ```bash
   docker-compose logs -f
   ```

## Troubleshooting

### "no configuration file provided: not found"
- Make sure `docker-compose.yml` is in `/media/nas/cache1/scui`
- Or specify the file: `docker-compose -f docker-compose.yml up -d`

### Config file location
- The docker-compose.yml expects `./api/config.json`
- If your config.json is at root level, update the volume mount in docker-compose.yml:
  ```yaml
  volumes:
    - ./config.json:/app/config.json:ro  # Instead of ./api/config.json
  ```

