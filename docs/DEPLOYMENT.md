# Deployment Guide

## Overview

This guide covers deploying the Security Camera UI system in various environments, from single-machine setups to distributed deployments.

## Quick Deployment

### Single Machine Setup

**Requirements:**
- Go 1.19+
- 4GB+ RAM
- 10GB+ free disk space
- Network access

**Steps:**
```bash
# 1. Clone and build
git clone <repository-url>
cd Security-Camera-UI/api
go build -o scui-api

# 2. Configure environment
export CAMERA_DIR="/path/to/camera/footage"
export PORT=8080

# 3. Run the service
./scui-api
```

**Access:**
- Web Interface: `http://localhost:8080`
- API: `http://localhost:8080/api/`

## Production Deployment

### System Requirements

**Minimum:**
- CPU: 4 cores, 2.0GHz
- RAM: 8GB
- Storage: 100GB SSD
- Network: 1Gbps

**Recommended:**
- CPU: 8+ cores, 3.0GHz+
- RAM: 16GB+
- Storage: 500GB+ NVMe SSD
- Network: 10Gbps
- GPU: CUDA-capable for AI processing

### Environment Configuration

**Production Environment Variables:**
```bash
# Core Configuration
CAMERA_DIR="/mnt/cameras"
PORT=8080
HOST="0.0.0.0"

# Database Configuration
DB_PATH="/var/lib/scui/events.db"
DB_BACKUP_ENABLED=true
DB_BACKUP_INTERVAL="24h"

# Performance Tuning
MAX_WORKERS=8
THUMBNAIL_QUALITY=85
THUMBNAIL_SIZE=320

# GPU Detection (Optional)
GPU_DETECTION_URL="http://gpu-service:8000"
GPU_DETECTION_TIMEOUT=30

# Security
CORS_ORIGINS="https://yourdomain.com"
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW="1m"
```

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY . .
RUN go build -o scui-api

FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/
COPY --from=builder /app/scui-api .
COPY --from=builder /app/ui ./ui
EXPOSE 8080
CMD ["./scui-api"]
```

**Docker Compose:**
```yaml
version: '3.8'
services:
  scui-api:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - /mnt/cameras:/mnt/cameras:ro
      - scui-data:/var/lib/scui
    environment:
      - CAMERA_DIR=/mnt/cameras
      - DB_PATH=/var/lib/scui/events.db
    restart: unless-stopped

  gpu-service:
    image: scui-gpu:latest
    volumes:
      - /mnt/cameras:/mnt/cameras:ro
    environment:
      - MODEL_PATH=/models
      - WORKERS=4
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  scui-data:
```

### Systemd Service

**Service File: `/etc/systemd/system/scui.service`**
```ini
[Unit]
Description=Security Camera UI
After=network.target

[Service]
Type=simple
User=scui
Group=scui
WorkingDirectory=/opt/scui
ExecStart=/opt/scui/scui-api
Environment=CAMERA_DIR=/mnt/cameras
Environment=PORT=8080
Environment=HOST=0.0.0.0
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

**Enable and Start:**
```bash
sudo systemctl enable scui
sudo systemctl start scui
sudo systemctl status scui
```

### Nginx Reverse Proxy

**Nginx Configuration:**
```nginx
server {
    listen 80;
    server_name yourdomain.com;
    
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support (if needed)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    # Static files caching
    location /ui/ {
        proxy_pass http://localhost:8080;
        expires 1h;
        add_header Cache-Control "public, immutable";
    }
}
```

### SSL/TLS Configuration

**Let's Encrypt with Certbot:**
```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d yourdomain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## Distributed Deployment

### Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Server    │    │  GPU Service    │    │  File Storage   │
│   (API + UI)    │◄──►│  (AI Processing)│◄──►│   (NAS/SAN)     │
│   Port: 8080    │    │   Port: 8000    │    │   /mnt/cameras  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   GPU Cluster   │    │  Backup System  │
│   (Nginx/HAProxy)│    │  (Multiple GPUs)│    │  (Automated)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Multi-Node Setup

**Node 1: Web Server**
```bash
# Install and configure API server
export CAMERA_DIR="/mnt/cameras"
export GPU_DETECTION_URL="http://gpu-node:8000"
./scui-api
```

**Node 2: GPU Service**
```bash
# Install GPU detection service
cd gpu-detection-python
pip install -r requirements.txt
python app.py --base /mnt/cameras --workers 4
```

**Node 3: File Storage**
```bash
# Configure NFS/SMB share
# Mount point: /mnt/cameras
# Ensure all nodes can access
```

### Load Balancing

**HAProxy Configuration:**
```
global
    daemon
    maxconn 4096

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms

frontend scui_frontend
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/scui.pem
    redirect scheme https if !{ ssl_fc }
    default_backend scui_backend

backend scui_backend
    balance roundrobin
    server web1 10.0.1.10:8080 check
    server web2 10.0.1.11:8080 check
    server web3 10.0.1.12:8080 check
```

## Monitoring and Logging

### Log Configuration

**Structured Logging:**
```bash
export LOG_LEVEL=info
export LOG_FORMAT=json
export LOG_FILE=/var/log/scui/scui.log
```

**Log Rotation:**
```bash
# /etc/logrotate.d/scui
/var/log/scui/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 scui scui
}
```

### Monitoring Setup

**Prometheus Metrics:**
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'scui'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

**Grafana Dashboard:**
- Import dashboard configuration
- Monitor system metrics
- Set up alerts for critical issues

### Health Checks

**Application Health:**
```bash
# Basic health check
curl http://localhost:8080/health

# Detailed system health
curl http://localhost:8080/system-metrics

# Database health
curl http://localhost:8080/db-health
```

**Automated Monitoring:**
```bash
#!/bin/bash
# health-check.sh
if ! curl -f http://localhost:8080/health > /dev/null 2>&1; then
    echo "Service is down, restarting..."
    systemctl restart scui
    # Send alert
    curl -X POST https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK \
         -H 'Content-type: application/json' \
         --data '{"text":"SCUI service restarted"}'
fi
```

## Backup and Recovery

### Database Backup

**Automated Backup:**
```bash
#!/bin/bash
# backup-db.sh
BACKUP_DIR="/var/backups/scui"
DB_PATH="/var/lib/scui/events.db"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
sqlite3 $DB_PATH ".backup $BACKUP_DIR/events_$DATE.db"

# Compress
gzip $BACKUP_DIR/events_$DATE.db

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "events_*.db.gz" -mtime +30 -delete
```

**Cron Job:**
```bash
# Add to crontab
0 2 * * * /opt/scui/scripts/backup-db.sh
```

### File System Backup

**Thumbnail Backup:**
```bash
#!/bin/bash
# backup-thumbs.sh
rsync -av --delete /var/lib/scui/thumbs/ /mnt/backup/scui/thumbs/
```

**Configuration Backup:**
```bash
#!/bin/bash
# backup-config.sh
tar -czf /mnt/backup/scui/config_$(date +%Y%m%d).tar.gz \
    /etc/systemd/system/scui.service \
    /etc/nginx/sites-available/scui \
    /opt/scui/
```

### Disaster Recovery

**Recovery Procedure:**
```bash
# 1. Restore database
cp /mnt/backup/scui/events_20240101_020000.db.gz /var/lib/scui/
gunzip /var/lib/scui/events_20240101_020000.db.gz
mv /var/lib/scui/events_20240101_020000.db /var/lib/scui/events.db

# 2. Restore thumbnails
rsync -av /mnt/backup/scui/thumbs/ /var/lib/scui/thumbs/

# 3. Restore configuration
tar -xzf /mnt/backup/scui/config_20240101.tar.gz -C /

# 4. Restart services
systemctl restart scui
systemctl restart nginx
```

## Security Considerations

### Network Security

**Firewall Rules:**
```bash
# UFW configuration
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw allow 8080/tcp  # API (internal only)
ufw deny 8000/tcp   # GPU service (internal only)
ufw enable
```

**VPN Access:**
- Use VPN for remote access
- Restrict API access to internal networks
- Implement proper authentication

### Application Security

**Environment Hardening:**
```bash
# Run as non-root user
useradd -r -s /bin/false scui
chown -R scui:scui /opt/scui
chown -R scui:scui /var/lib/scui
```

**File Permissions:**
```bash
# Secure file permissions
chmod 755 /opt/scui/scui-api
chmod 644 /opt/scui/ui/*
chmod 600 /var/lib/scui/events.db
```

### Data Protection

**Encryption at Rest:**
```bash
# Encrypt database
sqlite3 events.db "PRAGMA key='your-encryption-key';"
```

**Secure Communication:**
- Use HTTPS for all web traffic
- Implement proper CORS policies
- Use secure headers

## Performance Tuning

### Database Optimization

**SQLite Tuning:**
```sql
-- Optimize database
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=10000;
PRAGMA temp_store=MEMORY;
```

### Memory Optimization

**Go Runtime Tuning:**
```bash
export GOGC=100
export GOMEMLIMIT=8GiB
export GOMAXPROCS=8
```

### I/O Optimization

**Storage Tuning:**
```bash
# Use SSD for database and thumbnails
# Mount with noatime option
mount -o noatime /dev/sdb1 /var/lib/scui
```

## Troubleshooting Deployment

### Common Issues

**Port Conflicts:**
```bash
# Check port usage
netstat -tulpn | grep :8080
lsof -i :8080
```

**Permission Issues:**
```bash
# Check file permissions
ls -la /opt/scui/
ls -la /var/lib/scui/
```

**Service Won't Start:**
```bash
# Check logs
journalctl -u scui -f
tail -f /var/log/scui/scui.log
```

### Performance Issues

**High CPU Usage:**
- Check debug page for bottlenecks
- Reduce concurrent processing
- Optimize database queries

**Memory Issues:**
- Monitor memory usage
- Adjust Go runtime settings
- Consider increasing RAM

**Disk I/O Issues:**
- Use SSD storage
- Optimize file system settings
- Monitor disk usage

---

**For production deployments, consider professional support and comprehensive monitoring solutions.**
