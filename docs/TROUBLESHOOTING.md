# Troubleshooting Guide

## Common Issues and Solutions

### System Won't Start

#### Issue: "Port already in use"
**Solution:**
```bash
# Check what's using port 8080
lsof -i :8080
# or
netstat -tulpn | grep :8080

# Kill the process or use a different port
export PORT=8081
./scui-api
```

#### Issue: "Database locked"
**Solution:**
```bash
# Check for other instances
ps aux | grep scui-api
# Kill any running instances
pkill scui-api
# Restart the service
./scui-api
```

#### Issue: "Permission denied"
**Solution:**
```bash
# Make sure the binary is executable
chmod +x scui-api

# Check directory permissions
ls -la /path/to/camera/footage
# Fix permissions if needed
chmod -R 755 /path/to/camera/footage
```

### Database Issues

#### Issue: Database corruption
**Symptoms:**
- "Database is locked" errors
- Inconsistent data
- Slow queries

**Solution:**
```bash
# Backup current database
cp data/events.db data/events.db.backup

# Check database integrity
sqlite3 data/events.db "PRAGMA integrity_check;"

# If corrupted, restore from backup or rebuild
rm data/events.db
./scui-api  # Will create new database
```

#### Issue: Database growing too large
**Solution:**
```bash
# Check database size
du -h data/events.db

# Clean up old data (use the web interface)
# Go to About page → Clean Up Deleted Files
# Or manually:
sqlite3 data/events.db "DELETE FROM events WHERE path NOT LIKE '%' AND NOT EXISTS (SELECT 1 FROM files WHERE files.path = events.path);"
```

### File Processing Issues

#### Issue: Files not being detected
**Symptoms:**
- New camera files not appearing in UI
- File watcher not working

**Solution:**
1. **Check file watcher status:**
   - Go to Debug page
   - Check "File Watcher Status" section
   - Should show "Enabled" with green indicator

2. **Verify directory structure:**
   ```bash
   # Check if camera directory exists
   ls -la $CAMERA_DIR
   
   # Check file permissions
   ls -la $CAMERA_DIR/Camera_01/
   ```

3. **Restart file watcher:**
   - Restart the API service
   - Check debug page for watcher health

#### Issue: Thumbnails not generating
**Symptoms:**
- Images show but no thumbnails
- Thumbnail generation errors

**Solution:**
1. **Check thumbnail directory:**
   ```bash
   ls -la api/data/thumbs/
   ```

2. **Check file permissions:**
   ```bash
   chmod -R 755 api/data/thumbs/
   ```

3. **Clear thumbnail cache:**
   ```bash
   rm -rf api/data/thumbs/*
   # Restart service to regenerate
   ```

### AI Detection Issues

#### Issue: GPU detection not working
**Symptoms:**
- No AI detections appearing
- GPU service errors

**Solution:**
1. **Check GPU service status:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Verify GPU detection URL:**
   ```bash
   export GPU_DETECTION_URL="http://localhost:8000"
   ./scui-api
   ```

3. **Check GPU service logs:**
   ```bash
   # If running GPU service separately
   tail -f gpu-service.log
   ```

4. **Use new debug endpoints:**
   ```bash
   # Check detection processing status
   curl http://localhost:8080/debug-detections
   
   # Manually trigger detection processing
   curl -X POST http://localhost:8080/test/process-detections
   ```

#### Issue: New detections not appearing in UI
**Symptoms:**
- Detection processing working but UI not updating
- New detections not showing in real-time

**Solution:**
1. **Check detection processing status:**
   ```bash
   curl http://localhost:8080/debug-detections
   # Look for "recentDetectionUpdates" and "pendingDetectionSamples"
   ```

2. **Verify UI auto-refresh:**
   - Check browser console for JavaScript errors
   - Ensure `/scan-status` endpoint is responding
   - Look for "eventsWithDetectionData" count increasing

3. **Manual detection processing:**
   ```bash
   # Trigger detection processing manually
   curl -X POST http://localhost:8080/test/process-detections
   ```

4. **Check detection pipeline:**
   - Verify JSON files are being processed
   - Check that `detection_data` is being populated
   - Monitor DetectionProcessor logs

#### Issue: Low detection accuracy
**Symptoms:**
- Many false positives/negatives
- Poor object recognition

**Solution:**
1. **Adjust confidence threshold:**
   - Use debug page to monitor detection rates
   - Adjust GPU service confidence settings

2. **Check image quality:**
   - Ensure good lighting in camera footage
   - Check camera resolution and focus

3. **Update AI model:**
   - Use newer YOLO model versions
   - Retrain with your specific camera setup

### Performance Issues

#### Issue: Slow system performance
**Symptoms:**
- Slow page loading
- High CPU usage
- Memory issues

**Solution:**
1. **Check system resources:**
   - Go to Debug page
   - Monitor CPU, RAM, and disk usage
   - Look for any red indicators

2. **Optimize database:**
   ```bash
   # Vacuum database
   sqlite3 data/events.db "VACUUM;"
   
   # Analyze for optimization
   sqlite3 data/events.db "ANALYZE;"
   ```

3. **Reduce processing load:**
   - Limit concurrent file processing
   - Adjust thumbnail generation settings
   - Use SSD storage for better I/O

#### Issue: High memory usage
**Symptoms:**
- System running out of memory
- Slow performance

**Solution:**
1. **Check memory usage:**
   - Debug page → System Resources
   - Look for memory leaks

2. **Restart service:**
   ```bash
   # Restart to clear memory
   pkill scui-api
   ./scui-api
   ```

3. **Optimize settings:**
   - Reduce thumbnail cache size
   - Limit concurrent processing
   - Use smaller image sizes

### Web Interface Issues

#### Issue: Pages not loading
**Symptoms:**
- Blank pages
- JavaScript errors
- CSS not loading

**Solution:**
1. **Check browser console:**
   - Open Developer Tools (F12)
   - Look for JavaScript errors
   - Check Network tab for failed requests

2. **Clear browser cache:**
   - Hard refresh (Ctrl+F5)
   - Clear browser cache
   - Try different browser

3. **Check server logs:**
   ```bash
   # Look for error messages
   tail -f scui-api.log
   ```

#### Issue: Mobile interface issues
**Symptoms:**
- Poor mobile layout
- Touch issues
- Responsive problems

**Solution:**
1. **Check mobile viewport:**
   - Ensure proper meta viewport tag
   - Test on different devices

2. **Clear mobile cache:**
   - Clear browser data
   - Try different mobile browser

### Network Issues

#### Issue: Can't access from other devices
**Symptoms:**
- Only works on localhost
- Network access denied

**Solution:**
1. **Check firewall settings:**
   ```bash
   # Linux
   sudo ufw allow 8080
   
   # Windows
   # Add firewall rule for port 8080
   ```

2. **Bind to all interfaces:**
   ```bash
   export HOST="0.0.0.0"
   ./scui-api
   ```

3. **Check network configuration:**
   - Verify IP address
   - Test connectivity: `telnet <server-ip> 8080`

### Log Analysis

#### Enable Debug Logging
```bash
# Set debug level
export LOG_LEVEL=debug
./scui-api
```

#### Common Log Messages

**File Processing:**
```
INFO: Processing file: /path/to/file.jpg
ERROR: Failed to process file: permission denied
```

**Database Operations:**
```
INFO: Database query executed in 2.5ms
ERROR: Database connection failed
```

**AI Detection:**
```
INFO: AI detection completed: 3 objects found
ERROR: GPU service unavailable
```

### Recovery Procedures

#### Complete System Reset
```bash
# Stop all services
pkill scui-api
pkill python  # If GPU service running

# Backup data
cp -r api/data api/data.backup

# Clean database
rm api/data/events.db*

# Clean thumbnails
rm -rf api/data/thumbs/*

# Restart services
./scui-api
```

#### Partial Recovery
```bash
# Reset just the database
rm api/data/events.db*
./scui-api  # Will recreate database

# Reset just thumbnails
rm -rf api/data/thumbs/*
# Thumbnails will regenerate on next access
```

### Getting Help

#### Before Asking for Help

1. **Check the debug page** for system health
2. **Review logs** for error messages
3. **Try basic troubleshooting** steps above
4. **Document the issue** with:
   - Error messages
   - System configuration
   - Steps to reproduce

#### Useful Information to Include

- **System specs**: OS, CPU, RAM, storage
- **Configuration**: Environment variables
- **Logs**: Relevant error messages
- **Screenshots**: Debug page, error messages
- **Steps to reproduce**: What you were doing when it failed

#### Debug Information Collection

```bash
# System information
uname -a
df -h
free -h
ps aux | grep scui-api

# Configuration
env | grep -E "(CAMERA_DIR|PORT|HOST)"

# Logs
tail -100 scui-api.log
```

### Prevention

#### Regular Maintenance

1. **Monitor system health** via debug page
2. **Clean up old data** regularly
3. **Update system** components
4. **Backup database** periodically
5. **Monitor disk space** usage

#### Best Practices

1. **Use SSD storage** for better performance
2. **Keep system updated** with latest versions
3. **Monitor resource usage** regularly
4. **Test backups** periodically
5. **Document configuration** changes

---

**Still having issues?** Check the debug page for detailed system information and consider opening an issue with the collected debug information.
