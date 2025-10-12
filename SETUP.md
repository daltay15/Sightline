# 🚀 Quick Setup Guide

## **Getting Started in 5 Minutes**

### **1. Build the Application**
```bash
# Clone the repository
git clone <your-repo-url>
cd Security-Camera-UI

# Build the application
cd api
go build -o scui-api
```

### **2. Start the Application**
```bash
# Run the application
./scui-api
```

### **3. Open the Web Interface**
- Open your browser and go to: `http://localhost:8080`
- You'll be redirected to the AI Detections page

### **4. Configure the System**
- Click on **"Configuration"** in the navigation menu
- Set your camera directory path using the **📁 Browse** button to select the full path
- Configure other settings as needed
- Click **"Save Configuration"**

### **5. Test the Configuration**
- Click **"Test Configuration"** to verify your settings
- Check that your camera directory is accessible
- Verify database and GPU detection settings

## **🔧 Configuration Options**

### **Camera Directory**
- **Path**: Main directory where camera footage is stored
- **Example**: `/mnt/cameras` (Linux) or `C:\cameras` (Windows)
- **Structure**: Should contain subdirectories for each camera

### **Server Settings**
- **Host**: IP address to bind to (use `0.0.0.0` for all interfaces)
- **Port**: Web server port (default: 8080)
- **CORS**: Allowed origins for cross-origin requests

### **Database Settings**
- **Path**: SQLite database file location
- **Backup**: Enable automatic database backups
- **Interval**: How often to create backups

### **AI Detection Settings**
- **GPU Service URL**: URL of your GPU detection service
- **Timeout**: Request timeout in seconds
- **Confidence**: Minimum confidence threshold (0.0-1.0)
- **Classes**: Object types to detect (comma-separated)

### **Performance Settings**
- **Max Workers**: Number of concurrent file processors
- **Thumbnail Quality**: JPEG quality (1-100)
- **Thumbnail Size**: Maximum thumbnail size in pixels
- **Cache Size**: Cache size in MB

### **Security Settings**
- **Rate Limiting**: Enable/disable rate limiting
- **Requests**: Max requests per time window
- **Window**: Time window for rate limiting

## **📁 Directory Structure**

```
Security-Camera-UI/
├── api/
│   ├── scui-api          # Built application
│   ├── main.go          # Main application
│   ├── config/          # Configuration files
│   ├── internal/        # Internal packages
│   └── ui/              # Web interface files
├── docs/                # Documentation
└── SETUP.md            # This file
```

## **🌐 Web Interface Pages**

- **AI Detections**: View AI-detected objects and events
- **Events**: Browse all camera events
- **Stats**: Analytics and statistics
- **Debug**: System health and performance metrics
- **About**: System information and tools
- **Configuration**: System settings and configuration

## **🔍 Troubleshooting**

### **Common Issues**

1. **"Camera directory not found"**
   - Check that the path exists and is accessible
   - Ensure the application has read permissions

2. **"Database error"**
   - Verify the database path is writable
   - Check disk space availability

3. **"GPU detection service unavailable"**
   - Ensure the GPU service is running
   - Check the service URL and port

4. **"Port already in use"**
   - Change the server port in configuration
   - Stop other services using the same port

### **Getting Help**

- Check the **Debug** page for system health
- Review the **Troubleshooting** guide in `docs/`
- Check the application logs for error messages

## **⚡ Performance Tips**

1. **Use SSD storage** for better database performance
2. **Set appropriate worker counts** based on your CPU cores
3. **Enable caching** for better thumbnail performance
4. **Use GPU detection** for faster AI processing
5. **Regular database maintenance** to keep performance optimal

## **🔒 Security Considerations**

1. **Change default ports** in production
2. **Enable rate limiting** for public access
3. **Use HTTPS** in production environments
4. **Restrict CORS origins** to trusted domains
5. **Regular backups** of configuration and database

## **📈 Monitoring**

- Use the **Debug** page to monitor system health
- Check **Stats** page for usage analytics
- Monitor disk space and database size
- Watch for processing queue backlogs

## **🔄 Updates**

To update the application:
1. Stop the current application
2. Pull the latest changes
3. Rebuild: `go build -o scui-api`
4. Restart the application

---

**🎉 You're all set!** The Security Camera UI is now running and ready to process your camera footage with AI detection.
