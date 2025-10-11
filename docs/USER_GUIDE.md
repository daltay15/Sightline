# Security Camera UI User Guide

## Getting Started

### First Time Setup

1. **Access the Web Interface**
   - Open your browser to `http://localhost:8080`
   - You should see the main navigation menu

2. **Configure Your Camera Directory**
   - Go to the **About** page
   - Check that your camera directory is properly configured
   - Verify file permissions and structure

3. **Initial Scan**
   - The system will automatically start scanning for camera files
   - Monitor progress on the **Debug** page
   - Wait for initial scan to complete

## Main Navigation

### AI Detections
**Purpose**: View AI-detected objects with advanced filtering

**Key Features**:
- **Dynamic Grid Layout**: Adjust from 1-8 columns (desktop) or 1-2 columns (mobile)
- **Object Filtering**: Filter by person, car, truck, bicycle, motorcycle, etc.
- **Confidence Filtering**: Show only high-confidence detections
- **Date Range**: Filter by specific date ranges
- **Camera Selection**: View detections from specific cameras

**How to Use**:
1. Select your desired filters from the top controls
2. Click **Apply** to update the view
3. Use the grid size dropdown to adjust layout
4. Click on any detection to view full details

### Events
**Purpose**: Browse all camera events chronologically

**Key Features**:
- **Timeline View**: Chronological list of all events
- **Camera Filtering**: Filter by specific cameras
- **Date Navigation**: Jump to specific dates
- **Event Details**: View metadata and thumbnails

**How to Use**:
1. Use date controls to navigate to specific periods
2. Select cameras to focus on specific locations
3. Click on events to view full details
4. Use pagination to browse through results

### Stats
**Purpose**: View system statistics and performance metrics

**Key Features**:
- **Detection Trends**: Daily detection counts over time
- **Camera Statistics**: Per-camera event counts
- **System Performance**: Processing rates and efficiency
- **Recent Activity**: Latest detections and events

**How to Use**:
1. Review overall system statistics
2. Check detection trends for patterns
3. Monitor system performance
4. View recent activity for current status

### Debug
**Purpose**: Comprehensive system monitoring and troubleshooting

**Key Features**:
- **System Health**: Real-time system metrics
- **Database Status**: Database health and performance
- **File Processing**: Queue status and processing rates
- **Resource Monitoring**: CPU, RAM, and disk usage
- **Health Notifications**: Automatic alerts for issues

**How to Use**:
1. Monitor system health indicators
2. Check for any red warning indicators
3. Review processing queue status
4. Use refresh button to update all data

### About
**Purpose**: System information and maintenance tools

**Key Features**:
- **System Information**: Version and configuration details
- **Maintenance Tools**: Cleanup and optimization functions
- **Debug Access**: Quick access to debug page
- **System Status**: Overall system health

## Advanced Features

### Grid Layout Customization

**Desktop**:
- **1-8 columns**: Adjust based on your screen size and preference
- **Default**: 8 columns for maximum content visibility
- **Responsive**: Automatically adapts to screen size

**Mobile/Tablet**:
- **1-2 columns**: Optimized for touch interaction
- **Smart scaling**: Large grids automatically reduce on smaller screens
- **Touch-friendly**: Optimized for mobile devices

**How to Change**:
1. Use the **Grid Size** dropdown in the controls
2. Your preference is automatically saved
3. Changes apply immediately to the current view

### Filtering and Search

**Date Ranges**:
- **Quick Select**: Today, Yesterday, This Week, This Month, All Time
- **Custom Range**: Select specific start and end dates
- **Mobile Friendly**: Touch-optimized date pickers

**Object Types**:
- **People**: Human detection with confidence scoring
- **Vehicles**: Cars, trucks, buses, motorcycles, bicycles
- **Animals**: Dogs, birds, and other common animals
- **Custom**: High-confidence detections only

**Camera Selection**:
- **All Cameras**: View from all locations
- **Specific Camera**: Focus on one location
- **Multiple Selection**: Choose specific cameras (future feature)

### Fullscreen Viewing

**Image Viewing**:
- Click any thumbnail to enter fullscreen mode
- **Navigation**: Use arrow keys or click navigation arrows
- **Zoom**: Mouse wheel or pinch to zoom
- **Exit**: Press Escape or click the X button

**Video Viewing**:
- Click video thumbnails to play in fullscreen
- **Controls**: Standard video player controls
- **Seeking**: Click timeline to jump to specific times
- **Quality**: Automatic quality adjustment

### Mobile Optimization

**Responsive Design**:
- **Touch Navigation**: Swipe and tap optimized
- **Mobile Menu**: Hamburger menu for easy navigation
- **Touch Targets**: Large, easy-to-tap buttons
- **Orientation**: Works in both portrait and landscape

**Performance**:
- **Optimized Loading**: Faster page loads on mobile
- **Efficient Scrolling**: Smooth scrolling through large lists
- **Battery Friendly**: Optimized for mobile devices

## Tips and Best Practices

### Performance Optimization

1. **Grid Size**: Use fewer columns on slower devices
2. **Date Ranges**: Limit date ranges for faster loading
3. **Filters**: Use specific filters to reduce data load
4. **Regular Refresh**: Refresh data periodically for accuracy

### Mobile Usage

1. **Grid Layout**: Use 1-2 columns on mobile for best experience
2. **Touch Navigation**: Use swipe gestures for navigation
3. **Fullscreen**: Tap thumbnails for fullscreen viewing
4. **Orientation**: Rotate device for better viewing

### Troubleshooting

1. **Slow Loading**: Check debug page for system health
2. **Missing Data**: Verify camera directory configuration
3. **Display Issues**: Try different browser or clear cache
4. **Mobile Issues**: Check responsive design settings

### Data Management

1. **Regular Cleanup**: Use About page cleanup tools
2. **Storage Monitoring**: Check disk usage on debug page
3. **Backup**: Regular database backups recommended
4. **Maintenance**: Monitor system health regularly

## Keyboard Shortcuts

### Navigation
- **Arrow Keys**: Navigate through detections/events
- **Enter**: Open selected item
- **Escape**: Close fullscreen or dialogs
- **Tab**: Move between controls

### Viewing
- **Space**: Play/pause videos
- **+/-**: Zoom in/out
- **F**: Toggle fullscreen
- **R**: Refresh current view

### Filtering
- **Ctrl+A**: Select all cameras
- **Ctrl+D**: Clear all filters
- **Ctrl+R**: Reset to default view

## Common Tasks

### Finding Specific Events

1. **By Date**: Use date controls to jump to specific dates
2. **By Camera**: Select specific camera from dropdown
3. **By Object**: Use object type filters
4. **By Confidence**: Set minimum confidence threshold

### Monitoring System Health

1. **Debug Page**: Check for any red indicators
2. **System Resources**: Monitor CPU, RAM, disk usage
3. **Processing Queue**: Check for backlog issues
4. **Database Health**: Verify database performance

### Optimizing Performance

1. **Grid Size**: Adjust based on your screen size
2. **Date Ranges**: Use smaller date ranges for faster loading
3. **Filters**: Apply specific filters to reduce data
4. **Regular Maintenance**: Use cleanup tools regularly

## Troubleshooting Common Issues

### Page Not Loading
1. Check browser console for errors
2. Clear browser cache
3. Try different browser
4. Check server status on debug page

### Slow Performance
1. Reduce grid size
2. Limit date range
3. Apply filters
4. Check system resources on debug page

### Missing Data
1. Verify camera directory configuration
2. Check file permissions
3. Monitor file processing on debug page
4. Restart service if needed

### Mobile Issues
1. Check responsive design
2. Try different orientation
3. Clear mobile browser cache
4. Use mobile-optimized settings

## Getting Help

### Self-Service
1. **Debug Page**: Check system health and status
2. **About Page**: Review system configuration
3. **User Guide**: This document for common tasks
4. **Troubleshooting Guide**: For technical issues

### System Information
- **Version**: Check About page for version info
- **Configuration**: Review environment variables
- **Health**: Monitor debug page for system status
- **Logs**: Check system logs for error messages

### Support Resources
- **Documentation**: Comprehensive guides and API docs
- **Debug Tools**: Built-in system monitoring
- **Community**: GitHub issues and discussions
- **Professional Support**: Available for enterprise deployments

---

**Need more help?** Check the debug page for detailed system information and refer to the troubleshooting guide for technical issues.
