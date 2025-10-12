# Sightline

A comprehensive security camera management system with AI-powered object detection, real-time monitoring, and intelligent file organization.

## 🎯 Overview

This system provides a complete solution for managing security camera footage with advanced features:

- **Real-time File Monitoring**: Automatically detects and indexes new camera footage
- **AI Object Detection**: YOLO-powered detection of people, vehicles, and objects
- **Intelligent Organization**: Smart categorization and thumbnail generation
- **Web Interface**: Modern, responsive dashboard for viewing and managing footage
- **Performance Monitoring**: Comprehensive system health and debug tools
- **Distributed Processing**: Separate GPU service for AI processing
- **Easy Configuration**: Web-based configuration system for all settings

## 🚀 Quick Start

### Prerequisites

- **Go 1.19+** for the main API
- **Python 3.8+** for AI processing (optional)
- **CUDA-capable GPU** for AI acceleration (optional)
- **SQLite** database (included)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Security-Camera-UI
   ```

2. **Build the main API**:
   ```bash
   cd api
   go build -o scui-api
   ```

3. **Run the system**:
   ```bash
   ./scui-api
   ```

4. **Access the web interface**:
   - Open your browser to `http://localhost:8080`
   - Navigate to different sections using the top menu

5. **Configure the system**:
   - Click on **"Configuration"** in the navigation menu
   - Set your camera directory path and other settings
   - Click **"Save Configuration"** to apply changes
   - Use **"Test Configuration"** to verify your settings

## 📁 System Architecture

```
Security-Camera-UI/
├── api/                    # Main Go API server
│   ├── internal/          # Core system components
│   ├── ui/               # Web interface files
│   └── data/             # Database and thumbnails
├── gpu-detection-python/ # AI processing service
└── docs/                 # Documentation
```

## 🎛️ Web Interface

### Main Pages

- **AI Detections**: View AI-detected objects with filtering and search
- **Events**: Browse all camera events with timeline view
- **Stats**: System statistics and performance metrics
- **Debug**: Comprehensive system health monitoring
- **About**: System information and maintenance tools

### Key Features

- **Dynamic Grid Layout**: Adjustable 1-8 columns for optimal viewing
- **Mobile Responsive**: Optimized for all screen sizes
- **Real-time Updates**: Live data refresh and monitoring
- **Advanced Filtering**: Filter by camera, date, object type, and confidence
- **Fullscreen Viewing**: Immersive image and video viewing

## 🤖 AI Detection

### Supported Object Types

- **People**: Person detection with confidence scoring
- **Vehicles**: Cars, trucks, buses, motorcycles, bicycles
- **Animals**: Dogs, birds, and other common animals
- **Custom Classes**: Configurable detection categories

### AI Processing Options

1. **Local Processing**: Run AI detection on the same machine
2. **Distributed Processing**: Separate GPU service for AI processing
3. **Hybrid Mode**: Combine local and distributed processing

## 🔧 Configuration

### Environment Variables

```bash
# Camera directory (required)
CAMERA_DIR="/path/to/camera/footage"

# Database settings
DB_PATH="./data/events.db"

# GPU detection service (optional)
GPU_DETECTION_URL="http://localhost:8000"
GPU_DETECTION_TIMEOUT="30"

# Server settings
PORT="8080"
HOST="0.0.0.0"
```

### Camera Directory Structure

The system expects camera footage organized by:
```
/path/to/camera/footage/
├── Camera_01/
│   ├── 2024/
│   │   ├── 01/
│   │   │   ├── 01/
│   │   │   │   ├── image1.jpg
│   │   │   │   └── video1.mp4
│   │   │   └── 02/
│   │   └── 02/
│   └── 2023/
└── Camera_02/
```

## 📊 Monitoring & Debug

### System Health Monitoring

- **Real-time Metrics**: CPU, RAM, disk usage
- **Database Health**: Connection status and performance
- **File Processing**: Queue status and processing rates
- **AI Service Status**: GPU utilization and detection rates

### Debug Tools

- **System Overview**: Complete system status
- **Performance Metrics**: Detailed performance data
- **Raw API Responses**: Direct access to all endpoints
- **Health Notifications**: Automatic alerts for issues

## 🚀 Performance Optimization

### Recommended Hardware

- **CPU**: 4+ cores for optimal processing
- **RAM**: 8GB+ for smooth operation
- **Storage**: SSD for database and thumbnails
- **GPU**: CUDA-capable for AI acceleration (optional)

### Performance Tips

1. **Use SSD storage** for database and thumbnails
2. **Enable GPU acceleration** for AI processing
3. **Monitor system resources** via debug page
4. **Regular maintenance** using cleanup tools
5. **Optimize camera directory** structure

## 🔒 Security Considerations

- **Network Access**: Configure firewall rules appropriately
- **File Permissions**: Ensure proper directory permissions
- **Database Security**: Regular backups and access control
- **AI Processing**: Secure GPU service endpoints

## 📚 Documentation

- [GPU Detection Integration](api/GPU_DETECTION_INTEGRATION.md)
- [Parallel Processing Guide](gpu-detection-python/PARALLEL_PROCESSING.md)
- [API Documentation](docs/API.md)
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
2. Review the debug page for system health
3. Check system logs for error messages
4. Open an issue on GitHub

---

**Built with ❤️ for security professionals and home users alike.**
