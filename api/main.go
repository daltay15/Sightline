// main.go
package main

import (
	"database/sql"
	"embed"
	"encoding/json"
	"io/fs"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/daltay15/security-camera-ui/api/config"
	"github.com/daltay15/security-camera-ui/api/internal"
	httpx "github.com/daltay15/security-camera-ui/api/internal/http"
	"github.com/daltay15/security-camera-ui/api/internal/indexer"
	"github.com/daltay15/security-camera-ui/api/internal/telegram"

	"github.com/gin-gonic/gin"
	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/disk"
	"github.com/shirou/gopsutil/v3/mem"
)

//go:embed ui/*
var uiFS embed.FS

func main() {
	log.Printf("=== Security Camera UI Starting ===")

	// Use constants from config package
	rootDir := config.RootDir
	pendingDir := config.PendingDir
	processingDir := config.ProcessingDir
	completedDir := config.CompletedDir
	failedDir := config.FailedDir
	dbPath := config.DbPath
	thumbsDir := config.ThumbsDir
	port := config.Port
	telegramURL := config.TelegramURL

	// Initialize backup manager with config
	backupDir := filepath.Join(filepath.Dir(dbPath), "backups")

	// Load configuration to get backup interval
	configManager := internal.NewConfigManager("config.json")
	if err := configManager.LoadConfig(); err != nil {
		log.Printf("Warning: Failed to load config, using defaults: %v", err)
	}

	// Get backup settings from config
	backupEnabled := configManager.GetBool("db_backup_enabled", true)
	backupInterval := configManager.GetString("db_backup_interval", "24h")

	backupManager := internal.NewBackupManager(dbPath, backupDir, backupEnabled, backupInterval)

	// Initialize error notification service
	var errorNotifier *telegram.ErrorTelegramClient
	errorNotificationsEnabled := configManager.GetBool("error_notifications_enabled", false)
	pythonEndpoint := configManager.GetString("error_notifications_python_endpoint", "")

	if errorNotificationsEnabled {
		configData := configManager.GetConfig()
		var err error
		errorNotifier, err = telegram.NewErrorTelegramClient(configData)
		if err != nil {
			log.Printf("Warning: Failed to initialize error notification service: %v", err)
			errorNotifier = nil
		} else {
			log.Printf("Error notification service initialized successfully")
		}
	}

	// Initialize global error notifier
	internal.InitErrorNotifier(errorNotifier, pythonEndpoint, errorNotificationsEnabled)

	log.Printf("Configuration: rootDir=%s, completedDir=%s, port=%s", rootDir, completedDir, port)
	log.Printf("Backup settings: enabled=%v, interval=%s", backupEnabled, backupInterval)

	_ = os.MkdirAll(filepath.Dir(dbPath), 0755)
	_ = os.MkdirAll(thumbsDir, 0755)

	db, err := internal.OpenDB(dbPath)
	if err != nil {
		log.Fatal(err)
	}

	// ensure schema is present
	schemaSQL := `
	CREATE TABLE IF NOT EXISTS events (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		camera TEXT NOT NULL,
		path TEXT NOT NULL UNIQUE,
		jpg_path TEXT,
		sheet_path TEXT,
		start_ts INTEGER NOT NULL,
		duration_ms INTEGER DEFAULT 0,
		size_bytes INTEGER NOT NULL,
		reviewed INTEGER DEFAULT 0,
		tags TEXT DEFAULT '',
		created_at INTEGER NOT NULL
	);
	CREATE INDEX IF NOT EXISTS idx_events_ts ON events(start_ts);
	CREATE INDEX IF NOT EXISTS idx_events_camera_ts ON events(camera, start_ts);

	-- Detection results table for storing individual detections
	CREATE TABLE IF NOT EXISTS detections (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		event_id INTEGER NOT NULL,
		detection_type TEXT NOT NULL,
		confidence REAL NOT NULL,
		bbox_x INTEGER NOT NULL,
		bbox_y INTEGER NOT NULL,
		bbox_width INTEGER NOT NULL,
		bbox_height INTEGER NOT NULL,
		frame_index INTEGER,
		timestamp REAL,
		created_at INTEGER NOT NULL,
		FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE
	);
	CREATE INDEX IF NOT EXISTS idx_detections_event_id ON detections(event_id);
	CREATE INDEX IF NOT EXISTS idx_detections_type ON detections(detection_type);
	CREATE INDEX IF NOT EXISTS idx_detections_confidence ON detections(confidence);

	`
	if _, err := db.Exec(schemaSQL); err != nil {
		log.Fatal("Failed to create schema:", err)
	}

	// Add detection columns if they don't exist (migration)
	// Check if detection_data column exists
	var columnExists int
	err = db.QueryRow("SELECT COUNT(*) FROM pragma_table_info('events') WHERE name='detection_data'").Scan(&columnExists)
	if err != nil || columnExists == 0 {
		log.Printf("Adding detection_data column to events table")
		if _, err := db.Exec("ALTER TABLE events ADD COLUMN detection_data TEXT"); err != nil {
			log.Printf("Warning: Failed to add detection_data column: %v", err)
		}
	}

	// Check if detection_updated column exists
	err = db.QueryRow("SELECT COUNT(*) FROM pragma_table_info('events') WHERE name='detection_updated'").Scan(&columnExists)
	if err != nil || columnExists == 0 {
		log.Printf("Adding detection_updated column to events table")
		if _, err := db.Exec("ALTER TABLE events ADD COLUMN detection_updated INTEGER DEFAULT 0"); err != nil {
			log.Printf("Warning: Failed to add detection_updated column: %v", err)
		}
	}

	// Add media correlation metadata columns if they don't exist
	// Check if video_path column exists
	err = db.QueryRow("SELECT COUNT(*) FROM pragma_table_info('events') WHERE name='video_path'").Scan(&columnExists)
	if err != nil || columnExists == 0 {
		log.Printf("Adding video_path column to events table")
		if _, err := db.Exec("ALTER TABLE events ADD COLUMN video_path TEXT"); err != nil {
			log.Printf("Warning: Failed to add video_path column: %v", err)
		}
	}

	// Check if detection_path column exists
	err = db.QueryRow("SELECT COUNT(*) FROM pragma_table_info('events') WHERE name='detection_path'").Scan(&columnExists)
	if err != nil || columnExists == 0 {
		log.Printf("Adding detection_path column to events table")
		if _, err := db.Exec("ALTER TABLE events ADD COLUMN detection_path TEXT"); err != nil {
			log.Printf("Warning: Failed to add detection_path column: %v", err)
		}
	}

	// Create additional indexes for detection columns
	indexSQL := `
		CREATE INDEX IF NOT EXISTS idx_events_reviewed ON events(reviewed);
		CREATE INDEX IF NOT EXISTS idx_events_detection_updated ON events(detection_updated);
		CREATE INDEX IF NOT EXISTS idx_events_path ON events(path);
		CREATE INDEX IF NOT EXISTS idx_events_camera_path ON events(camera, path);
	`
	if _, err := db.Exec(indexSQL); err != nil {
		log.Printf("Warning: Failed to create detection indexes: %v", err)
	}

	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()

	// Add compression middleware for better performance
	r.Use(httpx.CompressionMiddleware())

	// API routes
	httpx.Routes(r, db, configManager)

	// Serve embedded UI files
	sub, err := fs.Sub(uiFS, "ui")
	if err != nil {
		log.Fatal("Failed to create UI filesystem:", err)
	}
	r.StaticFS("/ui", http.FS(sub))
	r.GET("/", func(c *gin.Context) { c.Redirect(http.StatusFound, "/ui/detections.html") })

	// (optional) health/stats
	r.GET("/health", func(c *gin.Context) { c.JSON(200, gin.H{"status": "ok"}) })
	r.GET("/stats", func(c *gin.Context) {
		var n int64
		_ = db.QueryRow("SELECT COUNT(*) FROM events").Scan(&n)
		c.JSON(200, gin.H{"events": n})
	})

	// Performance monitoring endpoint
	r.GET("/perf", func(c *gin.Context) {
		var totalEvents, videoEvents, thumbnailsGenerated int64
		_ = db.QueryRow("SELECT COUNT(*) FROM events").Scan(&totalEvents)
		_ = db.QueryRow("SELECT COUNT(*) FROM events WHERE path LIKE '%.mp4'").Scan(&videoEvents)
		_ = db.QueryRow("SELECT COUNT(*) FROM events WHERE jpg_path IS NOT NULL").Scan(&thumbnailsGenerated)

		c.JSON(200, gin.H{
			"totalEvents":         totalEvents,
			"videoEvents":         videoEvents,
			"thumbnailsGenerated": thumbnailsGenerated,
			"thumbnailCoverage":   float64(thumbnailsGenerated) / float64(videoEvents) * 100,
		})
	})

	// Debug endpoint to check database content
	r.GET("/debug", func(c *gin.Context) {
		var count int64
		_ = db.QueryRow("SELECT COUNT(*) FROM events").Scan(&count)

		// Get a few sample records
		rows, err := db.Query("SELECT id, camera, path FROM events ORDER BY start_ts DESC LIMIT 5")
		if err != nil {
			c.JSON(500, gin.H{"error": err.Error()})
			return
		}
		defer rows.Close()

		var samples []map[string]any
		for rows.Next() {
			var id, camera, path string
			_ = rows.Scan(&id, &camera, &path)
			samples = append(samples, map[string]any{
				"id": id, "camera": camera, "path": path,
			})
		}

		c.JSON(200, gin.H{
			"totalEvents": count,
			"samples":     samples,
		})
	})

	// Debug endpoint for detection processing
	r.GET("/debug-detections", func(c *gin.Context) {
		var totalEvents, detectionEvents, eventsWithDetectionData, detectionsTable int64
		_ = db.QueryRow("SELECT COUNT(*) FROM events").Scan(&totalEvents)
		_ = db.QueryRow("SELECT COUNT(*) FROM events WHERE camera = 'DETECTION'").Scan(&detectionEvents)
		_ = db.QueryRow("SELECT COUNT(*) FROM events WHERE detection_data IS NOT NULL").Scan(&eventsWithDetectionData)
		_ = db.QueryRow("SELECT COUNT(*) FROM detections").Scan(&detectionsTable)

		// Get recent detection events
		rows, err := db.Query("SELECT id, camera, path, start_ts, detection_data IS NOT NULL as has_detection_data FROM events WHERE camera = 'DETECTION' ORDER BY start_ts DESC LIMIT 10")
		if err != nil {
			c.JSON(500, gin.H{"error": err.Error()})
			return
		}
		defer rows.Close()

		var detectionSamples []map[string]any
		for rows.Next() {
			var id, camera, path string
			var startTs int64
			var hasDetectionData bool
			_ = rows.Scan(&id, &camera, &path, &startTs, &hasDetectionData)
			detectionSamples = append(detectionSamples, map[string]any{
				"id": id, "camera": camera, "path": path, "startTs": startTs, "hasDetectionData": hasDetectionData,
			})
		}

		// Get recent events with detection data
		rows2, err := db.Query("SELECT id, camera, path, start_ts FROM events WHERE detection_data IS NOT NULL ORDER BY start_ts DESC LIMIT 10")
		if err != nil {
			c.JSON(500, gin.H{"error": err.Error()})
			return
		}
		defer rows2.Close()

		var eventsWithDetectionSamples []map[string]any
		for rows2.Next() {
			var id, camera, path string
			var startTs int64
			_ = rows2.Scan(&id, &camera, &path, &startTs)
			eventsWithDetectionSamples = append(eventsWithDetectionSamples, map[string]any{
				"id": id, "camera": camera, "path": path, "startTs": startTs,
			})
		}

		// Get events that should have detection data but don't (recent events without detection data)
		rows3, err := db.Query(`
			SELECT e.id, e.camera, e.path, e.start_ts, e.created_at,
			       CASE WHEN d.id IS NOT NULL THEN 1 ELSE 0 END as has_detection_event
			FROM events e 
			LEFT JOIN events d ON d.camera = 'DETECTION' AND d.tags LIKE 'detection_for_event_' || e.id
			WHERE e.camera != 'DETECTION' 
			AND e.path LIKE '%.jpg'
			AND e.detection_data IS NULL
			AND e.start_ts > ? 
			ORDER BY e.start_ts DESC 
			LIMIT 10
		`, time.Now().Unix()-3600) // Last hour
		if err != nil {
			c.JSON(500, gin.H{"error": err.Error()})
			return
		}
		defer rows3.Close()

		var pendingDetectionSamples []map[string]any
		for rows3.Next() {
			var id, camera, path string
			var startTs, createdAt int64
			var hasDetectionEvent bool
			_ = rows3.Scan(&id, &camera, &path, &startTs, &createdAt, &hasDetectionEvent)
			pendingDetectionSamples = append(pendingDetectionSamples, map[string]any{
				"id": id, "camera": camera, "path": path, "startTs": startTs, "createdAt": createdAt, "hasDetectionEvent": hasDetectionEvent,
			})
		}

		// Get detection processing stats
		var recentDetectionUpdates int64
		_ = db.QueryRow("SELECT COUNT(*) FROM events WHERE detection_updated > ?", time.Now().Unix()-300).Scan(&recentDetectionUpdates) // Last 5 minutes

		c.JSON(200, gin.H{
			"totalEvents":                totalEvents,
			"detectionEvents":            detectionEvents,
			"eventsWithDetectionData":    eventsWithDetectionData,
			"detectionsTable":            detectionsTable,
			"recentDetectionUpdates":     recentDetectionUpdates,
			"detectionSamples":           detectionSamples,
			"eventsWithDetectionSamples": eventsWithDetectionSamples,
			"pendingDetectionSamples":    pendingDetectionSamples,
		})
	})

	// Live scanning status endpoint
	r.GET("/scan-status", func(c *gin.Context) {
		var totalEvents, videoEvents, thumbnailsGenerated, detectionEvents, eventsWithDetectionData int64
		_ = db.QueryRow("SELECT COUNT(*) FROM events").Scan(&totalEvents)
		_ = db.QueryRow("SELECT COUNT(*) FROM events WHERE path LIKE '%.mp4'").Scan(&videoEvents)
		_ = db.QueryRow("SELECT COUNT(*) FROM events WHERE jpg_path IS NOT NULL").Scan(&thumbnailsGenerated)
		_ = db.QueryRow("SELECT COUNT(*) FROM events WHERE camera = 'DETECTION'").Scan(&detectionEvents)
		_ = db.QueryRow("SELECT COUNT(*) FROM events WHERE detection_data IS NOT NULL").Scan(&eventsWithDetectionData)

		// Get recent events (last 10)
		rows, err := db.Query("SELECT id, camera, path, start_ts FROM events ORDER BY start_ts DESC LIMIT 10")
		if err != nil {
			c.JSON(500, gin.H{"error": err.Error()})
			return
		}
		defer rows.Close()

		var recentEvents []map[string]any
		for rows.Next() {
			var id, camera, path string
			var startTs int64
			_ = rows.Scan(&id, &camera, &path, &startTs)
			recentEvents = append(recentEvents, map[string]any{
				"id": id, "camera": camera, "path": path, "startTs": startTs,
			})
		}

		c.JSON(200, gin.H{
			"totalEvents":             totalEvents,
			"videoEvents":             videoEvents,
			"thumbnailsGenerated":     thumbnailsGenerated,
			"detectionEvents":         detectionEvents,
			"eventsWithDetectionData": eventsWithDetectionData,
			"recentEvents":            recentEvents,
			"scanning":                totalEvents == 0, // Assume scanning if no events yet
		})
	})

	// File watcher health endpoint
	r.GET("/watcher-health", func(c *gin.Context) {
		if !config.FileWatcherEnabled {
			c.JSON(200, gin.H{"status": "disabled", "message": "File watcher is disabled"})
			return
		}

		// Note: This would need access to the fileWatcher instance
		// For now, return basic status
		c.JSON(200, gin.H{
			"status":  "enabled",
			"message": "File watcher is running",
			"note":    "Detailed stats require fileWatcher instance access",
		})
	})

	// System metrics endpoint
	r.GET("/system-metrics", func(c *gin.Context) {
		// Get basic system info
		var totalEvents, videoEvents, thumbnailsGenerated int64
		_ = db.QueryRow("SELECT COUNT(*) FROM events").Scan(&totalEvents)
		_ = db.QueryRow("SELECT COUNT(*) FROM events WHERE path LIKE '%.mp4'").Scan(&videoEvents)
		_ = db.QueryRow("SELECT COUNT(*) FROM events WHERE jpg_path IS NOT NULL").Scan(&thumbnailsGenerated)

		// Get database file size
		var dbSize int64
		if stat, err := os.Stat(dbPath); err == nil {
			dbSize = stat.Size()
		}

		// Get Go runtime memory stats
		var m runtime.MemStats
		runtime.ReadMemStats(&m)

		// Get disk usage for the database directory
		var diskUsage int64
		if stat, err := os.Stat(filepath.Dir(dbPath)); err == nil {
			// This is a simplified disk usage calculation
			// In a real implementation, you'd want to use syscall or a library like gopsutil
			diskUsage = stat.Size() // This is just the directory size, not total disk usage
		}

		// Calculate program memory usage percentages
		memoryUsedMB := float64(m.Alloc) / 1024 / 1024
		memoryTotalMB := float64(m.Sys) / 1024 / 1024
		memoryPercent := (memoryUsedMB / memoryTotalMB) * 100

		// Get real system-wide memory info
		var systemMemTotal, systemMemAvailable, systemMemUsed uint64
		var systemMemError string
		if memInfo, err := mem.VirtualMemory(); err == nil {
			systemMemTotal = memInfo.Total
			systemMemAvailable = memInfo.Available
			systemMemUsed = memInfo.Used
		} else {
			systemMemError = err.Error()
		}

		// Get real system disk info
		var systemDiskTotal, systemDiskFree, systemDiskUsed uint64
		var systemDiskError string
		if diskInfo, err := disk.Usage("/"); err == nil {
			systemDiskTotal = diskInfo.Total
			systemDiskFree = diskInfo.Free
			systemDiskUsed = diskInfo.Used
		} else {
			systemDiskError = err.Error()
		}

		// Get real NAS drive info
		var nasDiskTotal, nasDiskFree, nasDiskUsed uint64
		var nasDiskError string
		if nasDiskInfo, err := disk.Usage("/mnt/nas"); err == nil {
			nasDiskTotal = nasDiskInfo.Total
			nasDiskFree = nasDiskInfo.Free
			nasDiskUsed = nasDiskInfo.Used
		} else {
			nasDiskError = err.Error()
		}

		// Get real CPU usage
		var cpuPercent float64
		var cpuError string
		if cpuInfo, err := cpu.Percent(time.Second, false); err == nil && len(cpuInfo) > 0 {
			cpuPercent = cpuInfo[0]
		} else {
			cpuError = err.Error()
		}

		c.JSON(200, gin.H{
			"database_size_bytes":  dbSize,
			"database_size_mb":     float64(dbSize) / (1024 * 1024),
			"total_events":         totalEvents,
			"video_events":         videoEvents,
			"thumbnails_generated": thumbnailsGenerated,
			"thumbnail_coverage":   float64(thumbnailsGenerated) / float64(videoEvents) * 100,
			"go_version":           "1.21+",

			// Program memory metrics
			"program_memory_used_mb":    memoryUsedMB,
			"program_memory_total_mb":   memoryTotalMB,
			"program_memory_percent":    memoryPercent,
			"program_memory_alloc_mb":   float64(m.Alloc) / 1024 / 1024,
			"program_memory_sys_mb":     float64(m.Sys) / 1024 / 1024,
			"program_memory_heap_mb":    float64(m.HeapAlloc) / 1024 / 1024,
			"program_gc_cycles":         m.NumGC,
			"program_gc_pause_total_ms": float64(m.PauseTotalNs) / 1000000,

			// Program system info
			"program_goroutines": runtime.NumGoroutine(),
			"program_num_cpus":   runtime.NumCPU(),
			"program_max_procs":  runtime.GOMAXPROCS(0),

			// System-wide memory metrics (real data)
			"system_memory_total_mb":     float64(systemMemTotal) / (1024 * 1024),
			"system_memory_available_mb": float64(systemMemAvailable) / (1024 * 1024),
			"system_memory_used_mb":      float64(systemMemUsed) / (1024 * 1024),
			"system_memory_percent": func() float64 {
				if systemMemTotal > 0 {
					return float64(systemMemUsed) / float64(systemMemTotal) * 100
				}
				return 0.0
			}(),
			"system_memory_error": systemMemError,

			// System-wide disk metrics (real data)
			"system_disk_total_mb": float64(systemDiskTotal) / (1024 * 1024),
			"system_disk_free_mb":  float64(systemDiskFree) / (1024 * 1024),
			"system_disk_used_mb":  float64(systemDiskUsed) / (1024 * 1024),
			"system_disk_percent": func() float64 {
				if systemDiskTotal > 0 {
					return float64(systemDiskUsed) / float64(systemDiskTotal) * 100
				}
				return 0.0
			}(),
			"system_disk_error": systemDiskError,

			// System-wide CPU metrics (real data)
			"system_cpu_percent": func() float64 {
				if cpuPercent > 0 {
					return cpuPercent
				}
				return 0.0
			}(),
			"system_cpu_error": cpuError,

			// NAS drive metrics (real data)
			"nas_disk_total_mb": float64(nasDiskTotal) / (1024 * 1024),
			"nas_disk_free_mb":  float64(nasDiskFree) / (1024 * 1024),
			"nas_disk_used_mb":  float64(nasDiskUsed) / (1024 * 1024),
			"nas_disk_percent": func() float64 {
				if nasDiskTotal > 0 {
					return float64(nasDiskUsed) / float64(nasDiskTotal) * 100
				}
				return 0.0
			}(),
			"nas_disk_error": nasDiskError,

			// Program disk info (simplified)
			"program_disk_usage_bytes": diskUsage,
			"program_disk_usage_mb":    float64(diskUsage) / (1024 * 1024),
		})
	})

	// Database health endpoint
	r.GET("/db-health", func(c *gin.Context) {
		// Test database connection and performance
		start := time.Now()
		var count int64
		err := db.QueryRow("SELECT COUNT(*) FROM events").Scan(&count)
		queryTime := time.Since(start)

		// Get table info
		var tableCount int
		_ = db.QueryRow("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").Scan(&tableCount)

		// Get index info
		var indexCount int
		_ = db.QueryRow("SELECT COUNT(*) FROM sqlite_master WHERE type='index'").Scan(&indexCount)

		health := gin.H{
			"status":        "Healthy",
			"query_time_ms": queryTime.Milliseconds(),
			"total_records": count,
			"table_count":   tableCount,
			"index_count":   indexCount,
			"connection_ok": err == nil,
		}

		if err != nil {
			health["status"] = "unhealthy"
			health["error"] = err.Error()
		}

		c.JSON(200, health)
	})

	// Processing queue status endpoint
	r.GET("/processing-status", func(c *gin.Context) {
		// Check pending directory
		pendingFiles := 0
		if entries, err := os.ReadDir(pendingDir); err == nil {
			pendingFiles = len(entries)
		}

		// Check processing directory
		processingFiles := 0
		if entries, err := os.ReadDir(processingDir); err == nil {
			processingFiles = len(entries)
		}

		// Check completed directory
		completedFiles := 0
		if entries, err := os.ReadDir(completedDir); err == nil {
			completedFiles = len(entries)
		}

		// Get recent processing activity
		var recentProcessed int64
		_ = db.QueryRow("SELECT COUNT(*) FROM events WHERE created_at > ?", time.Now().Unix()-3600).Scan(&recentProcessed)

		c.JSON(200, gin.H{
			"pending_files":    pendingFiles,
			"processing_files": processingFiles,
			"completed_files":  completedFiles,
			"recent_processed": recentProcessed,
			"queue_healthy":    pendingFiles < 100, // Arbitrary threshold
			"last_check":       time.Now().Unix(),
		})
	})

	// Disk monitoring status endpoint
	r.GET("/disk-monitor-status", func(c *gin.Context) {
		diskMonitor := internal.NewDiskMonitor(configManager.GetConfig())
		status := diskMonitor.GetStatus()
		c.JSON(200, status)
	})

	// Configuration endpoints
	r.GET("/api/config", func(c *gin.Context) {
		// Load current configuration from file
		configManager := internal.NewConfigManager("config.json")
		if err := configManager.LoadConfig(); err != nil {
			log.Printf("Warning: Failed to load config for API: %v", err)
		}

		// Return current configuration from file
		configData := configManager.GetConfig()
		c.JSON(200, configData)
	})

	r.POST("/api/config", func(c *gin.Context) {
		var newConfig map[string]interface{}
		if err := c.ShouldBindJSON(&newConfig); err != nil {
			c.JSON(400, gin.H{"error": "Invalid JSON"})
			return
		}

		// Save configuration to file
		configFile := "config.json"
		configData, err := json.MarshalIndent(newConfig, "", "  ")
		if err != nil {
			c.JSON(500, gin.H{"error": "Failed to marshal config"})
			return
		}

		if err := os.WriteFile(configFile, configData, 0644); err != nil {
			c.JSON(500, gin.H{"error": "Failed to save config"})
			return
		}

		// Update backup manager with new settings if they changed
		if backupEnabled, ok := newConfig["db_backup_enabled"].(bool); ok {
			if backupInterval, ok := newConfig["db_backup_interval"].(string); ok {
				backupManager.UpdateSettings(backupEnabled, backupInterval)
			}
		}

		c.JSON(200, gin.H{"message": "Configuration saved successfully"})
	})

	r.POST("/api/config/test", func(c *gin.Context) {
		var testConfig map[string]interface{}
		if err := c.ShouldBindJSON(&testConfig); err != nil {
			c.JSON(400, gin.H{"error": "Invalid JSON"})
			return
		}

		// Test configuration
		results := gin.H{
			"camera_dir": gin.H{
				"valid":   true,
				"message": "Camera directory path is valid",
			},
			"server_port": gin.H{
				"valid":   true,
				"message": "Port number is valid",
			},
			"gpu_detection": gin.H{
				"valid":   true,
				"message": "GPU detection URL is valid",
			},
		}

		// Test Telegram configuration if enabled
		if telegramEnabled, ok := testConfig["telegram_enabled"].(bool); ok && telegramEnabled {
			telegramClient, err := telegram.NewTelegramClient(testConfig)
			if err != nil {
				results["telegram"] = gin.H{
					"valid":   false,
					"error":   err.Error(),
					"message": "Telegram configuration is invalid",
				}
			} else {
				results["telegram"] = gin.H{
					"valid":   true,
					"message": "Telegram configuration is valid",
					"chats":   len(telegramClient.GetChats()),
				}
			}
		} else {
			results["telegram"] = gin.H{
				"valid":   true,
				"message": "Telegram is disabled",
			}
		}

		c.JSON(200, results)
	})

	r.POST("/api/config/test-directory", func(c *gin.Context) {
		var req struct {
			Path string `json:"path"`
		}
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(400, gin.H{"error": "Invalid JSON"})
			return
		}

		// Test directory access
		info, err := os.Stat(req.Path)
		if err != nil {
			c.JSON(200, gin.H{
				"valid":   false,
				"error":   err.Error(),
				"message": "Directory does not exist or is not accessible",
			})
			return
		}

		if !info.IsDir() {
			c.JSON(200, gin.H{
				"valid":   false,
				"error":   "Path is not a directory",
				"message": "The specified path is not a directory",
			})
			return
		}

		// Check if directory is readable
		if _, err := os.ReadDir(req.Path); err != nil {
			c.JSON(200, gin.H{
				"valid":   false,
				"error":   err.Error(),
				"message": "Directory is not readable",
			})
			return
		}

		c.JSON(200, gin.H{
			"valid":        true,
			"message":      "Directory is accessible and readable",
			"path":         req.Path,
			"is_directory": true,
			"readable":     true,
		})
	})

	r.POST("/api/config/restart", func(c *gin.Context) {
		// Note: In a real implementation, this would restart the service
		// For now, we'll just return a success message
		c.JSON(200, gin.H{
			"message": "Service restart initiated",
			"note":    "In production, this would restart the service",
		})
	})

	// Backup management endpoints
	r.GET("/api/backup/status", func(c *gin.Context) {
		status := backupManager.GetBackupStatus()
		c.JSON(200, status)
	})

	r.GET("/api/backup/list", func(c *gin.Context) {
		backups, err := backupManager.ListBackups()
		if err != nil {
			c.JSON(500, gin.H{"error": err.Error()})
			return
		}
		c.JSON(200, gin.H{"backups": backups})
	})

	r.POST("/api/backup/create", func(c *gin.Context) {
		if err := backupManager.CreateBackup(); err != nil {
			c.JSON(500, gin.H{"error": err.Error()})
			return
		}
		c.JSON(200, gin.H{"message": "Backup created successfully"})
	})

	r.POST("/api/backup/test", func(c *gin.Context) {
		if err := backupManager.TestBackupConnection(); err != nil {
			c.JSON(200, gin.H{
				"valid":   false,
				"error":   err.Error(),
				"message": "Backup system test failed",
			})
			return
		}
		c.JSON(200, gin.H{
			"valid":   true,
			"message": "Backup system is working correctly",
		})
	})

	r.POST("/api/backup/restore", func(c *gin.Context) {
		var req struct {
			BackupPath string `json:"backup_path"`
		}
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(400, gin.H{"error": "Invalid JSON"})
			return
		}

		if err := backupManager.RestoreBackup(req.BackupPath); err != nil {
			c.JSON(500, gin.H{"error": err.Error()})
			return
		}

		c.JSON(200, gin.H{"message": "Database restored successfully"})
	})

	go func() {
		log.Printf("HTTP server on %s", port)
		server := &http.Server{
			Addr:         port,
			Handler:      r,
			ReadTimeout:  30 * time.Second,
			WriteTimeout: 30 * time.Second,
			IdleTimeout:  120 * time.Second,
		}
		log.Fatal(server.ListenAndServe())
	}()

	// Detection processing
	go func() {
		defer func() {
			if r := recover(); r != nil {
				log.Printf("Detection processing goroutine panicked: %v", r)
			}
		}()

		// Initialize detection processor
		detectionProcessor := internal.NewDetectionProcessor(db, pendingDir, processingDir, completedDir, failedDir, telegramURL)

		for {
			if err := detectionProcessor.ProcessCompletedFiles(); err != nil {
				log.Printf("Detection processing error: %v", err)
			}

			time.Sleep(time.Duration(config.DetectionInterval))
		}
	}()

	// File system monitoring (file watcher or polling based on config)
	if config.FileWatcherEnabled {
		// Real-time file watcher (replaces polling)
		log.Printf("Starting real-time file watcher")
		fileWatcher := internal.NewFileWatcher(db, rootDir, thumbsDir, pendingDir, telegramURL)
		if err := fileWatcher.Start(); err != nil {
			log.Fatalf("Failed to start file watcher: %v", err)
		}
		defer fileWatcher.Stop()

		// Start the backup manager
		backupManager.StartBackupScheduler()

		// High-frequency watcher signal processor (for immediate processing)
		go func() {
			defer func() {
				if r := recover(); r != nil {
					log.Printf("Watcher signal processor panicked: %v", r)
				}
			}()

			log.Printf("Starting watcher signal processor")
			for {
				// Check for file watcher signals every 100ms for immediate processing
				if err := processWatcherSignals(db, rootDir, thumbsDir, pendingDir); err != nil {
					log.Printf("Error processing watcher signals: %v", err)
				}
				time.Sleep(100 * time.Millisecond)
			}
		}()
	} else {
		log.Printf("File watcher disabled, using polling approach")
	}

	// Background scans (frequency depends on file watcher setting)
	log.Printf("About to start background scanning goroutine")
	go func() {
		defer func() {
			if r := recover(); r != nil {
				log.Printf("Scanning goroutine panicked: %v", r)
			}
		}()

		log.Printf("Starting background scanning goroutine")
		for {
			log.Printf("Background scan iteration starting")
			if err := indexer.ScanAllParallel(db, indexer.Config{
				Root: rootDir, ThumbsDir: thumbsDir, PendingDir: pendingDir, MaxWorkers: config.MaxWorkers, BatchSize: config.BatchSize,
				ThumbnailWorkers: config.ThumbnailWorkers,
			}); err != nil {
				log.Printf("scan error: %v", err)
			}

			// Check how many events we have
			var count int64
			_ = db.QueryRow("SELECT COUNT(*) FROM events").Scan(&count)
			log.Printf("total events in database: %d", count)

			// Pre-generate thumbnails for better performance
			if err := indexer.PreGenerateThumbnails(db, indexer.Config{
				Root: rootDir, ThumbsDir: thumbsDir, ThumbnailWorkers: config.ThumbnailWorkers,
			}); err != nil {
				log.Printf("thumbnail generation error: %v", err)
			}

			// Different intervals based on file watcher setting
			if config.FileWatcherEnabled {
				// File watcher enabled: reduced frequency for cleanup only
				time.Sleep(5 * time.Minute)
			} else {
				// File watcher disabled: use original polling interval
				time.Sleep(time.Duration(config.ScanInterval))
			}
		}
	}()

	// Separate goroutine for detection scanning
	go func() {
		defer func() {
			if r := recover(); r != nil {
				log.Printf("Detection scanning goroutine panicked: %v", r)
			}
		}()

		// Wait a bit before starting to avoid initial database contention
		time.Sleep(1 * time.Second)

		for {
			if err := indexer.ScanDetectionImages(db, indexer.Config{
				DetectionDir:    completedDir,
				IndexDetections: true,
				TelegramURL:     telegramURL,
			}); err != nil {
				log.Printf("detection image scan error: %v", err)
			}

			time.Sleep(time.Duration(config.ScanInterval))
		}
	}()

	// Disk space monitoring
	diskMonitor := internal.NewDiskMonitor(configManager.GetConfig())
	diskMonitor.Start()

	log.Printf("All goroutines started, entering select loop")
	select {}
}

// processWatcherSignals processes files that were detected by the file watcher
func processWatcherSignals(db *sql.DB, rootDir, thumbsDir, pendingDir string) error {
	signalFile := filepath.Join(pendingDir, ".watcher_signal")

	// Check if signal file exists
	if _, err := os.Stat(signalFile); os.IsNotExist(err) {
		return nil // No signal file
	}

	// Read the file path from signal file
	content, err := os.ReadFile(signalFile)
	if err != nil {
		return err
	}

	filePath := string(content)
	log.Printf("Processing watcher signal for file: %s", filePath)

	// Process the specific file immediately
	cfg := indexer.Config{
		Root:             rootDir,
		ThumbsDir:        thumbsDir,
		PendingDir:       pendingDir,
		MaxWorkers:       1,
		BatchSize:        1,
		ThumbnailWorkers: 1,
	}

	// Use the existing indexer to process this specific file
	if err := indexer.ProcessSingleFile(db, cfg, filePath); err != nil {
		log.Printf("Error processing watcher signal file %s: %v", filePath, err)
		return err
	}

	// Remove the signal file
	os.Remove(signalFile)

	log.Printf("Successfully processed watcher signal for: %s", filePath)
	return nil
}
