// internal/http/routes.go
package httpx

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/daltay15/security-camera-ui/api/config"
	"github.com/daltay15/security-camera-ui/api/internal"
	"github.com/daltay15/security-camera-ui/api/internal/indexer"
	"github.com/gin-gonic/gin"
)

const (
	defaultLimit = 100
	maxLimit     = 500
)

// PreparedStatements holds cached prepared statements for better performance
type PreparedStatements struct {
	GetThumbPath   *sql.Stmt
	GetStreamPath  *sql.Stmt
	GetEvents      *sql.Stmt
	GetEventsCount *sql.Stmt
	UpdateReviewed *sql.Stmt
	mu             sync.RWMutex
}

var (
	preparedStmts *PreparedStatements
	once          sync.Once
)

// InitPreparedStatements initializes prepared statements for better performance
func InitPreparedStatements(db *sql.DB) error {
	var err error
	once.Do(func() {
		preparedStmts = &PreparedStatements{}

		preparedStmts.GetThumbPath, err = db.Prepare("SELECT COALESCE(jpg_path, path) FROM events WHERE id=?")
		if err != nil {
			return
		}

		preparedStmts.GetStreamPath, err = db.Prepare("SELECT COALESCE(video_path, path) FROM events WHERE id=?")
		if err != nil {
			return
		}

		preparedStmts.GetEvents, err = db.Prepare(`
			SELECT id,camera,path,jpg_path,sheet_path,start_ts,duration_ms,size_bytes,reviewed,tags,created_at
			FROM events
			WHERE camera=? AND start_ts>=? AND start_ts<=? AND reviewed=? AND size_bytes>=?
			ORDER BY start_ts DESC
			LIMIT ? OFFSET ?`)
		if err != nil {
			return
		}

		preparedStmts.GetEventsCount, err = db.Prepare("SELECT COUNT(*) FROM events WHERE camera=? AND start_ts>=? AND start_ts<=? AND reviewed=? AND size_bytes>=?")
		if err != nil {
			return
		}

		preparedStmts.UpdateReviewed, err = db.Prepare("UPDATE events SET reviewed=1 WHERE id=?")
		if err != nil {
			return
		}
	})
	return err
}

func Routes(r *gin.Engine, db *sql.DB) {
	// Initialize prepared statements for better performance
	if err := InitPreparedStatements(db); err != nil {
		panic("Failed to initialize prepared statements: " + err.Error())
	}

	// Initialize file-based GPU detection client
	gpuClient := internal.NewFileBasedGPUClient("Z:\\Cameras\\GPU_Processing")

	// Initialize detection processor
	detectionProcessor := internal.NewDetectionProcessor(db, config.PendingDir, config.ProcessingDir, config.CompletedDir, config.FailedDir, config.TelegramURL)

	// Initialize file copier for status endpoint
	r.GET("/events", func(c *gin.Context) {
		camera := c.Query("camera")
		from := c.Query("from")
		to := c.Query("to")
		minSizeStr := c.DefaultQuery("minSize", "0")

		limit, _ := strconv.Atoi(c.DefaultQuery("limit", strconv.Itoa(defaultLimit)))
		if limit <= 0 {
			limit = defaultLimit
		}
		if limit > maxLimit {
			limit = maxLimit
		}

		offset, _ := strconv.Atoi(c.DefaultQuery("offset", "0"))
		if offset < 0 {
			offset = 0
		}

		where := []string{"1=1"}
		args := []any{}

		if camera != "" {
			where = append(where, "e.camera=?")
			args = append(args, camera)
		}
		if from != "" {
			where = append(where, "e.start_ts>=?")
			args = append(args, from)
		}
		if to != "" {
			where = append(where, "e.start_ts<=?")
			args = append(args, to)
		}

		minSize, _ := strconv.ParseInt(minSizeStr, 10, 64)
		where = append(where, "e.size_bytes>=?")
		args = append(args, minSize)

		whereSQL := strings.Join(where, " AND ")

		// total count (for pagination UI) - only count raw events (not DETECTION events)
		var total int64
		countArgs := append([]any{}, args...)
		countQuery := "SELECT COUNT(*) FROM events e WHERE " + whereSQL + " AND e.camera != 'DETECTION'"
		if err := db.QueryRow(countQuery, countArgs...).Scan(&total); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		// Debug: log the query and total count
		fmt.Printf("DEBUG: Count query: %s\n", countQuery)
		fmt.Printf("DEBUG: Count args: %v\n", countArgs)
		fmt.Printf("DEBUG: Total events found: %d\n", total)

		// Debug: check if there are any events at all
		var totalEvents int64
		if err := db.QueryRow("SELECT COUNT(*) FROM events").Scan(&totalEvents); err != nil {
			fmt.Printf("DEBUG: Error counting all events: %v\n", err)
		} else {
			fmt.Printf("DEBUG: Total events in database: %d\n", totalEvents)
		}

		// Debug: check if there are any non-DETECTION events
		var nonDetectionEvents int64
		if err := db.QueryRow("SELECT COUNT(*) FROM events WHERE camera != 'DETECTION'").Scan(&nonDetectionEvents); err != nil {
			fmt.Printf("DEBUG: Error counting non-detection events: %v\n", err)
		} else {
			fmt.Printf("DEBUG: Non-detection events: %d\n", nonDetectionEvents)
		}

		// page query - only get raw events (base images and videos)
		pageArgs := append([]any{}, args...)
		pageArgs = append(pageArgs, limit, offset)

		q := `
			SELECT e.id, e.camera, e.path, e.jpg_path, e.sheet_path, e.start_ts, e.duration_ms, e.size_bytes, e.reviewed, e.tags, e.created_at, e.detection_data, e.detection_updated, e.video_path, e.detection_path
			FROM events e
			WHERE ` + whereSQL + `
			AND e.camera != 'DETECTION'
			ORDER BY e.start_ts DESC
			LIMIT ? OFFSET ?
		`

		// Debug: log the query and args
		fmt.Printf("DEBUG: Main query: %s\n", q)
		fmt.Printf("DEBUG: Query args: %v\n", pageArgs)

		rows, err := db.Query(q, pageArgs...)
		if err != nil {
			fmt.Printf("DEBUG: Query error: %v\n", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		defer rows.Close()

		items := make([]map[string]any, 0, limit)
		rowCount := 0
		for rows.Next() {
			rowCount++
			var id, camera, path string
			var jpg, sheet *string
			var startTs, durationMs, sizeBytes int64
			var reviewed int
			var tags, createdAt string
			var detectionData *string
			var detectionUpdated int64
			var videoPath, detectionPath *string

			if err := rows.Scan(
				&id, &camera, &path, &jpg, &sheet, &startTs,
				&durationMs, &sizeBytes, &reviewed, &tags, &createdAt, &detectionData, &detectionUpdated, &videoPath, &detectionPath,
			); err != nil {
				fmt.Printf("DEBUG: Row scan error: %v\n", err)
				continue
			}

			fmt.Printf("DEBUG: Processed row %d: id=%s, camera=%s, path=%s\n", rowCount, id, camera, path)

			e := map[string]any{
				"id":         id,
				"camera":     camera,
				"path":       path,
				"startTs":    startTs,
				"durationMs": durationMs,
				"sizeBytes":  sizeBytes,
				"reviewed":   reviewed,
				"tags":       tags,
				"createdAt":  createdAt,
			}

			if jpg != nil {
				e["jpgPath"] = *jpg
			}
			if sheet != nil {
				e["sheetPath"] = *sheet
			}
			if videoPath != nil {
				e["videoPath"] = *videoPath
			}

			items = append(items, e)
		}

		nextOffset := offset + len(items)
		if int64(nextOffset) >= total {
			c.JSON(200, gin.H{
				"items": items, "limit": limit, "offset": offset,
				"total": total, "nextOffset": nil,
			})
			return
		}
		c.JSON(200, gin.H{
			"items": items, "limit": limit, "offset": offset,
			"total": total, "nextOffset": nextOffset,
		})
	})

	r.GET("/stream/:id", func(c *gin.Context) {
		id := c.Param("id")
		var path string

		// Check if a specific path is provided as query parameter
		if customPath := c.Query("path"); customPath != "" {
			path = customPath
		} else {
			// Use prepared statement for better performance
			preparedStmts.mu.RLock()
			err := preparedStmts.GetStreamPath.QueryRow(id).Scan(&path)
			preparedStmts.mu.RUnlock()

			if err != nil {
				fmt.Printf("Stream endpoint: Event ID %s not found in database: %v\n", id, err)
				c.Status(http.StatusNotFound)
				return
			}
		}

		// Check if file actually exists
		if _, err := os.Stat(path); os.IsNotExist(err) {
			fmt.Printf("Stream endpoint: File not found for ID %s: %s (error: %v)\n", id, path, err)
			c.Status(http.StatusNotFound)
			return
		}

		// Add caching headers for video streams
		c.Header("Cache-Control", "public, max-age=3600") // 1 hour cache for videos
		c.Header("Accept-Ranges", "bytes")

		// Check if this is a range request (video seeking) or just metadata
		rangeHeader := c.GetHeader("Range")
		if rangeHeader == "" {
			// No range request - serve the full file for initial load
			http.ServeFile(c.Writer, c.Request, path)
			return
		}

		ServeFileRange(c.Writer, c.Request, path)
	})

	r.GET("/thumb/:id", func(c *gin.Context) {
		id := c.Param("id")
		var p string

		// Use prepared statement for better performance
		preparedStmts.mu.RLock()
		err := preparedStmts.GetThumbPath.QueryRow(id).Scan(&p)
		preparedStmts.mu.RUnlock()

		if err != nil {
			c.Status(http.StatusNotFound)
			return
		}

		// Check if file actually exists
		if _, err := os.Stat(p); os.IsNotExist(err) {
			c.Status(http.StatusNotFound)
			return
		}

		// Add caching headers for better performance
		c.Header("Cache-Control", "public, max-age=31536000") // 1 year cache
		c.Header("ETag", `"`+id+`"`)

		// Check if client has cached version
		if c.GetHeader("If-None-Match") == `"`+id+`"` {
			c.Status(http.StatusNotModified)
			return
		}

		http.ServeFile(c.Writer, c.Request, p)
	})

	// Route to serve annotated detection images
	r.GET("/detection/:id", func(c *gin.Context) {
		id := c.Param("id")

		// Check if this is a detection event ID (camera = 'DETECTION')
		var detectionPath string

		// First check if this ID is a detection event itself
		err := db.QueryRow("SELECT path FROM events WHERE id = ? AND camera = 'DETECTION'", id).Scan(&detectionPath)
		if err != nil {
			// If not a detection event, look for detection event linked to this original event
			err = db.QueryRow(`
				SELECT path FROM events 
				WHERE camera = 'DETECTION' 
				AND tags LIKE ? 
				ORDER BY created_at DESC 
				LIMIT 1
			`, "detection_for_event_"+id).Scan(&detectionPath)

			if err != nil {
				// Fallback: try to get detection data from the original event
				var detectionData string
				err = db.QueryRow("SELECT detection_data FROM events WHERE id = ? AND detection_data IS NOT NULL", id).Scan(&detectionData)
				if err != nil {
					c.Status(http.StatusNotFound)
					return
				}

				// Parse detection data to get annotated image path
				var detectionFile struct {
					AnnotatedImage string `json:"annotated_image"`
				}
				if err := json.Unmarshal([]byte(detectionData), &detectionFile); err != nil {
					c.Status(http.StatusNotFound)
					return
				}

				if detectionFile.AnnotatedImage == "" {
					c.Status(http.StatusNotFound)
					return
				}

				// Construct the full path to the annotated image
				detectionPath = filepath.Join("/mnt/nas/pool/Cameras/GPU_Processing/completed", detectionFile.AnnotatedImage)
			}
		}

		// Check if file actually exists
		if _, err := os.Stat(detectionPath); os.IsNotExist(err) {
			// Log the attempted path for debugging
			fmt.Printf("Detection image not found: %s\n", detectionPath)
			c.Status(http.StatusNotFound)
			return
		}

		// Add caching headers for better performance
		c.Header("Cache-Control", "public, max-age=3600") // 1 hour cache for detection images
		c.Header("ETag", `"detection_`+id+`"`)

		// Check if client has cached version
		if c.GetHeader("If-None-Match") == `"detection_`+id+`"` {
			c.Status(http.StatusNotModified)
			return
		}

		http.ServeFile(c.Writer, c.Request, detectionPath)
	})

	r.POST("/events/:id/reviewed", func(c *gin.Context) {
		id := c.Param("id")

		// Use prepared statement for better performance
		preparedStmts.mu.RLock()
		_, err := preparedStmts.UpdateReviewed.Exec(id)
		preparedStmts.mu.RUnlock()

		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.Status(http.StatusNoContent)
	})

	// Motion detection stats endpoint
	r.GET("/stats/motion", func(c *gin.Context) {
		days := c.DefaultQuery("days", "30")
		camera := c.Query("camera")

		daysInt, err := strconv.Atoi(days)
		if err != nil || daysInt <= 0 {
			daysInt = 30
		}
		if daysInt > 365 {
			daysInt = 365
		}

		// Calculate start date
		startTime := time.Now().AddDate(0, 0, -daysInt).Unix()

		where := []string{"start_ts >= ?"}
		args := []any{startTime}

		if camera != "" {
			where = append(where, "camera = ?")
			args = append(args, camera)
		}

		whereSQL := strings.Join(where, " AND ")

		// Get daily motion detection counts
		query := `
			SELECT 
				DATE(datetime(start_ts, 'unixepoch')) as date,
				COUNT(*) as count,
				COUNT(CASE WHEN path LIKE '%.mp4' THEN 1 END) as video_count,
				COUNT(CASE WHEN path LIKE '%.jpg' OR path LIKE '%.jpeg' THEN 1 END) as image_count
			FROM events 
			WHERE ` + whereSQL + `
			GROUP BY DATE(datetime(start_ts, 'unixepoch'))
			ORDER BY date DESC
		`

		rows, err := db.Query(query, args...)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		defer rows.Close()

		var dailyStats []map[string]any
		for rows.Next() {
			var date string
			var count, videoCount, imageCount int64

			if err := rows.Scan(&date, &count, &videoCount, &imageCount); err != nil {
				continue
			}

			dailyStats = append(dailyStats, map[string]any{
				"date":   date,
				"total":  count,
				"videos": videoCount,
				"images": imageCount,
			})
		}

		// Get camera breakdown
		cameraQuery := `
			SELECT 
				camera,
				COUNT(*) as count
			FROM events 
			WHERE ` + whereSQL + `
			GROUP BY camera
			ORDER BY count DESC
		`

		cameraRows, err := db.Query(cameraQuery, args...)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		defer cameraRows.Close()

		var cameraStats []map[string]any
		for cameraRows.Next() {
			var camera string
			var count int64

			if err := cameraRows.Scan(&camera, &count); err != nil {
				continue
			}

			cameraStats = append(cameraStats, map[string]any{
				"camera": camera,
				"count":  count,
			})
		}

		// Get hourly distribution for the last 7 days
		hourlyWhere := []string{"start_ts >= ?"}
		hourlyArgs := []any{time.Now().AddDate(0, 0, -7).Unix()}

		if camera != "" {
			hourlyWhere = append(hourlyWhere, "camera = ?")
			hourlyArgs = append(hourlyArgs, camera)
		}

		hourlyQuery := `
			SELECT 
				strftime('%H', datetime(start_ts, 'unixepoch')) as hour,
				COUNT(*) as count
			FROM events 
			WHERE ` + strings.Join(hourlyWhere, " AND ") + `
			GROUP BY strftime('%H', datetime(start_ts, 'unixepoch'))
			ORDER BY hour
		`

		hourlyRows, err := db.Query(hourlyQuery, hourlyArgs...)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		defer hourlyRows.Close()

		var hourlyStats []map[string]any
		for hourlyRows.Next() {
			var hour string
			var count int64

			if err := hourlyRows.Scan(&hour, &count); err != nil {
				continue
			}

			hourlyStats = append(hourlyStats, map[string]any{
				"hour":  hour,
				"count": count,
			})
		}

		c.JSON(200, gin.H{
			"dailyStats":  dailyStats,
			"cameraStats": cameraStats,
			"hourlyStats": hourlyStats,
			"period":      daysInt,
			"camera":      camera,
		})
	})

	// Database cleanup endpoint for deleted files
	r.POST("/cleanup", func(c *gin.Context) {
		// Find all events where the file no longer exists
		rows, err := db.Query("SELECT id, path, jpg_path, sheet_path FROM events")
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		defer rows.Close()

		var orphanedIDs []int64
		for rows.Next() {
			var id int64
			var path, jpgPath, sheetPath string
			if err := rows.Scan(&id, &path, &jpgPath, &sheetPath); err != nil {
				continue
			}

			// Check if main file exists
			if _, err := os.Stat(path); os.IsNotExist(err) {
				orphanedIDs = append(orphanedIDs, id)
				continue
			}

			// Check if thumbnail files exist (if they're separate files)
			if jpgPath != "" && jpgPath != path {
				if _, err := os.Stat(jpgPath); os.IsNotExist(err) {
					orphanedIDs = append(orphanedIDs, id)
					continue
				}
			}
			if sheetPath != "" && sheetPath != path {
				if _, err := os.Stat(sheetPath); os.IsNotExist(err) {
					orphanedIDs = append(orphanedIDs, id)
					continue
				}
			}
		}

		// Delete orphaned entries
		if len(orphanedIDs) > 0 {
			placeholders := strings.Repeat("?,", len(orphanedIDs)-1) + "?"
			query := "DELETE FROM events WHERE id IN (" + placeholders + ")"

			args := make([]any, len(orphanedIDs))
			for i, id := range orphanedIDs {
				args[i] = id
			}

			_, err := db.Exec(query, args...)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}
		}

		c.JSON(200, gin.H{
			"cleaned": len(orphanedIDs),
			"message": "Database cleanup completed",
		})
	})

	// GPU Detection Service Endpoints
	r.GET("/gpu/status", func(c *gin.Context) {
		status, err := gpuClient.GetStatus()
		if err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, status)
	})

	r.GET("/gpu/health", func(c *gin.Context) {
		if err := gpuClient.HealthCheck(); err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{"status": "unhealthy", "error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, gin.H{"status": "healthy"})
	})

	r.POST("/gpu/detect/image", func(c *gin.Context) {
		var request internal.DetectionRequest
		if err := c.ShouldBindJSON(&request); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		response, err := gpuClient.DetectImage(request)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, response)
	})

	r.POST("/gpu/detect/video", func(c *gin.Context) {
		var request internal.DetectionRequest
		if err := c.ShouldBindJSON(&request); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		response, err := gpuClient.DetectVideo(request)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, response)
	})

	// Convenience endpoint to detect objects in an event by ID
	r.POST("/events/:id/detect", func(c *gin.Context) {
		eventID := c.Param("id")

		// Get event details from database
		var path, fileType string
		err := db.QueryRow("SELECT path FROM events WHERE id = ?", eventID).Scan(&path)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "Event not found"})
			return
		}

		// Determine file type from extension
		if strings.HasSuffix(strings.ToLower(path), ".mp4") {
			fileType = "video"
		} else {
			fileType = "image"
		}

		// Create detection request
		request := internal.DetectionRequest{
			FilePath: path,
			FileType: fileType,
			Options: map[string]interface{}{
				"confidence_threshold": 0.45,
				"classes":              []string{"person", "car", "truck", "bus", "bicycle", "motorcycle"},
			},
		}

		// Perform detection based on file type
		var response *internal.DetectionResponse
		if fileType == "video" {
			response, err = gpuClient.DetectVideo(request)
		} else {
			response, err = gpuClient.DetectImage(request)
		}

		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, response)
	})

	// Detection API endpoints
	r.GET("/detections/stats", func(c *gin.Context) {
		daysStr := c.DefaultQuery("days", "30")
		days, err := strconv.Atoi(daysStr)
		if err != nil || days <= 0 {
			days = 30
		}
		if days > 365 {
			days = 365
		}

		stats, err := detectionProcessor.GetDetectionStats(days)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, stats)
	})

	r.GET("/detections/events/:id", func(c *gin.Context) {
		eventID := c.Param("id")
		eventIDInt, err := strconv.ParseInt(eventID, 10, 64)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid event ID"})
			return
		}

		detections, err := detectionProcessor.GetDetectionsForEvent(eventIDInt)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{"detections": detections})
	})

	r.GET("/detections/recent", func(c *gin.Context) {
		limitStr := c.DefaultQuery("limit", "50")
		limit, err := strconv.Atoi(limitStr)
		if err != nil || limit <= 0 {
			limit = 50
		}
		if limit > 200 {
			limit = 200
		}

		detections, err := detectionProcessor.GetRecentDetections(limit)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{"detections": detections})
	})

	// Test endpoint to manually trigger detection scanning
	r.POST("/test/scan-detections", func(c *gin.Context) {
		detectionDir := "/mnt/nas/pool/Cameras/GPU_Processing/completed"
		fmt.Printf("Manual detection scan triggered for: %s\n", detectionDir)

		// Check if directory exists
		if _, err := os.Stat(detectionDir); os.IsNotExist(err) {
			fmt.Printf("Directory does not exist: %s\n", detectionDir)
			c.JSON(http.StatusNotFound, gin.H{"error": "Directory does not exist: " + detectionDir})
			return
		}

		cfg := indexer.Config{
			DetectionDir:    detectionDir,
			IndexDetections: true,
		}

		if err := indexer.ScanDetectionImages(db, cfg); err != nil {
			fmt.Printf("Manual detection scan error: %v\n", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		var count int64
		_ = db.QueryRow("SELECT COUNT(*) FROM events WHERE camera = 'DETECTION'").Scan(&count)
		fmt.Printf("Manual detection scan completed, count: %d\n", count)

		c.JSON(http.StatusOK, gin.H{
			"message": "Detection scan completed",
			"count":   count,
		})
	})

	// Test endpoint to manually trigger detection processing
	r.POST("/test/process-detections", func(c *gin.Context) {
		fmt.Printf("Manual detection processing triggered\n")

		detectionProcessor := internal.NewDetectionProcessor(db, config.PendingDir, config.ProcessingDir, config.CompletedDir, config.FailedDir, config.TelegramURL)

		if err := detectionProcessor.ProcessCompletedFiles(); err != nil {
			fmt.Printf("Manual detection processing error: %v\n", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		var count int64
		_ = db.QueryRow("SELECT COUNT(*) FROM events WHERE detection_data IS NOT NULL").Scan(&count)
		fmt.Printf("Manual detection processing completed, events with detection data: %d\n", count)

		c.JSON(http.StatusOK, gin.H{
			"message":                 "Detection processing completed",
			"eventsWithDetectionData": count,
		})
	})

	// Enhanced events endpoint with detection data
	r.GET("/events/enhanced", func(c *gin.Context) {
		camera := c.Query("camera")
		from := c.Query("from")
		to := c.Query("to")
		minSizeStr := c.DefaultQuery("minSize", "0")
		reviewed := c.Query("reviewed")
		detectionType := c.Query("detectionType")
		minConfidenceStr := c.DefaultQuery("minConfidence", "0")

		limit, _ := strconv.Atoi(c.DefaultQuery("limit", strconv.Itoa(defaultLimit)))
		if limit <= 0 {
			limit = defaultLimit
		}
		if limit > maxLimit {
			limit = maxLimit
		}

		offset, _ := strconv.Atoi(c.DefaultQuery("offset", "0"))
		if offset < 0 {
			offset = 0
		}

		where := []string{"1=1"}
		args := []any{}

		if camera != "" {
			where = append(where, "e.camera=?")
			args = append(args, camera)
		}
		if from != "" {
			where = append(where, "e.start_ts>=?")
			args = append(args, from)
		}
		if to != "" {
			where = append(where, "e.start_ts<=?")
			args = append(args, to)
		}
		if reviewed != "" {
			where = append(where, "e.reviewed=?")
			args = append(args, reviewed)
		}

		minSize, _ := strconv.ParseInt(minSizeStr, 10, 64)
		where = append(where, "e.size_bytes>=?")
		args = append(args, minSize)

		// Add detection filters
		if detectionType != "" {
			where = append(where, "EXISTS (SELECT 1 FROM detections d WHERE d.event_id = e.id AND d.detection_type = ?)")
			args = append(args, detectionType)
		}

		minConfidence, _ := strconv.ParseFloat(minConfidenceStr, 64)
		if minConfidence > 0 {
			where = append(where, "EXISTS (SELECT 1 FROM detections d WHERE d.event_id = e.id AND d.confidence >= ?)")
			args = append(args, minConfidence)
		}

		// Add file type filtering
		fileType := c.Query("fileType")
		if fileType == "jpg" {
			where = append(where, "(e.path LIKE '%.jpg' OR e.path LIKE '%.jpeg')")
		}

		whereSQL := strings.Join(where, " AND ")

		// Get total count
		var total int64
		countArgs := append([]any{}, args...)
		if err := db.QueryRow("SELECT COUNT(*) FROM events e WHERE "+whereSQL+" AND e.camera != 'DETECTION'", countArgs...).Scan(&total); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		// Get events with detection data
		pageArgs := append([]any{}, args...)
		pageArgs = append(pageArgs, limit, offset)

		q := `
			SELECT e.id, e.camera, e.path, e.jpg_path, e.sheet_path, e.start_ts, e.duration_ms, e.size_bytes, e.reviewed, e.tags, e.created_at, e.detection_data, e.detection_updated, e.original_event_id, e.detection_image_path, e.video_path,
			       -- Get detection data from related detection events
			       (SELECT d.detection_data FROM events d WHERE d.camera = 'DETECTION' AND d.original_event_id = e.id LIMIT 1) as correlated_detection_data,
			       (SELECT d.detection_updated FROM events d WHERE d.camera = 'DETECTION' AND d.original_event_id = e.id LIMIT 1) as correlated_detection_updated,
			       (SELECT d.path FROM events d WHERE d.camera = 'DETECTION' AND d.original_event_id = e.id LIMIT 1) as correlated_detection_path
			FROM events e
			WHERE ` + whereSQL + `
			AND e.camera != 'DETECTION'
			ORDER BY e.start_ts DESC
			LIMIT ? OFFSET ?
		`
		rows, err := db.Query(q, pageArgs...)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		defer rows.Close()

		items := make([]map[string]any, 0, limit)
		for rows.Next() {
			var id, camera, path string
			var jpg, sheet, detectionData *string
			var startTs, durationMs, sizeBytes int64
			var reviewed int
			var tags, createdAt string
			var detectionUpdated int64
			var originalEventID *int64
			var detectionImagePath, videoPath *string

			if err := rows.Scan(
				&id, &camera, &path, &jpg, &sheet, &startTs,
				&durationMs, &sizeBytes, &reviewed, &tags, &createdAt, &detectionData, &detectionUpdated, &originalEventID, &detectionImagePath, &videoPath,
			); err != nil {
				continue
			}

			e := map[string]any{
				"id":         id,
				"camera":     camera,
				"path":       path,
				"startTs":    startTs,
				"durationMs": durationMs,
				"sizeBytes":  sizeBytes,
				"reviewed":   reviewed,
				"tags":       tags,
				"createdAt":  createdAt,
			}

			if jpg != nil {
				e["jpgPath"] = *jpg
			}
			if sheet != nil {
				e["sheetPath"] = *sheet
			}
			if detectionData != nil {
				e["detectionData"] = *detectionData
				e["detectionUpdated"] = detectionUpdated
			}
			if originalEventID != nil {
				e["originalEventId"] = *originalEventID
			}
			if detectionImagePath != nil {
				e["detectionImagePath"] = *detectionImagePath
			}
			if videoPath != nil {
				e["videoPath"] = *videoPath
			}

			// Get detections for this event
			eventIDInt, _ := strconv.ParseInt(id, 10, 64)
			detections, err := detectionProcessor.GetDetectionsForEvent(eventIDInt)
			if err == nil {
				e["detections"] = detections
			}

			items = append(items, e)
		}

		nextOffset := offset + len(items)
		if int64(nextOffset) >= total {
			c.JSON(200, gin.H{
				"items": items, "limit": limit, "offset": offset,
				"total": total, "nextOffset": nil,
			})
			return
		}
		c.JSON(200, gin.H{
			"items": items, "limit": limit, "offset": offset,
			"total": total, "nextOffset": nextOffset,
		})
	})

	// Metadata-based media correlation endpoint - returns events with their metadata
	r.GET("/events/metadata", func(c *gin.Context) {
		camera := c.Query("camera")
		from := c.Query("from")
		to := c.Query("to")
		minSizeStr := c.DefaultQuery("minSize", "0")
		reviewed := c.Query("reviewed")
		detectionType := c.Query("detectionType")
		minConfidenceStr := c.DefaultQuery("minConfidence", "0")

		limit, _ := strconv.Atoi(c.DefaultQuery("limit", strconv.Itoa(defaultLimit)))
		if limit <= 0 {
			limit = defaultLimit
		}
		if limit > maxLimit {
			limit = maxLimit
		}

		offset, _ := strconv.Atoi(c.DefaultQuery("offset", "0"))
		if offset < 0 {
			offset = 0
		}

		where := []string{"1=1"}
		args := []any{}

		if camera != "" {
			where = append(where, "e.camera=?")
			args = append(args, camera)
		}
		if from != "" {
			where = append(where, "e.start_ts>=?")
			args = append(args, from)
		}
		if to != "" {
			where = append(where, "e.start_ts<=?")
			args = append(args, to)
		}
		if reviewed != "" {
			where = append(where, "e.reviewed=?")
			args = append(args, reviewed)
		}

		minSize, _ := strconv.ParseInt(minSizeStr, 10, 64)
		where = append(where, "e.size_bytes>=?")
		args = append(args, minSize)

		// Add detection filters
		if detectionType != "" {
			where = append(where, "EXISTS (SELECT 1 FROM detections d WHERE d.event_id = e.id AND d.detection_type = ?)")
			args = append(args, detectionType)
		}

		minConfidence, _ := strconv.ParseFloat(minConfidenceStr, 64)
		if minConfidence > 0 {
			where = append(where, "EXISTS (SELECT 1 FROM detections d WHERE d.event_id = e.id AND d.confidence >= ?)")
			args = append(args, minConfidence)
		}

		// Add file type filtering
		fileType := c.Query("fileType")
		if fileType == "jpg" {
			where = append(where, "(e.path LIKE '%.jpg' OR e.path LIKE '%.jpeg')")
		}

		whereSQL := strings.Join(where, " AND ")

		// Get total count
		var total int64
		countArgs := append([]any{}, args...)
		if err := db.QueryRow("SELECT COUNT(*) FROM events e WHERE "+whereSQL+" AND e.camera != 'DETECTION'", countArgs...).Scan(&total); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		// Get events with correlated media
		pageArgs := append([]any{}, args...)
		pageArgs = append(pageArgs, limit, offset)

		q := `
			SELECT e.id, e.camera, e.path, e.jpg_path, e.sheet_path, e.start_ts, e.duration_ms, e.size_bytes, e.reviewed, e.tags, e.created_at, e.detection_data, e.detection_updated, e.video_path, e.detection_path
			FROM events e
			WHERE ` + whereSQL + `
			AND e.camera != 'DETECTION'
			-- Only show base image events (JPG/JPEG) as primary records
			AND (e.path LIKE '%.jpg' OR e.path LIKE '%.jpeg')
			ORDER BY e.start_ts DESC
			LIMIT ? OFFSET ?
		`
		rows, err := db.Query(q, pageArgs...)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		defer rows.Close()

		items := make([]map[string]any, 0, limit)
		for rows.Next() {
			var id, camera, path string
			var jpg, sheet, detectionData *string
			var startTs, durationMs, sizeBytes int64
			var reviewed int
			var tags, createdAt string
			var detectionUpdated int64
			var originalEventID *int64
			var detectionImagePath, videoPath *string

			if err := rows.Scan(
				&id, &camera, &path, &jpg, &sheet, &startTs,
				&durationMs, &sizeBytes, &reviewed, &tags, &createdAt, &detectionData, &detectionUpdated, &originalEventID, &detectionImagePath, &videoPath,
			); err != nil {
				continue
			}

			e := map[string]any{
				"id":         id,
				"camera":     camera,
				"path":       path,
				"startTs":    startTs,
				"durationMs": durationMs,
				"sizeBytes":  sizeBytes,
				"reviewed":   reviewed,
				"tags":       tags,
				"createdAt":  createdAt,
			}

			if jpg != nil {
				e["jpgPath"] = *jpg
			}
			if sheet != nil {
				e["sheetPath"] = *sheet
			}
			if detectionData != nil {
				e["detectionData"] = *detectionData
				e["detectionUpdated"] = detectionUpdated
			}
			if originalEventID != nil {
				e["originalEventId"] = *originalEventID
			}
			if detectionImagePath != nil {
				e["detectionImagePath"] = *detectionImagePath
			}
			if videoPath != nil {
				e["videoPath"] = *videoPath
			}

			// Get detections for this event
			eventIDInt, _ := strconv.ParseInt(id, 10, 64)
			detections, err := detectionProcessor.GetDetectionsForEvent(eventIDInt)
			if err == nil {
				e["detections"] = detections
			}

			items = append(items, e)
		}

		nextOffset := offset + len(items)
		if int64(nextOffset) >= total {
			c.JSON(200, gin.H{
				"items": items, "limit": limit, "offset": offset,
				"total": total, "nextOffset": nil,
			})
			return
		}
		c.JSON(200, gin.H{
			"items": items, "limit": limit, "offset": offset,
			"total": total, "nextOffset": nextOffset,
		})
	})

	// Simplified metadata-based endpoint for base images with video/detection metadata
	r.GET("/events/correlated", func(c *gin.Context) {
		camera := c.Query("camera")
		from := c.Query("from")
		to := c.Query("to")
		minSizeStr := c.DefaultQuery("minSize", "0")
		reviewed := c.Query("reviewed")
		detectionType := c.Query("detectionType")
		minConfidenceStr := c.DefaultQuery("minConfidence", "0")

		limit, _ := strconv.Atoi(c.DefaultQuery("limit", strconv.Itoa(defaultLimit)))
		if limit <= 0 {
			limit = defaultLimit
		}
		if limit > maxLimit {
			limit = maxLimit
		}

		offset, _ := strconv.Atoi(c.DefaultQuery("offset", "0"))
		if offset < 0 {
			offset = 0
		}

		where := []string{"1=1"}
		args := []any{}

		if camera != "" {
			where = append(where, "e.camera=?")
			args = append(args, camera)
		}
		if from != "" {
			where = append(where, "e.start_ts>=?")
			args = append(args, from)
		}
		if to != "" {
			where = append(where, "e.start_ts<=?")
			args = append(args, to)
		}
		if reviewed != "" {
			where = append(where, "e.reviewed=?")
			args = append(args, reviewed)
		}

		minSize, _ := strconv.ParseInt(minSizeStr, 10, 64)
		where = append(where, "e.size_bytes>=?")
		args = append(args, minSize)

		// Check if we have detection filters
		hasDetectionFilters := false
		if detectionType != "" {
			hasDetectionFilters = true
		}
		if minConfidenceStr != "" {
			minConfidence, _ := strconv.ParseFloat(minConfidenceStr, 64)
			if minConfidence > 0 {
				hasDetectionFilters = true
			}
		}

		whereSQL := strings.Join(where, " AND ")

		// Get total count - use different query based on whether we have detection filters
		var total int64
		var countQuery string
		countArgs := append([]any{}, args...)

		if hasDetectionFilters {
			// Use JOIN for detection filters
			detectionWhere := []string{}
			if detectionType != "" {
				detectionWhere = append(detectionWhere, "d.detection_type = ?")
				countArgs = append(countArgs, detectionType)
			}
			if minConfidenceStr != "" {
				minConfidence, _ := strconv.ParseFloat(minConfidenceStr, 64)
				if minConfidence > 0 {
					detectionWhere = append(detectionWhere, "d.confidence >= ?")
					countArgs = append(countArgs, minConfidence)
				}
			}

			detectionWhereSQL := ""
			if len(detectionWhere) > 0 {
				detectionWhereSQL = " AND " + strings.Join(detectionWhere, " AND ")
			}

			countQuery = `
				SELECT COUNT(DISTINCT e.id) 
				FROM events e 
				INNER JOIN detections d ON d.event_id = e.id 
				WHERE ` + whereSQL + ` 
				AND e.camera != 'DETECTION' 
				AND (e.path LIKE '%.jpg' OR e.path LIKE '%.jpeg')` + detectionWhereSQL
		} else {
			// Simple query without detection filters
			countQuery = "SELECT COUNT(*) FROM events e WHERE " + whereSQL + " AND e.camera != 'DETECTION' AND (e.path LIKE '%.jpg' OR e.path LIKE '%.jpeg')"
		}

		if err := db.QueryRow(countQuery, countArgs...).Scan(&total); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		// Get base image events with their metadata
		var q string
		var pageArgs []any

		if hasDetectionFilters {
			// Use JOIN for detection filters
			detectionWhere := []string{}
			detectionArgs := []any{}

			if detectionType != "" {
				detectionWhere = append(detectionWhere, "d.detection_type = ?")
				detectionArgs = append(detectionArgs, detectionType)
			}
			if minConfidenceStr != "" {
				minConfidence, _ := strconv.ParseFloat(minConfidenceStr, 64)
				if minConfidence > 0 {
					detectionWhere = append(detectionWhere, "d.confidence >= ?")
					detectionArgs = append(detectionArgs, minConfidence)
				}
			}

			detectionWhereSQL := ""
			if len(detectionWhere) > 0 {
				detectionWhereSQL = " AND " + strings.Join(detectionWhere, " AND ")
			}

			// Build args in correct order: base args + detection args + limit/offset
			pageArgs = append([]any{}, args...)
			pageArgs = append(pageArgs, detectionArgs...)
			pageArgs = append(pageArgs, limit, offset)

			q = `
				SELECT DISTINCT e.id, e.camera, e.path, e.jpg_path, e.sheet_path, e.start_ts, e.duration_ms, e.size_bytes, e.reviewed, e.tags, e.created_at, e.detection_data, e.detection_updated, e.video_path, e.detection_path
				FROM events e
				INNER JOIN detections d ON d.event_id = e.id
				WHERE ` + whereSQL + `
				AND e.camera != 'DETECTION'
				AND (e.path LIKE '%.jpg' OR e.path LIKE '%.jpeg')` + detectionWhereSQL + `
				ORDER BY e.start_ts DESC
				LIMIT ? OFFSET ?
			`
		} else {
			// Simple query without detection filters
			pageArgs = append([]any{}, args...)
			pageArgs = append(pageArgs, limit, offset)

			q = `
				SELECT e.id, e.camera, e.path, e.jpg_path, e.sheet_path, e.start_ts, e.duration_ms, e.size_bytes, e.reviewed, e.tags, e.created_at, e.detection_data, e.detection_updated, e.video_path, e.detection_path
				FROM events e
				WHERE ` + whereSQL + `
				AND e.camera != 'DETECTION'
				AND (e.path LIKE '%.jpg' OR e.path LIKE '%.jpeg')
				ORDER BY e.start_ts DESC
				LIMIT ? OFFSET ?
			`
		}

		rows, err := db.Query(q, pageArgs...)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		defer rows.Close()

		items := make([]map[string]any, 0, limit)
		eventIDs := make([]int64, 0, limit)

		// First pass: collect all events and their IDs
		for rows.Next() {
			var id, camera, path string
			var jpg, sheet, detectionData *string
			var startTs, durationMs, sizeBytes int64
			var reviewed int
			var tags, createdAt string
			var detectionUpdated int64
			var videoPath, detectionPath *string

			if err := rows.Scan(
				&id, &camera, &path, &jpg, &sheet, &startTs,
				&durationMs, &sizeBytes, &reviewed, &tags, &createdAt, &detectionData, &detectionUpdated, &videoPath, &detectionPath,
			); err != nil {
				continue
			}

			e := map[string]any{
				"id":         id,
				"camera":     camera,
				"path":       path,
				"startTs":    startTs,
				"durationMs": durationMs,
				"sizeBytes":  sizeBytes,
				"reviewed":   reviewed,
				"tags":       tags,
				"createdAt":  createdAt,
			}

			if jpg != nil {
				e["jpgPath"] = *jpg
			}
			if sheet != nil {
				e["sheetPath"] = *sheet
			}
			if detectionData != nil {
				e["detectionData"] = *detectionData
				e["detectionUpdated"] = detectionUpdated
			}
			if videoPath != nil {
				e["videoPath"] = *videoPath
			}
			if detectionPath != nil {
				e["detectionPath"] = *detectionPath
			}

			// Collect event ID for bulk detection query
			eventIDInt, _ := strconv.ParseInt(id, 10, 64)
			eventIDs = append(eventIDs, eventIDInt)

			items = append(items, e)
		}

		// Bulk fetch all detections for all events in a single query
		detectionsByEvent, err := detectionProcessor.GetDetectionsForEvents(eventIDs)
		if err != nil {
			// If bulk query fails, fall back to individual queries
			for i, e := range items {
				eventIDInt, _ := strconv.ParseInt(e["id"].(string), 10, 64)
				detections, err := detectionProcessor.GetDetectionsForEvent(eventIDInt)
				if err == nil {
					e["detections"] = detections
				}
				items[i] = e
			}
		} else {
			// Add detections to each event
			for i, e := range items {
				eventIDInt, _ := strconv.ParseInt(e["id"].(string), 10, 64)
				if detections, exists := detectionsByEvent[eventIDInt]; exists {
					e["detections"] = detections
				} else {
					e["detections"] = []any{} // Empty array if no detections
				}
				items[i] = e
			}
		}

		nextOffset := offset + len(items)
		if int64(nextOffset) >= total {
			c.JSON(200, gin.H{
				"items": items, "limit": limit, "offset": offset,
				"total": total, "nextOffset": nil,
			})
			return
		}
		c.JSON(200, gin.H{
			"items": items, "limit": limit, "offset": offset,
			"total": total, "nextOffset": nextOffset,
		})
	})

	// Camera list endpoint for dropdown
	r.GET("/cameras", func(c *gin.Context) {
		from := c.Query("from")
		to := c.Query("to")

		where := []string{"1=1"}
		args := []any{}

		if from != "" {
			where = append(where, "start_ts>=?")
			args = append(args, from)
		}
		if to != "" {
			where = append(where, "start_ts<=?")
			args = append(args, to)
		}

		whereSQL := strings.Join(where, " AND ")

		// Get camera counts
		query := `
			SELECT 
				camera,
				COUNT(*) as count
			FROM events 
			WHERE ` + whereSQL + `
			GROUP BY camera
			ORDER BY count DESC
		`

		rows, err := db.Query(query, args...)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		defer rows.Close()

		var cameraStats []map[string]any
		for rows.Next() {
			var camera string
			var count int64

			if err := rows.Scan(&camera, &count); err != nil {
				continue
			}

			cameraStats = append(cameraStats, map[string]any{
				"camera": camera,
				"count":  count,
			})
		}

		c.JSON(200, gin.H{
			"cameras": cameraStats,
		})
	})

	// Add endpoint to trigger video metadata updates
	r.POST("/update-video-metadata", func(c *gin.Context) {
		// This endpoint triggers the same logic as cmd/update_metadata
		// Get all base image events that don't have video_path
		rows, err := db.Query(`
			SELECT id, camera, path, start_ts 
			FROM events 
			WHERE camera != 'DETECTION' 
			AND (path LIKE '%.jpg' OR path LIKE '%.jpeg')
			AND video_path IS NULL
			ORDER BY start_ts DESC
			LIMIT 100
		`)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		defer rows.Close()

		updated := 0
		for rows.Next() {
			var baseID int64
			var camera, basePath string
			var startTs int64

			if err := rows.Scan(&baseID, &camera, &basePath, &startTs); err != nil {
				continue
			}

			// Look for related video file
			baseDir := filepath.Dir(basePath)
			baseName := strings.TrimSuffix(filepath.Base(basePath), filepath.Ext(basePath))

			// Try different video extensions
			videoExtensions := []string{".mp4", ".avi", ".mov", ".mkv"}
			var videoPath string

			for _, ext := range videoExtensions {
				videoFile := filepath.Join(baseDir, baseName+ext)
				if _, err := os.Stat(videoFile); err == nil {
					videoPath = videoFile
					break
				}
			}

			if videoPath != "" {
				// Update the database with video path
				_, err = db.Exec("UPDATE events SET video_path = ? WHERE id = ?", videoPath, baseID)
				if err == nil {
					updated++
				}
			}
		}

		c.JSON(200, gin.H{
			"message": "Video metadata update completed",
			"updated": updated,
		})
	})
}
