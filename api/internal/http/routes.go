// internal/http/routes.go
package httpx

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/daltay15/security-camera-ui/api/config"
	"github.com/daltay15/security-camera-ui/api/internal"
	"github.com/daltay15/security-camera-ui/api/internal/indexer"
	"github.com/daltay15/security-camera-ui/api/internal/telegram"
	"github.com/gin-gonic/gin"
)

const (
	defaultLimit = 100
	maxLimit     = 500
)

// formatBytes formats bytes into human-readable format (e.g., "30.03 GB")
func formatBytes(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	size := float64(bytes) / float64(div)
	units := []string{"KB", "MB", "GB", "TB", "PB"}
	if exp >= len(units) {
		return fmt.Sprintf("%.2f PB", float64(bytes)/float64(div*unit))
	}
	return fmt.Sprintf("%.2f %s", size, units[exp])
}

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

func Routes(r *gin.Engine, db *sql.DB, configManager *internal.ConfigManager) {
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
			internal.NotifyError("error", "http_routes", "Failed to count events", map[string]interface{}{
				"query": countQuery,
				"args":  countArgs,
				"error": err.Error(),
			})
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

	// Delete data by date range (day/week/month or explicit range)
	r.POST("/delete/range", func(c *gin.Context) {
		// Payload supports either explicit from/to (unix seconds) or a relative period
		var payload struct {
			From       *int64  `json:"from,omitempty"`
			To         *int64  `json:"to,omitempty"`
			Unit       string  `json:"unit,omitempty"`    // "day", "week", "month"
			Amount     int     `json:"amount,omitempty"`  // how many units to delete
			EndISO8601 *string `json:"end,omitempty"`     // optional ISO8601 end date for relative window
			Preview    bool    `json:"preview,omitempty"` // if true, don't delete, just return counts
		}

		if err := c.ShouldBindJSON(&payload); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		// Compute range
		var fromTs, toTs int64
		if payload.From != nil && payload.To != nil {
			fromTs = *payload.From
			toTs = *payload.To
		} else {
			// Relative window based on unit/amount
			if payload.Amount <= 0 {
				payload.Amount = 1
			}
			end := time.Now()
			hasExplicitEnd := false
			if payload.EndISO8601 != nil && *payload.EndISO8601 != "" {
				if t, err := time.Parse(time.RFC3339, *payload.EndISO8601); err == nil {
					end = t
					hasExplicitEnd = true
				}
			}
			start := end
			switch strings.ToLower(payload.Unit) {
			case "day", "days":
				if hasExplicitEnd && payload.Amount == 1 {
					// When deleting a specific day, the end date is the selected date at midnight UTC
					// Extract the date components from UTC (the date the user actually selected)
					year, month, day := end.UTC().Date()
					// Create start as beginning of selected day in local timezone
					start = time.Date(year, month, day, 0, 0, 0, 0, time.Local)
					// Create end as last second of selected day (23:59:59) to make it exclusive of next day
					end = time.Date(year, month, day, 23, 59, 59, 999999999, time.Local)
				} else {
					start = end.AddDate(0, 0, -payload.Amount)
				}
			case "week", "weeks":
				start = end.AddDate(0, 0, -7*payload.Amount)
			case "month", "months":
				start = end.AddDate(0, -payload.Amount, 0)
			default:
				c.JSON(http.StatusBadRequest, gin.H{"error": "Provide from/to or a valid unit: day|week|month"})
				return
			}
			fromTs = start.Unix()
			toTs = end.Unix()
		}

		// Collect base event candidates (exclude detection pseudo-camera rows)
		rows, err := db.Query(`
			SELECT id, path, COALESCE(jpg_path,''), COALESCE(sheet_path,''), COALESCE(video_path,''), COALESCE(detection_path,''), COALESCE(size_bytes, 0)
			FROM events
			WHERE start_ts >= ? AND start_ts <= ? AND camera != 'DETECTION'
		`, fromTs, toTs)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		defer rows.Close()

		type evt struct {
			id        int64
			p1        string
			p2        string
			p3        string
			p4        string
			p5        string
			sizeBytes int64
		}

		var baseEvents []evt
		var totalSizeBytes int64
		for rows.Next() {
			var e evt
			if err := rows.Scan(&e.id, &e.p1, &e.p2, &e.p3, &e.p4, &e.p5, &e.sizeBytes); err != nil {
				continue
			}
			baseEvents = append(baseEvents, e)
			totalSizeBytes += e.sizeBytes
		}

		// Build list of IDs for deletion (base events)
		ids := make([]int64, 0, len(baseEvents))
		for _, e := range baseEvents {
			ids = append(ids, e.id)
		}

		// Also collect related detection events for these base events
		var detectionEventIDs []int64
		if len(ids) > 0 {
			// Prefer explicit relation via original_event_id if present, else via tags pattern
			placeholders := strings.Repeat("?,", len(ids)-1) + "?"
			args := make([]any, len(ids))
			for i, id := range ids {
				args[i] = id
			}
			detRows, err := db.Query(`
				SELECT id FROM events WHERE camera='DETECTION' AND original_event_id IN (`+placeholders+`)
			`, args...)
			if err == nil {
				for detRows.Next() {
					var did int64
					if err := detRows.Scan(&did); err == nil {
						detectionEventIDs = append(detectionEventIDs, did)
					}
				}
				detRows.Close()
			}
			// Fallback via tags if original_event_id relation is not used
			if len(detectionEventIDs) == 0 {
				var tagArgs []any
				var tagConds []string
				for _, id := range ids {
					tagConds = append(tagConds, "tags LIKE ?")
					tagArgs = append(tagArgs, fmt.Sprintf("%%detection_for_event_%d%%", id))
				}
				q := "SELECT id FROM events WHERE camera='DETECTION' AND (" + strings.Join(tagConds, " OR ") + ")"
				detRows2, err2 := db.Query(q, tagArgs...)
				if err2 == nil {
					for detRows2.Next() {
						var did int64
						if err := detRows2.Scan(&did); err == nil {
							detectionEventIDs = append(detectionEventIDs, did)
						}
					}
					detRows2.Close()
				}
			}
		}

		// Group events by day folder for efficient deletion
		dayFolders := make(map[string]bool) // track unique day folders
		for _, e := range baseEvents {
			if e.p1 != "" {
				// Extract day folder: parent directory of the file
				dayFolder := filepath.Dir(e.p1)
				dayFolders[dayFolder] = true
			}
		}

		if payload.Preview {
			// Provide a list of day folders that would be deleted
			sampleFolders := make([]string, 0, len(dayFolders))
			for folder := range dayFolders {
				sampleFolders = append(sampleFolders, folder)
			}
			// Sort for consistent output
			sort.Strings(sampleFolders)
			// Limit to first 10 for preview
			if len(sampleFolders) > 10 {
				sampleFolders = sampleFolders[:10]
			}
			c.JSON(200, gin.H{
				"baseEvents":      len(baseEvents),
				"detectionEvents": len(detectionEventIDs),
				"dayFolders":      len(dayFolders),
				"sampleFolders":   sampleFolders,
				"totalSizeBytes":  totalSizeBytes,
				"totalSize":       formatBytes(totalSizeBytes),
			})
			return
		}

		// Delete entire day folders instead of individual files
		deletedFolders := 0
		attemptedFolders := 0
		var failedSamples []map[string]string
		// Track month folders that may become empty after deletion
		monthFolders := make(map[string]bool)
		for dayFolder := range dayFolders {
			if dayFolder == "" {
				continue
			}
			// Extract month folder (parent of day folder)
			monthFolder := filepath.Dir(dayFolder)
			if monthFolder != "" && monthFolder != dayFolder {
				monthFolders[monthFolder] = true
			}
			attemptedFolders++
			if err := os.RemoveAll(dayFolder); err == nil || os.IsNotExist(err) {
				deletedFolders++
				internal.LogDebug("Deleted day folder: %s", dayFolder)
			} else {
				if len(failedSamples) < 10 {
					failedSamples = append(failedSamples, map[string]string{"path": dayFolder, "error": err.Error()})
				}
				internal.LogError("Delete failed: %s (%v)", dayFolder, err)
			}
		}

		// After deleting day folders, check and delete empty month folders
		deletedMonthFolders := 0
		for monthFolder := range monthFolders {
			// Check if month folder is empty
			entries, err := os.ReadDir(monthFolder)
			if err != nil {
				// If we can't read it, it might not exist or we don't have permission - skip
				continue
			}
			// If folder is empty, delete it
			if len(entries) == 0 {
				if err := os.Remove(monthFolder); err == nil || os.IsNotExist(err) {
					deletedMonthFolders++
					internal.LogDebug("Deleted empty month folder: %s", monthFolder)
				} else {
					internal.LogError("Failed to delete empty month folder %s: %v", monthFolder, err)
				}
			}
		}

		// Best-effort: also remove GPU completed artifacts for each base event (image+json)
		// These are in a different directory, so read it once and batch delete matching files
		cleanedDetections := 0
		if len(ids) > 0 {
			completedDir := config.CompletedDir
			// Build a set of prefixes to match (eventID__)
			prefixMap := make(map[string]bool, len(ids))
			for _, id := range ids {
				prefixMap[fmt.Sprintf("%d__", id)] = true
			}

			// Read directory once
			entries, err := os.ReadDir(completedDir)
			if err == nil {
				var filesToDelete []string
				// Collect all matching files in one pass
				for _, e := range entries {
					name := e.Name()
					// Extract event ID from filename (format: {eventID}__{rest})
					if idx := strings.Index(name, "__"); idx > 0 {
						prefix := name[:idx+2] // Include the "__"
						// Check if this prefix matches any of our event IDs
						if prefixMap[prefix] {
							// Check if it's a detection file (ends with _det.jpg/_det.jpeg) or .json
							lowerName := strings.ToLower(name)
							if strings.HasSuffix(lowerName, "_det.jpg") ||
								strings.HasSuffix(lowerName, "_det.jpeg") ||
								strings.HasSuffix(lowerName, ".json") {
								filesToDelete = append(filesToDelete, filepath.Join(completedDir, name))
							}
						}
					}
				}

				// Delete all collected files
				for _, filePath := range filesToDelete {
					if err := os.Remove(filePath); err == nil || os.IsNotExist(err) {
						cleanedDetections++
					}
				}
			}
		}

		// Build combined list of all event IDs to delete records for
		allIDs := append([]int64{}, ids...)
		allIDs = append(allIDs, detectionEventIDs...)
		if len(allIDs) == 0 {
			c.JSON(200, gin.H{
				"deletedFolders":      deletedFolders,
				"deletedMonthFolders": deletedMonthFolders,
				"deletedEvents":       0,
				"deletedDetections":   0,
				"cleanedDetections":   cleanedDetections,
			})
			return
		}

		// Delete detections rows first
		placeholders := strings.Repeat("?,", len(allIDs)-1) + "?"
		args := make([]any, len(allIDs))
		for i, id := range allIDs {
			args[i] = id
		}
		_, _ = db.Exec("DELETE FROM detections WHERE event_id IN ("+placeholders+")", args...)

		// Delete events rows
		_, err = db.Exec("DELETE FROM events WHERE id IN ("+placeholders+")", args...)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(200, gin.H{
			"deletedFolders":      deletedFolders,
			"deletedMonthFolders": deletedMonthFolders,
			"attemptedFolders":    attemptedFolders,
			"deletedBaseEvents":   len(ids),
			"deletedDetections":   len(detectionEventIDs),
			"cleanedDetections":   cleanedDetections,
			"failedSamples":       failedSamples,
		})
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

	// Date stats for dynamic deletion UI
	r.GET("/data/date-stats", func(c *gin.Context) {
		// min/max start_ts for non-detection base events
		var minTs, maxTs sql.NullInt64
		if err := db.QueryRow("SELECT MIN(start_ts) FROM events WHERE camera != 'DETECTION'").Scan(&minTs); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		if err := db.QueryRow("SELECT MAX(start_ts) FROM events WHERE camera != 'DETECTION'").Scan(&maxTs); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		// total events
		var total int64
		_ = db.QueryRow("SELECT COUNT(*) FROM events WHERE camera != 'DETECTION'").Scan(&total)

		// daily buckets (date string + count)
		daily := make([]map[string]any, 0, 90)
		rows, err := db.Query(`
			SELECT DATE(datetime(start_ts,'unixepoch')) as d, COUNT(*)
			FROM events WHERE camera != 'DETECTION'
			GROUP BY d ORDER BY d ASC`)
		if err == nil {
			defer rows.Close()
			for rows.Next() {
				var d string
				var c int64
				if err := rows.Scan(&d, &c); err == nil {
					daily = append(daily, map[string]any{"date": d, "count": c})
				}
			}
		}

		// monthly buckets (YYYY-MM + count)
		monthly := make([]map[string]any, 0, 24)
		mrows, err := db.Query(`
			SELECT strftime('%Y-%m', datetime(start_ts,'unixepoch')) as ym, COUNT(*)
			FROM events WHERE camera != 'DETECTION'
			GROUP BY ym ORDER BY ym ASC`)
		if err == nil {
			defer mrows.Close()
			for mrows.Next() {
				var ym string
				var c int64
				if err := mrows.Scan(&ym, &c); err == nil {
					monthly = append(monthly, map[string]any{"month": ym, "count": c})
				}
			}
		}

		resp := gin.H{
			"minStartTs": func() any {
				if minTs.Valid {
					return minTs.Int64
				}
				return nil
			}(),
			"maxStartTs": func() any {
				if maxTs.Valid {
					return maxTs.Int64
				}
				return nil
			}(),
			"totalEvents": total,
			"daily":       daily,
			"monthly":     monthly,
		}
		c.JSON(200, resp)
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

	// Telegram endpoints - use passed configManager

	telegramClient, err := telegram.NewTelegramClient(configManager.GetConfig())
	if err != nil {
		fmt.Printf("Warning: Failed to initialize Telegram client: %v\n", err)
	}

	// Send message endpoint
	r.POST("/telegram/send", func(c *gin.Context) {
		if telegramClient == nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{"error": "Telegram not configured"})
			return
		}

		var payload struct {
			Message string `json:"message"`
			Chat    string `json:"chat,omitempty"`
		}

		if err := c.ShouldBindJSON(&payload); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		if payload.Message == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Missing 'message' in request body"})
			return
		}

		if err := telegramClient.SendMessage(payload.Message, payload.Chat); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(200, gin.H{"status": "success"})
	})

	// Health check endpoint
	r.GET("/telegram/health", func(c *gin.Context) {
		if telegramClient == nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{"ok": false, "error": "Telegram not configured"})
			return
		}
		c.JSON(200, gin.H{"ok": true})
	})

	// Test Telegram configuration
	r.POST("/telegram/test", func(c *gin.Context) {
		var testConfig map[string]interface{}
		if err := c.ShouldBindJSON(&testConfig); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON"})
			return
		}

		// Create a test Telegram client with the provided config
		testClient, err := telegram.NewTelegramClient(testConfig)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error":   "Telegram configuration invalid",
				"details": err.Error(),
			})
			return
		}

		// Test sending a message to the default chat
		testMessage := "🧪 **Telegram Test Message**\n\n" +
			"✅ Configuration is valid\n" +
			"✅ Bot token is working\n" +
			"✅ Chat ID is accessible\n\n" +
			"*This is a test message from your Security Camera UI system.*"

		// Try to send to default chat first, then security chat
		var sentTo string
		var sendErr error

		chats := testClient.GetChats()
		if defaultChat, exists := chats["default"]; exists && defaultChat != "" {
			sendErr = testClient.SendMessage(testMessage, "default")
			sentTo = "default chat"
		} else if securityChat, exists := chats["security_group"]; exists && securityChat != "" {
			sendErr = testClient.SendMessage(testMessage, "security_group")
			sentTo = "security group chat"
		} else {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": "No valid chat IDs configured",
			})
			return
		}

		if sendErr != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error":   "Failed to send test message",
				"details": sendErr.Error(),
				"sent_to": sentTo,
			})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"success": true,
			"message": "Test message sent successfully",
			"sent_to": sentTo,
			"chats":   testClient.GetChats(),
		})
	})

	// Test Error Notification system
	r.POST("/error-notifications/test", func(c *gin.Context) {
		var testConfig map[string]interface{}
		if err := c.ShouldBindJSON(&testConfig); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON"})
			return
		}

		// Create a test Error Telegram client with the provided config
		testClient, err := telegram.NewErrorTelegramClient(testConfig)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error":   "Error notification configuration invalid",
				"details": err.Error(),
			})
			return
		}

		// Test sending different types of error notifications
		testMessage := "🧪 **Error Notification Test**\n\n" +
			"✅ Configuration is valid\n" +
			"✅ Error notification system is working\n" +
			"✅ This is a test error notification\n\n" +
			"*This is a test message from your Security Camera UI error notification system.*"

		// Test different severity levels
		severities := []string{"info", "warning", "error", "critical"}
		var results []map[string]interface{}

		for _, severity := range severities {
			metadata := map[string]interface{}{
				"test_type": "error_notification_test",
				"severity":  severity,
				"timestamp": time.Now().Format(time.RFC3339),
			}

			var sendErr error
			switch severity {
			case "info":
				sendErr = testClient.SendErrorWithData(severity, "test_component", testMessage, metadata)
			case "warning":
				sendErr = testClient.SendErrorWithData(severity, "test_component", testMessage, metadata)
			case "error":
				sendErr = testClient.SendErrorWithData(severity, "test_component", testMessage, metadata)
			case "critical":
				sendErr = testClient.SendErrorWithData(severity, "test_component", testMessage, metadata)
			}

			errorMsg := ""
			if sendErr != nil {
				errorMsg = sendErr.Error()
			}
			results = append(results, map[string]interface{}{
				"severity": severity,
				"success":  sendErr == nil,
				"error":    errorMsg,
			})
		}

		// Test Python endpoint if configured
		pythonEndpoint := ""
		if endpoint, ok := testConfig["error_notifications_python_endpoint"].(string); ok {
			pythonEndpoint = endpoint
		}

		if pythonEndpoint != "" {
			metadata := map[string]interface{}{
				"test_type": "python_endpoint_test",
				"timestamp": time.Now().Format(time.RFC3339),
			}

			pythonErr := testClient.SendErrorToPythonEndpoint("info", "test_component", "Python endpoint test", metadata, pythonEndpoint)
			pythonErrorMsg := ""
			if pythonErr != nil {
				pythonErrorMsg = pythonErr.Error()
			}
			results = append(results, map[string]interface{}{
				"severity": "python_endpoint",
				"success":  pythonErr == nil,
				"error":    pythonErrorMsg,
			})
		}

		c.JSON(http.StatusOK, gin.H{
			"success":         true,
			"message":         "Error notification test completed",
			"results":         results,
			"python_endpoint": pythonEndpoint,
		})
	})

	// Detection ingestion endpoint for Python detection logic
	r.POST("/ingest_detection", func(c *gin.Context) {
		if telegramClient == nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{"error": "Telegram not configured"})
			return
		}

		var payload map[string]interface{}
		if err := c.ShouldBindJSON(&payload); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		// Extract detection data from Python payload
		detectionData := map[string]interface{}{
			"camera_name": "Python Detection",
			"timestamp":   time.Now().Unix(),
			"detections":  []interface{}{},
			"duration_ms": 0.0,
			"conf":        0.25,
			"iou":         0.45,
		}

		// Map Python detection data to Telegram format
		if originalPath, ok := payload["original_path"].(string); ok {
			// Extract camera name from path if possible
			if strings.Contains(originalPath, "/") {
				parts := strings.Split(originalPath, "/")
				if len(parts) > 1 {
					detectionData["camera_name"] = parts[len(parts)-2] // Use parent directory as camera name
				}
			}
		}

		if detectionDataPath, ok := payload["detection_data_path"].(map[string]interface{}); ok {
			// Map detection results
			if detections, ok := detectionDataPath["detections"].([]interface{}); ok {
				detectionData["detections"] = detections
			}
			if duration, ok := detectionDataPath["duration_ms"].(float64); ok {
				detectionData["duration_ms"] = duration
			}
			if conf, ok := detectionDataPath["conf"].(float64); ok {
				detectionData["conf"] = conf
			}
			if iou, ok := detectionDataPath["iou"].(float64); ok {
				detectionData["iou"] = iou
			}
		}

		// Add annotated image if available
		if annotatedPath, ok := payload["annotated_image_path"].(string); ok {
			detectionData["annotated_image_path"] = annotatedPath
		}

		// Determine chat based on detection confidence or type
		chat := "default"
		if detections, ok := detectionData["detections"].([]interface{}); ok && len(detections) > 0 {
			// Check if any detection has high confidence or is a person/vehicle
			hasHighConfidence := false
			hasPersonOrVehicle := false

			for _, det := range detections {
				if detMap, ok := det.(map[string]interface{}); ok {
					if conf, ok := detMap["score"].(float64); ok && conf > 0.7 {
						hasHighConfidence = true
					}
					if label, ok := detMap["label"].(string); ok {
						labelLower := strings.ToLower(label)
						if strings.Contains(labelLower, "person") ||
							strings.Contains(labelLower, "car") ||
							strings.Contains(labelLower, "truck") ||
							strings.Contains(labelLower, "bus") ||
							strings.Contains(labelLower, "bicycle") ||
							strings.Contains(labelLower, "motorcycle") {
							hasPersonOrVehicle = true
						}
					}
				}
			}

			// Use security group for high confidence or person/vehicle detections
			if hasHighConfidence || hasPersonOrVehicle {
				chat = "security_group"
			}
		}

		// Send detection notification via Telegram
		if err := telegramClient.SendDetection(detectionData, chat); err != nil {
			internal.NotifyError("error", "detection_ingest", "Failed to send detection notification", map[string]interface{}{
				"error":   err.Error(),
				"payload": payload,
			})
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"status":  "success",
			"message": "Detection notification sent",
			"chat":    chat,
		})
	})

	// Send telegram detection endpoint
	r.POST("/telegram/send_detection", func(c *gin.Context) {
		internal.LogDebug("Received Telegram detection request: %v", c.Request.Body)
		if telegramClient == nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{"error": "Telegram not configured"})
			return
		}

		var payload map[string]interface{}
		if err := c.ShouldBindJSON(&payload); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		if err := telegramClient.SendDetection(payload, "security_group"); err != nil {
			internal.LogError("Failed to send Telegram notification: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{"status": "success"})
	})
}
