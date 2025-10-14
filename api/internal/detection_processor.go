package internal

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// DetectionProcessor handles processing of detection files from the Python script
type DetectionProcessor struct {
	db            *sql.DB
	pendingDir    string
	processingDir string
	failedDir     string
	completedDir  string
	lastDebugTime time.Time
	logger        *log.Logger
	telegramURL   string // URL for the Telegram bot API
}

// DetectionFile represents the JSON structure from the Python script
type DetectionFile struct {
	Image          string          `json:"image"`
	DurationMs     float64         `json:"duration_ms"`
	Detections     []DetectionData `json:"detections"`
	Imgsz          int             `json:"imgsz"`
	Conf           float64         `json:"conf"`
	Iou            float64         `json:"iou"`
	AnnotatedImage string          `json:"annotated_image"`
}

type DetectionData struct {
	Label string    `json:"label"`
	Score float64   `json:"score"`
	Xyxy  []float64 `json:"xyxy"`
}

// NewDetectionProcessor creates a new detection processor
func NewDetectionProcessor(db *sql.DB, pendingDir, processingDir, completedDir, failedDir, telegramURL string) *DetectionProcessor {
	// Set up file logging for detection processor
	logFile, err := os.OpenFile("detection_processor.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	if err != nil {
		log.Printf("Warning: Failed to open detection processor log file: %v", err)
		// Fall back to standard logger if file logging fails
		return &DetectionProcessor{
			db:            db,
			pendingDir:    pendingDir,
			processingDir: processingDir,
			completedDir:  completedDir,
			failedDir:     failedDir,
			telegramURL:   telegramURL,
			logger:        log.New(os.Stdout, "[DETECTION] ", log.LstdFlags),
		}
	}

	// Create a multi-writer to log to both file and stdout
	multiWriter := io.MultiWriter(logFile, os.Stdout)
	detectionLogger := log.New(multiWriter, "[DETECTION] ", log.LstdFlags)

	return &DetectionProcessor{
		db:            db,
		pendingDir:    pendingDir,
		processingDir: processingDir,
		completedDir:  completedDir,
		failedDir:     failedDir,
		telegramURL:   telegramURL,
		logger:        detectionLogger,
	}
}

// ProcessCompletedFiles scans the completed directory for new detection files
func (dp *DetectionProcessor) ProcessCompletedFiles() error {
	dp.logger.Printf("=== DetectionProcessor: Starting scan of completed directory: %s", dp.completedDir)

	// Check if completed directory exists
	if _, err := os.Stat(dp.completedDir); os.IsNotExist(err) {
		dp.logger.Printf("Completed directory does not exist: %s", dp.completedDir)
		return nil
	}

	// Debug database contents only once per minute to avoid spam
	if time.Since(dp.lastDebugTime) > time.Minute {
		dp.DebugDatabaseContents()
		dp.lastDebugTime = time.Now()
	}

	jsonFileCount := 0
	processedCount := 0
	errorCount := 0

	// Walk through the completed directory looking for JSON files
	err := filepath.WalkDir(dp.completedDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			dp.logger.Printf("Error walking directory %s: %v", path, err)
			return err
		}

		// Skip directories
		if d.IsDir() {
			return nil
		}

		// Only process JSON files
		if !strings.HasSuffix(strings.ToLower(path), ".json") {
			return nil
		}

		jsonFileCount++

		// Process the detection file
		if err := dp.processDetectionFile(path); err != nil {
			dp.logger.Printf("Error processing detection file %s: %v", path, err)
			errorCount++
			return nil // Continue processing other files
		}

		processedCount++

		// Small delay to reduce database contention
		time.Sleep(10 * time.Millisecond)

		return nil
	})

	dp.logger.Printf("=== DetectionProcessor: Scan complete - Found %d JSON files, processed %d, errors %d", jsonFileCount, processedCount, errorCount)

	// Log recent detection updates for debugging
	if processedCount > 0 {
		var recentUpdates int64
		_ = dp.db.QueryRow("SELECT COUNT(*) FROM events WHERE detection_updated > ?", time.Now().Unix()-60).Scan(&recentUpdates)
		dp.logger.Printf("=== DetectionProcessor: Recent detection updates in last minute: %d", recentUpdates)
	}
	return err
}

// processDetectionFile processes a single detection JSON file
func (dp *DetectionProcessor) processDetectionFile(jsonPath string) error {
	// Read the JSON file
	data, err := os.ReadFile(jsonPath)
	if err != nil {
		dp.logger.Printf("Failed to read JSON file %s: %v", jsonPath, err)
		return fmt.Errorf("failed to read JSON file: %w", err)
	}

	// Parse the detection data
	var detectionFile DetectionFile
	if err := json.Unmarshal(data, &detectionFile); err != nil {
		dp.logger.Printf("Failed to parse JSON from %s: %v", jsonPath, err)
		return fmt.Errorf("failed to parse JSON: %w", err)
	}

	// Extract event ID from filename
	// Expected format: {event_id}__{original_filename}.json
	eventID, err := dp.extractEventIDFromFilename(jsonPath)
	if err != nil {
		dp.logger.Printf("Failed to extract event ID from filename %s: %v", jsonPath, err)
		return fmt.Errorf("failed to extract event ID from filename: %w", err)
	}

	// Verify the event exists in the database
	if err := dp.verifyEventExists(eventID); err != nil {
		dp.logger.Printf("Event ID %d does not exist in database: %v", eventID, err)
		return fmt.Errorf("event ID %d does not exist: %w", eventID, err)
	}

	// Check if this event already has detection data to prevent duplicates
	if dp.eventAlreadyHasDetections(eventID) {
		return nil
	}

	// Update the event with detection data
	detectionJSON, err := json.Marshal(detectionFile)
	if err != nil {
		dp.logger.Printf("Failed to marshal detection data for event %d: %v", eventID, err)
		return fmt.Errorf("failed to marshal detection data: %w", err)
	}

	// Update the event record with retry logic for database locks
	updateQuery := `
		UPDATE events 
		SET detection_data = ?, detection_updated = ?, detection_path = ?
		WHERE id = ?
	`

	// Get the detection image path from the detection file
	detectionImagePath := detectionFile.AnnotatedImage
	if detectionImagePath == "" {
		// Fallback: construct path from the JSON file path
		detectionImagePath = strings.TrimSuffix(jsonPath, ".json") + ".jpg"
	}

	// Retry logic for database locks
	maxRetries := 3
	for attempt := 0; attempt < maxRetries; attempt++ {
		_, err = dp.db.Exec(updateQuery, string(detectionJSON), time.Now().Unix(), detectionImagePath, eventID)
		if err == nil {
			break // Success
		}

		// Check if it's a database lock error
		if strings.Contains(err.Error(), "database is locked") || strings.Contains(err.Error(), "SQLITE_BUSY") {
			if attempt < maxRetries-1 {
				dp.logger.Printf("Database locked, retrying in 100ms (attempt %d/%d): %v", attempt+1, maxRetries, err)
				time.Sleep(100 * time.Millisecond)
				continue
			}
		}

		dp.logger.Printf("Failed to update event %d with detection data: %v", eventID, err)
		return fmt.Errorf("failed to update event with detection data: %w", err)
	}

	// Insert individual detections into the detections table
	for _, det := range detectionFile.Detections {
		if len(det.Xyxy) >= 4 {
			// Convert xyxy format [x1, y1, x2, y2] to bbox format
			x1, y1, x2, y2 := det.Xyxy[0], det.Xyxy[1], det.Xyxy[2], det.Xyxy[3]
			bboxX := int(x1)
			bboxY := int(y1)
			bboxWidth := int(x2 - x1)
			bboxHeight := int(y2 - y1)

			insertQuery := `
				INSERT INTO detections (event_id, detection_type, confidence, bbox_x, bbox_y, bbox_width, bbox_height, created_at)
				VALUES (?, ?, ?, ?, ?, ?, ?, ?)
			`
			_, err = dp.db.Exec(insertQuery, eventID, det.Label, det.Score, bboxX, bboxY, bboxWidth, bboxHeight, time.Now().Unix())
			if err != nil {
				dp.logger.Printf("Failed to insert detection for event %d: %v", eventID, err)
			}
		}
	}

	// Note: Telegram notifications are now handled by the indexer when processing detection images

	return nil
}

// extractEventIDFromFilename extracts the event ID from a filename
// Expected format: {event_id}__{original_filename}.json
func (dp *DetectionProcessor) extractEventIDFromFilename(jsonPath string) (int64, error) {
	filename := filepath.Base(jsonPath)

	// Remove .json extension
	baseName := strings.TrimSuffix(filename, ".json")

	// Split by double underscore to get event_id and original filename
	parts := strings.SplitN(baseName, "__", 2)
	if len(parts) < 2 {
		dp.logger.Printf("Filename does not contain double underscore separator: %s", baseName)
		return 0, fmt.Errorf("filename does not contain double underscore separator: %s", baseName)
	}

	eventIDStr := parts[0]

	// Parse event ID as integer
	eventID, err := strconv.ParseInt(eventIDStr, 10, 64)
	if err != nil {
		dp.logger.Printf("Failed to parse event ID '%s' as integer: %v", eventIDStr, err)
		return 0, fmt.Errorf("failed to parse event ID '%s' as integer: %w", eventIDStr, err)
	}

	return eventID, nil
}

// verifyEventExists checks if an event with the given ID exists in the database
func (dp *DetectionProcessor) verifyEventExists(eventID int64) error {
	var count int
	query := "SELECT COUNT(*) FROM events WHERE id = ?"
	err := dp.db.QueryRow(query, eventID).Scan(&count)
	if err != nil {
		dp.logger.Printf("Database error checking event ID %d: %v", eventID, err)
		return fmt.Errorf("database error checking event ID %d: %w", eventID, err)
	}

	if count == 0 {
		dp.logger.Printf("Event ID %d not found in database", eventID)
		return fmt.Errorf("event ID %d not found in database", eventID)
	}

	return nil
}

// eventAlreadyHasDetections checks if an event already has detection data
func (dp *DetectionProcessor) eventAlreadyHasDetections(eventID int64) bool {
	// Check if the event has detection_data in the events table
	var detectionDataExists bool
	query := "SELECT detection_data IS NOT NULL FROM events WHERE id = ?"
	err := dp.db.QueryRow(query, eventID).Scan(&detectionDataExists)
	if err != nil {
		dp.logger.Printf("Error checking detection data for event ID %d: %v", eventID, err)
		return false // If we can't check, assume it doesn't have detections
	}

	if detectionDataExists {
		return true
	}

	// Also check if there are individual detection records
	var detectionCount int
	query = "SELECT COUNT(*) FROM detections WHERE event_id = ?"
	err = dp.db.QueryRow(query, eventID).Scan(&detectionCount)
	if err != nil {
		dp.logger.Printf("Error checking detections count for event ID %d: %v", eventID, err)
		return false // If we can't check, assume it doesn't have detections
	}

	if detectionCount > 0 {
		return true
	}

	return false
}

// DebugDatabaseContents logs some sample events from the database for debugging
func (dp *DetectionProcessor) DebugDatabaseContents() {
	// Get total count
	var totalEvents int
	err := dp.db.QueryRow("SELECT COUNT(*) FROM events").Scan(&totalEvents)
	if err != nil {
		dp.logger.Printf("Error getting total events count: %v", err)
		return
	}

	if totalEvents == 0 {
		dp.logger.Printf("No events found in database - this is likely the issue!")
		return
	}

	// Only log if there are issues or for critical debugging
	if totalEvents < 10 {
		dp.logger.Printf("Low event count in database: %d events", totalEvents)
	}
}

// GetDetectionsForEvent retrieves all detections for a specific event
func (dp *DetectionProcessor) GetDetectionsForEvent(eventID int64) ([]DetectionResult, error) {
	query := `
		SELECT id, event_id, detection_type, confidence, bbox_x, bbox_y, bbox_width, bbox_height, frame_index, timestamp, created_at
		FROM detections 
		WHERE event_id = ?
		ORDER BY confidence DESC
	`
	rows, err := dp.db.Query(query, eventID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var detections []DetectionResult
	for rows.Next() {
		var det DetectionResult
		err := rows.Scan(&det.ID, &det.EventID, &det.Type, &det.Confidence, &det.BboxX, &det.BboxY, &det.BboxWidth, &det.BboxHeight, &det.FrameIndex, &det.Timestamp, &det.CreatedAt)
		if err != nil {
			return nil, err
		}
		detections = append(detections, det)
	}

	return detections, nil
}

// GetDetectionsForEvents retrieves all detections for multiple events in a single query
func (dp *DetectionProcessor) GetDetectionsForEvents(eventIDs []int64) (map[int64][]DetectionResult, error) {
	if len(eventIDs) == 0 {
		return make(map[int64][]DetectionResult), nil
	}

	// Create placeholders for the IN clause
	placeholders := make([]string, len(eventIDs))
	args := make([]interface{}, len(eventIDs))
	for i, id := range eventIDs {
		placeholders[i] = "?"
		args[i] = id
	}

	query := `
		SELECT id, event_id, detection_type, confidence, bbox_x, bbox_y, bbox_width, bbox_height, frame_index, timestamp, created_at
		FROM detections 
		WHERE event_id IN (` + strings.Join(placeholders, ",") + `)
		ORDER BY event_id, confidence DESC
	`
	rows, err := dp.db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	// Group detections by event ID
	detectionsByEvent := make(map[int64][]DetectionResult)
	for rows.Next() {
		var det DetectionResult
		err := rows.Scan(&det.ID, &det.EventID, &det.Type, &det.Confidence, &det.BboxX, &det.BboxY, &det.BboxWidth, &det.BboxHeight, &det.FrameIndex, &det.Timestamp, &det.CreatedAt)
		if err != nil {
			return nil, err
		}
		detectionsByEvent[det.EventID] = append(detectionsByEvent[det.EventID], det)
	}

	return detectionsByEvent, nil
}

// GetDetectionStats retrieves statistics about detections
func (dp *DetectionProcessor) GetDetectionStats(days int) (*DetectionStats, error) {
	// Calculate start time
	startTime := time.Now().AddDate(0, 0, -days).Unix()

	// Get total detections
	var totalDetections int
	err := dp.db.QueryRow("SELECT COUNT(*) FROM detections WHERE created_at >= ?", startTime).Scan(&totalDetections)
	if err != nil {
		return nil, err
	}

	// Get detections by type
	byType := make(map[string]int)
	rows, err := dp.db.Query(`
		SELECT detection_type, COUNT(*) 
		FROM detections 
		WHERE created_at >= ? 
		GROUP BY detection_type 
		ORDER BY COUNT(*) DESC
	`, startTime)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	for rows.Next() {
		var detectionType string
		var count int
		if err := rows.Scan(&detectionType, &count); err != nil {
			continue
		}
		byType[detectionType] = count
	}

	// Get detections by confidence range
	byConfidence := make(map[string]int)
	confRows, err := dp.db.Query(`
		SELECT 
			CASE 
				WHEN confidence >= 0.9 THEN 'Very High (0.9+)'
				WHEN confidence >= 0.7 THEN 'High (0.7-0.9)'
				WHEN confidence >= 0.5 THEN 'Medium (0.5-0.7)'
				ELSE 'Low (<0.5)'
			END as confidence_range,
			COUNT(*)
		FROM detections 
		WHERE created_at >= ? 
		GROUP BY confidence_range
	`, startTime)
	if err != nil {
		return nil, err
	}
	defer confRows.Close()

	for confRows.Next() {
		var confRange string
		var count int
		if err := confRows.Scan(&confRange, &count); err != nil {
			continue
		}
		byConfidence[confRange] = count
	}

	// Get recent detections
	recentDetections, err := dp.GetRecentDetections(10)
	if err != nil {
		return nil, err
	}

	// Get daily stats
	dailyStats, err := dp.getDailyDetectionStats(days)
	if err != nil {
		return nil, err
	}

	return &DetectionStats{
		TotalDetections:  totalDetections,
		ByType:           byType,
		ByConfidence:     byConfidence,
		RecentDetections: recentDetections,
		DailyStats:       dailyStats,
	}, nil
}

// GetRecentDetections gets the most recent detections
func (dp *DetectionProcessor) GetRecentDetections(limit int) ([]DetectionResult, error) {
	query := `
		SELECT id, event_id, detection_type, confidence, bbox_x, bbox_y, bbox_width, bbox_height, frame_index, timestamp, created_at
		FROM detections 
		ORDER BY created_at DESC 
		LIMIT ?
	`
	rows, err := dp.db.Query(query, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var detections []DetectionResult
	for rows.Next() {
		var det DetectionResult
		err := rows.Scan(&det.ID, &det.EventID, &det.Type, &det.Confidence, &det.BboxX, &det.BboxY, &det.BboxWidth, &det.BboxHeight, &det.FrameIndex, &det.Timestamp, &det.CreatedAt)
		if err != nil {
			return nil, err
		}
		detections = append(detections, det)
	}

	return detections, nil
}

// getDailyDetectionStats gets daily detection statistics
func (dp *DetectionProcessor) getDailyDetectionStats(days int) ([]DailyDetectionStats, error) {
	startTime := time.Now().AddDate(0, 0, -days).Unix()

	query := `
		SELECT 
			DATE(datetime(created_at, 'unixepoch')) as date,
			COUNT(*) as count,
			detection_type,
			COUNT(*) as type_count
		FROM detections 
		WHERE created_at >= ?
		GROUP BY DATE(datetime(created_at, 'unixepoch')), detection_type
		ORDER BY date DESC
	`
	rows, err := dp.db.Query(query, startTime)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	// Group by date
	dateStats := make(map[string]*DailyDetectionStats)

	for rows.Next() {
		var date, detectionType string
		var count, typeCount int
		if err := rows.Scan(&date, &count, &detectionType, &typeCount); err != nil {
			continue
		}

		if dateStats[date] == nil {
			dateStats[date] = &DailyDetectionStats{
				Date:   date,
				Count:  0,
				ByType: make(map[string]int),
			}
		}

		dateStats[date].Count += count
		dateStats[date].ByType[detectionType] = typeCount
	}

	// Convert to slice
	var result []DailyDetectionStats
	for _, stats := range dateStats {
		result = append(result, *stats)
	}

	return result, nil
}

// TelegramPayload represents the payload sent to the Telegram bot API
type TelegramPayload struct {
	CameraName         string                 `json:"camera_name"`
	Timestamp          int64                  `json:"timestamp"`
	Detections         []TelegramDetection    `json:"detections"`
	DurationMs         float64                `json:"duration_ms"`
	Imgsz              int                    `json:"imgsz"`
	Conf               float64                `json:"conf"`
	Iou                float64                `json:"iou"`
	DetectionJson      map[string]interface{} `json:"detection_json,omitempty"`
	DetectionJsonName  string                 `json:"detection_json_name,omitempty"`
	Chat               string                 `json:"chat,omitempty"`
	Message            string                 `json:"message,omitempty"`              // Custom message for Telegram
	AnnotatedImagePath string                 `json:"annotated_image_path,omitempty"` // Path to the detection image
	AnnotatedImageB64  string                 `json:"annotated_image_b64,omitempty"`  // Base64 encoded image data
	AnnotatedImageName string                 `json:"annotated_image_name,omitempty"` // Filename for the image
}

// TelegramDetection represents a detection in the Telegram payload format
type TelegramDetection struct {
	Label string  `json:"label"`
	Score float64 `json:"score"`
	X     int     `json:"x,omitempty"`
	Y     int     `json:"y,omitempty"`
	W     int     `json:"w,omitempty"`
	H     int     `json:"h,omitempty"`
}

// HasPersonDetection checks if any detection in the file is a person
func (dp *DetectionProcessor) HasPersonDetection(detectionFile DetectionFile) bool {
	for _, detection := range detectionFile.Detections {
		if strings.ToLower(detection.Label) == "person" {
			return true
		}
	}
	return false
}
