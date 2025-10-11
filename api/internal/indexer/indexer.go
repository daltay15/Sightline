package indexer

import (
	"bytes"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	// optional if you want EXIF/audio tags later
	"github.com/daltay15/security-camera-ui/api/config"
	"github.com/daltay15/security-camera-ui/api/internal"
)

type Config struct {
	Root             string
	ThumbsDir        string
	PendingDir       string
	MaxWorkers       int
	BatchSize        int
	ThumbnailWorkers int
	// Detection image indexing
	DetectionDir    string
	IndexDetections bool
	// Telegram notification
	TelegramURL string
}

func ScanAll(db *sql.DB, cfg Config) error {
	return filepath.WalkDir(cfg.Root, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		lower := strings.ToLower(p)
		if !(strings.HasSuffix(lower, ".mp4") || strings.HasSuffix(lower, ".jpg") || strings.HasSuffix(lower, ".jpeg")) {
			return nil
		}
		return indexFile(db, cfg, p)
	})
}

// ScanAllParallel processes files in parallel using worker pools
func ScanAllParallel(db *sql.DB, cfg Config) error {
	log.Printf("ScanAllParallel: Starting scan of %s", cfg.Root)

	// Set defaults if not provided
	if cfg.MaxWorkers <= 0 {
		cfg.MaxWorkers = 4
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 50
	}
	if cfg.ThumbnailWorkers <= 0 {
		cfg.ThumbnailWorkers = 2
	}

	// Collect all files first
	var files []string
	err := filepath.WalkDir(cfg.Root, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		lower := strings.ToLower(p)
		if strings.HasSuffix(lower, ".mp4") || strings.HasSuffix(lower, ".jpg") || strings.HasSuffix(lower, ".jpeg") {
			files = append(files, p)
		}
		return nil
	})
	if err != nil {
		log.Printf("ScanAllParallel: Error walking directory: %v", err)
		return err
	}

	// Filter out files that are already in the database and check file stability
	var newFiles []string
	log.Printf("ScanAllParallel: Found %d files to process", len(files))
	if len(files) == 0 {
		log.Printf("ScanAllParallel: No files to process")
		return nil
	}

	// First pass: separate existing files from new files
	for _, file := range files {
		var count int
		err := db.QueryRow("SELECT COUNT(*) FROM events WHERE path = ?", file).Scan(&count)
		if err != nil {
			log.Printf("Error checking if file exists in database: %v", err)
			continue
		}

		if count == 0 {
			// File is not in database, add to new files
			newFiles = append(newFiles, file)
		}
		// Skip files that already exist in database
	}

	// Second pass: only check stability for very recent files
	var stableFiles []string
	for _, file := range newFiles {
		// Only check stability for files modified very recently (within last 2 minutes)
		if isFileVeryRecent(file) {
			if isFileStable(file) {
				log.Printf("ScanAllParallel: Adding stable file to process: %s", file)
				stableFiles = append(stableFiles, file)
			} else {
				log.Printf("ScanAllParallel: Skipping unstable file (still being written): %s", file)
			}
		} else {
			// File is not recent, add directly without delay
			stableFiles = append(stableFiles, file)
		}
	}

	// Update newFiles to only include stable files
	newFiles = stableFiles

	log.Printf("ScanAllParallel: Found %d new files to process (skipping %d existing files)", len(newFiles), len(files)-len(newFiles))

	// Process only new files in parallel
	return processFilesParallel(db, cfg, newFiles)
}

// processFilesParallel processes files using worker pools and batch database operations
func processFilesParallel(db *sql.DB, cfg Config, files []string) error {
	// Channels for coordination
	fileChan := make(chan string, len(files))
	eventChan := make(chan *internal.Event, cfg.BatchSize*2)
	errorChan := make(chan error, cfg.MaxWorkers)

	// Start file processing workers
	var wg sync.WaitGroup
	for i := 0; i < cfg.MaxWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for file := range fileChan {
				event, err := processFileToEvent(cfg, file)
				if err != nil {
					errorChan <- err
					continue
				}
				eventChan <- event
			}
		}()
	}

	// Start database batch writer
	var dbWg sync.WaitGroup
	dbWg.Add(1)
	go func() {
		defer dbWg.Done()
		batchEvents(db, eventChan, cfg.BatchSize, cfg.PendingDir)
	}()

	// Send files to workers
	go func() {
		defer close(fileChan)
		for _, file := range files {
			fileChan <- file
		}
	}()

	// Wait for file processing to complete
	wg.Wait()
	close(eventChan)

	// Wait for database operations to complete
	dbWg.Wait()

	// Check for errors - but don't fail the entire scan for individual file errors
	// Only return an error if there's a critical system error
	select {
	case err := <-errorChan:
		// Log the error but don't fail the entire scan
		// Individual file errors should not stop the scanning process
		log.Printf("Error processing file: %v", err)
	default:
		return nil
	}
	return nil
}

// ScanDetectionImages scans and indexes detection images from the GPU processing completed directory
func ScanDetectionImages(db *sql.DB, cfg Config) error {
	if !cfg.IndexDetections || cfg.DetectionDir == "" {
		return nil
	}

	// Check if detection directory exists
	if _, err := os.Stat(cfg.DetectionDir); os.IsNotExist(err) {
		return nil
	}

	var detectionFiles []string
	err := filepath.WalkDir(cfg.DetectionDir, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		lower := strings.ToLower(p)
		// Look for detection images with _det suffix
		if strings.HasSuffix(lower, "_det.jpg") || strings.HasSuffix(lower, "_det.jpeg") {
			detectionFiles = append(detectionFiles, p)
		}
		return nil
	})
	if err != nil {
		log.Printf("Error walking detection directory: %v", err)
		return err
	}

	if len(detectionFiles) == 0 {
		return nil
	}

	// Process detection images
	for _, file := range detectionFiles {
		if err := indexDetectionFile(db, cfg, file); err != nil {
			log.Printf("Error indexing detection file %s: %v", file, err)
			// Continue processing other files
		}
	}

	return nil
}

// ProcessSingleFile processes a single file immediately (for file watcher)
func ProcessSingleFile(db *sql.DB, cfg Config, filePath string) error {
	log.Printf("ProcessSingleFile: Processing %s", filePath)

	// Check if file already exists in database
	var count int
	err := db.QueryRow("SELECT COUNT(*) FROM events WHERE path = ?", filePath).Scan(&count)
	if err != nil {
		return err
	}

	if count > 0 {
		log.Printf("ProcessSingleFile: File already in database, skipping: %s", filePath)
		return nil
	}

	// Process the file using existing logic
	event, err := processFileToEvent(cfg, filePath)
	if err != nil {
		return err
	}

	// Insert into database immediately
	eventChan := make(chan *internal.Event, 1)
	eventChan <- event
	close(eventChan)

	// Use batchEvents to insert the single event
	batchEvents(db, eventChan, 1, cfg.PendingDir)
	return nil
}

// indexDetectionFile indexes a single detection image file
func indexDetectionFile(db *sql.DB, cfg Config, p string) error {
	info, err := os.Stat(p)
	if err != nil {
		return err
	}

	// Parse the detection filename to extract event ID and original filename
	// Format: {event_id}__{original_filename}_det.jpg
	baseName := filepath.Base(p)
	detSuffix := "_det.jpg"
	if strings.HasSuffix(strings.ToLower(baseName), "_det.jpeg") {
		detSuffix = "_det.jpeg"
	}

	// Remove the _det suffix to get the original filename with event ID
	originalWithEventID := strings.TrimSuffix(baseName, detSuffix)

	// Split by double underscore to get event ID and original filename
	parts := strings.SplitN(originalWithEventID, "__", 2)
	if len(parts) != 2 {
		return fmt.Errorf("invalid detection filename format: %s", baseName)
	}

	eventIDStr := parts[0]
	_ = parts[1] // originalFilename - not used but kept for clarity

	// Parse event ID
	eventID, err := strconv.ParseInt(eventIDStr, 10, 64)
	if err != nil {
		return fmt.Errorf("invalid event ID: %s", eventIDStr)
	}

	// Check if the original event exists
	var originalEventID int64
	err = db.QueryRow("SELECT id FROM events WHERE id = ?", eventID).Scan(&originalEventID)
	if err != nil {
		return fmt.Errorf("original event %d not found", eventID)
	}

	// Check if the corresponding JSON file has actual detections
	jsonPath := strings.TrimSuffix(p, detSuffix) + ".json"
	if !hasDetectionsInJSON(jsonPath) {
		// Skip this detection image if it has no detections
		return nil
	}

	// Create a detection event entry
	// We'll store this as a separate event with a special camera name to distinguish it
	detectionEvent := internal.Event{
		Camera:    "DETECTION", // Special camera name for detection images
		Path:      p,
		StartTS:   0, // Will be set from original event
		SizeBytes: info.Size(),
		CreatedAt: time.Now().Unix(),
		Tags:      fmt.Sprintf("detection_for_event_%d", eventID), // Link to original event
	}

	// Get the start timestamp from the original event
	err = db.QueryRow("SELECT start_ts FROM events WHERE id = ?", eventID).Scan(&detectionEvent.StartTS)
	if err != nil {
		// Use current time as fallback
		detectionEvent.StartTS = time.Now().Unix()
	}

	// For detection images, the jpg_path is the same as the path
	detectionEvent.JpgPath = &p

	// Check if this detection image is already indexed
	var existingCount int
	err = db.QueryRow("SELECT COUNT(*) FROM events WHERE camera = 'DETECTION' AND path = ?", p).Scan(&existingCount)
	if err != nil {
		return err
	}

	if existingCount > 0 {
		return nil
	}

	// Insert the detection event
	err = internal.UpsertEvent(db, &detectionEvent)
	if err != nil {
		return err
	}

	// Send Telegram notification if person detected
	if err := sendTelegramNotificationIfPersonDetected(db, cfg, jsonPath, p, eventID); err != nil {
		log.Printf("Failed to send Telegram notification for event %d: %v", eventID, err)
		// Don't fail the indexing if Telegram fails
	}

	return nil
}

// hasDetectionsInJSON checks if a JSON file contains actual detections
func hasDetectionsInJSON(jsonPath string) bool {
	// Check if JSON file exists
	if _, err := os.Stat(jsonPath); os.IsNotExist(err) {
		return false
	}

	// Read the JSON file
	data, err := os.ReadFile(jsonPath)
	if err != nil {
		return false
	}

	// Parse the JSON to check if detections array has items
	var detectionFile struct {
		Detections []interface{} `json:"detections"`
	}

	if err := json.Unmarshal(data, &detectionFile); err != nil {
		return false
	}

	// Return true only if detections array has items
	return len(detectionFile.Detections) > 0
}

func indexFile(db *sql.DB, cfg Config, p string) error {
	info, err := os.Stat(p)
	if err != nil {
		return nil
	}
	parsed := Parse(cfg.Root, p)
	e := internal.Event{
		Camera:    parsed.Camera,
		Path:      p,
		StartTS:   parsed.StartTS,
		SizeBytes: info.Size(),
		CreatedAt: time.Now().Unix(),
	}
	lower := strings.ToLower(p)
	if strings.HasSuffix(lower, ".mp4") {
		e.DurationMS = 0 // (optional) call FFProbeDurationMS(p)
		// poster + sheet paths
		poster := filepath.Join(cfg.ThumbsDir, hashPath(p)+"_poster.jpg")
		sheet := filepath.Join(cfg.ThumbsDir, hashPath(p)+"_sheet.jpg")
		if _, err := os.Stat(poster); os.IsNotExist(err) {
			_ = MakePoster(p, poster)
		}
		if _, err := os.Stat(sheet); os.IsNotExist(err) {
			_ = MakeSheet(p, sheet)
		}
		e.JpgPath = &poster
		e.SheetPath = &sheet
	} else {
		// still image
		jp := p
		e.JpgPath = &jp
	}

	// Insert event into database first
	err = internal.UpsertEvent(db, &e)
	if err != nil {
		return err
	}

	return nil
}

func hashPath(s string) string {
	// short stable name for thumbnails
	// use a simple fnv:
	const off = 1469598103934665603
	const prime = 1099511628211
	var h uint64 = off
	for i := 0; i < len(s); i++ {
		h ^= uint64(s[i])
		h *= prime
	}
	return strings.ToLower(strings.ReplaceAll(filepath.Base(s), ".", "_")) + "_" + // readable + hash
		func(x uint64) string { return strings.ToUpper(fmt.Sprintf("%x", x)) }(h)
}

// processFileToEvent processes a single file and returns an Event (without database operations)
func processFileToEvent(cfg Config, p string) (*internal.Event, error) {
	info, err := os.Stat(p)
	if err != nil {
		return nil, err
	}

	parsed := Parse(cfg.Root, p)
	e := &internal.Event{
		Camera:    parsed.Camera,
		Path:      p,
		StartTS:   parsed.StartTS,
		SizeBytes: info.Size(),
		CreatedAt: time.Now().Unix(),
	}

	lower := strings.ToLower(p)
	if strings.HasSuffix(lower, ".mp4") {
		// Get duration for video files
		e.DurationMS = FFProbeDurationMS(p)

		// Generate thumbnails in parallel
		poster := filepath.Join(cfg.ThumbsDir, hashPath(p)+"_poster.jpg")
		sheet := filepath.Join(cfg.ThumbsDir, hashPath(p)+"_sheet.jpg")

		// Check if thumbnails exist, if not generate them
		posterExists := false
		sheetExists := false

		if _, err := os.Stat(poster); err == nil {
			posterExists = true
		}
		if _, err := os.Stat(sheet); err == nil {
			sheetExists = true
		}

		// Generate thumbnails in parallel if they don't exist
		var thumbWg sync.WaitGroup
		if !posterExists {
			thumbWg.Add(1)
			go func() {
				defer thumbWg.Done()
				_ = MakePoster(p, poster)
			}()
		}
		if !sheetExists {
			thumbWg.Add(1)
			go func() {
				defer thumbWg.Done()
				_ = MakeSheet(p, sheet)
			}()
		}
		thumbWg.Wait()

		e.JpgPath = &poster
		e.SheetPath = &sheet
	} else {
		// still image
		jp := p
		e.JpgPath = &jp
	}

	return e, nil
}

// batchEvents handles batch database operations
func batchEvents(db *sql.DB, eventChan <-chan *internal.Event, batchSize int, pendingDir string) {
	var batch []*internal.Event

	for event := range eventChan {
		batch = append(batch, event)

		if len(batch) >= batchSize {
			// Process batch
			_ = upsertEventsBatch(db, batch, pendingDir)
			batch = batch[:0] // reset slice
		}
	}

	// Process remaining events
	if len(batch) > 0 {
		_ = upsertEventsBatch(db, batch, pendingDir)
	}
}

// upsertEventsBatch performs batch upsert operations
func upsertEventsBatch(db *sql.DB, events []*internal.Event, pendingDir string) error {
	if len(events) == 0 {
		return nil
	}

	// Use a transaction for better performance
	tx, err := db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	stmt, err := tx.Prepare(`
		INSERT INTO events (camera, path, jpg_path, sheet_path, start_ts, duration_ms, size_bytes, reviewed, tags, created_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, 0, '', ?)
		ON CONFLICT(path) DO UPDATE SET
		  jpg_path=excluded.jpg_path,
		  sheet_path=excluded.sheet_path,
		  duration_ms=excluded.duration_ms,
		  size_bytes=excluded.size_bytes
	`)
	if err != nil {
		return err
	}
	defer stmt.Close()

	for _, e := range events {
		_, err := stmt.Exec(
			e.Camera, e.Path, e.JPGPathOrNil(), e.SheetPathOrNil(),
			e.StartTS, e.DurationMS, e.SizeBytes, time.Now().Unix(),
		)
		if err != nil {
			return err
		}
	}

	commitStart := time.Now()
	// Commit the transaction first to get the event IDs
	log.Printf("Committing transaction")
	if err := tx.Commit(); err != nil {
		log.Printf("Warning: Failed to commit transaction: %v", err)
		return err
	}
	commitEnd := time.Now()
	log.Printf("Committed transaction in %v", commitEnd.Sub(commitStart))

	// After successful database insertion, copy JPG files to processing directory
	// with event_id prepended to filename for easier processing
	copyStart := time.Now()
	log.Printf("Copying JPG files to processing directory")
	if err := copyJpgFilesToProcessing(db, events, pendingDir); err != nil {
		log.Printf("Warning: Failed to copy JPG files to processing directory: %v", err)
		// Don't return error here as the main indexing was successful
	}
	copyEnd := time.Now()
	log.Printf("Copied JPG files to processing directory in %v", copyEnd.Sub(copyStart))

	return nil
}

// PreGenerateThumbnails generates thumbnails for all video files in the background
func PreGenerateThumbnails(db *sql.DB, cfg Config) error {
	// Get all video files that don't have thumbnails yet
	rows, err := db.Query(`
		SELECT id, path FROM events 
		WHERE path LIKE '%.mp4' 
		AND (jpg_path IS NULL OR sheet_path IS NULL)
		ORDER BY start_ts DESC
		LIMIT 100
	`)
	if err != nil {
		return err
	}
	defer rows.Close()

	var wg sync.WaitGroup
	semaphore := make(chan struct{}, cfg.ThumbnailWorkers) // Limit concurrent thumbnail generation

	for rows.Next() {
		var id, path string
		if err := rows.Scan(&id, &path); err != nil {
			continue
		}

		wg.Add(1)
		go func(videoPath string) {
			defer wg.Done()
			semaphore <- struct{}{}        // Acquire semaphore
			defer func() { <-semaphore }() // Release semaphore

			// Generate thumbnails
			poster := filepath.Join(cfg.ThumbsDir, hashPath(videoPath)+"_poster.jpg")
			sheet := filepath.Join(cfg.ThumbsDir, hashPath(videoPath)+"_sheet.jpg")

			// Check if thumbnails already exist
			if _, err := os.Stat(poster); os.IsNotExist(err) {
				_ = MakePoster(videoPath, poster)
			}
			if _, err := os.Stat(sheet); os.IsNotExist(err) {
				_ = MakeSheet(videoPath, sheet)
			}

			// Update database with thumbnail paths
			_, _ = db.Exec(`
				UPDATE events 
				SET jpg_path=?, sheet_path=? 
				WHERE path=?
			`, poster, sheet, videoPath)
		}(path)
	}

	wg.Wait()
	return nil
}

// isFileVeryRecent checks if a file was modified very recently (within last 2 minutes)
func isFileVeryRecent(filePath string) bool {
	info, err := os.Stat(filePath)
	if err != nil {
		return false
	}

	// Check if file was modified within the last 30 seconds
	// This is the only time we need to check stability
	recentThreshold := 30 * time.Second
	timeSinceModification := time.Since(info.ModTime())

	return timeSinceModification < recentThreshold
}

// isFileStable checks if a file is stable (not being written to)
func isFileStable(filePath string) bool {
	// Get file info
	info1, err := os.Stat(filePath)
	if err != nil {
		return false
	}

	// Wait a short time and check again
	time.Sleep(500 * time.Millisecond)

	info2, err := os.Stat(filePath)
	if err != nil {
		return false
	}

	// File is stable if size and modification time haven't changed
	return info1.Size() == info2.Size() && info1.ModTime().Equal(info2.ModTime())
}

// copyJpgFilesToProcessing copies JPG files to the processing directory with event_id prepended
func copyJpgFilesToProcessing(db *sql.DB, events []*internal.Event, pendingDir string) error {
	// Filter only JPG events and exclude cameras from detection processing
	var jpgEvents []*internal.Event
	for _, event := range events {
		lower := strings.ToLower(event.Path)
		if strings.HasSuffix(lower, ".jpg") || strings.HasSuffix(lower, ".jpeg") {
			// Check if camera is excluded from detection processing
			if IsCameraExcludedFromDetection(event.Camera) {
				log.Printf("Skipping detection processing for excluded camera: %s", event.Camera)
				continue
			}
			jpgEvents = append(jpgEvents, event)
		}
	}

	if len(jpgEvents) == 0 {
		return nil // No JPG files to process
	}

	// Get event IDs for the JPG files
	eventIDs := make(map[string]int64) // path -> event_id mapping
	for _, event := range jpgEvents {
		var eventID int64
		err := db.QueryRow("SELECT id FROM events WHERE path = ?", event.Path).Scan(&eventID)
		if err != nil {
			log.Printf("Warning: Failed to get event ID for path %s: %v", event.Path, err)
			continue
		}
		eventIDs[event.Path] = eventID
	}

	// Copy files to processing directory with event_id prepended
	for _, event := range jpgEvents {
		eventID, exists := eventIDs[event.Path]
		if !exists {
			continue
		}

		// Create filename with event_id prepended
		originalFilename := filepath.Base(event.Path)
		newFilename := fmt.Sprintf("%d__%s", eventID, originalFilename)

		// Ensure processing directory exists
		if err := os.MkdirAll(pendingDir, 0755); err != nil {
			log.Printf("Warning: Failed to create processing directory: %v, %s", err, pendingDir)
			continue
		}

		destPath := filepath.Join(pendingDir, newFilename)

		// Check if file already exists
		if _, err := os.Stat(destPath); err == nil {
			log.Printf("Processing file already exists, skipping: %s", destPath)
			continue
		}

		// Copy the file
		if err := copyFileWithEventID(event.Path, destPath); err != nil {
			log.Printf("Warning: Failed to copy file %s to processing directory: %v", event.Path, err)
			continue
		}
	}

	return nil
}

// copyFileWithEventID copies a file from source to destination
func copyFileWithEventID(src, dst string) error {
	// Open source file
	sourceFile, err := os.Open(src)
	if err != nil {
		return fmt.Errorf("failed to open source file: %w", err)
	}
	defer sourceFile.Close()

	// Create destination file
	destFile, err := os.Create(dst)
	if err != nil {
		return fmt.Errorf("failed to create destination file: %w", err)
	}
	defer destFile.Close()

	// Copy the file
	_, err = io.Copy(destFile, sourceFile)
	if err != nil {
		return fmt.Errorf("failed to copy file: %w", err)
	}

	// Sync to ensure data is written to disk
	return destFile.Sync()
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
	Message            string                 `json:"message,omitempty"`
	AnnotatedImagePath string                 `json:"annotated_image_path,omitempty"`
	AnnotatedImageB64  string                 `json:"annotated_image_b64,omitempty"`
	AnnotatedImageName string                 `json:"annotated_image_name,omitempty"`
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

// sendTelegramNotificationIfPersonDetected checks for person detections and sends notification
func sendTelegramNotificationIfPersonDetected(db *sql.DB, cfg Config, jsonPath, imagePath string, eventID int64) error {
	if cfg.TelegramURL == "" {
		return nil // No Telegram URL configured
	}

	// Read and parse the JSON file
	data, err := os.ReadFile(jsonPath)
	if err != nil {
		return fmt.Errorf("failed to read JSON file: %w", err)
	}

	var detectionFile struct {
		Image          string          `json:"image"`
		DurationMs     float64         `json:"duration_ms"`
		Detections     []DetectionData `json:"detections"`
		Imgsz          int             `json:"imgsz"`
		Conf           float64         `json:"conf"`
		Iou            float64         `json:"iou"`
		AnnotatedImage string          `json:"annotated_image"`
	}

	if err := json.Unmarshal(data, &detectionFile); err != nil {
		return fmt.Errorf("failed to parse JSON: %w", err)
	}

	// Check if any detection is a person
	hasPerson := false
	for _, detection := range detectionFile.Detections {
		if strings.ToLower(detection.Label) == "person" {
			hasPerson = true
			break
		}
	}

	if !hasPerson {
		return nil // No person detected, no notification needed
	}

	// Get camera name from the original event
	var cameraName string
	err = db.QueryRow("SELECT camera FROM events WHERE id = ?", eventID).Scan(&cameraName)
	if err != nil {
		cameraName = "Unknown Camera" // Fallback
	}

	// Create formatted message
	message := fmt.Sprintf("🚨 A person was detected at %s", cameraName)

	// Add detection details
	if len(detectionFile.Detections) > 0 {
		message += "\n\nDetections:"
		for _, det := range detectionFile.Detections {
			if strings.ToLower(det.Label) == "person" {
				message += fmt.Sprintf("\n• %s (confidence: %.2f)", det.Label, det.Score)
			}
		}
	}

	message += fmt.Sprintf("\n\nTime: %s", time.Now().Format("2006-01-02 15:04:05"))

	// Convert detections to Telegram format
	var telegramDetections []TelegramDetection
	for _, det := range detectionFile.Detections {
		telegramDet := TelegramDetection{
			Label: det.Label,
			Score: det.Score,
		}

		// Add bounding box if available
		if len(det.Xyxy) >= 4 {
			x1, y1, x2, y2 := det.Xyxy[0], det.Xyxy[1], det.Xyxy[2], det.Xyxy[3]
			telegramDet.X = int(x1)
			telegramDet.Y = int(y1)
			telegramDet.W = int(x2 - x1)
			telegramDet.H = int(y2 - y1)
		}

		telegramDetections = append(telegramDetections, telegramDet)
	}

	// Read and encode the image
	var imageB64 string
	var imageName string
	if imagePath != "" {
		imageData, err := os.ReadFile(imagePath)
		if err != nil {
			log.Printf("Failed to read image file %s: %v", imagePath, err)
		} else {
			imageB64 = base64.StdEncoding.EncodeToString(imageData)
			imageName = filepath.Base(imagePath)
		}
	}

	// Create detection JSON data
	detectionJson := map[string]interface{}{
		"image":           detectionFile.Image,
		"duration_ms":     detectionFile.DurationMs,
		"detections":      detectionFile.Detections,
		"imgsz":           detectionFile.Imgsz,
		"conf":            detectionFile.Conf,
		"iou":             detectionFile.Iou,
		"annotated_image": detectionFile.AnnotatedImage,
	}

	// Create the payload
	payload := TelegramPayload{
		CameraName:         cameraName,
		Timestamp:          time.Now().Unix(),
		Detections:         telegramDetections,
		DurationMs:         detectionFile.DurationMs,
		Imgsz:              detectionFile.Imgsz,
		Conf:               detectionFile.Conf,
		Iou:                detectionFile.Iou,
		DetectionJson:      detectionJson,
		DetectionJsonName:  fmt.Sprintf("detection_%d.json", eventID),
		Chat:               "security_group",
		Message:            message,
		AnnotatedImagePath: imagePath,
		AnnotatedImageB64:  imageB64,
		AnnotatedImageName: imageName,
	}

	// Convert to JSON
	jsonData, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal Telegram payload: %w", err)
	}

	// Send HTTP POST request
	resp, err := http.Post(cfg.TelegramURL, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to send Telegram notification: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("telegram API returned status %d: %s", resp.StatusCode, string(body))
	}

	log.Printf("Successfully sent Telegram notification for event %d", eventID)
	return nil
}

// DetectionData represents a single detection (matching the structure from detection_processor.go)
type DetectionData struct {
	Label string    `json:"label"`
	Score float64   `json:"score"`
	Xyxy  []float64 `json:"xyxy"`
}

// IsCameraExcludedFromDetection checks if a camera is excluded from detection processing
func IsCameraExcludedFromDetection(camera string) bool {
	for _, excludedCamera := range config.GetExcludedCameras() {
		if camera == excludedCamera {
			return true
		}
	}
	return false
}
