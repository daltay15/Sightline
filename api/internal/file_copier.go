package internal

import (
	"database/sql"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/daltay15/security-camera-ui/api/config"
)

// FileCopier handles copying new files to the GPU processing pipeline
type FileCopier struct {
	db            *sql.DB
	cameraRoot    string
	pendingDir    string
	processingDir string
	completedDir  string
	failedDir     string
	lastCopyTime  time.Time
	lastCopiedTs  int64 // Track the last copied start_ts for chronological ordering
}

// NewFileCopier creates a new FileCopier instance
func NewFileCopier(db *sql.DB, cameraRoot, pendingDir, processingDir, completedDir, failedDir string) *FileCopier {
	// Ensure all directories exist
	for _, dir := range []string{pendingDir, processingDir, completedDir, failedDir} {
		if err := os.MkdirAll(dir, 0755); err != nil {
			log.Fatalf("Failed to create directory %s: %v", dir, err)
		}
	}

	// Load the last copied timestamp from the database
	lastCopiedTs := loadLastCopiedTimestamp(db)

	return &FileCopier{
		db:            db,
		cameraRoot:    cameraRoot,
		pendingDir:    pendingDir,
		processingDir: processingDir,
		completedDir:  completedDir,
		failedDir:     failedDir,
		lastCopyTime:  time.Now().Add(-24 * time.Hour), // Start from 24 hours ago
		lastCopiedTs:  lastCopiedTs,                    // Load from database or start from beginning
	}
}

// CopyNewFilesToPending copies new JPG files from the camera root to the pending directory
func (fc *FileCopier) CopyNewFilesToPending() error {
	// If this is the first run (lastCopiedTs = 0), only copy a small batch to avoid overwhelming the system
	var query string
	var args []interface{}

	// Build query with camera exclusions
	excludedCameras := strings.Join(config.GetExcludedCameras(), "','")
	query = fmt.Sprintf(`
			SELECT path, camera, start_ts 
			FROM events 
			WHERE start_ts > ? 
			AND (path LIKE '%%%s' OR path LIKE '%%%s')
			AND camera NOT IN ('%s')
			ORDER BY start_ts ASC
		`, ".jpg", ".jpeg", excludedCameras)
	args = []interface{}{fc.lastCopiedTs}

	rows, err := fc.db.Query(query, args...)
	if err != nil {
		return fmt.Errorf("failed to query new files: %w", err)
	}
	defer rows.Close()

	var filesToCopy []string
	var maxTs int64 = fc.lastCopiedTs
	for rows.Next() {
		var path, camera string
		var startTs int64
		if err := rows.Scan(&path, &camera, &startTs); err != nil {
			continue
		}
		filesToCopy = append(filesToCopy, path)
		if startTs > maxTs {
			maxTs = startTs
		}
	}

	if len(filesToCopy) == 0 {
		return nil // No new files to copy
	}

	log.Printf("FileCopier: Found %d new JPG files to copy to pending directory (lastCopiedTs: %d)", len(filesToCopy), fc.lastCopiedTs)

	// Ensure pending directory exists
	if err := os.MkdirAll(fc.pendingDir, 0755); err != nil {
		return fmt.Errorf("failed to create pending directory: %w", err)
	}

	// Copy files to pending directory
	copiedCount := 0
	for _, sourcePath := range filesToCopy {
		if err := fc.copyFileToPending(sourcePath); err != nil {
			log.Printf("FileCopier: Failed to copy %s: %v", sourcePath, err)
			continue
		}
		copiedCount++
	}

	// Update last copied timestamp for chronological ordering
	fc.lastCopiedTs = maxTs
	fc.lastCopyTime = time.Now()

	// Save the timestamp for persistence across restarts
	if err := fc.saveLastCopiedTimestamp(); err != nil {
		log.Printf("FileCopier: Warning - failed to save timestamp: %v", err)
	}

	log.Printf("FileCopier: Successfully copied %d/%d files to pending directory (up to timestamp %d)", copiedCount, len(filesToCopy), maxTs)
	return nil
}

// copyFileToPending copies a single file to the pending directory
func (fc *FileCopier) copyFileToPending(sourcePath string) error {
	// Check if source file exists
	if _, err := os.Stat(sourcePath); os.IsNotExist(err) {
		return fmt.Errorf("source file does not exist: %s", sourcePath)
	}

	// Generate destination filename
	// Use the original filename to maintain compatibility with GPU processing
	fileName := filepath.Base(sourcePath)
	destPath := filepath.Join(fc.pendingDir, fileName)

	// Check if file already exists in pending (avoid duplicates)
	if _, err := os.Stat(destPath); err == nil {
		log.Printf("FileCopier: File already exists in pending: %s", fileName)
		return nil
	}

	// Copy the file (keep original in place)
	if err := fc.copyFile(sourcePath, destPath); err != nil {
		return fmt.Errorf("failed to copy file: %w", err)
	}

	// log.Printf("FileCopier: Copied %s to pending directory (original preserved)", fileName)
	return nil
}

// copyFile performs the actual file copy operation
func (fc *FileCopier) copyFile(src, dst string) error {
	// Open source file
	sourceFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer sourceFile.Close()

	// Create destination file
	destFile, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer destFile.Close()

	// Copy the file
	_, err = io.Copy(destFile, sourceFile)
	if err != nil {
		return err
	}

	// Sync to ensure data is written to disk
	return destFile.Sync()
}

// GetProcessingStatus returns the current status of the GPU processing pipeline
func (fc *FileCopier) GetProcessingStatus() map[string]int {
	status := make(map[string]int)

	// Count files in each directory
	dirs := map[string]string{
		"pending":    fc.pendingDir,
		"processing": fc.processingDir,
		"completed":  fc.completedDir,
		"failed":     fc.failedDir,
	}

	for name, dir := range dirs {
		if entries, err := os.ReadDir(dir); err == nil {
			status[name] = len(entries)
		} else {
			status[name] = 0
		}
	}

	return status
}

// CleanupOldFiles removes files older than 7 days from processing directories
func (fc *FileCopier) CleanupOldFiles() error {
	cutoffTime := time.Now().AddDate(0, 0, -7) // 7 days ago

	dirs := []string{fc.processingDir, fc.completedDir, fc.failedDir}

	for _, dir := range dirs {
		if _, err := os.Stat(dir); os.IsNotExist(err) {
			continue
		}

		entries, err := os.ReadDir(dir)
		if err != nil {
			continue
		}

		for _, entry := range entries {
			if entry.IsDir() {
				continue
			}

			filePath := filepath.Join(dir, entry.Name())
			info, err := entry.Info()
			if err != nil {
				continue
			}

			if info.ModTime().Before(cutoffTime) {
				if err := os.Remove(filePath); err == nil {
					log.Printf("FileCopier: Cleaned up old file: %s", filePath)
				}
			}
		}
	}

	return nil
}

// loadLastCopiedTimestamp loads the last copied timestamp from the database
func loadLastCopiedTimestamp(db *sql.DB) int64 {
	// Try to get the highest start_ts from events that have been copied to pending
	// We'll use a simple approach: get the highest start_ts from events table
	// In a more sophisticated system, we could track this in a separate table
	var lastTs int64

	// Build query with camera exclusions
	excludedCameras := strings.Join(config.GetExcludedCameras(), "','")
	query := fmt.Sprintf(`
		SELECT MAX(start_ts) 
		FROM events 
		WHERE (path LIKE '%%%s' OR path LIKE '%%%s')
		AND start_ts > 0
		AND camera NOT IN ('%s')
	`, ".jpg", ".jpeg", excludedCameras)

	err := db.QueryRow(query).Scan(&lastTs)
	if err != nil || lastTs == 0 {
		log.Printf("FileCopier: No previous timestamp found, starting from beginning")
		return 0
	}

	log.Printf("FileCopier: Resuming from timestamp %d", lastTs)
	return lastTs
}

// saveLastCopiedTimestamp saves the last copied timestamp to the database
func (fc *FileCopier) saveLastCopiedTimestamp() error {
	// For now, we'll just log it. In a production system, you might want to store this
	// in a separate table or configuration file
	log.Printf("FileCopier: Last copied timestamp: %d", fc.lastCopiedTs)
	return nil
}
