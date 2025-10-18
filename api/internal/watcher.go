package internal

import (
	"database/sql"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/fsnotify/fsnotify"
)

// FileWatcher handles real-time file system monitoring
type FileWatcher struct {
	db             *sql.DB
	watcher        *fsnotify.Watcher
	rootDir        string
	thumbsDir      string
	pendingDir     string
	telegramURL    string
	processedFiles map[string]time.Time
	logger         *log.Logger
	// Health monitoring
	lastCleanup time.Time
	eventCount  int64
	errorCount  int64
}

// NewFileWatcher creates a new file system watcher
func NewFileWatcher(db *sql.DB, rootDir, thumbsDir, pendingDir, telegramURL string) *FileWatcher {
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		NotifyCriticalError("file_watcher", "Failed to create file watcher", map[string]interface{}{
			"error": err.Error(),
		})
		log.Fatalf("Failed to create file watcher: %v", err)
	}

	return &FileWatcher{
		db:             db,
		watcher:        watcher,
		rootDir:        rootDir,
		thumbsDir:      thumbsDir,
		pendingDir:     pendingDir,
		telegramURL:    telegramURL,
		processedFiles: make(map[string]time.Time),
		logger:         log.New(os.Stdout, "[WATCHER] ", log.LstdFlags),
		lastCleanup:    time.Now(),
		eventCount:     0,
		errorCount:     0,
	}
}

// Start begins monitoring the file system for changes
func (fw *FileWatcher) Start() error {
	fw.logger.Printf("Starting file watcher for directory: %s", fw.rootDir)

	// Add the root directory to watch
	if err := fw.watcher.Add(fw.rootDir); err != nil {
		return err
	}

	// Recursively add all subdirectories
	if err := fw.addSubdirectories(fw.rootDir); err != nil {
		fw.logger.Printf("Warning: Failed to add some subdirectories: %v", err)
	}

	// Start the event processing goroutine
	go fw.processEvents()

	return nil
}

// addSubdirectories recursively adds all subdirectories to the watcher
func (fw *FileWatcher) addSubdirectories(dir string) error {
	return filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil // Skip errors, continue walking
		}
		if d.IsDir() {
			if err := fw.watcher.Add(path); err != nil {
				fw.logger.Printf("Warning: Could not watch directory %s: %v", path, err)
			}
		}
		return nil
	})
}

// processEvents handles file system events
func (fw *FileWatcher) processEvents() {
	for {
		select {
		case event, ok := <-fw.watcher.Events:
			if !ok {
				fw.logger.Printf("File watcher events channel closed")
				return
			}
			fw.handleEvent(event)

		case err, ok := <-fw.watcher.Errors:
			if !ok {
				fw.logger.Printf("File watcher errors channel closed")
				return
			}
			fw.logger.Printf("File watcher error: %v", err)
		}
	}
}

// handleEvent processes individual file system events
func (fw *FileWatcher) handleEvent(event fsnotify.Event) {
	fw.eventCount++

	// Only process write events (file creation/modification)
	if !event.Has(fsnotify.Write) {
		return
	}

	filePath := event.Name
	lowerPath := strings.ToLower(filePath)

	// Only process video and image files
	if !strings.HasSuffix(lowerPath, ".mp4") &&
		!strings.HasSuffix(lowerPath, ".jpg") &&
		!strings.HasSuffix(lowerPath, ".jpeg") {
		return
	}

	// Check if we've already processed this file recently (avoid duplicates)
	if fw.isRecentlyProcessed(filePath) {
		return
	}

	fw.logger.Printf("New file detected: %s", filePath)

	// Process the file immediately
	go fw.processFile(filePath)
}

// isRecentlyProcessed checks if a file was processed recently to avoid duplicates
func (fw *FileWatcher) isRecentlyProcessed(filePath string) bool {
	now := time.Now()
	if lastProcessed, exists := fw.processedFiles[filePath]; exists {
		// If processed within last 5 seconds, skip
		if now.Sub(lastProcessed) < 5*time.Second {
			return true
		}
	}
	fw.processedFiles[filePath] = now

	// Clean up old entries to prevent memory leak
	fw.cleanupOldProcessedFiles(now)

	return false
}

// cleanupOldProcessedFiles removes entries older than 1 minute to prevent memory leak
func (fw *FileWatcher) cleanupOldProcessedFiles(now time.Time) {
	// Only clean up every 100th call to avoid overhead
	if len(fw.processedFiles)%100 != 0 {
		return
	}

	cutoff := now.Add(-1 * time.Minute)
	cleanedCount := 0
	for filePath, lastProcessed := range fw.processedFiles {
		if lastProcessed.Before(cutoff) {
			delete(fw.processedFiles, filePath)
			cleanedCount++
		}
	}

	if cleanedCount > 0 {
		fw.logger.Printf("Cleaned up %d old processed file entries, %d remaining", cleanedCount, len(fw.processedFiles))
	}
	fw.lastCleanup = now
}

// GetHealthStats returns health statistics for monitoring
func (fw *FileWatcher) GetHealthStats() map[string]interface{} {
	return map[string]interface{}{
		"processed_files_count": len(fw.processedFiles),
		"total_events":          fw.eventCount,
		"error_count":           fw.errorCount,
		"last_cleanup":          fw.lastCleanup,
		"watcher_active":        fw.watcher != nil,
	}
}

// processFile processes a single file for immediate indexing and alerting
func (fw *FileWatcher) processFile(filePath string) {
	defer func() {
		if r := recover(); r != nil {
			fw.errorCount++
			fw.logger.Printf("Panic in processFile for %s: %v", filePath, r)
		}
	}()

	// Wait a moment for file to be stable (reduced from 500ms to 100ms for speed)
	time.Sleep(100 * time.Millisecond)

	// Check if file is stable
	if !fw.isFileStable(filePath) {
		fw.logger.Printf("File not stable yet, skipping: %s", filePath)
		return
	}

	// Check if file already exists in database
	var count int
	err := fw.db.QueryRow("SELECT COUNT(*) FROM events WHERE path = ?", filePath).Scan(&count)
	if err != nil {
		fw.errorCount++
		fw.logger.Printf("Error checking file in database: %v", err)
		return
	}

	if count > 0 {
		fw.logger.Printf("File already in database, skipping: %s", filePath)
		return
	}

	fw.logger.Printf("Processing new file: %s", filePath)

	// Trigger immediate processing by creating a signal file
	// This will cause the background scanner to process this file immediately
	fw.triggerImmediateProcessing(filePath)
}

// triggerImmediateProcessing creates a signal to trigger immediate processing
func (fw *FileWatcher) triggerImmediateProcessing(filePath string) {
	// Create a signal file that the background scanner can detect
	signalFile := filepath.Join(fw.pendingDir, ".watcher_signal")

	// Write the file path to the signal file
	if err := os.WriteFile(signalFile, []byte(filePath), 0644); err != nil {
		fw.logger.Printf("Failed to create signal file: %v", err)
		return
	}

	fw.logger.Printf("Triggered immediate processing for: %s", filePath)
}

// isFileStable checks if a file is stable (not being written to)
func (fw *FileWatcher) isFileStable(filePath string) bool {
	info1, err := os.Stat(filePath)
	if err != nil {
		return false
	}

	// Wait a very short time and check again
	time.Sleep(100 * time.Millisecond)

	info2, err := os.Stat(filePath)
	if err != nil {
		return false
	}

	// File is stable if size and modification time haven't changed
	return info1.Size() == info2.Size() && info1.ModTime().Equal(info2.ModTime())
}

// Stop stops the file watcher
func (fw *FileWatcher) Stop() error {
	return fw.watcher.Close()
}
