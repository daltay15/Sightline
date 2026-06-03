package internal

import (
	"crypto/sha1"
	"database/sql"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"sync"
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
	inFlightFiles  map[string]struct{}
	mu             sync.Mutex
	logger         *Logger
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
		LogFatal("Failed to create file watcher: %v", err)
	}

	return &FileWatcher{
		db:             db,
		watcher:        watcher,
		rootDir:        rootDir,
		thumbsDir:      thumbsDir,
		pendingDir:     pendingDir,
		telegramURL:    telegramURL,
		processedFiles: make(map[string]time.Time),
		inFlightFiles:  make(map[string]struct{}),
		logger:         NewLogger(INFO, "[WATCHER] ", os.Stdout),
		lastCleanup:    time.Now(),
		eventCount:     0,
		errorCount:     0,
	}
}

// Start begins monitoring the file system for changes
func (fw *FileWatcher) Start() error {
	fw.logger.Info("Starting file watcher for directory: %s", fw.rootDir)

	// Add the root directory to watch
	if err := fw.watcher.Add(fw.rootDir); err != nil {
		return err
	}

	// Recursively add all subdirectories
	if err := fw.addSubdirectories(fw.rootDir); err != nil {
		fw.logger.Warn("Failed to add some subdirectories: %v", err)
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
				fw.logger.Warn("Could not watch directory %s: %v", path, err)
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
				fw.logger.Warn("File watcher events channel closed")
				return
			}
			fw.handleEvent(event)

		case err, ok := <-fw.watcher.Errors:
			if !ok {
				fw.logger.Warn("File watcher errors channel closed")
				return
			}
			fw.logger.Error("File watcher error: %v", err)
		}
	}
}

// handleEvent processes individual file system events
func (fw *FileWatcher) handleEvent(event fsnotify.Event) {
	fw.eventCount++

	if event.Has(fsnotify.Create) {
		if info, err := os.Stat(event.Name); err == nil && info.IsDir() {
			if err := fw.addSubdirectories(event.Name); err != nil {
				fw.logger.Warn("Failed to watch new directory %s: %v", event.Name, err)
			}
			return
		}
	}

	// Process file creation and write events. Some filesystems emit only one of
	// these reliably for camera output, especially across network mounts.
	if !event.Has(fsnotify.Write) && !event.Has(fsnotify.Create) {
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

	// Images should be processed as quickly as possible. Videos are commonly
	// written over time by cameras, so debounce repeated write events and allow
	// only one processor goroutine per path.
	if strings.HasSuffix(lowerPath, ".mp4") {
		if fw.shouldSkipVideoEvent(filePath) {
			return
		}
	} else if fw.isRecentlyProcessed(filePath) {
		return
	}

	fw.logger.Info("New file detected: %s", filePath)

	// Process the file immediately
	go fw.processFile(filePath)
}

// shouldSkipVideoEvent debounces repeated .mp4 write events and prevents
// multiple goroutines from waiting on the same still-writing video file.
func (fw *FileWatcher) shouldSkipVideoEvent(filePath string) bool {
	now := time.Now()

	fw.mu.Lock()
	defer fw.mu.Unlock()

	if _, exists := fw.inFlightFiles[filePath]; exists {
		return true
	}

	if lastProcessed, exists := fw.processedFiles[filePath]; exists {
		if now.Sub(lastProcessed) < 2*time.Minute {
			return true
		}
	}

	fw.processedFiles[filePath] = now
	fw.inFlightFiles[filePath] = struct{}{}
	fw.cleanupOldProcessedFilesLocked(now)

	return false
}

// isRecentlyProcessed checks if a file was processed recently to avoid duplicates.
// This path is intentionally tuned for images so they still process quickly.
func (fw *FileWatcher) isRecentlyProcessed(filePath string) bool {
	now := time.Now()

	fw.mu.Lock()
	defer fw.mu.Unlock()

	if lastProcessed, exists := fw.processedFiles[filePath]; exists {
		// If processed within last 5 seconds, skip. This keeps images responsive
		// while still avoiding duplicate create/write events.
		if now.Sub(lastProcessed) < 5*time.Second {
			return true
		}
	}

	fw.processedFiles[filePath] = now
	fw.cleanupOldProcessedFilesLocked(now)

	return false
}

// cleanupOldProcessedFilesLocked removes entries older than 5 minutes to prevent memory leak.
// fw.mu must already be held by the caller.
func (fw *FileWatcher) cleanupOldProcessedFilesLocked(now time.Time) {
	// Only clean up every 100th tracked file to avoid overhead.
	if len(fw.processedFiles)%100 != 0 {
		return
	}

	cutoff := now.Add(-5 * time.Minute)
	cleanedCount := 0
	for filePath, lastProcessed := range fw.processedFiles {
		if lastProcessed.Before(cutoff) {
			delete(fw.processedFiles, filePath)
			cleanedCount++
		}
	}

	if cleanedCount > 0 {
		fw.logger.Debug("Cleaned up %d old processed file entries, %d remaining", cleanedCount, len(fw.processedFiles))
	}
	fw.lastCleanup = now
}

// GetHealthStats returns health statistics for monitoring
func (fw *FileWatcher) GetHealthStats() map[string]interface{} {
	fw.mu.Lock()
	processedCount := len(fw.processedFiles)
	inFlightCount := len(fw.inFlightFiles)
	fw.mu.Unlock()

	return map[string]interface{}{
		"processed_files_count": processedCount,
		"in_flight_files_count": inFlightCount,
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
			fw.logger.Error("Panic in processFile for %s: %v", filePath, r)
		}
	}()
	defer fw.markProcessingFinished(filePath)

	stableTimeout := 5 * time.Second
	if strings.HasSuffix(strings.ToLower(filePath), ".mp4") {
		stableTimeout = 2 * time.Minute
	}

	if !fw.waitForFileStable(filePath, stableTimeout) {
		fw.logger.Debug("File not stable after retry window, leaving for fallback scan: %s", filePath)
		return
	}

	// Check if file already exists in database
	var count int
	err := fw.db.QueryRow("SELECT COUNT(*) FROM events WHERE path = ?", filePath).Scan(&count)
	if err != nil {
		fw.errorCount++
		fw.logger.Error("Error checking file in database: %v", err)
		return
	}

	if count > 0 {
		fw.logger.Debug("File already in database, skipping: %s", filePath)
		return
	}

	fw.logger.Info("Processing new file: %s", filePath)

	// Trigger immediate processing by creating a signal file
	// This will cause the background scanner to process this file immediately
	fw.triggerImmediateProcessing(filePath)
}

func (fw *FileWatcher) markProcessingFinished(filePath string) {
	if !strings.HasSuffix(strings.ToLower(filePath), ".mp4") {
		return
	}

	fw.mu.Lock()
	delete(fw.inFlightFiles, filePath)
	fw.mu.Unlock()
}

// triggerImmediateProcessing creates a signal to trigger immediate processing
func (fw *FileWatcher) triggerImmediateProcessing(filePath string) {
	signalDir := filepath.Join(fw.pendingDir, ".watcher_signals")
	if err := os.MkdirAll(signalDir, 0755); err != nil {
		fw.logger.Error("Failed to create signal directory: %v", err)
		return
	}

	sum := sha1.Sum([]byte(filePath))
	signalFile := filepath.Join(signalDir, time.Now().Format("20060102150405.000000000")+"_"+hex.EncodeToString(sum[:])+".signal")
	if err := os.WriteFile(signalFile, []byte(filePath), 0644); err != nil {
		fw.logger.Error("Failed to create signal file: %v", err)
		return
	}

	fw.logger.Debug("Triggered immediate processing for: %s", filePath)
}

func (fw *FileWatcher) waitForFileStable(filePath string, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for {
		if fw.isFileStable(filePath) {
			return true
		}
		if time.Now().After(deadline) {
			return false
		}
		time.Sleep(250 * time.Millisecond)
	}
}

// isFileStable checks if a file is stable (not being written to)
func (fw *FileWatcher) isFileStable(filePath string) bool {
	lowerPath := strings.ToLower(filePath)
	if strings.HasSuffix(lowerPath, ".mp4") {
		return fw.isFileStableAcrossChecks(filePath, 3, 2*time.Second)
	}

	return fw.isFileStableAcrossChecks(filePath, 1, 100*time.Millisecond)
}

func (fw *FileWatcher) isFileStableAcrossChecks(filePath string, requiredStableChecks int, delay time.Duration) bool {
	info, err := os.Stat(filePath)
	if err != nil {
		return false
	}

	lastSize := info.Size()
	lastModTime := info.ModTime()
	stableChecks := 0

	for stableChecks < requiredStableChecks {
		time.Sleep(delay)

		info, err := os.Stat(filePath)
		if err != nil {
			return false
		}

		if info.Size() == lastSize && info.ModTime().Equal(lastModTime) {
			stableChecks++
			continue
		}

		lastSize = info.Size()
		lastModTime = info.ModTime()
		stableChecks = 0
	}

	return true
}

// Stop stops the file watcher
func (fw *FileWatcher) Stop() error {
	return fw.watcher.Close()
}
