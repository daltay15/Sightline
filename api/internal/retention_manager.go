package internal

import (
	"database/sql"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/daltay15/security-camera-ui/api/config"
)

// RetentionManager handles automatic data retention cleanup
type RetentionManager struct {
	db            *sql.DB
	configManager *ConfigManager
	enabled       bool
	unit          string // "day", "week", "month"
	amount        int    // how many units to keep
	lastRun       time.Time
}

// NewRetentionManager creates a new retention manager
func NewRetentionManager(db *sql.DB, configManager *ConfigManager) *RetentionManager {
	return &RetentionManager{
		db:            db,
		configManager: configManager,
		enabled:       false,
		unit:          "day",
		amount:        30,
	}
}

// UpdateSettings updates the retention manager settings from config
func (rm *RetentionManager) UpdateSettings() {
	rm.enabled = rm.configManager.GetBool("retention_enabled", false)
	rm.unit = rm.configManager.GetString("retention_unit", "day")
	rm.amount = rm.configManager.GetInt("retention_amount", 30)
	log.Printf("Retention settings updated: enabled=%v, keep last %d %s(s)", rm.enabled, rm.amount, rm.unit)
}

// StartRetentionScheduler starts the retention scheduler (runs nightly)
func (rm *RetentionManager) StartRetentionScheduler() {
	if !rm.enabled {
		log.Printf("Data retention is disabled")
		return
	}

	log.Printf("Data retention scheduler started (keeping last %d %s(s))", rm.amount, rm.unit)

	// Start retention goroutine
	go func() {
		defer func() {
			if r := recover(); r != nil {
				log.Printf("Retention manager panicked: %v", r)
			}
		}()

		for {
			// Calculate next run time (next day at 2 AM local time)
			now := time.Now()
			nextRun := time.Date(now.Year(), now.Month(), now.Day(), 2, 0, 0, 0, time.Local)
			if nextRun.Before(now) || nextRun.Equal(now) {
				// If 2 AM has passed today, schedule for tomorrow
				nextRun = nextRun.AddDate(0, 0, 1)
			}

			waitTime := time.Until(nextRun)
			log.Printf("Next data retention check scheduled in %v (at %s)", waitTime, nextRun.Format("2006-01-02 15:04:05"))

			time.Sleep(waitTime)

			// Update settings from config before running (in case user changed them)
			rm.UpdateSettings()

			if rm.enabled {
				// Perform retention cleanup
				if err := rm.PerformRetentionCleanup(); err != nil {
					log.Printf("Data retention cleanup failed: %v", err)
				} else {
					log.Printf("Data retention cleanup completed successfully")
					rm.lastRun = time.Now()
				}
			}
		}
	}()
}

// PerformRetentionCleanup performs the actual retention cleanup
func (rm *RetentionManager) PerformRetentionCleanup() error {
	if !rm.enabled {
		return fmt.Errorf("retention is disabled")
	}

	log.Printf("Starting data retention cleanup: keeping last %d %s(s)", rm.amount, rm.unit)

	// Calculate the cutoff timestamp (everything before this should be deleted)
	now := time.Now()
	var cutoffTime time.Time

	switch strings.ToLower(rm.unit) {
	case "day", "days":
		cutoffTime = now.AddDate(0, 0, -rm.amount)
	case "week", "weeks":
		cutoffTime = now.AddDate(0, 0, -7*rm.amount)
	case "month", "months":
		cutoffTime = now.AddDate(0, -rm.amount, 0)
	default:
		return fmt.Errorf("invalid retention unit: %s", rm.unit)
	}

	cutoffTs := cutoffTime.Unix()

	log.Printf("Deleting data older than %s (timestamp: %d)", cutoffTime.Format("2006-01-02 15:04:05"), cutoffTs)

	// Use the same deletion logic as the manual delete endpoint
	return rm.deleteDataByTimestamp(cutoffTs)
}

// deleteDataByTimestamp deletes all data older than the given timestamp
// This mirrors the logic from the /delete/range endpoint
func (rm *RetentionManager) deleteDataByTimestamp(cutoffTs int64) error {
	// Collect base event candidates (exclude detection pseudo-camera rows)
	rows, err := rm.db.Query(`
		SELECT id, path, COALESCE(jpg_path,''), COALESCE(sheet_path,''), COALESCE(video_path,''), COALESCE(detection_path,''), COALESCE(size_bytes, 0)
		FROM events
		WHERE start_ts < ? AND camera != 'DETECTION'
	`, cutoffTs)
	if err != nil {
		return fmt.Errorf("failed to query events: %w", err)
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
	for rows.Next() {
		var e evt
		if err := rows.Scan(&e.id, &e.p1, &e.p2, &e.p3, &e.p4, &e.p5, &e.sizeBytes); err != nil {
			continue
		}
		baseEvents = append(baseEvents, e)
	}

	if len(baseEvents) == 0 {
		log.Printf("No data to delete (all data is within retention period)")
		return nil
	}

	log.Printf("Found %d events to delete", len(baseEvents))

	// Build list of IDs for deletion (base events)
	ids := make([]int64, 0, len(baseEvents))
	for _, e := range baseEvents {
		ids = append(ids, e.id)
	}

	// Also collect related detection events for these base events
	var detectionEventIDs []int64
	if len(ids) > 0 {
		placeholders := strings.Repeat("?,", len(ids)-1) + "?"
		args := make([]any, len(ids))
		for i, id := range ids {
			args[i] = id
		}
		detRows, err := rm.db.Query(`
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
			detRows2, err2 := rm.db.Query(q, tagArgs...)
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
	dayFolders := make(map[string]bool)
	monthFolders := make(map[string]bool)
	for _, e := range baseEvents {
		if e.p1 != "" {
			dayFolder := filepath.Dir(e.p1)
			dayFolders[dayFolder] = true
			// Track month folders
			monthFolder := filepath.Dir(dayFolder)
			if monthFolder != "" && monthFolder != dayFolder {
				monthFolders[monthFolder] = true
			}
		}
	}

	// Delete entire day folders
	deletedFolders := 0
	for dayFolder := range dayFolders {
		if dayFolder == "" {
			continue
		}
		if err := os.RemoveAll(dayFolder); err == nil || os.IsNotExist(err) {
			deletedFolders++
		} else {
			log.Printf("Failed to delete day folder %s: %v", dayFolder, err)
		}
	}

	// Delete empty month folders
	deletedMonthFolders := 0
	for monthFolder := range monthFolders {
		entries, err := os.ReadDir(monthFolder)
		if err != nil {
			continue
		}
		if len(entries) == 0 {
			if err := os.Remove(monthFolder); err == nil || os.IsNotExist(err) {
				deletedMonthFolders++
			}
		}
	}

	// Clean up GPU completed artifacts
	cleanedDetections := 0
	if len(ids) > 0 {
		completedDir := config.CompletedDir
		prefixMap := make(map[string]bool, len(ids))
		for _, id := range ids {
			prefixMap[fmt.Sprintf("%d__", id)] = true
		}

		entries, err := os.ReadDir(completedDir)
		if err == nil {
			var filesToDelete []string
			for _, e := range entries {
				name := e.Name()
				if idx := strings.Index(name, "__"); idx > 0 {
					prefix := name[:idx+2]
					if prefixMap[prefix] {
						lowerName := strings.ToLower(name)
						if strings.HasSuffix(lowerName, "_det.jpg") ||
							strings.HasSuffix(lowerName, "_det.jpeg") ||
							strings.HasSuffix(lowerName, ".json") {
							filesToDelete = append(filesToDelete, filepath.Join(completedDir, name))
						}
					}
				}
			}

			for _, filePath := range filesToDelete {
				if err := os.Remove(filePath); err == nil || os.IsNotExist(err) {
					cleanedDetections++
				}
			}
		}
	}

	// Delete database records
	allIDs := append([]int64{}, ids...)
	allIDs = append(allIDs, detectionEventIDs...)
	if len(allIDs) > 0 {
		placeholders := strings.Repeat("?,", len(allIDs)-1) + "?"
		args := make([]any, len(allIDs))
		for i, id := range allIDs {
			args[i] = id
		}
		_, _ = rm.db.Exec("DELETE FROM detections WHERE event_id IN ("+placeholders+")", args...)
		_, err = rm.db.Exec("DELETE FROM events WHERE id IN ("+placeholders+")", args...)
		if err != nil {
			return fmt.Errorf("failed to delete database records: %w", err)
		}
	}

	log.Printf("Retention cleanup completed: deleted %d day folder(s), %d month folder(s), %d base events, %d detection events, %d detection artifacts",
		deletedFolders, deletedMonthFolders, len(ids), len(detectionEventIDs), cleanedDetections)

	return nil
}
