package internal

import (
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// BackupManager handles database backup operations
type BackupManager struct {
	dbPath      string
	backupDir   string
	enabled     bool
	interval    string
	lastBackup  time.Time
	backupCount int
	maxBackups  int
}

// NewBackupManager creates a new backup manager
func NewBackupManager(dbPath string, backupDir string, enabled bool, interval string) *BackupManager {
	return &BackupManager{
		dbPath:     dbPath,
		backupDir:  backupDir,
		enabled:    enabled,
		interval:   interval,
		maxBackups: 10, // Keep last 10 backups
	}
}

// UpdateSettings updates the backup manager settings
func (bm *BackupManager) UpdateSettings(enabled bool, interval string) {
	bm.enabled = enabled
	bm.interval = interval
	log.Printf("Backup settings updated: enabled=%v, interval=%s", enabled, interval)
}

// StartBackupScheduler starts the backup scheduler
func (bm *BackupManager) StartBackupScheduler() {
	if !bm.enabled {
		log.Printf("Database backup is disabled")
		return
	}

	// Create backup directory if it doesn't exist
	if err := os.MkdirAll(bm.backupDir, 0755); err != nil {
		log.Printf("Failed to create backup directory: %v", err)
		return
	}

	log.Printf("Database backup scheduler started (interval: %s)", bm.interval)

	// Start backup goroutine
	go func() {
		for {
			// Calculate next backup time
			nextBackup := bm.calculateNextBackup()
			waitTime := time.Until(nextBackup)

			log.Printf("Next database backup scheduled in %v", waitTime)
			time.Sleep(waitTime)

			// Perform backup
			if err := bm.CreateBackup(); err != nil {
				log.Printf("Database backup failed: %v", err)
			} else {
				log.Printf("Database backup completed successfully")
			}
		}
	}()
}

// CreateBackup creates a backup of the database
func (bm *BackupManager) CreateBackup() error {
	if !bm.enabled {
		return fmt.Errorf("backup is disabled")
	}

	// Check if database file exists
	if _, err := os.Stat(bm.dbPath); os.IsNotExist(err) {
		return fmt.Errorf("database file does not exist: %s", bm.dbPath)
	}

	// Generate backup filename with timestamp
	timestamp := time.Now().Format("2006-01-02_15-04-05")
	backupFileName := fmt.Sprintf("events_backup_%s.db", timestamp)
	backupPath := filepath.Join(bm.backupDir, backupFileName)

	// Create backup
	if err := bm.copyFile(bm.dbPath, backupPath); err != nil {
		return fmt.Errorf("failed to create backup: %v", err)
	}

	// Update backup info
	bm.lastBackup = time.Now()
	bm.backupCount++

	// Clean up old backups
	if err := bm.cleanupOldBackups(); err != nil {
		log.Printf("Warning: failed to cleanup old backups: %v", err)
	}

	log.Printf("Database backup created: %s", backupPath)
	return nil
}

// copyFile copies a file from src to dst
func (bm *BackupManager) copyFile(src, dst string) error {
	sourceFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer sourceFile.Close()

	destFile, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer destFile.Close()

	_, err = io.Copy(destFile, sourceFile)
	if err != nil {
		return err
	}

	// Copy file permissions
	sourceInfo, err := os.Stat(src)
	if err != nil {
		return err
	}

	return os.Chmod(dst, sourceInfo.Mode())
}

// cleanupOldBackups removes old backup files
func (bm *BackupManager) cleanupOldBackups() error {
	entries, err := os.ReadDir(bm.backupDir)
	if err != nil {
		return err
	}

	// Find all backup files
	var backupFiles []os.FileInfo
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), "events_backup_") && strings.HasSuffix(entry.Name(), ".db") {
			if info, err := entry.Info(); err == nil {
				backupFiles = append(backupFiles, info)
			}
		}
	}

	// Sort by modification time (newest first)
	for i := 0; i < len(backupFiles)-1; i++ {
		for j := i + 1; j < len(backupFiles); j++ {
			if backupFiles[i].ModTime().Before(backupFiles[j].ModTime()) {
				backupFiles[i], backupFiles[j] = backupFiles[j], backupFiles[i]
			}
		}
	}

	// Remove old backups if we have more than maxBackups
	if len(backupFiles) > bm.maxBackups {
		for i := bm.maxBackups; i < len(backupFiles); i++ {
			oldBackupPath := filepath.Join(bm.backupDir, backupFiles[i].Name())
			if err := os.Remove(oldBackupPath); err != nil {
				log.Printf("Failed to remove old backup: %s", oldBackupPath)
			} else {
				log.Printf("Removed old backup: %s", oldBackupPath)
			}
		}
	}

	return nil
}

// calculateNextBackup calculates the next backup time based on interval
func (bm *BackupManager) calculateNextBackup() time.Time {
	now := time.Now()

	switch bm.interval {
	case "1h":
		return now.Add(1 * time.Hour)
	case "6h":
		return now.Add(6 * time.Hour)
	case "12h":
		return now.Add(12 * time.Hour)
	case "24h":
		return now.Add(24 * time.Hour)
	default:
		return now.Add(24 * time.Hour) // Default to daily
	}
}

// GetBackupStatus returns the current backup status
func (bm *BackupManager) GetBackupStatus() map[string]interface{} {
	return map[string]interface{}{
		"enabled":      bm.enabled,
		"interval":     bm.interval,
		"last_backup":  bm.lastBackup,
		"backup_count": bm.backupCount,
		"backup_dir":   bm.backupDir,
		"max_backups":  bm.maxBackups,
	}
}

// ListBackups returns a list of available backups
func (bm *BackupManager) ListBackups() ([]map[string]interface{}, error) {
	entries, err := os.ReadDir(bm.backupDir)
	if err != nil {
		return nil, err
	}

	var backups []map[string]interface{}
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), "events_backup_") && strings.HasSuffix(entry.Name(), ".db") {
			info, err := entry.Info()
			if err != nil {
				continue
			}

			backups = append(backups, map[string]interface{}{
				"filename": entry.Name(),
				"size":     info.Size(),
				"created":  info.ModTime(),
				"path":     filepath.Join(bm.backupDir, entry.Name()),
			})
		}
	}

	return backups, nil
}

// RestoreBackup restores the database from a backup
func (bm *BackupManager) RestoreBackup(backupPath string) error {
	if _, err := os.Stat(backupPath); os.IsNotExist(err) {
		return fmt.Errorf("backup file does not exist: %s", backupPath)
	}

	// Create a backup of current database before restoring
	currentBackupPath := fmt.Sprintf("%s.before_restore_%s", bm.dbPath, time.Now().Format("2006-01-02_15-04-05"))
	if err := bm.copyFile(bm.dbPath, currentBackupPath); err != nil {
		log.Printf("Warning: failed to backup current database before restore: %v", err)
	}

	// Restore from backup
	if err := bm.copyFile(backupPath, bm.dbPath); err != nil {
		return fmt.Errorf("failed to restore from backup: %v", err)
	}

	log.Printf("Database restored from backup: %s", backupPath)
	return nil
}

// TestBackupConnection tests if the backup system can access the database
func (bm *BackupManager) TestBackupConnection() error {
	// Test database file access
	if _, err := os.Stat(bm.dbPath); os.IsNotExist(err) {
		return fmt.Errorf("database file does not exist: %s", bm.dbPath)
	}

	// Test backup directory access
	if err := os.MkdirAll(bm.backupDir, 0755); err != nil {
		return fmt.Errorf("cannot create backup directory: %s", bm.backupDir)
	}

	// Test write access to backup directory
	testFile := filepath.Join(bm.backupDir, "test_backup_access.tmp")
	if err := os.WriteFile(testFile, []byte("test"), 0644); err != nil {
		return fmt.Errorf("cannot write to backup directory: %s", bm.backupDir)
	}

	// Clean up test file
	os.Remove(testFile)

	return nil
}
