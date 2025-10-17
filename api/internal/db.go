package internal

import (
	"database/sql"
	"time"

	_ "modernc.org/sqlite" // pure Go sqlite
)

func OpenDB(path string) (*sql.DB, error) {
	db, err := sql.Open("sqlite", path+"?_pragma=journal_mode(WAL)")
	if err != nil {
		NotifyCriticalError("database", "Failed to open database", map[string]interface{}{
			"path":  path,
			"error": err.Error(),
		})
		return nil, err
	}

	// Optimize connection pool for better performance
	db.SetMaxOpenConns(25)
	db.SetMaxIdleConns(5)
	db.SetConnMaxLifetime(5 * time.Minute)

	// Configure SQLite for better performance
	pragmas := []string{
		"PRAGMA busy_timeout=5000",
		"PRAGMA cache_size=10000",
		"PRAGMA temp_store=memory",
		"PRAGMA mmap_size=268435456", // 256MB
		"PRAGMA synchronous=NORMAL",
		"PRAGMA journal_mode=WAL",
		"PRAGMA wal_autocheckpoint=1000",
	}

	for _, pragma := range pragmas {
		if _, err := db.Exec(pragma); err != nil {
			NotifyCriticalError("database", "Failed to configure database pragma", map[string]interface{}{
				"pragma": pragma,
				"error":  err.Error(),
			})
			return nil, err
		}
	}

	return db, nil
}

func UpsertEvent(db *sql.DB, e *Event) error {
	_, err := db.Exec(`
INSERT INTO events (camera, path, jpg_path, sheet_path, start_ts, duration_ms, size_bytes, reviewed, tags, created_at)
VALUES (?, ?, ?, ?, ?, ?, ?, 0, '', ?)
ON CONFLICT(path) DO UPDATE SET
  jpg_path=excluded.jpg_path,
  sheet_path=excluded.sheet_path,
  duration_ms=excluded.duration_ms,
  size_bytes=excluded.size_bytes
  `,
		e.Camera, e.Path, e.JPGPathOrNil(), e.SheetPathOrNil(),
		e.StartTS, e.DurationMS, e.SizeBytes, time.Now().Unix())
	return err
}

func (e *Event) JPGPathOrNil() any {
	if e.JpgPath == nil {
		return nil
	}
	return *e.JpgPath
}
func (e *Event) SheetPathOrNil() any {
	if e.SheetPath == nil {
		return nil
	}
	return *e.SheetPath
}
