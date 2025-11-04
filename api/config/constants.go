package config

// Application constants
const (
	// Directory paths
	RootDir       = "/mnt/nas/pool/Cameras/House"
	PendingDir    = "/media/nas/cache1/scui/data/object_detection/pending"
	ProcessingDir = "/media/nas/cache1/scui/data/object_detection/processing"
	CompletedDir  = "/media/nas/cache1/scui/data/object_detection/completed"
	FailedDir     = "/media/nas/cache1/scui/data/object_detection/failed"
	DbPath        = "/media/nas/cache1/scui/data/events.db"
	ThumbsDir     = "/media/nas/cache1/scui/data/thumbs"

	// Network configuration
	Port        = ":8080"
	TelegramURL = "http://localhost:8080"

	// Processing configuration
	MaxWorkers        = 4
	BatchSize         = 50
	ThumbnailWorkers  = 4
	ScanInterval      = 500 * 1000000 // 500ms in nanoseconds (fallback only)
	DetectionInterval = 200 * 1000000 // 200ms in nanoseconds

	// File watcher configuration
	FileWatcherEnabled = true
	FileStabilityDelay = 100 * 1000000 // 100ms in nanoseconds

	// Camera exclusions for detection processing
)

// GetExcludedCameras returns the list of cameras to exclude from detection processing
func GetExcludedCameras() []string {
	return []string{"1_00"}
}
