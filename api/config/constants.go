package config

// Application constants
const (
	// Directory paths
	RootDir       = "/mnt/nas/pool/Cameras/House"
	PendingDir    = "/mnt/nas/pool/Cameras/GPU_Processing/pending"
	ProcessingDir = "/mnt/nas/pool/Cameras/GPU_Processing/processing"
	CompletedDir  = "/mnt/nas/pool/Cameras/GPU_Processing/completed"
	FailedDir     = "/mnt/nas/pool/Cameras/GPU_Processing/failed"
	DbPath        = "/media/nas/cache1/scui/data/events.db"
	ThumbsDir     = "/media/nas/cache1/scui/data/thumbs"

	// Network configuration
	Port        = ":8080"
	TelegramURL = "http://localhost:5000/send_detection"

	// Processing configuration
	MaxWorkers        = 4
	BatchSize         = 50
	ThumbnailWorkers  = 4
	ScanInterval      = 500 * 1000000 // 500ms in nanoseconds (fallback only)
	DetectionInterval = 500 * 1000000 // 500ms in nanoseconds

	// File watcher configuration
	FileWatcherEnabled = true
	FileStabilityDelay = 100 * 1000000 // 100ms in nanoseconds

	// Camera exclusions for detection processing
)

// GetExcludedCameras returns the list of cameras to exclude from detection processing
func GetExcludedCameras() []string {
	return []string{"1_00"}
}
