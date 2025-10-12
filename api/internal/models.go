package internal

type Event struct {
	ID                 int64             `db:"id" json:"id"`
	Camera             string            `db:"camera" json:"camera"`
	Path               string            `db:"path" json:"path"`
	JpgPath            *string           `db:"jpg_path" json:"jpgPath,omitempty"`
	SheetPath          *string           `db:"sheet_path" json:"sheetPath,omitempty"`
	StartTS            int64             `db:"start_ts" json:"startTs"`
	DurationMS         int64             `db:"duration_ms" json:"durationMs"`
	SizeBytes          int64             `db:"size_bytes" json:"sizeBytes"`
	Reviewed           int               `db:"reviewed" json:"reviewed"`
	Tags               string            `db:"tags" json:"tags"`
	CreatedAt          int64             `db:"created_at" json:"createdAt"`
	DetectionData      *string           `db:"detection_data" json:"detectionData,omitempty"`
	DetectionUpdated   int64             `db:"detection_updated" json:"detectionUpdated"`
	OriginalEventID    *int64            `db:"original_event_id" json:"originalEventId,omitempty"`
	DetectionImagePath *string           `db:"detection_image_path" json:"detectionImagePath,omitempty"`
	VideoPath          *string           `db:"video_path" json:"videoPath,omitempty"`
	DetectionPath      *string           `db:"detection_path" json:"detectionPath,omitempty"`
	Detections         []DetectionResult `json:"detections,omitempty"`
}

type EventsResp struct {
	Items []Event `json:"items"`
	Next  *string `json:"next,omitempty"`
}

// DetectionResult represents a single object detection
type DetectionResult struct {
	ID         int64    `db:"id" json:"id"`
	EventID    int64    `db:"event_id" json:"eventId"`
	Type       string   `db:"detection_type" json:"type"`
	Confidence float64  `db:"confidence" json:"confidence"`
	BboxX      int      `db:"bbox_x" json:"bboxX"`
	BboxY      int      `db:"bbox_y" json:"bboxY"`
	BboxWidth  int      `db:"bbox_width" json:"bboxWidth"`
	BboxHeight int      `db:"bbox_height" json:"bboxHeight"`
	FrameIndex *int     `db:"frame_index" json:"frameIndex,omitempty"`
	Timestamp  *float64 `db:"timestamp" json:"timestamp,omitempty"`
	CreatedAt  int64    `db:"created_at" json:"createdAt"`
}

// DetectionStats represents statistics about detections
type DetectionStats struct {
	TotalDetections  int                   `json:"totalDetections"`
	ByType           map[string]int        `json:"byType"`
	ByConfidence     map[string]int        `json:"byConfidence"`
	RecentDetections []DetectionResult     `json:"recentDetections"`
	DailyStats       []DailyDetectionStats `json:"dailyStats"`
}

type DailyDetectionStats struct {
	Date   string         `json:"date"`
	Count  int            `json:"count"`
	ByType map[string]int `json:"byType"`
}

// GPU Detection Service Models
type DetectionRequest struct {
	FilePath    string                 `json:"file_path"`
	FileType    string                 `json:"file_type"`
	Options     map[string]interface{} `json:"options,omitempty"`
	CallbackURL *string                `json:"callback_url,omitempty"`
	Metadata    map[string]string      `json:"metadata,omitempty"`
}

type Detection struct {
	Type       string   `json:"type"`
	Confidence float64  `json:"confidence"`
	Bbox       Bbox     `json:"bbox"`
	FrameIndex *int     `json:"frame_index,omitempty"`
	Timestamp  *float64 `json:"timestamp,omitempty"`
}

type Bbox struct {
	X      int `json:"x"`
	Y      int `json:"y"`
	Width  int `json:"width"`
	Height int `json:"height"`
}

type DetectionResponse struct {
	RequestID        string      `json:"request_id"`
	FilePath         string      `json:"file_path"`
	FileType         string      `json:"file_type"`
	Detections       []Detection `json:"detections"`
	ProcessingTimeMs float64     `json:"processing_time_ms"`
	FrameCount       *int        `json:"frame_count,omitempty"`
	Success          bool        `json:"success"`
	Error            *string     `json:"error,omitempty"`
	Timestamp        string      `json:"timestamp"`
}

type GPUStatus struct {
	Enabled            bool    `json:"enabled"`
	DeviceID           int     `json:"device_id"`
	MemoryTotalMB      int     `json:"memory_total_mb"`
	MemoryUsedMB       int     `json:"memory_used_mb"`
	UtilizationPercent float64 `json:"utilization_percent"`
}

type PerformanceStats struct {
	ActiveJobs       int     `json:"active_jobs"`
	MaxConcurrent    int     `json:"max_concurrent"`
	AverageLatencyMs float64 `json:"average_latency_ms"`
	TotalProcessed   int     `json:"total_processed"`
	ErrorRate        float64 `json:"error_rate"`
}

type StatusResponse struct {
	Service     string           `json:"service"`
	Status      string           `json:"status"`
	Version     string           `json:"version"`
	GPU         GPUStatus        `json:"gpu"`
	Performance PerformanceStats `json:"performance"`
	Timestamp   string           `json:"timestamp"`
}
