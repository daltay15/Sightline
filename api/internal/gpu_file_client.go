package internal

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"
)

// FileBasedGPUClient handles GPU detection through file system operations
type FileBasedGPUClient struct {
	basePath string
}

// NewFileBasedGPUClient creates a new file-based GPU client
func NewFileBasedGPUClient(basePath string) *FileBasedGPUClient {
	return &FileBasedGPUClient{
		basePath: basePath,
	}
}

// GPUProcessingRequest represents a request for GPU processing
type GPUProcessingRequest struct {
	ID          string                 `json:"id"`
	FilePath    string                 `json:"file_path"`
	FileType    string                 `json:"file_type"`
	Options     map[string]interface{} `json:"options"`
	RequestedAt time.Time              `json:"requested_at"`
}

// GPUProcessingResponse represents the result of GPU processing
type GPUProcessingResponse struct {
	ID               string      `json:"id"`
	FilePath         string      `json:"file_path"`
	FileType         string      `json:"file_type"`
	Detections       []Detection `json:"detections"`
	Success          bool        `json:"success"`
	Error            string      `json:"error,omitempty"`
	ProcessedAt      time.Time   `json:"processed_at"`
	ProcessingTimeMs float64     `json:"processing_time_ms"`
}

// Note: Detection and Bbox types are already defined in models.go

// SubmitProcessingRequest submits a file for GPU processing
func (c *FileBasedGPUClient) SubmitProcessingRequest(request GPUProcessingRequest) error {
	// Create the pending directory if it doesn't exist
	pendingDir := filepath.Join(c.basePath, "pending")
	if err := os.MkdirAll(pendingDir, 0755); err != nil {
		return fmt.Errorf("failed to create pending directory: %w", err)
	}

	// Create request file
	requestFile := filepath.Join(pendingDir, request.ID+".json")

	// Write request to file
	data, err := json.MarshalIndent(request, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	if err := os.WriteFile(requestFile, data, 0644); err != nil {
		return fmt.Errorf("failed to write request file: %w", err)
	}

	log.Printf("GPU File Client: Submitted processing request %s for file %s", request.ID, request.FilePath)
	return nil
}

// CheckProcessingResult checks if a processing result is available
func (c *FileBasedGPUClient) CheckProcessingResult(requestID string) (*GPUProcessingResponse, error) {
	completedDir := filepath.Join(c.basePath, "completed")
	resultFile := filepath.Join(completedDir, requestID+".json")

	// Check if result file exists
	if _, err := os.Stat(resultFile); os.IsNotExist(err) {
		return nil, nil // No result yet
	}

	// Read result file
	data, err := os.ReadFile(resultFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read result file: %w", err)
	}

	var response GPUProcessingResponse
	if err := json.Unmarshal(data, &response); err != nil {
		return nil, fmt.Errorf("failed to unmarshal result: %w", err)
	}

	return &response, nil
}

// WaitForProcessingResult waits for a processing result with timeout
func (c *FileBasedGPUClient) WaitForProcessingResult(requestID string, timeout time.Duration) (*GPUProcessingResponse, error) {
	start := time.Now()

	for time.Since(start) < timeout {
		result, err := c.CheckProcessingResult(requestID)
		if err != nil {
			return nil, err
		}
		if result != nil {
			return result, nil
		}

		// Wait a bit before checking again
		time.Sleep(1 * time.Second)
	}

	return nil, fmt.Errorf("timeout waiting for processing result")
}

// DetectImage submits an image for detection processing
func (c *FileBasedGPUClient) DetectImage(request DetectionRequest) (*DetectionResponse, error) {
	// Generate unique request ID
	requestID := fmt.Sprintf("img_%d", time.Now().UnixNano())

	// Create processing request
	processingRequest := GPUProcessingRequest{
		ID:          requestID,
		FilePath:    request.FilePath,
		FileType:    "image",
		Options:     request.Options,
		RequestedAt: time.Now(),
	}

	// Submit request
	if err := c.SubmitProcessingRequest(processingRequest); err != nil {
		return nil, err
	}

	// Wait for result (with timeout)
	result, err := c.WaitForProcessingResult(requestID, 5*time.Minute)
	if err != nil {
		return nil, err
	}

	// Convert to DetectionResponse
	var errorPtr *string
	if result.Error != "" {
		errorPtr = &result.Error
	}

	response := &DetectionResponse{
		RequestID:        result.ID,
		FilePath:         result.FilePath,
		FileType:         result.FileType,
		Detections:       result.Detections,
		ProcessingTimeMs: result.ProcessingTimeMs,
		FrameCount:       nil,
		Success:          result.Success,
		Error:            errorPtr,
		Timestamp:        result.ProcessedAt.Format(time.RFC3339),
	}

	return response, nil
}

// DetectVideo submits a video for detection processing
func (c *FileBasedGPUClient) DetectVideo(request DetectionRequest) (*DetectionResponse, error) {
	// Generate unique request ID
	requestID := fmt.Sprintf("vid_%d", time.Now().UnixNano())

	// Create processing request
	processingRequest := GPUProcessingRequest{
		ID:          requestID,
		FilePath:    request.FilePath,
		FileType:    "video",
		Options:     request.Options,
		RequestedAt: time.Now(),
	}

	// Submit request
	if err := c.SubmitProcessingRequest(processingRequest); err != nil {
		return nil, err
	}

	// Wait for result (with longer timeout for videos)
	result, err := c.WaitForProcessingResult(requestID, 10*time.Minute)
	if err != nil {
		return nil, err
	}

	// Count unique frames
	frameCount := 0
	if len(result.Detections) > 0 {
		frames := make(map[int]bool)
		for _, det := range result.Detections {
			if det.FrameIndex != nil {
				frames[*det.FrameIndex] = true
			}
		}
		frameCount = len(frames)
	}

	// Convert to DetectionResponse
	var errorPtr *string
	if result.Error != "" {
		errorPtr = &result.Error
	}

	response := &DetectionResponse{
		RequestID:        result.ID,
		FilePath:         result.FilePath,
		FileType:         result.FileType,
		Detections:       result.Detections,
		ProcessingTimeMs: result.ProcessingTimeMs,
		FrameCount:       &frameCount,
		Success:          result.Success,
		Error:            errorPtr,
		Timestamp:        result.ProcessedAt.Format(time.RFC3339),
	}

	return response, nil
}

// GetStatus returns the status of the file-based GPU service
func (c *FileBasedGPUClient) GetStatus() (*StatusResponse, error) {
	// Check if base path exists and is accessible
	if _, err := os.Stat(c.basePath); os.IsNotExist(err) {
		return nil, fmt.Errorf("GPU processing directory does not exist: %s", c.basePath)
	}

	// Count pending and completed files
	pendingDir := filepath.Join(c.basePath, "pending")
	completedDir := filepath.Join(c.basePath, "completed")

	pendingCount := 0
	completedCount := 0

	if entries, err := os.ReadDir(pendingDir); err == nil {
		pendingCount = len(entries)
	}

	if entries, err := os.ReadDir(completedDir); err == nil {
		completedCount = len(entries)
	}

	return &StatusResponse{
		Service: "gpu-detection-file-based",
		Status:  "running",
		Version: "1.0.0",
		GPU: GPUStatus{
			Enabled:            true,
			DeviceID:           0,
			MemoryTotalMB:      0,
			MemoryUsedMB:       0,
			UtilizationPercent: 0.0,
		},
		Performance: PerformanceStats{
			ActiveJobs:       pendingCount,
			MaxConcurrent:    4,
			AverageLatencyMs: 0.0,
			TotalProcessed:   completedCount,
			ErrorRate:        0.0,
		},
		Timestamp: time.Now().Format(time.RFC3339),
	}, nil
}

// HealthCheck checks if the file-based GPU service is healthy
func (c *FileBasedGPUClient) HealthCheck() error {
	// Check if base path exists and is writable
	if _, err := os.Stat(c.basePath); os.IsNotExist(err) {
		return fmt.Errorf("GPU processing directory does not exist: %s", c.basePath)
	}

	// Try to create a test file
	testFile := filepath.Join(c.basePath, "health_check.tmp")
	if err := os.WriteFile(testFile, []byte("health check"), 0644); err != nil {
		return fmt.Errorf("GPU processing directory is not writable: %w", err)
	}

	// Clean up test file
	os.Remove(testFile)

	return nil
}
