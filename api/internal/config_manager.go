package internal

import (
	"encoding/json"
	"os"
	"path/filepath"
	"time"
)

// ConfigManager handles loading and saving configuration
type ConfigManager struct {
	configFile string
	config     map[string]interface{}
}

// NewConfigManager creates a new configuration manager
func NewConfigManager(configFile string) *ConfigManager {
	return &ConfigManager{
		configFile: configFile,
		config:     make(map[string]interface{}),
	}
}

// LoadConfig loads configuration from file
func (cm *ConfigManager) LoadConfig() error {
	// Check if config file exists
	if _, err := os.Stat(cm.configFile); os.IsNotExist(err) {
		// Create default config if it doesn't exist
		cm.setDefaults()
		return cm.SaveConfig()
	}

	// Read config file
	data, err := os.ReadFile(cm.configFile)
	if err != nil {
		return err
	}

	// Parse JSON
	if err := json.Unmarshal(data, &cm.config); err != nil {
		return err
	}

	// Merge with defaults for any missing keys
	cm.mergeWithDefaults()

	return nil
}

// SaveConfig saves configuration to file
func (cm *ConfigManager) SaveConfig() error {
	// Ensure directory exists
	dir := filepath.Dir(cm.configFile)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	// Marshal to JSON
	data, err := json.MarshalIndent(cm.config, "", "  ")
	if err != nil {
		return err
	}

	// Write to file
	return os.WriteFile(cm.configFile, data, 0644)
}

// GetConfig returns the current configuration
func (cm *ConfigManager) GetConfig() map[string]interface{} {
	return cm.config
}

// SetConfig sets the configuration
func (cm *ConfigManager) SetConfig(config map[string]interface{}) {
	cm.config = config
}

// GetString gets a string value from config
func (cm *ConfigManager) GetString(key string, defaultValue string) string {
	if value, exists := cm.config[key]; exists {
		if str, ok := value.(string); ok {
			return str
		}
	}
	return defaultValue
}

// GetInt gets an integer value from config
func (cm *ConfigManager) GetInt(key string, defaultValue int) int {
	if value, exists := cm.config[key]; exists {
		switch v := value.(type) {
		case int:
			return v
		case float64:
			return int(v)
		}
	}
	return defaultValue
}

// GetBool gets a boolean value from config
func (cm *ConfigManager) GetBool(key string, defaultValue bool) bool {
	if value, exists := cm.config[key]; exists {
		if b, ok := value.(bool); ok {
			return b
		}
	}
	return defaultValue
}

// setDefaults sets default configuration values
func (cm *ConfigManager) setDefaults() {
	cm.config = map[string]interface{}{
		"camera_dir":             "/mnt/cameras",
		"server_host":            "0.0.0.0",
		"server_port":            8080,
		"cors_origins":           "*",
		"db_path":                "./data/events.db",
		"db_backup_enabled":      true,
		"db_backup_interval":     "24h",
		"gpu_detection_url":      "http://localhost:8000",
		"gpu_detection_timeout":  30,
		"detection_confidence":   0.45,
		"detection_classes":      "person,car,truck,bus,bicycle,motorcycle,dog,bird",
		"max_workers":            8,
		"thumbnail_quality":      85,
		"thumbnail_size":         320,
		"cache_size":             100,
		"rate_limit_enabled":     false,
		"rate_limit_requests":    100,
		"rate_limit_window":      "1m",
		"telegram_enabled":       false,
		"telegram_token":         "",
		"telegram_default_chat":  "",
		"telegram_security_chat": "",
		"retention_enabled":      false,
		"retention_unit":         "day",
		"retention_amount":       30,
		"created_at":             time.Now().Unix(),
		"updated_at":             time.Now().Unix(),
	}
}

// mergeWithDefaults merges current config with defaults for missing keys
func (cm *ConfigManager) mergeWithDefaults() {
	defaults := map[string]interface{}{
		"camera_dir":             "/mnt/cameras",
		"server_host":            "0.0.0.0",
		"server_port":            8080,
		"cors_origins":           "*",
		"db_path":                "./data/events.db",
		"db_backup_enabled":      true,
		"db_backup_interval":     "24h",
		"gpu_detection_url":      "http://localhost:8000",
		"gpu_detection_timeout":  30,
		"detection_confidence":   0.45,
		"detection_classes":      "person,car,truck,bus,bicycle,motorcycle,dog,bird",
		"max_workers":            8,
		"thumbnail_quality":      85,
		"thumbnail_size":         320,
		"cache_size":             100,
		"rate_limit_enabled":     false,
		"rate_limit_requests":    100,
		"rate_limit_window":      "1m",
		"telegram_enabled":       false,
		"telegram_token":         "",
		"telegram_default_chat":  "",
		"telegram_security_chat": "",
		"retention_enabled":      false,
		"retention_unit":         "day",
		"retention_amount":       30,
	}

	// Add missing keys from defaults
	for key, value := range defaults {
		if _, exists := cm.config[key]; !exists {
			cm.config[key] = value
		}
	}

	// Update timestamp
	cm.config["updated_at"] = time.Now().Unix()
}

// ValidateConfig validates the configuration
func (cm *ConfigManager) ValidateConfig() []string {
	var errors []string

	// Validate camera directory
	if cameraDir := cm.GetString("camera_dir", ""); cameraDir == "" {
		errors = append(errors, "Camera directory is required")
	}

	// Validate server port
	if port := cm.GetInt("server_port", 0); port < 1 || port > 65535 {
		errors = append(errors, "Server port must be between 1 and 65535")
	}

	// Validate detection confidence
	if confidence := cm.GetFloat("detection_confidence", 0); confidence < 0 || confidence > 1 {
		errors = append(errors, "Detection confidence must be between 0 and 1")
	}

	// Validate max workers
	if workers := cm.GetInt("max_workers", 0); workers < 1 || workers > 32 {
		errors = append(errors, "Max workers must be between 1 and 32")
	}

	// Validate thumbnail quality
	if quality := cm.GetInt("thumbnail_quality", 0); quality < 1 || quality > 100 {
		errors = append(errors, "Thumbnail quality must be between 1 and 100")
	}

	// Validate thumbnail size
	if size := cm.GetInt("thumbnail_size", 0); size < 64 || size > 1024 {
		errors = append(errors, "Thumbnail size must be between 64 and 1024")
	}

	// Validate cache size
	if cache := cm.GetInt("cache_size", 0); cache < 10 || cache > 1000 {
		errors = append(errors, "Cache size must be between 10 and 1000 MB")
	}

	return errors
}

// GetFloat gets a float value from config
func (cm *ConfigManager) GetFloat(key string, defaultValue float64) float64 {
	if value, exists := cm.config[key]; exists {
		switch v := value.(type) {
		case float64:
			return v
		case int:
			return float64(v)
		}
	}
	return defaultValue
}

// TestConfig tests the configuration by validating paths and connections
func (cm *ConfigManager) TestConfig() map[string]interface{} {
	results := make(map[string]interface{})

	// Test camera directory
	cameraDir := cm.GetString("camera_dir", "")
	if cameraDir != "" {
		if info, err := os.Stat(cameraDir); err != nil {
			results["camera_dir"] = map[string]interface{}{
				"valid":   false,
				"error":   err.Error(),
				"message": "Camera directory does not exist or is not accessible",
			}
		} else if !info.IsDir() {
			results["camera_dir"] = map[string]interface{}{
				"valid":   false,
				"error":   "Not a directory",
				"message": "Camera directory path is not a directory",
			}
		} else {
			results["camera_dir"] = map[string]interface{}{
				"valid":   true,
				"message": "Camera directory is accessible",
			}
		}
	}

	// Test database path
	dbPath := cm.GetString("db_path", "")
	if dbPath != "" {
		dir := filepath.Dir(dbPath)
		if info, err := os.Stat(dir); err != nil {
			results["db_path"] = map[string]interface{}{
				"valid":   false,
				"error":   err.Error(),
				"message": "Database directory does not exist",
			}
		} else if !info.IsDir() {
			results["db_path"] = map[string]interface{}{
				"valid":   false,
				"error":   "Not a directory",
				"message": "Database path is not a directory",
			}
		} else {
			results["db_path"] = map[string]interface{}{
				"valid":   true,
				"message": "Database directory is accessible",
			}
		}
	}

	// Test GPU detection URL
	gpuUrl := cm.GetString("gpu_detection_url", "")
	if gpuUrl != "" {
		results["gpu_detection"] = map[string]interface{}{
			"valid":   true,
			"message": "GPU detection URL is configured",
			"url":     gpuUrl,
		}
	}

	return results
}
