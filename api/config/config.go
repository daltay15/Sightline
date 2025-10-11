package config

import (
	"os"
)

type Config struct {
	PendingDir    string
	ProcessingDir string
	CompletedDir  string
	FailedDir     string
}

func LoadConfig() *Config {
	config := &Config{
		PendingDir:    getEnv("PENDING_DIR", "/mnt/nas/pool/Cameras/GPU_Processing/pending"),
		ProcessingDir: getEnv("PROCESSING_DIR", "/mnt/nas/pool/Cameras/GPU_Processing/processing"),
		CompletedDir:  getEnv("COMPLETED_DIR", "/mnt/nas/pool/Cameras/GPU_Processing/completed"),
		FailedDir:     getEnv("FAILED_DIR", "/mnt/nas/pool/Cameras/GPU_Processing/failed"),
	}
	return config
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
