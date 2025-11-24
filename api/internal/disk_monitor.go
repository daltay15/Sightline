package internal

import (
	"fmt"
	"time"

	"github.com/shirou/gopsutil/v3/disk"
)

// DiskMonitor handles periodic disk space monitoring
type DiskMonitor struct {
	enabled                 bool
	interval                time.Duration
	systemWarningThreshold  float64
	systemCriticalThreshold float64
	nasWarningThreshold     float64
	nasCriticalThreshold    float64
	nasPath                 string // Path to NAS mount point (configurable)
	lastSystemWarning       time.Time
	lastSystemCritical      time.Time
	lastNasWarning          time.Time
	lastNasCritical         time.Time
	notificationCooldown    time.Duration
}

// NewDiskMonitor creates a new disk space monitor
func NewDiskMonitor(config map[string]interface{}) *DiskMonitor {
	enabled, _ := config["disk_monitoring_enabled"].(bool)
	if !enabled {
		return &DiskMonitor{enabled: false}
	}

	// Parse interval (default 5 minutes)
	intervalStr, _ := config["disk_monitoring_interval"].(string)
	interval, err := time.ParseDuration(intervalStr)
	if err != nil {
		interval = 5 * time.Minute
	}

	// Parse thresholds with defaults
	systemWarning, _ := config["disk_monitoring_system_warning_threshold"].(float64)
	if systemWarning == 0 {
		systemWarning = 80
	}
	systemCritical, _ := config["disk_monitoring_system_critical_threshold"].(float64)
	if systemCritical == 0 {
		systemCritical = 90
	}
	nasWarning, _ := config["disk_monitoring_nas_warning_threshold"].(float64)
	if nasWarning == 0 {
		nasWarning = 85
	}
	nasCritical, _ := config["disk_monitoring_nas_critical_threshold"].(float64)
	if nasCritical == 0 {
		nasCritical = 95
	}

	// Get NAS path from config, with fallback logic
	// Try camera_dir first (e.g., /mnt/nas/pool/Cameras/House), then nas_path, then default
	nasPath := "/mnt/nas"
	if cameraDir, ok := config["camera_dir"].(string); ok && cameraDir != "" {
		nasPath = cameraDir
	} else if nasPathConfig, ok := config["nas_path"].(string); ok && nasPathConfig != "" {
		nasPath = nasPathConfig
	}

	return &DiskMonitor{
		enabled:                 enabled,
		interval:                interval,
		systemWarningThreshold:  systemWarning,
		systemCriticalThreshold: systemCritical,
		nasWarningThreshold:     nasWarning,
		nasCriticalThreshold:    nasCritical,
		nasPath:                 nasPath,
		notificationCooldown:    30 * time.Minute, // Prevent spam
	}
}

// Start begins the disk monitoring goroutine
func (dm *DiskMonitor) Start() {
	if !dm.enabled {
		LogInfo("Disk monitoring is disabled")
		return
	}

	LogInfo("Starting disk space monitoring (interval: %v)", dm.interval)

	go func() {
		defer func() {
			if r := recover(); r != nil {
				LogError("Disk monitoring goroutine panicked: %v", r)
			}
		}()

		for {
			dm.checkDiskSpace()
			time.Sleep(dm.interval)
		}
	}()
}

// checkDiskSpace checks both system and NAS disk space
func (dm *DiskMonitor) checkDiskSpace() {
	// Check system disk space
	dm.checkSystemDiskSpace()

	// Check NAS disk space
	dm.checkNASDiskSpace()
}

// checkSystemDiskSpace monitors the system disk (/)
func (dm *DiskMonitor) checkSystemDiskSpace() {
	diskInfo, err := disk.Usage("/")
	if err != nil {
		NotifyError("error", "disk_monitor", "Failed to get system disk usage", map[string]interface{}{
			"error": err.Error(),
		})
		return
	}

	usedPercent := float64(diskInfo.Used) / float64(diskInfo.Total) * 100
	now := time.Now()

	// Check critical threshold
	if usedPercent >= dm.systemCriticalThreshold {
		// Only send if we haven't sent a critical notification recently
		if now.Sub(dm.lastSystemCritical) > dm.notificationCooldown {
			NotifyCriticalError("disk_monitor", "System disk space critically low", map[string]interface{}{
				"used_percent": fmt.Sprintf("%.1f%%", usedPercent),
				"used_gb":      fmt.Sprintf("%.1f GB", float64(diskInfo.Used)/(1024*1024*1024)),
				"free_gb":      fmt.Sprintf("%.1f GB", float64(diskInfo.Free)/(1024*1024*1024)),
				"total_gb":     fmt.Sprintf("%.1f GB", float64(diskInfo.Total)/(1024*1024*1024)),
				"threshold":    fmt.Sprintf("%.1f%%", dm.systemCriticalThreshold),
			})
			dm.lastSystemCritical = now
		}
	} else if usedPercent >= dm.systemWarningThreshold {
		// Check warning threshold
		if now.Sub(dm.lastSystemWarning) > dm.notificationCooldown {
			NotifyWarning("disk_monitor", "System disk space getting low", map[string]interface{}{
				"used_percent": fmt.Sprintf("%.1f%%", usedPercent),
				"used_gb":      fmt.Sprintf("%.1f GB", float64(diskInfo.Used)/(1024*1024*1024)),
				"free_gb":      fmt.Sprintf("%.1f GB", float64(diskInfo.Free)/(1024*1024*1024)),
				"total_gb":     fmt.Sprintf("%.1f GB", float64(diskInfo.Total)/(1024*1024*1024)),
				"threshold":    fmt.Sprintf("%.1f%%", dm.systemWarningThreshold),
			})
			dm.lastSystemWarning = now
		}
	}
}

// checkNASDiskSpace monitors the NAS disk
func (dm *DiskMonitor) checkNASDiskSpace() {
	// Use configured NAS path, with fallback to /mnt/nas
	nasPath := dm.nasPath
	if nasPath == "" {
		nasPath = "/mnt/nas"
	}

	diskInfo, err := disk.Usage(nasPath)
	if err != nil {
		// Fall back to /mnt/nas if configured path fails
		if nasPath != "/mnt/nas" {
			if diskInfo, err = disk.Usage("/mnt/nas"); err != nil {
				NotifyError("error", "disk_monitor", "Failed to get NAS disk usage", map[string]interface{}{
					"error":    err.Error(),
					"nas_path": nasPath,
				})
				return
			}
		} else {
			NotifyError("error", "disk_monitor", "Failed to get NAS disk usage", map[string]interface{}{
				"error":    err.Error(),
				"nas_path": nasPath,
			})
			return
		}
	}

	usedPercent := float64(diskInfo.Used) / float64(diskInfo.Total) * 100
	now := time.Now()

	// Check critical threshold
	if usedPercent >= dm.nasCriticalThreshold {
		// Only send if we haven't sent a critical notification recently
		if now.Sub(dm.lastNasCritical) > dm.notificationCooldown {
			NotifyCriticalError("disk_monitor", "NAS disk space critically low", map[string]interface{}{
				"used_percent": fmt.Sprintf("%.1f%%", usedPercent),
				"used_tb":      fmt.Sprintf("%.1f TB", float64(diskInfo.Used)/(1024*1024*1024*1024)),
				"free_tb":      fmt.Sprintf("%.1f TB", float64(diskInfo.Free)/(1024*1024*1024*1024)),
				"total_tb":     fmt.Sprintf("%.1f TB", float64(diskInfo.Total)/(1024*1024*1024*1024)),
				"threshold":    fmt.Sprintf("%.1f%%", dm.nasCriticalThreshold),
			})
			dm.lastNasCritical = now
		}
	} else if usedPercent >= dm.nasWarningThreshold {
		// Check warning threshold
		if now.Sub(dm.lastNasWarning) > dm.notificationCooldown {
			NotifyWarning("disk_monitor", "NAS disk space getting low", map[string]interface{}{
				"used_percent": fmt.Sprintf("%.1f%%", usedPercent),
				"used_tb":      fmt.Sprintf("%.1f TB", float64(diskInfo.Used)/(1024*1024*1024*1024)),
				"free_tb":      fmt.Sprintf("%.1f TB", float64(diskInfo.Free)/(1024*1024*1024*1024)),
				"total_tb":     fmt.Sprintf("%.1f TB", float64(diskInfo.Total)/(1024*1024*1024*1024)),
				"threshold":    fmt.Sprintf("%.1f%%", dm.nasWarningThreshold),
			})
			dm.lastNasWarning = now
		}
	}
}

// GetStatus returns the current disk monitoring status
func (dm *DiskMonitor) GetStatus() map[string]interface{} {
	if !dm.enabled {
		return map[string]interface{}{
			"enabled": false,
		}
	}

	// Get current disk usage
	systemInfo, systemErr := disk.Usage("/")

	// Use configured NAS path, with fallback to /mnt/nas
	nasPath := dm.nasPath
	if nasPath == "" {
		nasPath = "/mnt/nas"
	}
	nasInfo, nasErr := disk.Usage(nasPath)
	// Fall back to /mnt/nas if configured path fails
	if nasErr != nil && nasPath != "/mnt/nas" {
		nasInfo, nasErr = disk.Usage("/mnt/nas")
	}

	status := map[string]interface{}{
		"enabled":                   dm.enabled,
		"interval":                  dm.interval.String(),
		"system_warning_threshold":  dm.systemWarningThreshold,
		"system_critical_threshold": dm.systemCriticalThreshold,
		"nas_warning_threshold":     dm.nasWarningThreshold,
		"nas_critical_threshold":    dm.nasCriticalThreshold,
		"notification_cooldown":     dm.notificationCooldown.String(),
	}

	if systemErr == nil {
		systemUsedPercent := float64(systemInfo.Used) / float64(systemInfo.Total) * 100
		status["system_disk"] = map[string]interface{}{
			"used_percent": fmt.Sprintf("%.1f%%", systemUsedPercent),
			"used_gb":      fmt.Sprintf("%.1f GB", float64(systemInfo.Used)/(1024*1024*1024)),
			"free_gb":      fmt.Sprintf("%.1f GB", float64(systemInfo.Free)/(1024*1024*1024)),
			"total_gb":     fmt.Sprintf("%.1f GB", float64(systemInfo.Total)/(1024*1024*1024)),
		}
	} else {
		status["system_disk"] = map[string]interface{}{
			"error": systemErr.Error(),
		}
	}

	if nasErr == nil {
		nasUsedPercent := float64(nasInfo.Used) / float64(nasInfo.Total) * 100
		status["nas_disk"] = map[string]interface{}{
			"used_percent": fmt.Sprintf("%.1f%%", nasUsedPercent),
			"used_tb":      fmt.Sprintf("%.1f TB", float64(nasInfo.Used)/(1024*1024*1024*1024)),
			"free_tb":      fmt.Sprintf("%.1f TB", float64(nasInfo.Free)/(1024*1024*1024*1024)),
			"total_tb":     fmt.Sprintf("%.1f TB", float64(nasInfo.Total)/(1024*1024*1024*1024)),
		}
	} else {
		status["nas_disk"] = map[string]interface{}{
			"error": nasErr.Error(),
		}
	}

	return status
}
