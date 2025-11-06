package internal

import (
	"fmt"
	"runtime"
	"time"

	"github.com/daltay15/security-camera-ui/api/internal/telegram"
)

// GlobalErrorNotifier is a global error notification service
type GlobalErrorNotifier struct {
	telegramClient *telegram.ErrorTelegramClient
	pythonEndpoint string
	enabled        bool
}

var globalErrorNotifier *GlobalErrorNotifier

// InitErrorNotifier initializes the global error notification service
func InitErrorNotifier(telegramClient *telegram.ErrorTelegramClient, pythonEndpoint string, enabled bool) {
	globalErrorNotifier = &GlobalErrorNotifier{
		telegramClient: telegramClient,
		pythonEndpoint: pythonEndpoint,
		enabled:        enabled,
	}
}

// NotifyError sends an error notification if the service is enabled
func NotifyError(severity, component, message string, metadata map[string]interface{}) {
	if globalErrorNotifier == nil || !globalErrorNotifier.enabled {
		return
	}

	// Add runtime information to metadata
	if metadata == nil {
		metadata = make(map[string]interface{})
	}
	metadata["timestamp"] = time.Now().Format(time.RFC3339)
	metadata["goroutine_id"] = getGoroutineID()

	// Send to Python endpoint (which forwards to Telegram)
	if globalErrorNotifier.telegramClient != nil {
		if err := globalErrorNotifier.telegramClient.SendErrorWithData(severity, component, message, metadata); err != nil {
			LogError("Failed to send error notification via Python endpoint: %v", err)
		}
	}
}

// NotifyErrorWithStack sends an error notification with stack trace
func NotifyErrorWithStack(severity, component, message string, metadata map[string]interface{}, stack string) {
	if globalErrorNotifier == nil || !globalErrorNotifier.enabled {
		return
	}

	// Add runtime information to metadata
	if metadata == nil {
		metadata = make(map[string]interface{})
	}
	metadata["timestamp"] = time.Now().Format(time.RFC3339)
	metadata["goroutine_id"] = getGoroutineID()

	// Send to Python endpoint (which forwards to Telegram)
	if globalErrorNotifier.telegramClient != nil {
		if err := globalErrorNotifier.telegramClient.SendErrorNotification(severity, component, message, metadata, stack, ""); err != nil {
			LogError("Failed to send error notification via Python endpoint: %v", err)
		}
	}
}

// NotifyCriticalError sends a critical error notification
func NotifyCriticalError(component, message string, metadata map[string]interface{}) {
	NotifyError("critical", component, message, metadata)
}

// NotifyWarning sends a warning notification
func NotifyWarning(component, message string, metadata map[string]interface{}) {
	NotifyError("warning", component, message, metadata)
}

// NotifyInfo sends an info notification
func NotifyInfo(component, message string, metadata map[string]interface{}) {
	NotifyError("info", component, message, metadata)
}

// getGoroutineID gets the current goroutine ID (approximate)
func getGoroutineID() int {
	var buf [64]byte
	n := runtime.Stack(buf[:], false)
	id := 0
	fmt.Sscanf(string(buf[:n]), "goroutine %d", &id)
	return id
}

// IsErrorNotifierEnabled returns whether error notifications are enabled
func IsErrorNotifierEnabled() bool {
	return globalErrorNotifier != nil && globalErrorNotifier.enabled
}
