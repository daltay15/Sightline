package telegram

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// TelegramClient handles sending messages and photos to Telegram
type TelegramClient struct {
	token  string
	chats  map[string]string
	client *http.Client
}

// NewTelegramClient creates a new Telegram client from config
func NewTelegramClient(config map[string]interface{}) (*TelegramClient, error) {
	// Check if Telegram is enabled
	enabled, ok := config["telegram_enabled"].(bool)
	if !ok || !enabled {
		return nil, fmt.Errorf("telegram not enabled")
	}

	// Get token
	token, ok := config["telegram_token"].(string)
	if !ok || token == "" {
		return nil, fmt.Errorf("telegram token not configured")
	}

	// Build chats map
	chats := make(map[string]string)
	if defaultChat, ok := config["telegram_default_chat"].(string); ok && defaultChat != "" {
		chats["default"] = defaultChat
	}
	if securityChat, ok := config["telegram_security_chat"].(string); ok && securityChat != "" {
		chats["security_group"] = securityChat
	}

	if len(chats) == 0 {
		return nil, fmt.Errorf("no telegram chats configured")
	}

	return &TelegramClient{
		token:  token,
		chats:  chats,
		client: &http.Client{Timeout: 40 * time.Second},
	}, nil
}

// resolveChatID resolves a chat alias to a chat ID
func (tc *TelegramClient) resolveChatID(chat string) (string, error) {
	if chat == "" {
		if defaultChat, exists := tc.chats["default"]; exists && defaultChat != "" {
			return defaultChat, nil
		}
		return "", fmt.Errorf("no default chat configured")
	}

	// Check if it's an alias
	if chatID, exists := tc.chats[chat]; exists {
		return chatID, nil
	}

	// Assume it's a raw chat ID
	return chat, nil
}

// formatMessage formats a message with HTML markup
func formatMessage(msg string) string {
	msg = strings.TrimSpace(msg)

	if strings.HasPrefix(msg, "Starting:") {
		return fmt.Sprintf("🚀 <b>Starting</b>: %s", strings.TrimSpace(msg[9:]))
	} else if strings.HasPrefix(msg, "Succeeded:") {
		return fmt.Sprintf("✅ <b>Succeeded</b>: %s", strings.TrimSpace(msg[10:]))
	} else if strings.HasPrefix(msg, "Failure:") {
		content := strings.TrimSpace(msg[7:])
		return fmt.Sprintf("❌❌❌ <b>FAILURE DETECTED!</b> ❌❌❌\n<pre>%s</pre>", content)
	} else if strings.HasPrefix(msg, "Conflict:") {
		return fmt.Sprintf("⚠️ <b>Conflict</b>: %s", strings.TrimSpace(msg[9:]))
	} else if strings.HasPrefix(msg, "Cancelled:") {
		return fmt.Sprintf("🛑 <b>Cancelled</b>: %s", strings.TrimSpace(msg[10:]))
	}

	return fmt.Sprintf("ℹ️ %s", msg)
}

// formatDetectionCaption formats a detection caption
func formatDetectionCaption(payload map[string]interface{}) string {
	cam := "Camera"
	if cameraName, ok := payload["camera_name"].(string); ok && cameraName != "" {
		cam = cameraName
	} else if camera, ok := payload["camera"].(string); ok && camera != "" {
		cam = camera
	}

	// Format timestamp
	tsStr := time.Now().Format("2006-01-02 15:04:05")
	if ts, ok := payload["timestamp"]; ok {
		switch v := ts.(type) {
		case float64:
			tsStr = time.Unix(int64(v), 0).Format("2006-01-02 15:04:05")
		case int64:
			tsStr = time.Unix(v, 0).Format("2006-01-02 15:04:05")
		case string:
			tsStr = v
		}
	}

	// Process detections
	detections, _ := payload["detections"].([]interface{})
	if detections == nil {
		detections = []interface{}{}
	}

	// Count detections by label
	counts := make(map[string]int)
	for _, det := range detections {
		if detMap, ok := det.(map[string]interface{}); ok {
			if label, ok := detMap["label"].(string); ok {
				counts[label]++
			}
		}
	}

	// Get top 5 classes
	var topLines []string
	for label, count := range counts {
		topLines = append(topLines, fmt.Sprintf("%s×%d", label, count))
	}

	total := len(detections)
	metaParts := []string{}

	if duration, ok := payload["duration_ms"].(float64); ok {
		metaParts = append(metaParts, fmt.Sprintf("%.0fms", duration))
	}
	if conf, ok := payload["conf"].(float64); ok {
		metaParts = append(metaParts, fmt.Sprintf("conf≥%.2f", conf))
	}
	if iou, ok := payload["iou"].(float64); ok {
		metaParts = append(metaParts, fmt.Sprintf("IoU=%.2f", iou))
	}

	metaStr := ""
	if len(metaParts) > 0 {
		metaStr = fmt.Sprintf(" (%s)", strings.Join(metaParts, ", "))
	}

	lines := []string{
		fmt.Sprintf("📷 <b>%s</b> — %s", cam, tsStr),
		fmt.Sprintf("👀 Detections: <b>%d</b>%s", total, metaStr),
	}

	if len(topLines) > 0 {
		lines = append(lines, "• "+strings.Join(topLines, ", "))
	}

	return strings.Join(lines, "\n")
}

// SendMessage sends a text message to Telegram
func (tc *TelegramClient) SendMessage(message, chat string) error {
	chatID, err := tc.resolveChatID(chat)
	if err != nil {
		return err
	}

	url := fmt.Sprintf("https://api.telegram.org/bot%s/sendMessage", tc.token)

	payload := map[string]interface{}{
		"chat_id":                  chatID,
		"text":                     formatMessage(message),
		"parse_mode":               "HTML",
		"disable_web_page_preview": true,
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal payload: %w", err)
	}

	resp, err := tc.client.Post(url, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to send message: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("telegram API error: %s", string(body))
	}

	return nil
}

// SendPhoto sends a photo to Telegram
func (tc *TelegramClient) SendPhoto(photoPath, caption, chat string) error {
	chatID, err := tc.resolveChatID(chat)
	if err != nil {
		return err
	}

	url := fmt.Sprintf("https://api.telegram.org/bot%s/sendPhoto", tc.token)

	// Check if file exists
	if _, err := os.Stat(photoPath); os.IsNotExist(err) {
		return fmt.Errorf("photo file does not exist: %s", photoPath)
	}

	// Create multipart form
	var b bytes.Buffer
	w := multipart.NewWriter(&b)

	// Add chat_id
	if err := w.WriteField("chat_id", chatID); err != nil {
		return fmt.Errorf("failed to write chat_id: %w", err)
	}

	// Add parse_mode
	if err := w.WriteField("parse_mode", "HTML"); err != nil {
		return fmt.Errorf("failed to write parse_mode: %w", err)
	}

	// Add caption if provided
	if caption != "" {
		if err := w.WriteField("caption", caption); err != nil {
			return fmt.Errorf("failed to write caption: %w", err)
		}
	}

	// Add photo file
	file, err := os.Open(photoPath)
	if err != nil {
		return fmt.Errorf("failed to open photo: %w", err)
	}
	defer file.Close()

	fw, err := w.CreateFormFile("photo", filepath.Base(photoPath))
	if err != nil {
		return fmt.Errorf("failed to create form file: %w", err)
	}

	if _, err = io.Copy(fw, file); err != nil {
		return fmt.Errorf("failed to copy file: %w", err)
	}

	w.Close()

	req, err := http.NewRequest("POST", url, &b)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", w.FormDataContentType())

	resp, err := tc.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send photo: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("telegram API error: %s", string(body))
	}

	return nil
}

// SendPhotoBytes sends photo bytes to Telegram
func (tc *TelegramClient) SendPhotoBytes(imgBytes []byte, filename, caption, chat string) error {
	chatID, err := tc.resolveChatID(chat)
	if err != nil {
		return err
	}

	url := fmt.Sprintf("https://api.telegram.org/bot%s/sendPhoto", tc.token)

	// Create multipart form
	var b bytes.Buffer
	w := multipart.NewWriter(&b)

	// Add chat_id
	if err := w.WriteField("chat_id", chatID); err != nil {
		return fmt.Errorf("failed to write chat_id: %w", err)
	}

	// Add parse_mode
	if err := w.WriteField("parse_mode", "HTML"); err != nil {
		return fmt.Errorf("failed to write parse_mode: %w", err)
	}

	// Add caption if provided
	if caption != "" {
		if err := w.WriteField("caption", caption); err != nil {
			return fmt.Errorf("failed to write caption: %w", err)
		}
	}

	// Add photo bytes
	fw, err := w.CreateFormFile("photo", filename)
	if err != nil {
		return fmt.Errorf("failed to create form file: %w", err)
	}

	if _, err = fw.Write(imgBytes); err != nil {
		return fmt.Errorf("failed to write photo bytes: %w", err)
	}

	w.Close()

	req, err := http.NewRequest("POST", url, &b)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", w.FormDataContentType())

	resp, err := tc.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send photo: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("telegram API error: %s", string(body))
	}

	return nil
}

// SendDocumentBytes sends document bytes to Telegram
func (tc *TelegramClient) SendDocumentBytes(fileBytes []byte, filename, mimeType, caption, chat string) error {
	chatID, err := tc.resolveChatID(chat)
	if err != nil {
		return err
	}

	url := fmt.Sprintf("https://api.telegram.org/bot%s/sendDocument", tc.token)

	// Create multipart form
	var b bytes.Buffer
	w := multipart.NewWriter(&b)

	// Add chat_id
	if err := w.WriteField("chat_id", chatID); err != nil {
		return fmt.Errorf("failed to write chat_id: %w", err)
	}

	// Add parse_mode
	if err := w.WriteField("parse_mode", "HTML"); err != nil {
		return fmt.Errorf("failed to write parse_mode: %w", err)
	}

	// Add caption if provided
	if caption != "" {
		if err := w.WriteField("caption", caption); err != nil {
			return fmt.Errorf("failed to write caption: %w", err)
		}
	}

	// Add document bytes
	fw, err := w.CreateFormFile("document", filename)
	if err != nil {
		return fmt.Errorf("failed to create form file: %w", err)
	}

	if _, err = fw.Write(fileBytes); err != nil {
		return fmt.Errorf("failed to write document bytes: %w", err)
	}

	w.Close()

	req, err := http.NewRequest("POST", url, &b)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", w.FormDataContentType())

	resp, err := tc.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send document: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("telegram API error: %s", string(body))
	}

	return nil
}

// GetChats returns the configured chats
func (tc *TelegramClient) GetChats() map[string]string {
	return tc.chats
}

// SendDetection sends a detection notification to Telegram
func (tc *TelegramClient) SendDetection(payload map[string]interface{}, chat string) error {
	caption := formatDetectionCaption(payload)

	// Check for base64 encoded image
	if imgB64, ok := payload["annotated_image_b64"].(string); ok {
		imgBytes, err := decodeBase64(imgB64)
		if err != nil {
			return fmt.Errorf("failed to decode image_b64: %w", err)
		}
		filename := "frame.jpg"
		if name, ok := payload["annotated_image_name"].(string); ok {
			filename = name
		}
		return tc.SendPhotoBytes(imgBytes, filename, caption, chat)
	}

	// Check for base64 encoded JSON
	if jsonB64, ok := payload["detection_json_b64"].(string); ok {
		jsonBytes, err := decodeBase64(jsonB64)
		if err != nil {
			return fmt.Errorf("failed to decode detection_json_b64: %w", err)
		}
		filename := "detection.json"
		if name, ok := payload["detection_json_name"].(string); ok {
			filename = name
		}
		return tc.SendDocumentBytes(jsonBytes, filename, "application/json", caption, chat)
	}

	// Check for regular JSON
	if jsonData, ok := payload["detection_json"]; ok {
		jsonBytes, err := json.MarshalIndent(jsonData, "", "  ")
		if err != nil {
			return fmt.Errorf("failed to marshal detection_json: %w", err)
		}
		filename := "detection.json"
		if name, ok := payload["detection_json_name"].(string); ok {
			filename = name
		}
		return tc.SendDocumentBytes(jsonBytes, filename, "application/json", caption, chat)
	}

	// Check for image path/URL
	if imagePath, ok := payload["annotated_image_path"].(string); ok {
		return tc.SendPhoto(imagePath, caption, chat)
	}
	if imageURL, ok := payload["annotated_image_url"].(string); ok {
		return tc.SendPhoto(imageURL, caption, chat)
	}

	// Fallback to text message
	return tc.SendMessage(caption, chat)
}

// decodeBase64 decodes a base64 string
func decodeBase64(s string) ([]byte, error) {
	return base64.StdEncoding.DecodeString(s)
}

// ErrorTelegramClient handles sending error notifications via Python endpoint
type ErrorTelegramClient struct {
	pythonEndpoint string
	client         *http.Client
}

// NewErrorTelegramClient creates a new error notification client from config
func NewErrorTelegramClient(config map[string]interface{}) (*ErrorTelegramClient, error) {
	// Check if error notifications are enabled
	enabled, ok := config["error_notifications_enabled"].(bool)
	if !ok || !enabled {
		return nil, fmt.Errorf("error notifications not enabled")
	}

	// Get Python endpoint
	pythonEndpoint, ok := config["error_notifications_python_endpoint"].(string)
	if !ok || pythonEndpoint == "" {
		return nil, fmt.Errorf("error notifications python endpoint not configured")
	}

	return &ErrorTelegramClient{
		pythonEndpoint: pythonEndpoint,
		client:         &http.Client{Timeout: 40 * time.Second},
	}, nil
}

// SendError sends a basic error notification via Python endpoint
func (etc *ErrorTelegramClient) SendError(severity, component, message string) error {
	return etc.SendErrorToPythonEndpoint(severity, component, message, nil, etc.pythonEndpoint)
}

// SendErrorWithData sends an error notification with additional metadata via Python endpoint
func (etc *ErrorTelegramClient) SendErrorWithData(severity, component, message string, metadata map[string]interface{}) error {
	return etc.SendErrorToPythonEndpoint(severity, component, message, metadata, etc.pythonEndpoint)
}

// SendErrorNotification sends a structured error notification via Python endpoint
func (etc *ErrorTelegramClient) SendErrorNotification(severity, component, message string, metadata map[string]interface{}, stack, context string) error {
	// Add stack trace and context to metadata
	if stack != "" {
		metadata["stack_trace"] = stack
	}
	if context != "" {
		metadata["context"] = context
	}

	return etc.SendErrorToPythonEndpoint(severity, component, message, metadata, etc.pythonEndpoint)
}

// SendErrorToPythonEndpoint sends error data to the Python endpoint
func (etc *ErrorTelegramClient) SendErrorToPythonEndpoint(severity, component, message string, metadata map[string]interface{}, pythonEndpoint string) error {
	url := fmt.Sprintf("%s/send", pythonEndpoint)

	// Format the error message for the Python endpoint
	formattedMessage := formatErrorForPython(severity, component, message, metadata)

	// Create payload that matches the Python endpoint structure
	payload := map[string]interface{}{
		"message": formattedMessage,
		"chat":    "security_group", // Use security_group for error notifications
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal payload: %w", err)
	}

	resp, err := etc.client.Post(url, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to send to Python endpoint: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("Python endpoint error: %s", string(body))
	}

	return nil
}

// formatErrorForPython formats an error message for the Python endpoint
func formatErrorForPython(severity, component, message string, metadata map[string]interface{}) string {
	var emoji string
	switch severity {
	case "critical":
		emoji = "🚨🚨🚨"
	case "error":
		emoji = "❌"
	case "warning":
		emoji = "⚠️"
	case "info":
		emoji = "ℹ️"
	default:
		emoji = "🔍"
	}

	// Create a formatted message for the Python endpoint
	formattedMessage := fmt.Sprintf("%s *%s* in *%s*\n\n%s",
		emoji, strings.ToUpper(severity), component, message)

	// Add metadata if available
	if len(metadata) > 0 {
		formattedMessage += "\n\n*Additional Data:*\n"
		for key, value := range metadata {
			// Handle special formatting for stack traces and context
			if key == "stack_trace" {
				formattedMessage += fmt.Sprintf("*Stack Trace:*\n```\n%v\n```\n", value)
			} else if key == "context" {
				formattedMessage += fmt.Sprintf("*Context:*\n```\n%v\n```\n", value)
			} else {
				formattedMessage += fmt.Sprintf("• %s: %v\n", key, value)
			}
		}
	}

	// Add timestamp
	formattedMessage += fmt.Sprintf("\n_Time: %s_", time.Now().Format("2006-01-02 15:04:05"))

	return formattedMessage
}
