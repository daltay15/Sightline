# Error Notifications System

This document describes the error notification system that sends error messages via your Python endpoint.

## Configuration

Add these settings to your `config.json`:

```json
{
  "error_notifications_enabled": true,
  "error_notifications_python_endpoint": "http://localhost:5000"
}
```

## Python Endpoint Integration

The error notification system sends data **only** to your Python endpoint at `http://localhost:5000/send` with this structure:

```json
{
  "message": "🚨 CRITICAL in database\n\nDatabase connection failed\n\nAdditional Data:\n• path: /path/to/db\n• error: connection timeout\n\n_Time: 2024-01-15 10:30:00_",
  "chat": "security_group"
}
```

### Example Python Endpoint Response

Your Python endpoint should return:
```json
{
  "status": "success",
  "response": "Message sent successfully"
}
```

## Testing

Use the test endpoint to verify the system:

```bash
curl -X POST http://localhost:8080/error-notifications/test \
  -H "Content-Type: application/json" \
  -d '{
    "error_notifications_enabled": true,
    "error_notifications_python_endpoint": "http://localhost:5000"
  }'
```

## Error Severity Levels

- **info** ℹ️ - Informational messages
- **warning** ⚠️ - Warning messages  
- **error** ❌ - Error messages
- **critical** 🚨🚨🚨 - Critical error messages

## Usage in Code

```go
// Send a basic error
internal.NotifyError("error", "database", "Connection failed", map[string]interface{}{
    "host": "localhost",
    "port": 5432,
})

// Send a critical error
internal.NotifyCriticalError("system", "Out of memory", map[string]interface{}{
    "memory_used": "95%",
    "processes": 150,
})

// Send a warning
internal.NotifyWarning("detection", "Low confidence detection", map[string]interface{}{
    "confidence": 0.3,
    "threshold": 0.5,
})
```

## Message Format

The system formats messages with:
- Emoji based on severity level
- Component name in bold
- Error message
- Additional metadata (if provided)
- Timestamp

Example output:
```
🚨 CRITICAL in database

Database connection failed

Additional Data:
• path: /path/to/db
• error: connection timeout
• timestamp: 2024-01-15T10:30:00Z

_Time: 2024-01-15 10:30:00_
```
