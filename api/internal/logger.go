package internal

import (
	"fmt"
	"io"
	"log"
	"os"
	"strings"
)

// LogLevel represents the severity of a log message
type LogLevel int

const (
	DEBUG LogLevel = iota
	INFO
	WARN
	ERROR
	FATAL
)

// String returns the string representation of the log level
func (l LogLevel) String() string {
	switch l {
	case DEBUG:
		return "DEBUG"
	case INFO:
		return "INFO"
	case WARN:
		return "WARN"
	case ERROR:
		return "ERROR"
	case FATAL:
		return "FATAL"
	default:
		return "UNKNOWN"
	}
}

// Logger provides structured logging with levels
type Logger struct {
	level  LogLevel
	logger *log.Logger
	prefix string
}

var defaultLogger *Logger

// InitLogger initializes the default logger
func InitLogger(level LogLevel, prefix string) {
	defaultLogger = NewLogger(level, prefix, os.Stdout)
}

// NewLogger creates a new logger instance
func NewLogger(level LogLevel, prefix string, writer io.Writer) *Logger {
	if prefix != "" && !strings.HasSuffix(prefix, " ") {
		prefix += " "
	}
	return &Logger{
		level:  level,
		logger: log.New(writer, "", log.LstdFlags),
		prefix: prefix,
	}
}

// SetLevel sets the log level for the logger
func (l *Logger) SetLevel(level LogLevel) {
	l.level = level
}

// shouldLog checks if a message at the given level should be logged
func (l *Logger) shouldLog(level LogLevel) bool {
	return level >= l.level
}

// format formats a log message with level and prefix
func (l *Logger) format(level LogLevel, format string, args ...interface{}) string {
	msg := format
	if len(args) > 0 {
		msg = fmt.Sprintf(format, args...)
	}
	return fmt.Sprintf("[%s]%s%s", level.String(), l.prefix, msg)
}

// Debug logs a debug message
func (l *Logger) Debug(format string, args ...interface{}) {
	if l.shouldLog(DEBUG) {
		l.logger.Print(l.format(DEBUG, format, args...))
	}
}

// Info logs an info message
func (l *Logger) Info(format string, args ...interface{}) {
	if l.shouldLog(INFO) {
		l.logger.Print(l.format(INFO, format, args...))
	}
}

// Warn logs a warning message
func (l *Logger) Warn(format string, args ...interface{}) {
	if l.shouldLog(WARN) {
		l.logger.Print(l.format(WARN, format, args...))
	}
}

// Error logs an error message
func (l *Logger) Error(format string, args ...interface{}) {
	if l.shouldLog(ERROR) {
		l.logger.Print(l.format(ERROR, format, args...))
	}
}

// Fatal logs a fatal message and exits
func (l *Logger) Fatal(format string, args ...interface{}) {
	l.logger.Fatal(l.format(FATAL, format, args...))
}

// Package-level convenience functions that use the default logger
func LogDebug(format string, args ...interface{}) {
	if defaultLogger != nil {
		defaultLogger.Debug(format, args...)
	} else {
		log.Printf("[DEBUG] "+format, args...)
	}
}

func LogInfo(format string, args ...interface{}) {
	if defaultLogger != nil {
		defaultLogger.Info(format, args...)
	} else {
		log.Printf("[INFO] "+format, args...)
	}
}

func LogWarn(format string, args ...interface{}) {
	if defaultLogger != nil {
		defaultLogger.Warn(format, args...)
	} else {
		log.Printf("[WARN] "+format, args...)
	}
}

func LogError(format string, args ...interface{}) {
	if defaultLogger != nil {
		defaultLogger.Error(format, args...)
	} else {
		log.Printf("[ERROR] "+format, args...)
	}
}

func LogFatal(format string, args ...interface{}) {
	if defaultLogger != nil {
		defaultLogger.Fatal(format, args...)
	} else {
		log.Fatalf("[FATAL] "+format, args...)
	}
}

// GetLogLevelFromString parses a log level from a string
func GetLogLevelFromString(levelStr string) LogLevel {
	switch strings.ToUpper(levelStr) {
	case "DEBUG":
		return DEBUG
	case "INFO":
		return INFO
	case "WARN", "WARNING":
		return WARN
	case "ERROR":
		return ERROR
	case "FATAL":
		return FATAL
	default:
		return INFO // Default to INFO
	}
}
