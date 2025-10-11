package httpx

import (
	"compress/gzip"
	"io"
	"strings"

	"github.com/gin-gonic/gin"
)

// CompressionMiddleware adds gzip compression for better performance
func CompressionMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// Skip compression for already compressed content
		if strings.Contains(c.GetHeader("Content-Type"), "image/") ||
			strings.Contains(c.GetHeader("Content-Type"), "video/") {
			c.Next()
			return
		}

		// Check if client supports gzip
		if !strings.Contains(c.GetHeader("Accept-Encoding"), "gzip") {
			c.Next()
			return
		}

		// Set gzip headers
		c.Header("Content-Encoding", "gzip")
		c.Header("Vary", "Accept-Encoding")

		// Create gzip writer
		gz := gzip.NewWriter(c.Writer)
		defer gz.Close()

		// Wrap the response writer
		c.Writer = &gzipResponseWriter{
			ResponseWriter: c.Writer,
			Writer:         gz,
		}

		c.Next()
	}
}

type gzipResponseWriter struct {
	gin.ResponseWriter
	Writer io.Writer
}

func (w *gzipResponseWriter) Write(data []byte) (int, error) {
	return w.Writer.Write(data)
}
