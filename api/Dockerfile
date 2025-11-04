# Runtime image - uses pre-built binary
# Using Debian instead of Alpine in case binary was compiled for glibc
FROM debian:bookworm-slim

# Install runtime dependencies
# wget is needed for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    tzdata \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -m -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy pre-built binary from api directory
# The binary on server is named scui-api, but docker-compose expects scui
COPY api/scui-api /app/scui

# Copy config template (optional, config.json will be mounted)
COPY api/config.json.template /app/config.json.template

# Make binary executable and verify it exists
RUN chmod +x /app/scui && \
    ls -lh /app/scui && \
    file /app/scui || true && \
    ldd /app/scui 2>/dev/null || echo "Binary check complete"

# Change ownership to app user
RUN chown -R appuser:appuser /app

# Switch to app user
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health 2>&1 || exit 1

# Run the application
# Note: docker-compose adds a sleep command, but this is the base command
CMD ["./scui"]
