FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY config.json.template .

# Create directories for processing
RUN mkdir -p /app/pending /app/processing /app/completed /app/failed

# Copy and setup entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Health check script
RUN echo '#!/bin/bash\npython -c "import torch; print(\"CUDA available:\", torch.cuda.is_available())" && exit 0 || exit 1' > /healthcheck.sh && \
    chmod +x /healthcheck.sh

# Expose port (if needed for health checks)
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD /healthcheck.sh || exit 1

# Use entrypoint script
ENTRYPOINT ["/docker-entrypoint.sh"]

# Run the application
CMD ["python", "app.py"]

