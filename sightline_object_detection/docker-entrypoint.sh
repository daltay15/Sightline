#!/bin/bash
set -e

# Update config.json with environment variables if provided
if [ -f /app/config.json ]; then
    # Use Python to update the JSON config with environment variables
    python3 << EOF
import json
import os

# Load config
with open('/app/config.json', 'r') as f:
    config = json.load(f)

# Update API endpoint if provided
api_endpoint = os.getenv('API_ENDPOINT')
if api_endpoint:
    config['cpu_config']['api_endpoint'] = api_endpoint
    config['gpu_config']['api_endpoint'] = api_endpoint

# Update Telegram endpoint if provided
telegram_endpoint = os.getenv('TELEGRAM_ENDPOINT')
if telegram_endpoint:
    config['cpu_config']['telegram_endpoint'] = telegram_endpoint
    config['gpu_config']['telegram_endpoint'] = telegram_endpoint

# Save updated config
with open('/app/config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("✅ Config updated with environment variables")
EOF
fi

# Execute the main command
exec "$@"

