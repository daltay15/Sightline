# YOLO Detection App - Simplified Configuration

This Python app now uses **JSON-only configuration** - no more command line arguments needed!

## Quick Start

1. **Edit `config.json`** to customize your settings
2. **Run the app**: `python app.py`
3. **That's it!** The app automatically detects GPU/CPU and applies the appropriate configuration

## Configuration

All settings are now in `config.json`:

### CPU Configuration (`cpu_config`)
- Optimized for CPU processing
- Lower batch sizes, no mixed precision
- Thread limiting for stability

### GPU Configuration (`gpu_config`) 
- Optimized for GPU processing
- Higher batch sizes, mixed precision enabled
- VRAM fraction limiting

## Key Settings

| Setting | Description | CPU Default | GPU Default |
|---------|-------------|-------------|-------------|
| `task` | "detect" or "classify" | detect | detect |
| `model` | YOLO model file | yolo11x.pt | yolo11x.pt |
| `batch_size` | Images per batch | 1 | 6 |
| `imgsz` | Input image size | 1280 | 1280 |
| `api_endpoint` | API integration URL | localhost:8080 | localhost:8080 |
| `telegram_endpoint` | Telegram alerts URL | localhost:8080 | localhost:8080 |
| `disable_api` | Disable API integration | true | true |
| `disable_telegram` | Disable Telegram alerts | false | false |
| `once` | Process once vs watch mode | false | false |

## Automatic Mode Selection

The app automatically chooses CPU or GPU configuration based on:
- **GPU Available**: Uses `gpu_config`
- **No GPU**: Uses `cpu_config`

## Example Custom Configuration

```json
{
  "cpu_config": {
    "task": "detect",
    "model": "yolo11n.pt",
    "batch_size": 2,
    "disable_telegram": true
  },
  "gpu_config": {
    "task": "detect", 
    "model": "yolo11x.pt",
    "batch_size": 8,
    "vram_fraction": 0.5,
    "telegram_endpoint": "http://production-server:8080/telegram/send_detection"
  }
}
```

## Benefits

✅ **Simplified usage** - No complex command line arguments  
✅ **Environment-specific configs** - Different settings for dev/prod  
✅ **Version control** - Track configuration changes  
✅ **Team sharing** - Share configs without code changes  
✅ **Automatic fallback** - Works even if config file is missing  

## Migration from Command Line

| Old Command Line | New JSON Config |
|------------------|-----------------|
| `--cpu` | Automatically detected |
| `--batch-size 8` | `"batch_size": 8` |
| `--telegram-endpoint URL` | `"telegram_endpoint": "URL"` |
| `--disable-telegram` | `"disable_telegram": true` |
| `--once` | `"once": true` |

The app is now much simpler to use and configure!