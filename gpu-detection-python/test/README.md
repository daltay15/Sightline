# YOLO Detection/Classification Watcher

Watches a folder for `.jpg`/`.jpeg` images, processes them with an Ultralytics YOLO model (classification or detection), and moves results to `completed/`. Now includes API integration to send detection data to a remote service.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Choose a base folder (it will create pending/, processing/, completed/, failed/)
mkdir -p /path/to/base/pending

# Run the watcher (downloads model on first run)
python app.py --base /path/to/base --model yolov8n-cls.pt --device auto --topk 5

# For detection mode with API integration
python app.py --base /path/to/base --model yolov8n.pt --task detect --api-endpoint http://your-api:8080/ingest_detection

# CPU processing (optimized for strong CPUs)
python app.py --base /path/to/base --model yolo11x.pt --task detect --cpu --imgsz 1280
```

Drop `.jpg` files into `/path/to/base/pending/` and watch them flow to `completed/` with a JSON + TXT summary.

## API Integration

The watcher can now send detection data to a remote API endpoint after processing images. This is useful for integrating with security camera systems.

### API Configuration

- `--api-endpoint`: URL of the API endpoint (default: `http://your-security-camera-ui:8080/ingest_detection`)
- `--disable-api`: Disable API calls (useful for testing or when API is unavailable)

### API Payload Format

The API receives detection data in the following format:

```json
{
  "original_path": "/path/to/original/file.jpg",
  "annotated_image": "/path/to/annotated/image.jpg",
  "detection_data": {
    "detections": [
      {
        "label": "person",
        "score": 0.95,
        "xyxy": [100, 100, 200, 300]
      }
    ],
    "duration_ms": 1000.0,
    "imgsz": 640,
    "conf": 0.45,
    "iou": 0.5
  },
  "success": true,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Notes

- The first time you run, Ultralytics will download the model weights. You can use different models (e.g., `yolov8n.pt`, `yolov8s.pt`, etc.).
- If you have a CUDA GPU, ensure you install a CUDA-enabled PyTorch build that matches your driver.
- Use `--once` to process current backlog and exit.
- Use `--task detect` for object detection or `--task classify` for image classification.
- API calls are made asynchronously and won't block processing if they fail.

## CPU Processing

For systems without GPU or when you want to use CPU processing, use the `--cpu` flag:

```bash
# CPU processing with optimized settings
python app.py --base /path/to/base --model yolo11x.pt --task detect --cpu --imgsz 1280

# CPU processing with API integration
python app.py --base /path/to/base --model yolo11x.pt --task detect --cpu --imgsz 1280 --api-endpoint http://your-api:8080/ingest_detection
```

### CPU Optimizations

When using `--cpu` flag, the following optimizations are automatically applied:
- **Device**: Forces `cpu` device
- **Batch Size**: Set to 1 (optimal for CPU)
- **Image Size**: Minimum 640px (auto-adjusted if lower)
- **Mixed Precision**: Disabled (not supported on CPU)
- **Compilation**: Disabled for stability
- **Threads**: Limited to 6 threads for stability
- **Yield**: Disabled (not needed on CPU)

### Configuration Constants

The application uses configurable constants for CPU and GPU optimization:

**CPU_CONFIG:**
- `max_threads`: 6
- `batch_size`: 1
- `min_imgsz`: 640
- `mixed_precision`: False
- `compile`: False
- `yield_ms`: 0
- `base`: "/mnt/nas/pool/Cameras/GPU_Processing"
- `model`: "yolo11x.pt"
- `imgsz`: 1280
- `task`: "detect"
- `api_endpoint`: "http://localhost:8080/ingest_detection"
- `disable_api`: False

**GPU_CONFIG:**
- `max_threads`: None (use all available)
- `batch_size`: 6 (optimized for your setup)
- `min_imgsz`: 224
- `mixed_precision`: True (can be enabled)
- `compile`: True (can be enabled)
- `yield_ms`: 20
- `vram_fraction`: 0.75 (optimized for your setup)
- `device`: "auto"
- `base`: "/mnt/nas/pool/Cameras/GPU_Processing"
- `model`: "yolo11x.pt"
- `imgsz`: 1280
- `task`: "detect"
- `api_endpoint`: "http://localhost:8080/ingest_detection"
- `disable_api`: False

### CPU Performance Tips

- Use smaller models (yolov8n, yolov8s) for faster CPU processing
- Larger models (yolo11x) work but are slower on CPU
- Image size 640-1280 works well for CPU processing
- Consider using `--once` for batch processing instead of continuous watching

### Command Mapping

Your original command:
```bash
python3 app.py --task detect --base /mnt/z/Cameras/GPU_Processing --model yolo11x.pt --device auto --imgsz 1280 --batch-size 6 --mixed-precision --device auto --vram-fraction 0.75 --yield-ms 20 --api-endpoint http://localhost:8080/ingest_detection
```

Is now equivalent to:
```bash
python3 app.py
```

The GPU_CONFIG automatically applies:
- `--task detect` (from config)
- `--base /mnt/nas/Cameras/GPU_Processing` (from config)
- `--model yolo11x.pt` (from config)
- `--imgsz 1280` (from config)
- `--device auto` (from config)
- `--batch-size 6` (from config) 
- `--mixed-precision` (from config)
- `--vram-fraction 0.75` (from config)
- `--yield-ms 20` (from config)
- `--api-endpoint http://localhost:8080/ingest_detection` (from config)
