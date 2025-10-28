#!/usr/bin/env python3
""" 
YOLO Classification/Detection Watcher
-------------------------------------
- Watches <base>/pending for .jpg/.jpeg images.
- Moves new files to <base>/processing, runs YOLO (classify or detect).
- Writes JSON (and annotated image for detect), then moves to completed/failed.
- Deletes the original image file on success (unless --skip-delete).

Usage:
  python app.py --base /path/to/base --model yolov8n-cls.pt --device auto --topk 5

Folder layout (auto-created):
  base/
    pending/
    processing/
    completed/
    failed/

Notes:
- First run may download weights.
- Classification emits a JSON (+TXT optional); detection also writes annotated image.
"""

import logging
import signal
import sys
import shutil
import time
import json
import os  # NEW: env config must happen before torch import
import requests
import base64
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

# ---- GPU allocator tuning BEFORE importing torch ----
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.75"
)

# Third-party
from ultralytics import YOLO
import torch  # import after allocator env is set
import cv2

# --------- Configuration Management ---------

def load_config(config_path: str) -> dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to the JSON configuration file
        
    Returns:
        Dictionary containing CPU and GPU configurations
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Validate required sections
        if "cpu_config" not in config or "gpu_config" not in config:
            raise ValueError("Configuration file must contain 'cpu_config' and 'gpu_config' sections")
        
        logging.info(f"✅ Configuration loaded from: {config_path}")
        return config
        
    except FileNotFoundError:
        logging.warning(f"⚠️ Configuration file not found: {config_path}")
        logging.info("Using default configuration...")
        return get_default_config()
    except json.JSONDecodeError as e:
        logging.error(f"❌ Invalid JSON in configuration file {config_path}: {e}")
        logging.info("Using default configuration...")
        return get_default_config()
    except Exception as e:
        logging.error(f"❌ Error loading configuration from {config_path}: {e}")
        logging.info("Using default configuration...")
        return get_default_config()

def get_default_config() -> dict:
    """
    Get default configuration as fallback.
    
    Returns:
        Dictionary containing default CPU and GPU configurations
    """
    return {
        "cpu_config": {
            "max_threads": 6,
            "batch_size": 1,
            "min_imgsz": 640,
            "mixed_precision": False,
            "compile": False,
            "yield_ms": 0,
            "vram_fraction": None,
            "device": "cpu",
            "base": "/mnt/nas/pool/Cameras/GPU_Processing",
            "model": "yolo11x.pt",
            "imgsz": 1280,
            "task": "detect",
            "api_endpoint": "http://localhost:8080/ingest_detection",
            "telegram_endpoint": "http://localhost:8080/telegram/send_detection",
            "disable_api": True,
            "disable_telegram": False,
            "topk": 5,
            "poll": 0.5,
            "workers": 1,
            "skip_delete": False,
            "once": False,
            "conf": 0.25,
            "iou": 0.45,
            "tta": False
        },
        "gpu_config": {
            "max_threads": None,
            "batch_size": 6,
            "min_imgsz": 224,
            "mixed_precision": True,
            "compile": True,
            "yield_ms": 20,
            "vram_fraction": 0.75,
            "device": "auto",
            "base": "/mnt/nas/pool/Cameras/GPU_Processing",
            "model": "yolo11x.pt",
            "imgsz": 1280,
            "task": "detect",
            "api_endpoint": "http://localhost:8080/ingest_detection",
            "telegram_endpoint": "http://localhost:8080/telegram/send_detection",
            "disable_api": True,
            "disable_telegram": False,
            "topk": 5,
            "poll": 0.5,
            "workers": 1,
            "skip_delete": False,
            "once": False,
            "conf": 0.25,
            "iou": 0.45,
            "tta": False
        }
    }

# --------- Model Wrapper ---------
@dataclass
class ClassificationResult:
    path: str
    topk: List[Dict[str, Any]]  # [{label, score}]
    duration_ms: float

def cap_vram(fraction: float = 0.45, device_index: int = 0):
    """Hard-cap VRAM usage to keep Windows/WSL responsive while gaming."""
    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(fraction, device_index)
            torch.cuda.empty_cache()
            logging.info(f"VRAM cap applied: {fraction:.2f} on cuda:{device_index}")
        except Exception as e:
            logging.warning(f"VRAM cap not applied: {e}")

def check_gpu_info():
    """Check GPU information and CUDA availability"""
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        logging.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
        logging.info(f"CUDA version: {torch.version.cuda}")
        if "5070" in gpu_name or "RTX 5070" in gpu_name:
            logging.info("🚀 RTX 5070 detected - enabling tuned defaults")
            return "rtx5070"
        return True
    else:
        logging.warning("CUDA not available, falling back to CPU")
        return False

def optimize_gpu_settings():
    """Optimize GPU settings for better perf + stability"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logging.info("✅ GPU optimizations: cuDNN benchmark, TF32, allocator tuned")

def warmup_gpu():
    """Light warmup to initialize CUDA context"""
    if torch.cuda.is_available():
        try:
            dummy = torch.randn(512, 512, device='cuda:0')
            _ = dummy @ dummy
            torch.cuda.synchronize()
            logging.info("🔥 GPU warmed up")
        except Exception as e:
            logging.warning(f"GPU warmup failed: {e}")

def monitor_gpu_usage():
    """Log basic VRAM usage"""
    if torch.cuda.is_available():
        try:
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            utilization = (allocated / total) * 100
            logging.info(f"📊 GPU Memory Allocated: {allocated:.1f}GB / {total:.1f}GB ({utilization:.1f}%) | Reserved: {reserved:.1f}GB")
        except Exception as e:
            logging.warning(f"Could not monitor GPU usage: {e}")

def get_optimal_batch_size(model_name: str, imgsz: int, gpu_type: str = None) -> int:
    """Heuristic for batch size"""
    if gpu_type == "rtx5070":
        mn = model_name.lower()
        if "yolo11n" in mn or "yolov8n" in mn:
            return min(64, max(16, 2048 // max(imgsz, 1)))
        elif "yolo11s" in mn or "yolov8s" in mn:
            return min(48, max(12, 1536 // max(imgsz, 1)))
        elif "yolo11m" in mn or "yolov8m" in mn:
            return min(32, max(8, 1024 // max(imgsz, 1)))
        elif "yolo11l" in mn or "yolov8l" in mn:
            return min(24, max(4, 768 // max(imgsz, 1)))
        elif "yolo11x" in mn or "yolov8x" in mn:
            return min(16, max(2, 512 // max(imgsz, 1)))
    return min(16, max(4, 1024 // max(imgsz, 1)))

class YOLOClassifier:
    def __init__(self, weights: str, device: str = "auto", imgsz: int = 640, tta: bool = False, batch_size: int = 8, 
                 mixed_precision: bool = False, compile_model: bool = False, yield_ms: int = 20):
        logging.info("Loading CLASSIFY model: %s (device=%s, batch_size=%d)", weights, device, batch_size)
        self.model = YOLO(weights)
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.imgsz = imgsz
        self.tta = tta
        self.batch_size = batch_size
        self.mixed_precision = mixed_precision
        self.yield_ms = yield_ms

        if self.device.startswith("cuda"):
            optimize_gpu_settings()
            warmup_gpu()
            if mixed_precision:
                logging.info("🚀 Mixed precision (FP16) enabled")
            if compile_model and hasattr(torch, 'compile'):
                try:
                    logging.info("⏳ Compiling model (reduce-overhead)...")
                    self.model.model = torch.compile(self.model.model, mode="reduce-overhead")
                    logging.info("✅ torch.compile enabled")
                except Exception as e:
                    logging.warning(f"Model compile failed: {e}. Continuing without.")
        elif self.device == "cpu":
            # CPU-specific optimizations
            max_threads = 6  # Default CPU thread limit
            if max_threads:
                torch.set_num_threads(min(max_threads, torch.get_num_threads()))
                logging.info(f"🖥️ CPU optimizations: limited to {max_threads} threads, no mixed precision")
            else:
                logging.info("🖥️ CPU optimizations: using all available threads, no mixed precision")

        logging.info("✅ Classify model ready")

    def _yield_gpu(self):
        if self.device.startswith("cuda"):
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            time.sleep(self.yield_ms / 1000.0)

    def classify(self, img_path: Path, topk: int = 5) -> ClassificationResult:
        t0 = time.time()
        results = self.model.predict(
            source=str(img_path),
            device=self.device,
            verbose=False,
            imgsz=self.imgsz,
            augment=self.tta,
            half=self.mixed_precision  # NEW: actually use FP16
        )
        dt_ms = (time.time() - t0) * 1000.0
        self._yield_gpu()

        r = results[0]
        probs = r.probs

        if topk <= 1:
            idxs = [int(probs.top1)]
            confs = [float(probs.top1conf)]
        else:
            idxs = list(probs.top5)[:topk]
            confs = [float(c) for c in probs.top5conf[: len(idxs)]]

        names = r.names
        def idx_to_label(i):
            if isinstance(names, (list, tuple)):
                return names[i] if 0 <= i < len(names) else str(i)
            elif isinstance(names, dict):
                return names.get(i, str(i))
            return str(i)

        items = [{"label": idx_to_label(i), "score": c} for i, c in zip(idxs, confs)]
        return ClassificationResult(path=str(img_path), topk=items, duration_ms=dt_ms)

    def classify_batch(self, img_paths: List[Path], topk: int = 5) -> List[ClassificationResult]:
        t0 = time.time()
        source_paths = [str(p) for p in img_paths]
        results = self.model.predict(
            source=source_paths,
            device=self.device,
            verbose=False,
            imgsz=self.imgsz,
            augment=self.tta,
            half=self.mixed_precision  # NEW
        )
        dt_ms = (time.time() - t0) * 1000.0
        self._yield_gpu()

        batch_results = []
        for img_path, r in zip(img_paths, results):
            probs = r.probs
            if topk <= 1:
                idxs = [int(probs.top1)]
                confs = [float(probs.top1conf)]
            else:
                idxs = list(probs.top5)[:topk]
                confs = [float(c) for c in probs.top5conf[: len(idxs)]]
            names = r.names
            def idx_to_label(i):
                if isinstance(names, (list, tuple)):
                    return names[i] if 0 <= i < len(names) else str(i)
                elif isinstance(names, dict):
                    return names.get(i, str(i))
                return str(i)
            items = [{"label": idx_to_label(i), "score": c} for i, c in zip(idxs, confs)]
            batch_results.append(ClassificationResult(path=str(img_path), topk=items, duration_ms=dt_ms/len(img_paths)))
        return batch_results

@dataclass
class DetectionResult:
    path: str
    duration_ms: float
    dets: List[Dict[str, Any]]  # [{label, score, xyxy:[x1,y1,x2,y2]}]
    annotated_path: str

class YOLODetector:
    def __init__(self, weights: str, device: str = "auto", imgsz: int = 640, conf: float = 0.25, iou: float = 0.45, 
                 batch_size: int = 8, mixed_precision: bool = False, compile_model: bool = False, yield_ms: int = 20):
        logging.info("Loading DETECT model: %s (device=%s, batch_size=%d)", weights, device, batch_size)
        self.model = YOLO(weights)
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.batch_size = batch_size
        self.mixed_precision = mixed_precision
        self.yield_ms = yield_ms

        if self.device.startswith("cuda"):
            optimize_gpu_settings()
            warmup_gpu()
            if mixed_precision:
                logging.info("🚀 Mixed precision (FP16) enabled")
            if compile_model and hasattr(torch, 'compile'):
                try:
                    logging.info("⏳ Compiling model (reduce-overhead)...")
                    self.model.model = torch.compile(self.model.model, mode="reduce-overhead")
                    logging.info("✅ torch.compile enabled")
                except Exception as e:
                    logging.warning(f"Model compile failed: {e}. Continuing without.")
        elif self.device == "cpu":
            # CPU-specific optimizations
            max_threads = 6  # Default CPU thread limit
            if max_threads:
                torch.set_num_threads(min(max_threads, torch.get_num_threads()))
                logging.info(f"🖥️ CPU optimizations: limited to {max_threads} threads, no mixed precision")
            else:
                logging.info("🖥️ CPU optimizations: using all available threads, no mixed precision")

        logging.info("✅ Detect model ready")

    def _yield_gpu(self):
        if self.device.startswith("cuda"):
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            time.sleep(self.yield_ms / 1000.0)

    def detect(self, img_path: Path) -> DetectionResult:
        t0 = time.time()
        results = self.model.predict(
            source=str(img_path),
            device=self.device,
            verbose=False,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            agnostic_nms=False,
            max_det=300,
            half=self.mixed_precision  # NEW
        )
        dt_ms = (time.time() - t0) * 1000.0
        self._yield_gpu()

        r = results[0]
        names = r.names
        dets = []
        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                cls_id = int(b.cls.item())
                conf = float(b.conf.item())
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                label = names[cls_id] if isinstance(names, (list, tuple)) else names.get(cls_id, str(cls_id))
                dets.append({"label": label, "score": conf, "xyxy": [x1, y1, x2, y2]})

        annotated = r.plot()  # numpy array (BGR)
        ann_path = img_path.with_suffix("").parent / f"{img_path.stem}_det.jpg"
        cv2.imwrite(str(ann_path), annotated)

        return DetectionResult(
            path=str(img_path),
            duration_ms=dt_ms,
            dets=dets,
            annotated_path=str(ann_path),
        )

    def detect_batch(self, img_paths: List[Path]) -> List[DetectionResult]:
        t0 = time.time()
        source_paths = [str(p) for p in img_paths]
        results = self.model.predict(
            source=source_paths,
            device=self.device,
            verbose=False,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            agnostic_nms=False,
            max_det=300,
            half=self.mixed_precision  # NEW
        )
        dt_ms = (time.time() - t0) * 1000.0
        self._yield_gpu()

        batch_results = []
        for img_path, r in zip(img_paths, results):
            names = r.names
            dets = []
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    cls_id = int(b.cls.item())
                    conf = float(b.conf.item())
                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                    label = names[cls_id] if isinstance(names, (list, tuple)) else names.get(cls_id, str(cls_id))
                    dets.append({"label": label, "score": conf, "xyxy": [x1, y1, x2, y2]})
            annotated = r.plot()
            ann_path = img_path.with_suffix("").parent / f"{img_path.stem}_det.jpg"
            cv2.imwrite(str(ann_path), annotated)
            batch_results.append(DetectionResult(
                path=str(img_path),
                duration_ms=dt_ms/len(img_paths),
                dets=dets,
                annotated_path=str(ann_path),
            ))
        return batch_results

# --------- API Integration ---------

def send_detection_to_telegram(detection_result: DetectionResult, original_path: str, 
                              telegram_endpoint: str, camera_name: str = None) -> bool:
    """
    Send detection data directly to Telegram endpoint for immediate alerting.
    
    Args:
        detection_result: DetectionResult object with detection data
        original_path: Path to the original image file
        telegram_endpoint: Telegram endpoint URL
        camera_name: Optional camera name for the alert
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Extract camera name from path if not provided
        if not camera_name:
            # Try to extract camera name from filename (format: ID__CameraName_XX_timestamp.jpg)
            filename = Path(original_path).stem  # Get filename without extension
            if "__" in filename:
                # Split by "__" and take the second part, then split by "_" and take the first part
                parts = filename.split("__")
                if len(parts) >= 2:
                    camera_part = parts[1].split("_")[0]  # Get camera name before first underscore
                    if camera_part:
                        camera_name = camera_part.replace("_", " ")  # Replace underscores with spaces
                    else:
                        camera_name = "Unknown Camera"
                else:
                    camera_name = "Unknown Camera"
            else:
                # Fallback to parent directory name
                path_parts = Path(original_path).parts
                if len(path_parts) > 1:
                    camera_name = path_parts[-2]  # Use parent directory as camera name
                else:
                    camera_name = "Unknown Camera"
        
        # Check if any detection is a person (same logic as Go code)
        has_person = False
        person_detections = []
        for detection in detection_result.dets:
            if detection.get("label", "").lower() == "person":
                has_person = True
                person_detections.append(detection)
        
        if not has_person:
            logging.info("No person detected, skipping Telegram alert")
            return True  # Not an error, just no alert needed
        
        # Read and encode the annotated image (same logic as Go code)
        image_b64 = ""
        image_name = ""
        if detection_result.annotated_path and Path(detection_result.annotated_path).exists():
            try:
                with open(detection_result.annotated_path, 'rb') as img_file:
                    image_data = img_file.read()
                    image_b64 = base64.b64encode(image_data).decode('utf-8')
                    image_name = Path(detection_result.annotated_path).name
                    logging.info(f"Encoded annotated image: {image_name} ({len(image_data)} bytes)")
            except Exception as e:
                logging.warning(f"Failed to read annotated image {detection_result.annotated_path}: {e}")
        
        # Format payload for Telegram endpoint (same structure as Go code)
        payload = {
            "camera_name": camera_name,
            "timestamp": int(time.time()),
            "detections": person_detections,  # Only send person detections
            "duration_ms": detection_result.duration_ms,
            "chat": "security_group"  # Default chat group
        }
        
        # Add image data if available (same priority as Go code)
        if image_b64:
            payload["annotated_image_b64"] = image_b64
            payload["annotated_image_name"] = image_name
        elif detection_result.annotated_path:
            payload["annotated_image_path"] = detection_result.annotated_path
        
        logging.info(f"Sending detection alert to Telegram: {payload}")
        response = requests.post(telegram_endpoint, json=payload, timeout=30)
        logging.info(f"Telegram response: {response}")
        
        if response.status_code == 200:
            result = response.json()
            logging.info(f"✅ Telegram alert sent successfully: {result.get('status', 'Success')}")
            return True
        else:
            logging.error(f"❌ Telegram request failed with status {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Failed to send Telegram alert: {e}")
        return False
    except Exception as e:
        logging.error(f"❌ Unexpected error sending Telegram alert: {e}")
        return False

def send_detection_to_api(original_path: str, annotated_path: str, detection_data_path: Dict[str, Any], 
                         api_endpoint: str, success: bool = True) -> Dict[str, Any]:
    """
    Send detection data to the API endpoint with retry logic.
    Will halt everything and retry until it succeeds.
    
    Args:
        original_path: Path to the original image file
        annotated_path: Path to the annotated image file  
        detection_data: Detection results data
        api_endpoint: API endpoint URL
        success: Whether the detection was successful
        
    Returns:
        API response data (guaranteed to succeed)
    """
    payload = {
        "original_path": original_path,
        "annotated_image_path": annotated_path,
        "detection_data_path": detection_data_path,
        "success": success,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    retry_count = 0
    
    while True:  # Infinite retries
        try:
            logging.info(f"📡 Sending detection data to API (attempt {retry_count + 1}): {api_endpoint}")
            response = requests.post(api_endpoint, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                logging.info(f"✅ API response: {result.get('message', 'Success')} (event_id: {result.get('event_id', 'N/A')})")
                return result
            else:
                logging.error(f"❌ API request failed with status {response.status_code}: {response.text}")
                logging.warning(f"🔄 Retrying in 5 seconds... (attempt {retry_count + 1})")
                time.sleep(5)
                retry_count += 1
                
        except requests.exceptions.RequestException as e:
            logging.error(f"❌ Failed to send data to API: {e}")
            logging.warning(f"🔄 Retrying in 5 seconds... (attempt {retry_count + 1})")
            time.sleep(5)
            retry_count += 1
        except Exception as e:
            logging.error(f"❌ Unexpected error sending data to API: {e}")
            logging.warning(f"🔄 Retrying in 5 seconds... (attempt {retry_count + 1})")
            time.sleep(5)
            retry_count += 1

# --------- Pipeline ---------

IMAGE_EXTS = {".jpg", ".jpeg"}

def ensure_dirs(base: Path):
    for name in ("pending", "processing", "completed", "failed"):
        (base / name).mkdir(parents=True, exist_ok=True)

def move(src: Path, dst: Path):
    if not src.exists():
        raise FileNotFoundError(f"Source file does not exist: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        return shutil.move(str(src), str(dst))
    except Exception as e:
        raise RuntimeError(f"Failed to move {src} to {dst}: {e}")

def write_json(path: Path, data: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def write_txt(path: Path, lines: List[str]):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def process_one(img_in_pending: Path, base: Path, runner, topk: int, api_endpoint: str = None, disable_api: bool = False, telegram_endpoint: str = None, disable_telegram: bool = False) -> None:
    processing_dir = base / "processing"
    completed_dir = base / "completed"
    failed_dir = base / "failed"

    try:
        if not img_in_pending.exists():
            logging.warning("File does not exist, skipping: %s", img_in_pending)
            return
        
        logging.info("Processing file: %s", img_in_pending)
        processing_path = processing_dir / img_in_pending.name
        move(img_in_pending, processing_path)
        logging.info("Moved to processing: %s", processing_path)

        if isinstance(runner, YOLODetector):
            det = runner.detect(processing_path)
            logging.info(f"Detection result: {det}")
            # Send Telegram alert immediately after detection (ASAP alerting)
            if not disable_telegram and telegram_endpoint:
                logging.info(f"Sending Telegram alert to: {telegram_endpoint}")
                telegram_success = send_detection_to_telegram(
                    detection_result=det,
                    original_path=str(processing_path),
                    telegram_endpoint=telegram_endpoint
                )
                if telegram_success:
                    logging.info("🚨 Telegram alert sent immediately for detection")
                else:
                    logging.error("❌ Telegram alert failed")
            else:
                logging.warning(f"Telegram alert skipped - disable_telegram: {disable_telegram}, telegram_endpoint: {telegram_endpoint}")

            stem = processing_path.stem
            json_path = processing_path.with_suffix("").parent / f"{stem}.json"
            ann_path  = Path(det.annotated_path)

            payload = {
                "image": processing_path.name,
                "duration_ms": round(det.duration_ms, 2),
                "detections": det.dets,
                "imgsz": runner.imgsz,
                "conf": runner.conf,
                "iou": runner.iou,
                "annotated_image": ann_path.name,
            }
            write_json(json_path, payload)

            # Send detection data to API if enabled
            if not disable_api and api_endpoint:
                detection_data = {
                    "detections": det.dets,
                    "duration_ms": round(det.duration_ms, 2),
                    "imgsz": runner.imgsz,
                    "conf": runner.conf,
                    "iou": runner.iou,
                }
                api_result = send_detection_to_api(
                    original_path=str(processing_path),
                    annotated_path=str(ann_path),
                    detection_data=detection_data,
                    api_endpoint=api_endpoint,
                    success=True
                )
                logging.info("✅ Detection data sent to API successfully")

            for p in (json_path, ann_path):
                dest = completed_dir / p.name
                move(p, dest)
            
            if not getattr(runner, "skip_delete", False) and processing_path.exists():
                processing_path.unlink()
                logging.info("✓ Deleted original image: %s", processing_path.name)

            logging.info("✓ Completed (detect): %s", processing_path.name)

        else:
            res = runner.classify(processing_path, topk=topk)

            stem = processing_path.stem
            json_path = processing_path.with_suffix("").parent / f"{stem}.json"

            payload = {
                "image": processing_path.name,
                "duration_ms": round(res.duration_ms, 2),
                "topk": res.topk,
            }
            write_json(json_path, payload)

            dest = completed_dir / json_path.name
            move(json_path, dest)
            
            if processing_path.exists():
                processing_path.unlink()
                logging.info("✓ Deleted original image: %s", processing_path.name)

            logging.info("✓ Completed (classify): %s", processing_path.name)

    except Exception as e:
        logging.exception("Failed processing: %s", img_in_pending.name)
        try:
            processing_path = processing_dir / img_in_pending.name
            if processing_path.exists():
                move(processing_path, failed_dir / img_in_pending.name)
            elif img_in_pending.exists():
                move(img_in_pending, failed_dir / img_in_pending.name)
        except Exception:
            pass
        err_txt = failed_dir / f"{img_in_pending.stem}_error.txt"
        write_txt(err_txt, [f"Error: {e}"])

def process_batch(batch_imgs: List[Path], base: Path, runner, topk: int, skip_delete: bool = False, stop_flag: bool = False, api_endpoint: str = None, disable_api: bool = False, telegram_endpoint: str = None, disable_telegram: bool = False) -> None:
    """Process a batch of images (true batch)"""
    processing_dir = base / "processing"
    completed_dir = base / "completed"
    failed_dir = base / "failed"
    
    import signal
    def timeout_handler(signum, frame):
        raise TimeoutError("Batch processing timeout")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300)  # 5 minutes timeout
    
    if stop_flag:
        logging.info("Stopping batch processing before starting...")
        return
    
    try:
        processing_paths = []
        for img in batch_imgs:
            if not img.exists():
                logging.warning("File does not exist, skipping: %s", img)
                continue
            processing_path = processing_dir / img.name
            move(img, processing_path)
            processing_paths.append(processing_path)
        
        if not processing_paths:
            logging.warning("No valid images in batch")
            return
            
        logging.info("Processing batch of %d images", len(processing_paths))
        logging.info("🔥 TRUE BATCH INFERENCE")
        
        monitor_gpu_usage()
        
        logging.info(f"🚀 Starting batch inference of {len(processing_paths)} images...")
        start_time = time.time()
        if isinstance(runner, YOLODetector):
            batch_results = runner.detect_batch(processing_paths)
            inference_time = time.time() - start_time
            logging.info(f"⚡ Batch inference completed in {inference_time:.2f}s")
            monitor_gpu_usage()
            
            import concurrent.futures
            io_start_time = time.time()
            logging.info("📁 Starting parallel I/O processing...")

            def process_single_result(args):
                processing_path, det, completed_dir = args
                
                # Send Telegram alert immediately after detection (ASAP alerting)
                if not disable_telegram and telegram_endpoint:
                    logging.info(f"Sending Telegram alert to: {telegram_endpoint}")
                    telegram_success = send_detection_to_telegram(
                        detection_result=det,
                        original_path=str(processing_path),
                        telegram_endpoint=telegram_endpoint
                    )
                    if telegram_success:
                        logging.info("🚨 Telegram alert sent immediately for detection")
                    else:
                        logging.error("❌ Telegram alert failed")
                else:
                    logging.warning(f"Telegram alert skipped - disable_telegram: {disable_telegram}, telegram_endpoint: {telegram_endpoint}")
                
                stem = processing_path.stem
                json_path = processing_path.with_suffix("").parent / f"{stem}.json"
                ann_path = Path(det.annotated_path)

                payload = {
                    "image": processing_path.name,
                    "duration_ms": round(det.duration_ms, 2),
                    "detections": det.dets,
                    "imgsz": runner.imgsz,
                    "conf": runner.conf,
                    "iou": runner.iou,
                    "annotated_image": ann_path.name,
                }
                write_json(json_path, payload)

                # Send detection data to API if enabled
                if not disable_api and api_endpoint:
                    detection_data = {
                        "detections": det.dets,
                        "duration_ms": round(det.duration_ms, 2),
                        "imgsz": runner.imgsz,
                        "conf": runner.conf,
                        "iou": runner.iou,
                    }
                    api_result = send_detection_to_api(
                        original_path=str(processing_path),
                        annotated_path=str(ann_path),
                        detection_data=detection_data,
                        api_endpoint=api_endpoint,
                        success=True
                    )
                    logging.info("✅ Detection data sent to API successfully")

                for p in (json_path, ann_path):
                    dest = completed_dir / p.name
                    move(p, dest)
                
                if not skip_delete and processing_path.exists():
                    try:
                        processing_path.unlink()
                        return f"✓ Deleted: {processing_path.name}"
                    except Exception as e:
                        return f"⚠️ Failed to delete: {processing_path.name} - {e}"
                return None
            
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    results = list(zip(processing_paths, batch_results, [completed_dir] * len(processing_paths)))
                    futures = [executor.submit(process_single_result, args) for args in results]
                    completed_deletions = 0
                    failed_deletions = 0
                    for future in concurrent.futures.as_completed(futures):
                        if stop_flag:
                            logging.info("Stopping I/O processing due to signal...")
                            break
                        result = future.result()
                        if result:
                            if "✓ Deleted:" in result:
                                completed_deletions += 1
                            elif "⚠️ Failed" in result:
                                failed_deletions += 1
                                logging.warning(result)
            except KeyboardInterrupt:
                logging.info("I/O processing interrupted by user")
                return
            
            io_time = time.time() - io_start_time
            logging.info(f"📁 Parallel I/O completed in {io_time:.2f}s - {completed_deletions} deleted, {failed_deletions} failed")

        else:
            batch_results = runner.classify_batch(processing_paths, topk=topk)
            inference_time = time.time() - start_time
            logging.info(f"⚡ Batch inference completed in {inference_time:.2f}s")
            monitor_gpu_usage()
            
            import concurrent.futures
            def process_single_classification(args):
                processing_path, res, completed_dir = args
                stem = processing_path.stem
                json_path = processing_path.with_suffix("").parent / f"{stem}.json"

                payload = {
                    "image": processing_path.name,
                    "duration_ms": round(res.duration_ms, 2),
                    "topk": res.topk,
                }
                write_json(json_path, payload)

                dest = completed_dir / json_path.name
                move(json_path, dest)
                
                if not skip_delete and processing_path.exists():
                    try:
                        processing_path.unlink()
                        return f"✓ Deleted: {processing_path.name}"
                    except Exception as e:
                        return f"⚠️ Failed to delete: {processing_path.name} - {e}"
                return None
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                results = list(zip(processing_paths, batch_results, [completed_dir] * len(processing_paths)))
                futures = [executor.submit(process_single_classification, args) for args in results]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        logging.info(result)
            
            io_time = time.time() - io_start_time
            logging.info(f"📁 Parallel I/O completed in {io_time:.2f}s")

        logging.info("✓ Completed batch of %d images", len(processing_paths))
        
    except Exception as e:
        logging.exception("Failed processing batch")
        for img in batch_imgs:
            try:
                processing_path = processing_dir / img.name
                if processing_path.exists():
                    move(processing_path, failed_dir / img.name)
                elif img.exists():
                    move(img, failed_dir / img.name)
            except Exception:
                pass
        err_txt = failed_dir / f"batch_error_{int(time.time())}.txt"
        write_txt(err_txt, [f"Batch processing error: {e}"])
    finally:
        signal.alarm(0)

def find_pending_images(base: Path) -> list[Path]:
    pending_dir = base / "pending"
    files = []
    if not pending_dir.exists():
        logging.warning("Pending directory does not exist: %s", pending_dir)
        return files
    try:
        for p in pending_dir.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() in IMAGE_EXTS:
                if p.name.endswith(".part") or p.name.startswith("~"):
                    continue
                files.append(p)
    except Exception as e:
        logging.error("Error scanning pending directory %s: %s", pending_dir, e)
    return sorted(files)

def run_once(base: Path, runner, topk: int, workers: int = 1, batch_size: int = 8, skip_delete: bool = False, api_endpoint: str = None, disable_api: bool = False, telegram_endpoint: str = None, disable_telegram: bool = False):
    imgs = find_pending_images(base)
    if not imgs:
        logging.info("No pending images to process.")
        return
    logging.info("Found %d image(s) to process.", len(imgs))
    
    stop = False
    executor = None
    
    def handle_sig(sig, frame):
        nonlocal stop, executor
        print(f"\nReceived signal {sig}, stopping batch processing...")
        logging.info("Received signal %s, stopping batch processing...", sig)
        stop = True
        if executor:
            executor.shutdown(wait=False)
    
    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)
    
    try:
        for i in range(0, len(imgs), batch_size):
            if stop:
                logging.info("Stopping batch processing due to signal")
                break
            batch_imgs = imgs[i:i + batch_size]
            logging.info("Processing batch %d/%d (%d images)", i//batch_size + 1, (len(imgs) + batch_size - 1)//batch_size, len(batch_imgs))
            if batch_size > 1:
                try:
                    process_batch(batch_imgs, base, runner, topk, skip_delete, stop, api_endpoint, disable_api, telegram_endpoint, disable_telegram)
                except Exception as e:
                    logging.error(f"Batch processing failed: {e}")
                    if stop:
                        break
            else:
                for img in batch_imgs:
                    if stop:
                        break
                    process_one(img, base, runner, topk, api_endpoint, disable_api, telegram_endpoint, disable_telegram)
                    
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, stopping batch processing...")
        logging.info("KeyboardInterrupt received, stopping batch processing...")
    except Exception as e:
        logging.error(f"Unexpected error in batch processing: {e}")
    finally:
        logging.info("Batch processing complete")
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)

def run_watch(base: Path, runner, topk: int, poll: float, workers: int, api_endpoint: str = None, disable_api: bool = False, telegram_endpoint: str = None, disable_telegram: bool = False):
    logging.info("Watching %s for new images... (poll=%.2fs)", (base / 'pending'), poll)
    stop = False

    def handle_sig(sig, frame):
        nonlocal stop
        print(f"\nReceived signal {sig}, shutting down...")
        logging.info("Received signal %s, shutting down...", sig)
        stop = True

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    try:
        while not stop:
            new_files = find_pending_images(base)
            for img in new_files:
                if stop:
                    break
                process_one(img, base, runner, topk, api_endpoint, disable_api, telegram_endpoint, disable_telegram)
            if not stop:
                time.sleep(poll)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, shutting down...")
        logging.info("KeyboardInterrupt received, shutting down...")
    finally:
        logging.info("Shutdown complete")

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Load configuration from JSON file
    config_data = load_config("config.json")
    
    # Determine if we should use CPU or GPU mode
    use_cpu = False
    if torch.cuda.is_available():
        # Check GPU availability and info
        gpu_type = check_gpu_info()
        if gpu_type:
            logging.info("🚀 GPU detected - using GPU configuration")
            config = config_data["gpu_config"]
        else:
            logging.info("🖥️ No GPU available - using CPU configuration")
            use_cpu = True
            config = config_data["cpu_config"]
    else:
        logging.info("🖥️ CUDA not available - using CPU configuration")
        use_cpu = True
        config = config_data["cpu_config"]
    
    # Apply configuration
    if use_cpu:
        logging.info("🖥️ CPU mode - applying CPU optimizations")
        device = config["device"]
        batch_size = config["batch_size"]
        model = config["model"]
        imgsz = config["imgsz"]
        task = config["task"]
        mixed_precision = config["mixed_precision"]
        compile_model = config["compile"]
        yield_ms = config["yield_ms"]
        vram_fraction = None  # Not applicable to CPU
    else:
        logging.info("🚀 GPU mode - applying GPU optimizations")
        device = config["device"]
        batch_size = config["batch_size"]
        model = config["model"]
        imgsz = config["imgsz"]
        task = config["task"]
        mixed_precision = config["mixed_precision"]
        compile_model = config["compile"]
        yield_ms = config["yield_ms"]
        vram_fraction = config["vram_fraction"]
        
        # Cap VRAM early to keep system responsive
        if torch.cuda.is_available() and vram_fraction:
            cap_vram(fraction=vram_fraction, device_index=0)
    
    # Get other configuration values
    base_path = config["base"]
    api_endpoint = config["api_endpoint"]
    disable_api = config["disable_api"]
    telegram_endpoint = config["telegram_endpoint"]
    disable_telegram = config["disable_telegram"]
    topk = config["topk"]
    poll = config["poll"]
    workers = config["workers"]
    skip_delete = config["skip_delete"]
    once = config["once"]
    conf = config["conf"]
    iou = config["iou"]
    tta = config["tta"]
    
    logging.info(f"✅ Configuration: device={device}, task={task}, model={model}, imgsz={imgsz}, batch_size={batch_size}, mixed_precision={mixed_precision}, compile={compile_model}, base={base_path}, api_endpoint={api_endpoint}, telegram_endpoint={telegram_endpoint}")

    # Setup base directory
    base = Path(base_path).expanduser().resolve()
    ensure_dirs(base)

    # Create model runner
    if task == "detect":
        runner = YOLODetector(
            weights=model,
            device=device,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            batch_size=batch_size,
            mixed_precision=mixed_precision,
            compile_model=compile_model,
            yield_ms=yield_ms,
        )
    else:
        runner = YOLOClassifier(
            weights=model,
            device=device,
            imgsz=imgsz,
            tta=tta,
            batch_size=batch_size,
            mixed_precision=mixed_precision,
            compile_model=compile_model,
            yield_ms=yield_ms,
        )

    # Run processing
    if once:
        run_once(base, runner, topk, workers=workers, batch_size=batch_size, skip_delete=skip_delete, api_endpoint=api_endpoint, disable_api=disable_api, telegram_endpoint=telegram_endpoint, disable_telegram=disable_telegram)
    else:
        run_once(base, runner, topk, workers=workers, batch_size=batch_size, skip_delete=skip_delete, api_endpoint=api_endpoint, disable_api=disable_api, telegram_endpoint=telegram_endpoint, disable_telegram=disable_telegram)
        run_watch(base, runner, topk, poll, workers, api_endpoint=api_endpoint, disable_api=disable_api, telegram_endpoint=telegram_endpoint, disable_telegram=disable_telegram)

if __name__ == "__main__":
    main()
