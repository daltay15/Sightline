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

import argparse
import logging
import signal
import sys
import shutil
import time
import json
import os  # NEW: env config must happen before torch import
import requests
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

# --------- CLI ---------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Watch a folder and run YOLO on images.")
    p.add_argument("--base", help="Base folder (contains pending/processing/completed/failed)")
    p.add_argument("--model", default="yolov8n-cls.pt", help="Ultralytics weights (e.g., yolov8n-cls.pt or yolov8n.pt)")
    p.add_argument("--device", default="auto", help="Device: 'cpu', 'cuda:0', or 'auto'")
    p.add_argument("--task", choices=["classify", "detect"], default="classify", help="Classification or detection run mode")
    p.add_argument("--topk", type=int, default=5, help="Top-K classes for classification")
    p.add_argument("--poll", type=float, default=0.5, help="Polling interval (seconds)")
    p.add_argument("--workers", type=int, default=1, help="Parallel workers (unused; single-threaded pipeline)")
    p.add_argument("--once", action="store_true", help="Process current pending files then exit (no watch)")
    p.add_argument("--imgsz", type=int, default=640, help="Inference size (e.g., 224/320/640). Lower reduces VRAM.")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size (reduce if VRAM tight)")
    p.add_argument("--max-batch", type=int, default=64, help="Upper bound for auto-batching logic")
    p.add_argument("--mixed-precision", action="store_true", help="Enable FP16/half precision on CUDA")
    p.add_argument("--compile", action="store_true", help="Enable torch.compile (PT2). Slower first run.")
    p.add_argument("--no-compile", action="store_true", help="Force disable compile")
    p.add_argument("--skip-delete", action="store_true", help="Do not delete originals after success")
    p.add_argument("--tta", action="store_true", help="Test-time augmentation (slower, sometimes better)")
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="Detection IoU threshold")
    # NEW knobs for stability while gaming
    p.add_argument("--vram-fraction", type=float, default=0.45, help="Max VRAM fraction this process can use")
    p.add_argument("--yield-ms", type=int, default=20, help="Sleep this many ms after each inference to keep desktop responsive")
    # API endpoint configuration
    p.add_argument("--api-endpoint", default="http://your-security-camera-ui:8080/ingest_detection", help="API endpoint URL for sending detection data")
    p.add_argument("--disable-api", action="store_true", help="Disable sending data to API endpoint")
    # CPU optimization
    p.add_argument("--cpu", action="store_true", help="Force CPU processing with optimized settings (imgsz 640+, batch-size 1)")
    return p.parse_args()

# --------- Configuration Constants ---------

# CPU optimization constants
CPU_CONFIG = {
    "max_threads": 6,
    "batch_size": 1,
    "min_imgsz": 640,
    "mixed_precision": False,
    "compile": False,
    "yield_ms": 0,  # No yield needed on CPU
    "vram_fraction": None,  # Not applicable to CPU
    "device": "cpu",
    "base": "/mnt/nas/pool/Cameras/GPU_Processing",
    "model": "yolo11x.pt",
    "imgsz": 1280,
    "task": "detect",
    "api_endpoint": "http://localhost:8080/ingest_detection",
    "disable_api": True
}

# GPU optimization constants  
GPU_CONFIG = {
    "max_threads": None,  # Use all available threads
    "batch_size": 6,  # Your preferred batch size
    "min_imgsz": 224,
    "mixed_precision": True,  # Can be enabled
    "compile": True,  # Can be enabled
    "yield_ms": 20,  # Yield to keep desktop responsive
    "vram_fraction": 0.75,  # Your preferred VRAM fraction
    "device": "auto",
    "base": "/mnt/nas/pool/Cameras/GPU_Processing",
    "model": "yolo11x.pt",
    "imgsz": 1280,
    "task": "detect",
    "api_endpoint": "http://localhost:8080/ingest_detection",
    "disable_api": True
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
            max_threads = CPU_CONFIG["max_threads"]
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
            max_threads = CPU_CONFIG["max_threads"]
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

def process_one(img_in_pending: Path, base: Path, runner, topk: int, api_endpoint: str = None, disable_api: bool = False) -> None:
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

def process_batch(batch_imgs: List[Path], base: Path, runner, topk: int, skip_delete: bool = False, stop_flag: bool = False, api_endpoint: str = None, disable_api: bool = False) -> None:
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

def run_once(base: Path, runner, topk: int, workers: int = 1, batch_size: int = 8, skip_delete: bool = False, api_endpoint: str = None, disable_api: bool = False):
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
                    process_batch(batch_imgs, base, runner, topk, skip_delete, stop, api_endpoint, disable_api)
                except Exception as e:
                    logging.error(f"Batch processing failed: {e}")
                    if stop:
                        break
            else:
                for img in batch_imgs:
                    if stop:
                        break
                    process_one(img, base, runner, topk, api_endpoint, disable_api)
                    
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, stopping batch processing...")
        logging.info("KeyboardInterrupt received, stopping batch processing...")
    except Exception as e:
        logging.error(f"Unexpected error in batch processing: {e}")
    finally:
        logging.info("Batch processing complete")
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)

def run_watch(base: Path, runner, topk: int, poll: float, workers: int, api_endpoint: str = None, disable_api: bool = False):
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
                process_one(img, base, runner, topk, api_endpoint, disable_api)
            if not stop:
                time.sleep(poll)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, shutting down...")
        logging.info("KeyboardInterrupt received, shutting down...")
    finally:
        logging.info("Shutdown complete")

def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Cap VRAM early to keep system responsive
    if torch.cuda.is_available():
        cap_vram(fraction=args.vram_fraction, device_index=0)

    # Check GPU availability and info
    gpu_type = check_gpu_info()
    
    # Apply CPU or GPU configuration first
    if args.cpu:
        config = CPU_CONFIG
        logging.info("🖥️ CPU mode enabled - applying CPU optimizations")
        args.device = config["device"]
        args.batch_size = config["batch_size"]
        args.model = config["model"]
        args.imgsz = config["imgsz"]
        args.task = config["task"]
        args.mixed_precision = config["mixed_precision"]
        args.compile = config["compile"]
        args.yield_ms = config["yield_ms"]
        # Apply path and API configuration
        args.base = config["base"]
        args.api_endpoint = config["api_endpoint"]
        args.disable_api = config["disable_api"]
        # VRAM fraction not applicable to CPU
        logging.info(f"✅ CPU config: device={config['device']}, task={config['task']}, model={config['model']}, imgsz={config['imgsz']}, batch_size={config['batch_size']}, mixed_precision={config['mixed_precision']}, compile={config['compile']}, base={config['base']}, api_endpoint={config['api_endpoint']}")
    else:
        config = GPU_CONFIG
        logging.info("🚀 GPU mode - applying GPU optimizations")
        args.device = config["device"]
        args.model = config["model"]
        args.imgsz = config["imgsz"]
        args.task = config["task"]
        # Apply GPU batch size if not explicitly set by user
        if args.batch_size == 8:  # Default value, apply config
            args.batch_size = config["batch_size"]
        # Auto-optimize batch size for 5070 if user left default "big" batch
        elif args.batch_size == 32:  # legacy default case
            optimal_batch = get_optimal_batch_size(args.model, args.imgsz, gpu_type)
            if optimal_batch != args.batch_size:
                logging.info(f"🚀 Auto-optimized batch size: {args.batch_size} → {optimal_batch} for {args.model}")
                args.batch_size = optimal_batch
        # Apply GPU defaults if not explicitly set
        if not hasattr(args, 'mixed_precision') or args.mixed_precision is None:
            args.mixed_precision = config["mixed_precision"]
        if not hasattr(args, 'compile') or args.compile is None:
            args.compile = config["compile"]
        args.yield_ms = config["yield_ms"]
        # Apply VRAM fraction from config if not explicitly set
        if args.vram_fraction == 0.45:  # Default value, apply config
            args.vram_fraction = config["vram_fraction"]
        # Apply path and API configuration
        args.base = config["base"]
        args.api_endpoint = config["api_endpoint"]
        args.disable_api = config["disable_api"]
        logging.info(f"✅ GPU config: device={config['device']}, task={config['task']}, model={config['model']}, imgsz={config['imgsz']}, batch_size={args.batch_size}, mixed_precision={config['mixed_precision']}, compile={config['compile']}, vram_fraction={args.vram_fraction}, base={config['base']}, api_endpoint={config['api_endpoint']}")

    # Base path is already set from config above
    base = Path(args.base).expanduser().resolve()
    ensure_dirs(base)

    compile_model = args.compile and not args.no_compile
    
    if args.task == "detect":
        runner = YOLODetector(
            weights=args.model,
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            batch_size=args.batch_size,
            mixed_precision=args.mixed_precision,
            compile_model=compile_model,
            yield_ms=args.yield_ms,
        )
    else:
        runner = YOLOClassifier(
            weights=args.model,
            device=args.device,
            imgsz=args.imgsz,
            tta=args.tta,
            batch_size=args.batch_size,
            mixed_precision=args.mixed_precision,
            compile_model=compile_model,
            yield_ms=args.yield_ms,
        )

    if args.once:
        run_once(base, runner, args.topk, workers=args.workers, batch_size=args.batch_size, skip_delete=args.skip_delete, api_endpoint=args.api_endpoint, disable_api=args.disable_api)
    else:
        run_once(base, runner, args.topk, workers=args.workers, batch_size=args.batch_size, skip_delete=args.skip_delete, api_endpoint=args.api_endpoint, disable_api=args.disable_api)
        run_watch(base, runner, args.topk, args.poll, args.workers, api_endpoint=args.api_endpoint, disable_api=args.disable_api)

if __name__ == "__main__":
    main()




# #!/usr/bin/env python3
# """ 
# YOLO Classification Watcher
# ---------------------------
# - Watches <base>/pending for .jpg/.jpeg images.
# - Moves new files to <base>/processing, runs YOLO classification (top-K).
# - Writes a JSON result and a human-readable TXT summary.
# - Moves outputs to <base>/completed on success, or to <base>/failed on error.
# - Deletes the original image file after successful processing and extraction.

# Usage:
#   python app.py --base /path/to/base --model yolov8n-cls.pt --device auto --topk 5

# Folder layout (auto-created):
#   base/
#     pending/
#     processing/
#     completed/
#     failed/

# Notes:
# - The first run will download the model weights (internet required).
# - Classification models don't draw boxes; we emit a JSON (+TXT) with top-K labels.

# """

# import argparse
# import logging
# import signal
# import sys
# import shutil
# import time
# import json
# from dataclasses import dataclass
# from pathlib import Path
# from typing import List, Dict, Any

# # Third-party
# from ultralytics import YOLO

# # --------- CLI ---------

# def parse_args() -> argparse.Namespace:
#     p = argparse.ArgumentParser(description="Watch a folder and classify images with YOLO.")
#     p.add_argument("--base", required=True, help="Base folder (will contain pending/processing/completed/failed)")
#     p.add_argument("--model", default="yolov8n-cls.pt", help="Ultralytics classification model weights (e.g., yolov8n-cls.pt)")
#     p.add_argument("--device", default="auto", help="Device: 'cpu', 'cuda:0', or 'auto'")
#     p.add_argument("--topk", type=int, default=5, help="How many top classes to return")
#     p.add_argument("--poll", type=float, default=0.5, help="Polling interval (seconds)")
#     p.add_argument("--workers", type=int, default=1, help="Number of parallel workers (ignored - single-threaded only)")
#     p.add_argument("--once", action="store_true", help="Process current pending files then exit (no watch)")
#     p.add_argument("--imgsz", type=int, default=1280, help="Inference image size (e.g. 224/320/640)")
#     p.add_argument("--batch-size", type=int, default=8, help="Batch size for RTX 5070 optimization (32+ for max utilization)")
#     p.add_argument("--max-batch", type=int, default=64, help="Maximum batch size for RTX 5070 (64+ for 12GB VRAM)")
#     p.add_argument("--mixed-precision", action="store_true", help="Enable mixed precision for faster inference")
#     p.add_argument("--compile", action="store_true", help="Enable PyTorch compilation for faster inference (SLOW for large models)")
#     p.add_argument("--no-compile", action="store_true", help="Disable compilation (recommended for YOLO11x)")
#     p.add_argument("--skip-delete", action="store_true", help="Skip deleting original images for faster processing")
#     p.add_argument("--tta", action="store_true", help="Enable test-time augmentation")  # <-- add this
#     p.add_argument("--task", choices=["classify", "detect"], default="classify", help="Run classification or detection (demo uses detect).")
#     p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detection")
#     p.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS in detection")
#     return p.parse_args()


# # --------- Model Wrapper ---------

# @dataclass
# class ClassificationResult:
#     path: str
#     topk: List[Dict[str, Any]]  # [{label, score}]
#     duration_ms: float

# import torch

# def check_gpu_info():
#     """Check GPU information and CUDA availability"""
#     if torch.cuda.is_available():
#         gpu_count = torch.cuda.device_count()
#         current_device = torch.cuda.current_device()
#         gpu_name = torch.cuda.get_device_name(current_device)
#         gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
#         logging.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
#         logging.info(f"CUDA version: {torch.version.cuda}")
        
#         # Check if RTX 5070 for optimization
#         if "RTX 5070" in gpu_name or "5070" in gpu_name:
#             logging.info("🚀 RTX 5070 detected - enabling maximum performance optimizations")
#             return "rtx5070"
#         return True
#     else:
#         logging.warning("CUDA not available, falling back to CPU")
#         return False

# def optimize_gpu_settings():
#     """Optimize GPU settings for maximum performance"""
#     if torch.cuda.is_available():
#         # Enable cuDNN benchmarking for optimal performance
#         torch.backends.cudnn.benchmark = True
#         torch.backends.cudnn.deterministic = False
        
#         # Set memory allocation strategy
#         torch.cuda.empty_cache()
        
#         # Enable TensorFloat-32 (TF32) for faster training on Ampere GPUs
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True
        
#         logging.info("✅ GPU optimizations enabled: cuDNN benchmark, TF32, memory optimization")

# def warmup_gpu():
#     """Warm up GPU with a dummy operation to ensure it stays active"""
#     if torch.cuda.is_available():
#         try:
#             # Create a dummy tensor and perform operations to warm up GPU
#             dummy_tensor = torch.randn(1000, 1000, device='cuda:0')
#             _ = torch.matmul(dummy_tensor, dummy_tensor)
#             torch.cuda.synchronize()
#             logging.info("🔥 GPU warmed up for optimal performance")
#         except Exception as e:
#             logging.warning(f"GPU warmup failed: {e}")

# def monitor_gpu_usage():
#     """Monitor GPU usage and provide recommendations"""
#     if torch.cuda.is_available():
#         try:
#             # Get GPU memory info
#             total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
#             allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
#             cached_memory = torch.cuda.memory_reserved(0) / 1024**3
            
#             utilization = (allocated_memory / total_memory) * 100
            
#             logging.info(f"📊 GPU Memory: {allocated_memory:.1f}GB/{total_memory:.1f}GB ({utilization:.1f}%)")
#             logging.info(f"📊 GPU Cached: {cached_memory:.1f}GB")
            
#             # Provide recommendations
#             if utilization < 20:
#                 logging.warning("⚠️  Low GPU utilization! Consider increasing --batch-size")
#             elif utilization < 50:
#                 logging.info("💡 GPU utilization could be higher. Try larger batch sizes")
#             elif utilization > 90:
#                 logging.warning("⚠️  High GPU memory usage! Consider reducing batch size")
#             else:
#                 logging.info("✅ Good GPU utilization")
                
#         except Exception as e:
#             logging.warning(f"Could not monitor GPU usage: {e}")

# def get_optimal_batch_size(model_name: str, imgsz: int, gpu_type: str = None) -> int:
#     """Get optimal batch size based on model and GPU for maximum utilization"""
#     if gpu_type == "rtx5070":
#         # RTX 5070 optimized batch sizes for maximum GPU utilization
#         if "yolo11n" in model_name.lower() or "yolov8n" in model_name.lower():
#             return min(64, max(32, 4096 // imgsz))  # Much larger batches
#         elif "yolo11s" in model_name.lower() or "yolov8s" in model_name.lower():
#             return min(48, max(24, 3072 // imgsz))
#         elif "yolo11m" in model_name.lower() or "yolov8m" in model_name.lower():
#             return min(32, max(16, 2048 // imgsz))
#         elif "yolo11l" in model_name.lower() or "yolov8l" in model_name.lower():
#             return min(24, max(12, 1536 // imgsz))
#         elif "yolo11x" in model_name.lower() or "yolov8x" in model_name.lower():
#             return min(16, max(8, 1024 // imgsz))
    
#     # Default batch sizes
#     return min(16, max(8, 1024 // imgsz))

# class YOLOClassifier:
#     def __init__(self, weights: str, device: str = "auto", imgsz: int = 640, tta: bool = False, batch_size: int = 8, 
#                  mixed_precision: bool = False, compile_model: bool = False):
#         logging.info("Loading model: %s (device=%s, batch_size=%d)", weights, device, batch_size)
#         logging.info("⏳ Downloading/loading model... This may take a moment for large models.")
        
#         # Show model size info
#         if "yolo11x" in weights.lower() or "yolov8x" in weights.lower():
#             logging.info("⚠️  Large model detected. Consider using smaller models for faster loading.")
        
#         self.model = YOLO(weights)
#         if device == "auto":
#             self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
#         else:
#             self.device = device
#         self.imgsz = imgsz
#         self.tta = tta
#         self.batch_size = batch_size
#         self.mixed_precision = mixed_precision
        
#         # Apply GPU optimizations
#         if self.device.startswith("cuda"):
#             optimize_gpu_settings()
            
#             # Warm up GPU for optimal performance
#             warmup_gpu()
            
#             # Enable mixed precision if requested
#             if mixed_precision:
#                 logging.info("🚀 Mixed precision enabled for faster inference")
                
#             # Compile model if requested (PyTorch 2.0+)
#             if compile_model and hasattr(torch, 'compile'):
#                 try:
#                     logging.info("⏳ Compiling model... This may take 1-2 minutes for large models like YOLO11x")
#                     self.model.model = torch.compile(self.model.model, mode="reduce-overhead")
#                     logging.info("🚀 Model compilation enabled for faster inference")
#                 except Exception as e:
#                     logging.warning(f"Model compilation failed: {e}")
#                     logging.info("Continuing without compilation...")
#             else:
#                 logging.info("ℹ️  Compilation disabled - using standard inference")
        
#         logging.info("✅ Model loaded successfully!")

#     def classify(self, img_path: Path, topk: int = 5) -> ClassificationResult:
#         t0 = time.time()
#         results = self.model.predict(
#             source=str(img_path),
#             device=self.device,
#             verbose=False,
#             imgsz=self.imgsz,
#             augment=self.tta
#         )
#         dt_ms = (time.time() - t0) * 1000.0

#         r = results[0]
#         probs = r.probs

#         # Use the new API: top1 / top5
#         if topk <= 1:
#             idxs = [int(probs.top1)]
#             confs = [float(probs.top1conf)]
#         else:
#             idxs = list(probs.top5)[:topk]
#             confs = [float(c) for c in probs.top5conf[: len(idxs)]]

#         names = r.names
#         def idx_to_label(i):
#             if isinstance(names, (list, tuple)):
#                 return names[i] if 0 <= i < len(names) else str(i)
#             elif isinstance(names, dict):
#                 return names.get(i, str(i))
#             return str(i)

#         items = [{"label": idx_to_label(i), "score": c} for i, c in zip(idxs, confs)]
#         return ClassificationResult(path=str(img_path), topk=items, duration_ms=dt_ms)

#     def classify_batch(self, img_paths: List[Path], topk: int = 5) -> List[ClassificationResult]:
#         """Process multiple images in a single batch for RTX 5070 optimization"""
#         t0 = time.time()
        
#         # Convert paths to strings for batch processing
#         source_paths = [str(p) for p in img_paths]
        
#         results = self.model.predict(
#             source=source_paths,
#             device=self.device,
#             verbose=False,
#             imgsz=self.imgsz,
#             augment=self.tta
#         )
#         dt_ms = (time.time() - t0) * 1000.0
        
#         batch_results = []
#         for i, (img_path, r) in enumerate(zip(img_paths, results)):
#             probs = r.probs
            
#             # Use the new API: top1 / top5
#             if topk <= 1:
#                 idxs = [int(probs.top1)]
#                 confs = [float(probs.top1conf)]
#             else:
#                 idxs = list(probs.top5)[:topk]
#                 confs = [float(c) for c in probs.top5conf[: len(idxs)]]

#             names = r.names
#             def idx_to_label(i):
#                 if isinstance(names, (list, tuple)):
#                     return names[i] if 0 <= i < len(names) else str(i)
#                 elif isinstance(names, dict):
#                     return names.get(i, str(i))
#                 return str(i)

#             items = [{"label": idx_to_label(i), "score": c} for i, c in zip(idxs, confs)]
#             batch_results.append(ClassificationResult(path=str(img_path), topk=items, duration_ms=dt_ms/len(img_paths)))
        
#         return batch_results


# import cv2

# @dataclass
# class DetectionResult:
#     path: str
#     duration_ms: float
#     dets: List[Dict[str, Any]]  # [{label, score, xyxy:[x1,y1,x2,y2]}]
#     annotated_path: str

# class YOLODetector:
#     def __init__(self, weights: str, device: str = "auto", imgsz: int = 640, conf: float = 0.25, iou: float = 0.45, 
#                  batch_size: int = 8, mixed_precision: bool = False, compile_model: bool = False):
#         logging.info("Loading DETECT model: %s (device=%s, batch_size=%d)", weights, device, batch_size)
#         logging.info("⏳ Downloading/loading model... This may take a moment for large models.")
        
#         # Show model size info
#         if "yolo11x" in weights.lower() or "yolov8x" in weights.lower():
#             logging.info("⚠️  Large model detected. Consider using smaller models for faster loading.")
        
#         self.model = YOLO(weights)
#         if device == "auto":
#             import torch
#             self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
#         else:
#             self.device = device
#         self.imgsz = imgsz
#         self.conf = conf
#         self.iou = iou
#         self.batch_size = batch_size
#         self.mixed_precision = mixed_precision
        
#         # Apply GPU optimizations
#         if self.device.startswith("cuda"):
#             optimize_gpu_settings()
            
#             # Warm up GPU for optimal performance
#             warmup_gpu()
            
#             # Enable mixed precision if requested
#             if mixed_precision:
#                 logging.info("🚀 Mixed precision enabled for faster inference")
                
#             # Compile model if requested (PyTorch 2.0+)
#             if compile_model and hasattr(torch, 'compile'):
#                 try:
#                     logging.info("⏳ Compiling model... This may take 1-2 minutes for large models like YOLO11x")
#                     self.model.model = torch.compile(self.model.model, mode="reduce-overhead")
#                     logging.info("🚀 Model compilation enabled for faster inference")
#                 except Exception as e:
#                     logging.warning(f"Model compilation failed: {e}")
#                     logging.info("Continuing without compilation...")
#             else:
#                 logging.info("ℹ️  Compilation disabled - using standard inference")
        
#         logging.info("✅ Model loaded successfully!")

#     def detect(self, img_path: Path) -> DetectionResult:
#         t0 = time.time()
#         results = self.model.predict(
#             source=str(img_path),
#             device=self.device,
#             verbose=False,
#             imgsz=self.imgsz,
#             conf=self.conf,
#             iou=self.iou,
#             agnostic_nms=False,
#             max_det=300,
#         )
#         dt_ms = (time.time() - t0) * 1000.0

#         r = results[0]
#         names = r.names
#         dets = []
#         if r.boxes is not None and len(r.boxes) > 0:
#             for b in r.boxes:
#                 cls_id = int(b.cls.item())
#                 conf = float(b.conf.item())
#                 x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
#                 label = names[cls_id] if isinstance(names, (list, tuple)) else names.get(cls_id, str(cls_id))
#                 dets.append({"label": label, "score": conf, "xyxy": [x1, y1, x2, y2]})

#         # render annotated image
#         annotated = r.plot()  # numpy array (BGR)
#         ann_path = img_path.with_suffix("").parent / f"{img_path.stem}_det.jpg"
#         cv2.imwrite(str(ann_path), annotated)

#         return DetectionResult(
#             path=str(img_path),
#             duration_ms=dt_ms,
#             dets=dets,
#             annotated_path=str(ann_path),
#         )

#     def detect_batch(self, img_paths: List[Path]) -> List[DetectionResult]:
#         """Process multiple images in a single batch for RTX 5070 optimization"""
#         t0 = time.time()
        
#         # Convert paths to strings for batch processing
#         source_paths = [str(p) for p in img_paths]
        
#         results = self.model.predict(
#             source=source_paths,
#             device=self.device,
#             verbose=False,
#             imgsz=self.imgsz,
#             conf=self.conf,
#             iou=self.iou,
#             agnostic_nms=False,
#             max_det=300,
#         )
#         dt_ms = (time.time() - t0) * 1000.0
        
#         batch_results = []
#         for i, (img_path, r) in enumerate(zip(img_paths, results)):
#             names = r.names
#             dets = []
#             if r.boxes is not None and len(r.boxes) > 0:
#                 for b in r.boxes:
#                     cls_id = int(b.cls.item())
#                     conf = float(b.conf.item())
#                     x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
#                     label = names[cls_id] if isinstance(names, (list, tuple)) else names.get(cls_id, str(cls_id))
#                     dets.append({"label": label, "score": conf, "xyxy": [x1, y1, x2, y2]})

#             # render annotated image
#             annotated = r.plot()  # numpy array (BGR)
#             ann_path = img_path.with_suffix("").parent / f"{img_path.stem}_det.jpg"
#             cv2.imwrite(str(ann_path), annotated)

#             batch_results.append(DetectionResult(
#                 path=str(img_path),
#                 duration_ms=dt_ms/len(img_paths),
#                 dets=dets,
#                 annotated_path=str(ann_path),
#             ))
        
#         return batch_results

# # --------- Pipeline ---------

# IMAGE_EXTS = {".jpg", ".jpeg"}

# def ensure_dirs(base: Path):
#     for name in ("pending", "processing", "completed", "failed"):
#         (base / name).mkdir(parents=True, exist_ok=True)

# def move(src: Path, dst: Path):
#     if not src.exists():
#         raise FileNotFoundError(f"Source file does not exist: {src}")
#     dst.parent.mkdir(parents=True, exist_ok=True)
#     try:
#         return shutil.move(str(src), str(dst))
#     except Exception as e:
#         raise RuntimeError(f"Failed to move {src} to {dst}: {e}")


# def write_json(path: Path, data: Dict[str, Any]):
#     path.parent.mkdir(parents=True, exist_ok=True)
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=2)

# def write_txt(path: Path, lines: List[str]):
#     with open(path, "w", encoding="utf-8") as f:
#         f.write("\n".join(lines) + "\n")

# def process_one(img_in_pending: Path, base: Path, runner, topk: int) -> None:
#     processing_dir = base / "processing"
#     completed_dir = base / "completed"
#     failed_dir = base / "failed"

#     try:
#         # Check if the file actually exists before processing
#         if not img_in_pending.exists():
#             logging.warning("File does not exist, skipping: %s", img_in_pending)
#             return
        
#         logging.info("Processing file: %s", img_in_pending)
#         # Move to processing
#         processing_path = processing_dir / img_in_pending.name
#         move(img_in_pending, processing_path)
#         logging.info("Moved to processing: %s", processing_path)

#         # ===== Run inference (detect OR classify) =====
#         if isinstance(runner, YOLODetector):
#             det = runner.detect(processing_path)

#             # outputs
#             stem = processing_path.stem
#             json_path = processing_path.with_suffix("").parent / f"{stem}.json"
#             ann_path  = Path(det.annotated_path)

#             payload = {
#                 "image": processing_path.name,
#                 "duration_ms": round(det.duration_ms, 2),
#                 "detections": det.dets,  # [{label, score, xyxy}]
#                 "imgsz": runner.imgsz,
#                 "conf": runner.conf,
#                 "iou": runner.iou,
#                 "annotated_image": ann_path.name,
#             }
#             write_json(json_path, payload)


#             # Move to completed (json + annotated image)
#             for p in (json_path, ann_path):
#                 dest = completed_dir / p.name
#                 move(p, dest)
            
#             # Delete the original image file after successful processing
#             if processing_path.exists():
#                 processing_path.unlink()
#                 logging.info("✓ Deleted original image: %s", processing_path.name)

#             logging.info("✓ Completed (detect): %s", processing_path.name)

#         else:
#             # Classification branch (unchanged)
#             res = runner.classify(processing_path, topk=topk)

#             stem = processing_path.stem
#             json_path = processing_path.with_suffix("").parent / f"{stem}.json"

#             payload = {
#                 "image": processing_path.name,
#                 "duration_ms": round(res.duration_ms, 2),
#                 "topk": res.topk,
#             }
#             write_json(json_path, payload)


#             # Move to completed (json only)
#             dest = completed_dir / json_path.name
#             move(json_path, dest)
            
#             # Delete the original image file after successful processing
#             if processing_path.exists():
#                 processing_path.unlink()
#                 logging.info("✓ Deleted original image: %s", processing_path.name)

#             logging.info("✓ Completed (classify): %s", processing_path.name)

#     except Exception as e:
#         logging.exception("Failed processing: %s", img_in_pending.name)
#         try:
#             # The file has been moved to processing, so we need to move it from there
#             processing_path = processing_dir / img_in_pending.name
#             if processing_path.exists():
#                 move(processing_path, failed_dir / img_in_pending.name)
#             elif img_in_pending.exists():
#                 # Fallback: if somehow the file is still in pending
#                 move(img_in_pending, failed_dir / img_in_pending.name)
#         except Exception:
#             pass
#         err_txt = failed_dir / f"{img_in_pending.stem}_error.txt"
#         write_txt(err_txt, [f"Error: {e}"])

# def process_batch(batch_imgs: List[Path], base: Path, runner, topk: int, skip_delete: bool = False, stop_flag: bool = False) -> None:
#     """Process a batch of images for RTX 5070 optimization"""
#     processing_dir = base / "processing"
#     completed_dir = base / "completed"
#     failed_dir = base / "failed"
    
#     # Set a timeout for the entire batch processing
#     import signal
    
#     def timeout_handler(signum, frame):
#         raise TimeoutError("Batch processing timeout")
    
#     # Set timeout to 5 minutes per batch
#     signal.signal(signal.SIGALRM, timeout_handler)
#     signal.alarm(300)  # 5 minutes timeout
    
#     # Check if we should stop before starting
#     if stop_flag:
#         logging.info("Stopping batch processing before starting...")
#         return
    
#     try:
#         # Move all images to processing
#         processing_paths = []
#         for img in batch_imgs:
#             if not img.exists():
#                 logging.warning("File does not exist, skipping: %s", img)
#                 continue
#             processing_path = processing_dir / img.name
#             move(img, processing_path)
#             processing_paths.append(processing_path)
        
#         if not processing_paths:
#             logging.warning("No valid images in batch")
#             return
            
#         logging.info("Processing batch of %d images", len(processing_paths))
#         logging.info("🔥 Using TRUE BATCH PROCESSING for RTX 5070 optimization!")
        
#         # Monitor GPU usage before processing
#         monitor_gpu_usage()
        
#         # Run batch inference with GPU monitoring
#         logging.info(f"🚀 Starting batch inference of {len(processing_paths)} images...")
#         start_time = time.time()
#         if isinstance(runner, YOLODetector):
#             batch_results = runner.detect_batch(processing_paths)
            
#             # Monitor GPU usage after batch inference
#             inference_time = time.time() - start_time
#             logging.info(f"⚡ Batch inference completed in {inference_time:.2f}s")
#             monitor_gpu_usage()
            
#             # Process results in parallel for faster I/O
#             import concurrent.futures
#             import threading
            
#             io_start_time = time.time()
#             logging.info("📁 Starting parallel I/O processing...")
            
#             def process_single_result(args):
#                 processing_path, det, completed_dir = args
#                 stem = processing_path.stem
#                 json_path = processing_path.with_suffix("").parent / f"{stem}.json"
#                 ann_path = Path(det.annotated_path)

#                 payload = {
#                     "image": processing_path.name,
#                     "duration_ms": round(det.duration_ms, 2),
#                     "detections": det.dets,
#                     "imgsz": runner.imgsz,
#                     "conf": runner.conf,
#                     "iou": runner.iou,
#                     "annotated_image": ann_path.name,
#                 }
#                 write_json(json_path, payload)

#                 # Move to completed
#                 for p in (json_path, ann_path):
#                     dest = completed_dir / p.name
#                     move(p, dest)
                
#                 # Delete original image (optimized for speed)
#                 if not skip_delete and processing_path.exists():
#                     try:
#                         processing_path.unlink()
#                         return f"✓ Deleted: {processing_path.name}"
#                     except Exception as e:
#                         return f"⚠️ Failed to delete: {processing_path.name} - {e}"
#                 return None
            
#             # Process all results in parallel with interrupt handling
#             try:
#                 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
#                     results = list(zip(processing_paths, batch_results, [completed_dir] * len(processing_paths)))
#                     futures = [executor.submit(process_single_result, args) for args in results]
                    
#                     # Collect results without individual logging for speed
#                     completed_deletions = 0
#                     failed_deletions = 0
                    
#                     for future in concurrent.futures.as_completed(futures):
#                         if stop_flag:  # Check for stop signal during processing
#                             logging.info("Stopping I/O processing due to signal...")
#                             break
#                         result = future.result()
#                         if result:
#                             if "✓ Deleted:" in result:
#                                 completed_deletions += 1
#                             elif "⚠️ Failed" in result:
#                                 failed_deletions += 1
#                                 logging.warning(result)
#             except KeyboardInterrupt:
#                 logging.info("I/O processing interrupted by user")
#                 return
            
#             io_time = time.time() - io_start_time
#             logging.info(f"📁 Parallel I/O completed in {io_time:.2f}s - {completed_deletions} deleted, {failed_deletions} failed")

#         else:
#             # Classification batch processing
#             batch_results = runner.classify_batch(processing_paths, topk=topk)
            
#             # Monitor GPU usage after batch inference
#             inference_time = time.time() - start_time
#             logging.info(f"⚡ Batch inference completed in {inference_time:.2f}s")
#             monitor_gpu_usage()
            
#             # Process results in parallel for faster I/O
#             import concurrent.futures
            
#             def process_single_classification(args):
#                 processing_path, res, completed_dir = args
#                 stem = processing_path.stem
#                 json_path = processing_path.with_suffix("").parent / f"{stem}.json"

#                 payload = {
#                     "image": processing_path.name,
#                     "duration_ms": round(res.duration_ms, 2),
#                     "topk": res.topk,
#                 }
#                 write_json(json_path, payload)

#                 # Move to completed
#                 dest = completed_dir / json_path.name
#                 move(json_path, dest)
                
#                 # Delete original image (optimized for speed)
#                 if not skip_delete and processing_path.exists():
#                     try:
#                         processing_path.unlink()
#                         return f"✓ Deleted: {processing_path.name}"
#                     except Exception as e:
#                         return f"⚠️ Failed to delete: {processing_path.name} - {e}"
#                 return None
            
#             # Process all results in parallel
#             with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
#                 results = list(zip(processing_paths, batch_results, [completed_dir] * len(processing_paths)))
#                 futures = [executor.submit(process_single_classification, args) for args in results]
                
#                 for future in concurrent.futures.as_completed(futures):
#                     result = future.result()
#                     if result:
#                         logging.info(result)
            
#             io_time = time.time() - io_start_time
#             logging.info(f"📁 Parallel I/O completed in {io_time:.2f}s")

#         logging.info("✓ Completed batch of %d images", len(processing_paths))
        
#     except Exception as e:
#         logging.exception("Failed processing batch")
#         # Move failed images to failed directory
#         for img in batch_imgs:
#             try:
#                 processing_path = processing_dir / img.name
#                 if processing_path.exists():
#                     move(processing_path, failed_dir / img.name)
#                 elif img.exists():
#                     move(img, failed_dir / img.name)
#             except Exception:
#                 pass
#         err_txt = failed_dir / f"batch_error_{int(time.time())}.txt"
#         write_txt(err_txt, [f"Batch processing error: {e}"])
#     finally:
#         # Cancel timeout
#         signal.alarm(0)

# def find_pending_images(base: Path) -> list[Path]:
#     pending_dir = base / "pending"
#     files = []
    
#     # Check if pending directory exists
#     if not pending_dir.exists():
#         logging.warning("Pending directory does not exist: %s", pending_dir)
#         return files
    
#     try:
#         for p in pending_dir.iterdir():
#             if not p.is_file():
#                 continue
#             if p.suffix.lower() in {".jpg", ".jpeg"}:
#                 # Skip if a tmp/incomplete file (common pattern: ends with .part or ~)
#                 if p.name.endswith(".part") or p.name.startswith("~"):
#                     continue
#                 files.append(p)
#     except Exception as e:
#         logging.error("Error scanning pending directory %s: %s", pending_dir, e)

#     return sorted(files)

# def run_once(base: Path, runner, topk: int, workers: int = 1, batch_size: int = 8, skip_delete: bool = False):
#     imgs = find_pending_images(base)
#     if not imgs:
#         logging.info("No pending images to process.")
#         return
#     logging.info("Found %d image(s) to process.", len(imgs))
    
#     # Set up signal handler for graceful shutdown during batch processing
#     stop = False
#     executor = None
    
#     def handle_sig(sig, frame):
#         nonlocal stop, executor
#         print(f"\nReceived signal {sig}, stopping batch processing...")
#         logging.info("Received signal %s, stopping batch processing...", sig)
#         stop = True
#         # Cancel any running executor
#         if executor:
#             executor.shutdown(wait=False)
    
#     signal.signal(signal.SIGINT, handle_sig)
#     signal.signal(signal.SIGTERM, handle_sig)
    
#     try:
#         # Process images in batches for RTX 5070 optimization
#         for i in range(0, len(imgs), batch_size):
#             if stop:
#                 logging.info("Stopping batch processing due to signal")
#                 break
            
#             batch_imgs = imgs[i:i + batch_size]
#             logging.info("Processing batch %d/%d (%d images)", i//batch_size + 1, (len(imgs) + batch_size - 1)//batch_size, len(batch_imgs))
            
#             # Use batch processing if available and batch size > 1
#             if batch_size > 1:
#                 try:
#                     process_batch(batch_imgs, base, runner, topk, skip_delete, stop)
#                 except Exception as e:
#                     logging.error(f"Batch processing failed: {e}")
#                     if stop:
#                         break
#             else:
#                 # Fall back to individual processing
#                 for img in batch_imgs:
#                     if stop:
#                         break
#                     process_one(img, base, runner, topk)
                    
#     except KeyboardInterrupt:
#         print("\nKeyboardInterrupt received, stopping batch processing...")
#         logging.info("KeyboardInterrupt received, stopping batch processing...")
#     except Exception as e:
#         logging.error(f"Unexpected error in batch processing: {e}")
#     finally:
#         logging.info("Batch processing complete")
#         # Reset signal handlers
#         signal.signal(signal.SIGINT, signal.SIG_DFL)
#         signal.signal(signal.SIGTERM, signal.SIG_DFL)

# def run_watch(base: Path, runner, topk: int, poll: float, workers: int):
#     logging.info("Watching %s for new images... (poll=%.2fs)", (base / 'pending'), poll)
#     stop = False

#     def handle_sig(sig, frame):
#         nonlocal stop
#         print(f"\nReceived signal {sig}, shutting down...")
#         logging.info("Received signal %s, shutting down...", sig)
#         stop = True

#     # Set up signal handlers
#     signal.signal(signal.SIGINT, handle_sig)
#     signal.signal(signal.SIGTERM, handle_sig)

#     try:
#         while not stop:
#             new_files = find_pending_images(base)
#             for img in new_files:
#                 if stop:  # Check for stop signal between files
#                     break
#                 process_one(img, base, runner, topk)
#             if not stop:
#                 time.sleep(poll)
#     except KeyboardInterrupt:
#         print("\nKeyboardInterrupt received, shutting down...")
#         logging.info("KeyboardInterrupt received, shutting down...")
#     finally:
#         logging.info("Shutdown complete")

# def main():
#     args = parse_args()
#     logging.basicConfig(
#         level=logging.INFO,
#         format="[%(asctime)s] [%(levelname)s] %(message)s",
#         datefmt="%Y-%m-%d %H:%M:%S",
#     )
    
#     # Check GPU availability and info
#     gpu_type = check_gpu_info()
    
#     base = Path(args.base).expanduser().resolve()
#     ensure_dirs(base)

#     # Auto-optimize batch size for RTX 5070
#     if args.batch_size == 32:  # Default value
#         optimal_batch = get_optimal_batch_size(args.model, args.imgsz, gpu_type)
#         if optimal_batch != args.batch_size:
#             logging.info(f"🚀 Auto-optimized batch size: {args.batch_size} → {optimal_batch} for {args.model}")
#             args.batch_size = optimal_batch

#     # Handle compilation flags
#     compile_model = args.compile and not args.no_compile
    
#     # Choose runner by task
#     if args.task == "detect":
#         runner = YOLODetector(
#             weights=args.model,
#             device=args.device,
#             imgsz=args.imgsz,
#             conf=args.conf,
#             iou=args.iou,
#             batch_size=args.batch_size,
#             mixed_precision=args.mixed_precision,
#             compile_model=compile_model,
#         )
#     else:
#         runner = YOLOClassifier(
#             weights=args.model,
#             device=args.device,
#             imgsz=args.imgsz,
#             tta=args.tta,
#             batch_size=args.batch_size,
#             mixed_precision=args.mixed_precision,
#             compile_model=compile_model,
#         )

#     if args.once:
#         run_once(base, runner, args.topk, workers=args.workers, batch_size=args.batch_size, skip_delete=args.skip_delete)
#     else:
#         run_once(base, runner, args.topk, workers=args.workers, batch_size=args.batch_size, skip_delete=args.skip_delete)
#         run_watch(base, runner, args.topk, args.poll, args.workers)


# if __name__ == "__main__":
#     main()
