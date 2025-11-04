#!/usr/bin/env python3
"""
Configuration Example for CPU/GPU Optimization
==============================================

This file shows how to modify the CPU_CONFIG and GPU_CONFIG constants
in app.py to customize the optimization settings.
"""

# Example: Custom CPU configuration for high-end CPU
CUSTOM_CPU_CONFIG = {
    "max_threads": 8,        # Use more threads for high-end CPU
    "batch_size": 1,        # Keep batch size 1 for CPU
    "min_imgsz": 640,       # Minimum image size
    "mixed_precision": False,  # Not supported on CPU
    "compile": False,       # Disable compilation for stability
    "yield_ms": 0          # No yield needed on CPU
}

# Example: Custom GPU configuration for RTX 5070
CUSTOM_GPU_CONFIG = {
    "max_threads": None,     # Use all available threads
    "batch_size": 16,       # Larger batch for RTX 5070
    "min_imgsz": 224,       # Lower minimum for GPU
    "mixed_precision": True,  # Enable mixed precision
    "compile": True,        # Enable compilation
    "yield_ms": 10         # Shorter yield for better performance
}

# Example: Conservative GPU configuration for older GPU
CONSERVATIVE_GPU_CONFIG = {
    "max_threads": None,
    "batch_size": 4,        # Smaller batch for older GPU
    "min_imgsz": 320,       # Higher minimum for stability
    "mixed_precision": False,  # Disable for older GPU
    "compile": False,       # Disable compilation
    "yield_ms": 50         # Longer yield for stability
}

# Example: High-performance CPU configuration
HIGH_PERF_CPU_CONFIG = {
    "max_threads": 12,      # Use more threads
    "batch_size": 1,        # Still keep batch size 1
    "min_imgsz": 512,       # Lower minimum for speed
    "mixed_precision": False,
    "compile": False,
    "yield_ms": 0
}

if __name__ == "__main__":
    print("Configuration Examples:")
    print("1. Custom CPU Config:", CUSTOM_CPU_CONFIG)
    print("2. Custom GPU Config:", CUSTOM_GPU_CONFIG)
    print("3. Conservative GPU Config:", CONSERVATIVE_GPU_CONFIG)
    print("4. High-Performance CPU Config:", HIGH_PERF_CPU_CONFIG)
    print("\nTo use these configurations, copy the values to the CPU_CONFIG and GPU_CONFIG")
    print("constants in app.py and modify as needed for your hardware.")
