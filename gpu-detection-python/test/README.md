# YOLO Classification Watcher

Watches a folder for `.jpg`/`.jpeg` images, classifies them with an Ultralytics YOLO *classification* model, and moves results to `completed/`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Choose a base folder (it will create pending/, processing/, completed/, failed/)
mkdir -p /path/to/base/pending

# Run the watcher (downloads model on first run)
python app.py --base /path/to/base --model yolov8n-cls.pt --device auto --topk 5
```

Drop `.jpg` files into `/path/to/base/pending/` and watch them flow to `completed/` with a JSON + TXT summary.

### Notes

- The first time you run, Ultralytics will download `yolov8n-cls.pt`. You can swap to a different classification model (e.g., `yolov8s-cls.pt`) if you want.
- If you have a CUDA GPU, ensure you install a CUDA-enabled PyTorch build that matches your driver.
- Use `--once` to process current backlog and exit.
- Use `--workers N` to process multiple images in parallel.
