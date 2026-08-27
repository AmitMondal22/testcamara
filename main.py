"""
main.py
-------
Raspberry Pi 4 Display Data Extractor — Continuous Loop.

Captures camera frames every N seconds (from config.json),
extracts all visible text/numbers from the display,
and prints clean JSON to stdout.

No image storage. No sample matching. Just live OCR → JSON.

Usage:
    python main.py                  # Run continuous loop (interval from config.json)
    python main.py --interval 3     # Override to 3 seconds
    python main.py --camera 1       # Use camera index 1
"""

import json
import os
import sys
import time
import signal
import argparse
from datetime import datetime

from dotenv import load_dotenv

from src.extractor import extract_from_frame, open_camera, load_dataset_file

# Load .env file
ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(ENV_FILE)

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
DATASET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.json")


def load_config():
    cfg = {
        "interval_seconds": 5,
        "camera_index": 0,
        "camera_resolution": [1280, 720],
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                cfg.update(json.load(f))
        except Exception:
            pass

    # dataset.json overrides fields definition if present
    dataset = load_dataset_file(DATASET_FILE)
    if dataset:
        cfg["fields"] = dataset

    # .env overrides config.json
    if os.getenv("INTERVAL_SECONDS"):
        cfg["interval_seconds"] = int(os.getenv("INTERVAL_SECONDS"))
    if os.getenv("CAMERA_INDEX"):
        cfg["camera_index"] = int(os.getenv("CAMERA_INDEX"))

    return cfg


def main():
    config = load_config()

    parser = argparse.ArgumentParser(description="Display Data Extractor for Raspberry Pi 4")
    parser.add_argument("-i", "--interval", type=float, help="Capture interval in seconds")
    parser.add_argument("-c", "--camera", type=int, help="Camera index")
    args = parser.parse_args()

    interval = args.interval or config.get("interval_seconds", 5)
    cam_idx = args.camera if args.camera is not None else config.get("camera_index", 0)
    resolution = tuple(config.get("camera_resolution", [1280, 720]))

    print("=" * 55)
    print("  DISPLAY DATA EXTRACTOR — Raspberry Pi 4")
    print(f"  Interval: {interval}s | Camera: {cam_idx}")
    print("  Press Ctrl+C to stop")
    print("=" * 55, flush=True)

    # Open camera
    try:
        cap = open_camera(cam_idx, resolution)
    except RuntimeError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # Graceful shutdown
    running = True
    def stop(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop)

    reading_num = 0
    try:
        while running:
            t0 = time.time()
            reading_num += 1

            # Flush camera buffer → grab freshest frame
            for _ in range(3):
                cap.grab()
            ret, frame = cap.read()

            if not ret or frame is None:
                print(f"[Warning] Frame capture failed", file=sys.stderr)
                time.sleep(1)
                continue

            data, item_count = extract_from_frame(frame, fields_config=config.get("fields"))

            output = {
                "reading": reading_num,
                "timestamp": datetime.now().isoformat(),
                "items_detected": item_count,
                "data": data,
            }

            print(json.dumps(output, indent=2, ensure_ascii=False), flush=True)

            elapsed = time.time() - t0
            sleep_time = max(0.1, interval - elapsed)
            time.sleep(sleep_time)

    finally:
        cap.release()
        print("\nCamera released. Exiting.")


if __name__ == "__main__":
    main()
