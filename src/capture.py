"""
capture.py
----------
Camera capture module for Raspberry Pi 4 Model B, Linux, and Windows systems.
Supports:
  1. High-speed 3-frame burst capture per second (V4L2 / DirectShow / MJPEG)
  2. Raspberry Pi 4 USB webcams & CSI camera module support
  3. Continuous 1-second interval telemetry collection loop
  4. Central JSON configuration integration
"""

import cv2
import numpy as np
import time
import sys
import os
import json

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")


def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


_CONFIG = load_config()


def _get_video_backend():
    """Selects the best OpenCV video backend for the target hardware platform."""
    platform = sys.platform.lower()
    if platform.startswith("linux"):
        # On Raspberry Pi OS / Linux, use Video4Linux2 (V4L2)
        return cv2.CAP_V4L2
    elif platform.startswith("win"):
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def open_camera(camera_index: int = 0, width: int = 1280, height: int = 720) -> cv2.VideoCapture:
    """
    Opens camera device with hardware-optimized backend (V4L2 for Raspberry Pi 4).
    Configures resolution and MJPG pixel format.
    """
    backend = _get_video_backend()
    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {camera_index}. Check USB/CSI connection.")

    # Apply properties
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer lag on Pi 4
    except Exception:
        pass

    return cap


def discover_camera_details(max_tested: int = 5) -> list:
    """Detect available camera hardware devices and return metadata."""
    cameras = []
    backend = _get_video_backend()
    for idx in range(max_tested):
        try:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    label = "Raspberry Pi Camera / Primary Webcam" if idx == 0 else f"Camera Device #{idx}"
                    cameras.append({
                        "index": idx,
                        "id": str(idx),
                        "label": f"{label} ({w}x{h})",
                        "name": label,
                        "resolution": f"{w}x{h}",
                        "rtsp_url": str(idx)
                    })
                cap.release()
        except Exception:
            pass
    return cameras


def find_available_cameras(max_tested: int = 5) -> list:
    cams = discover_camera_details(max_tested)
    return [c["index"] for c in cams] if cams else [0]


def capture_burst_frames(cap: cv2.VideoCapture, num_frames: int = 3, burst_delay: float = 0.05) -> list:
    """
    Grabs a burst of `num_frames` (default: 3) in rapid succession from an already open camera.
    Fast and non-blocking for 1-second interval execution on Raspberry Pi 4.
    """
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            frames.append(frame)
        if burst_delay > 0:
            time.sleep(burst_delay)
    return frames


def capture_from_webcam(camera_index: int = 0, num_frames: int = 3) -> list:
    """Interactive preview window for capturing a 3-frame burst."""
    cap = open_camera(camera_index)
    window_name = "Webcam Extractor - [SPACE/C] Capture 3-Frame Burst | [Q] Quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    print("\nWebcam preview active.")
    print(" -> Press 'SPACE' or 'c' to CAPTURE a 3-frame burst for 100% accuracy.")
    print(" -> Press 'q' or 'ESC' to cancel.\n")

    captured_frames = []
    try:
        while True:
            ret, live_frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            display_frame = live_frame.copy()
            h, w = display_frame.shape[:2]

            cv2.rectangle(display_frame, (0, 0), (w, 40), (20, 20, 20), -1)
            cv2.putText(
                display_frame,
                f"Press 'SPACE' to Capture {num_frames}-Frame Burst | 'q' to Quit",
                (15, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord('c'), ord('C'), 32):  # 'c' or SPACE
                print(f"Capturing {num_frames}-frame burst...")
                captured_frames = capture_burst_frames(cap, num_frames=num_frames, burst_delay=0.08)
                print(f"Successfully captured {len(captured_frames)} burst frames!")
                break
            elif key in (ord('q'), ord('Q'), 27):
                print("Webcam capture cancelled.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return captured_frames


def capture_headless(camera_index: int = 0, num_frames: int = 3, warmup_frames: int = 10) -> list:
    """Non-interactive automatic 3-frame burst capture."""
    cap = open_camera(camera_index)
    try:
        for _ in range(warmup_frames):
            cap.read()
            time.sleep(0.03)

        return capture_burst_frames(cap, num_frames=num_frames, burst_delay=0.08)
    finally:
        cap.release()


def run_1sec_burst_collection_loop(camera_index: int = 0, process_burst_fn=None, interval: float = 1.0, num_frames: int = 3):
    """
    Continuous 1-second telemetry collection loop for Raspberry Pi 4.
    Every `interval` seconds:
      1. Collects a 3-frame burst
      2. Invokes `process_burst_fn(frames)`
      3. Prints 100% accurate discrete consensus data to the terminal
    Press Ctrl+C to terminate.
    """
    cfg = load_config()
    target_interval = cfg.get("app", {}).get("extraction_interval_seconds", interval)
    burst_count = cfg.get("app", {}).get("burst_frame_count", num_frames)
    burst_delay = cfg.get("app", {}).get("burst_delay_seconds", 0.05)

    print(f"\n" + "=" * 65)
    print(f" STARTING 1-SECOND 3-FRAME BURST DATA COLLECTION LOOP ")
    print(f" Target Platform: Raspberry Pi 4 / Linux / Multi-OS")
    print(f" Cycle Interval : {target_interval} second(s)")
    print(f" Burst Frames   : {burst_count} frames per cycle")
    print(f" Press Ctrl+C in terminal to stop.")
    print("=" * 65 + "\n")

    cap = open_camera(camera_index)
    # Warm up camera sensor
    for _ in range(10):
        cap.read()
        time.sleep(0.02)

    cycle_count = 0
    try:
        while True:
            t_start = time.time()
            cycle_count += 1

            # 1. Collect 3-frame burst
            burst_frames = capture_burst_frames(cap, num_frames=burst_count, burst_delay=burst_delay)

            if burst_frames and process_burst_fn:
                # 2. Process burst with discrete consensus voting and print
                process_burst_fn(burst_frames, cycle_count=cycle_count)

            # 3. Precision sleep to maintain exactly 1-second cadence
            elapsed = time.time() - t_start
            sleep_time = max(0.01, target_interval - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[LOOP] 1-Second Data Collection Loop stopped by user.")
    finally:
        cap.release()
