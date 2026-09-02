"""
capture.py
----------
Captures live frames or continuous video streams from a webcam device
for image data scraping.
"""

import cv2
import numpy as np
import time
import sys


def _get_backend():
    """Return platform-appropriate OpenCV VideoCapture backend (V4L2 on Linux/Raspberry Pi, DSHOW on Windows)."""
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    elif sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


def discover_camera_details(max_tested: int = 5) -> list:
    """Detect available hardware camera devices and return rich metadata."""
    cameras = []
    backend = _get_backend()
    for idx in range(max_tested):
        try:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx)

            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    label = "IMX477 IR Camera / Webcam #0" if idx == 0 else f"Attached Camera #{idx}"
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
    """Detect available camera indices connected to system."""
    cams = discover_camera_details(max_tested)
    return [c["index"] for c in cams] if cams else [0]



def capture_from_webcam(camera_index: int = 0, num_frames: int = 1) -> list:
    """
    Opens an interactive webcam GUI window.
    When SPACE/c is pressed, captures 1 crisp HD frame for instant high-speed OCR.
    """
    backend = _get_backend()
    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {camera_index}. Check camera connection.")

    # Request HD 1080p resolution for high OCR clarity with MJPG codec
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    window_name = "Webcam Extractor - [SPACE/C] Capture 3-Frame Burst | [Q] Quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    print("\nWebcam preview active.")
    print(" -> Press 'SPACE' or 'c' to CAPTURE a 3-frame burst for maximum accuracy.")
    print(" -> Press 'q' or 'ESC' to cancel.\n")

    captured_frames = []
    try:
        while True:
            ret, live_frame = cap.read()
            if not ret:
                print("Warning: Failed to read frame from webcam.")
                time.sleep(0.1)
                continue

            display_frame = live_frame.copy()
            h, w = display_frame.shape[:2]

            cv2.rectangle(display_frame, (0, 0), (w, 40), (20, 20, 20), -1)
            cv2.putText(
                display_frame,
                "Press 'SPACE' / 'c' to CAPTURE (3-Frame Burst) | Press 'q' to QUIT",
                (15, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord('c'), ord('C'), 32):  # 'c' or SPACE
                print(f"Capturing {num_frames} frames over 1.5 seconds for multi-frame accuracy...")
                for idx in range(num_frames):
                    ret_b, burst_frame = cap.read()
                    if ret_b and burst_frame is not None:
                        captured_frames.append(burst_frame)

                        # Flash progress overlay
                        flash_frame = burst_frame.copy()
                        cv2.rectangle(flash_frame, (0, 0), (w, 50), (0, 165, 255), -1)
                        cv2.putText(
                            flash_frame,
                            f"CAPTURING FRAME {idx+1}/{num_frames} - HOLD STILL...",
                            (15, 33),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )
                        cv2.imshow(window_name, flash_frame)
                        cv2.waitKey(400)
                print(f"Successfully captured {len(captured_frames)} burst frames!")
                break
            elif key in (ord('q'), ord('Q'), 27):  # 'q' or ESC
                print("Webcam capture cancelled by user.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return captured_frames if captured_frames else []


def capture_headless(camera_index: int = 0, num_frames: int = 3, warmup_frames: int = 15) -> list:
    """
    Non-interactive automatic capture without GUI window.
    Captures `num_frames` automatically in ~1.5 seconds.
    """
    backend = _get_backend()
    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {camera_index}")

    frames = []
    try:
        # Allow auto-exposure and auto-focus to settle
        for _ in range(warmup_frames):
            cap.read()
            time.sleep(0.03)

        for _ in range(num_frames):
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
            time.sleep(0.5)

        return frames
    finally:
        cap.release()


def capture_live_stream(camera_index: int = 0, process_fn=None, frame_interval: float = 1.0):
    """
    Continuous live webcam scraping mode.
    Runs OCR on webcam frames every `frame_interval` seconds and calls `process_fn(frame)`.
    Press 'q' in the window to stop live streaming.
    """
    backend = _get_backend()
    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {camera_index}")

    # Set camera resolution to HD 1280x720 with MJPG for sharp OCR readability
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    except Exception:
        pass

    window_name = "Live Webcam Data Extractor - [Q] Stop Stream"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    print("\nStarting Live Webcam OCR Stream...")
    print("Press 'q' or ESC in the preview window to exit live mode.\n")

    last_process_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            current_time = time.time()

            # Run OCR callback periodically on active frame
            if process_fn and (current_time - last_process_time >= frame_interval):
                process_fn(frame)
                last_process_time = current_time

            # Show live preview
            display_frame = frame.copy()
            cv2.putText(
                display_frame,
                "LIVE SCRAPING ACTIVE - Press 'q' to stop",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(20) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                print("Live webcam stream stopped.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
