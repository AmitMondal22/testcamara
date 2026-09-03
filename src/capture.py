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


import threading

_CAMERA_SINGLETONS = {}
_CAMERA_LOCK = threading.Lock()


class UnifiedCameraCapture:
    """
    Unified camera wrapper that automatically tries Picamera2 (Raspberry Pi IMX477 IR Camera)
    and seamlessly falls back to OpenCV VideoCapture (Windows / USB Webcams / GStreamer).
    """

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.picam2 = None
        self.cap = None
        self.is_picamera = False

        # 1. Try Picamera2 (Primary hardware stack for Raspberry Pi 4 + IMX477 Camera)
        if not sys.platform.startswith("win"):
            try:
                # pyrefly: ignore [missing-import]
                from picamera2 import Picamera2
                try:
                    self.picam2 = Picamera2(camera_num=camera_index)
                except Exception:
                    self.picam2 = Picamera2()

                # Configure preview configuration (1280x720 RGB888 for Raspberry Pi IMX477)
                try:
                    config = self.picam2.create_preview_configuration(
                        main={"size": (width, height), "format": "RGB888"}
                    )
                    self.picam2.configure(config)
                except Exception:
                    try:
                        config = self.picam2.create_video_configuration(
                            main={"size": (width, height), "format": "RGB888"}
                        )
                        self.picam2.configure(config)
                    except Exception:
                        config = self.picam2.create_still_configuration(
                            main={"size": (width, height), "format": "RGB888"}
                        )
                        self.picam2.configure(config)

                self.picam2.start()
                time.sleep(1.0)
                self.is_picamera = True
                print(f"[Camera] Initialized via Picamera2 (Raspberry Pi IMX477 #{camera_index})", flush=True)
            except Exception as err:
                print(f"[Camera Note] Picamera2 init note ({err}). Trying OpenCV fallback...", flush=True)
                self.picam2 = None
                self.is_picamera = False

        # 2. Fallback to OpenCV VideoCapture (GStreamer / V4L2)
        if not self.is_picamera:
            if sys.platform.startswith("linux"):
                try:
                    gst_pipeline = f"libcamerasrc ! video/x-raw, width={width}, height={height} ! videoconvert ! appsink"
                    self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
                except Exception:
                    self.cap = None

            if self.cap is None or not self.cap.isOpened():
                backend = _get_backend()
                self.cap = cv2.VideoCapture(camera_index, backend)

            if not self.cap or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(camera_index)

            if self.cap and self.cap.isOpened():
                try:
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                except Exception:
                    pass
                print(f"[Camera] Initialized via OpenCV VideoCapture (#{camera_index})", flush=True)

    def isOpened(self) -> bool:
        if self.is_picamera:
            return self.picam2 is not None
        return self.cap is not None and self.cap.isOpened()

    def read(self):
        """Returns tuple (ret: bool, frame: np.ndarray in BGR format)."""
        if self.is_picamera and self.picam2 is not None:
            try:
                frame_rgb = self.picam2.capture_array()
                if frame_rgb is not None and frame_rgb.size > 0:
                    # Make explicit heap copy to avoid DMA buffer recycling SIGBUS
                    frame_bgr = cv2.cvtColor(np.ascontiguousarray(frame_rgb.copy()), cv2.COLOR_RGB2BGR)
                    return True, frame_bgr
            except Exception:
                return False, None
        elif self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return True, np.ascontiguousarray(frame.copy())
            return ret, frame
        return False, None

    def release(self):
        if self.is_picamera and self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                pass
            self.picam2 = None
            self.is_picamera = False

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None


def get_unified_camera(camera_index: int = 0, width: int = 1280, height: int = 720) -> UnifiedCameraCapture:
    """Thread-safe singleton getter to prevent multiple Picamera2 instances from locking camera hardware on Raspberry Pi."""
    global _CAMERA_SINGLETONS
    with _CAMERA_LOCK:
        cam = _CAMERA_SINGLETONS.get(camera_index)
        if cam is None or not cam.isOpened():
            cam = UnifiedCameraCapture(camera_index, width, height)
            _CAMERA_SINGLETONS[camera_index] = cam
        return cam


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
    for idx in range(max_tested):
        try:
            cam = get_unified_camera(idx, width=1280, height=720)
            if cam.isOpened():
                ret, frame = cam.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    cam_type = "Picamera2 IMX477" if cam.is_picamera else "USB/Built-in Webcam"
                    label = f"Camera #{idx} ({cam_type} {w}x{h})"
                    cameras.append({
                        "index": idx,
                        "id": str(idx),
                        "label": label,
                        "name": f"Camera #{idx} ({cam_type})",
                        "resolution": f"{w}x{h}",
                        "rtsp_url": str(idx)
                    })
        except Exception:
            pass
    return cameras


def find_available_cameras(max_tested: int = 5) -> list:
    """Detect available camera indices connected to system."""
    cams = discover_camera_details(max_tested)
    return [c["index"] for c in cams] if cams else [0]


def capture_from_webcam(camera_index: int = 0, num_frames: int = 1) -> list:
    """
    Opens an interactive webcam GUI window using UnifiedCameraCapture.
    When SPACE/c is pressed, captures HD frames for instant high-speed OCR.
    """
    cam = UnifiedCameraCapture(camera_index, width=1920, height=1080)
    if not cam.isOpened():
        raise RuntimeError(f"Could not open camera at index {camera_index}. Check hardware connection.")

    window_name = "Webcam Extractor - [SPACE/C] Capture 3-Frame Burst | [Q] Quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    print("\nCamera preview active.")
    print(" -> Press 'SPACE' or 'c' to CAPTURE a 3-frame burst for maximum accuracy.")
    print(" -> Press 'q' or 'ESC' to cancel.\n")

    captured_frames = []
    try:
        while True:
            ret, live_frame = cam.read()
            if not ret or live_frame is None:
                print("Warning: Failed to read frame from camera.")
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
                    ret_b, burst_frame = cam.read()
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
                print("Camera capture cancelled by user.")
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

    return captured_frames if captured_frames else []


def capture_headless(camera_index: int = 0, num_frames: int = 3, warmup_frames: int = 15) -> list:
    """
    Non-interactive automatic capture without GUI window.
    Captures `num_frames` automatically in ~1.5 seconds.
    """
    cam = UnifiedCameraCapture(camera_index, width=1280, height=720)
    if not cam.isOpened():
        raise RuntimeError(f"Could not open camera at index {camera_index}")

    frames = []
    try:
        # Allow auto-exposure and auto-focus to settle
        for _ in range(warmup_frames):
            cam.read()
            time.sleep(0.03)

        for _ in range(num_frames):
            ret, frame = cam.read()
            if ret and frame is not None:
                frames.append(frame)
            time.sleep(0.5)

        return frames
    finally:
        cam.release()


def capture_live_stream(camera_index: int = 0, process_fn=None, frame_interval: float = 1.0):
    """
    Continuous live camera scraping mode using UnifiedCameraCapture.
    Runs OCR on camera frames every `frame_interval` seconds and calls `process_fn(frame)`.
    Press 'q' in the window to stop live streaming.
    """
    cam = UnifiedCameraCapture(camera_index, width=1280, height=720)
    if not cam.isOpened():
        raise RuntimeError(f"Could not open camera at index {camera_index}")

    window_name = "Live Camera Data Extractor - [Q] Stop Stream"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    print("\nStarting Live Camera OCR Stream...")
    print("Press 'q' or ESC in the preview window to exit live mode.\n")

    last_process_time = 0.0

    try:
        while True:
            ret, frame = cam.read()
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
                print("Live camera stream stopped.")
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
