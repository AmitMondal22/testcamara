"""
capture.py
----------
Captures live frames or continuous video streams from Raspberry Pi IMX477 camera (Picamera2)
with fallback for desktop testing.
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
    Dedicated Raspberry Pi Camera wrapper powered by Picamera2.
    Configures preview configuration in RGB888 and converts to OpenCV BGR.
    Includes OpenCV / synthetic fallbacks for local desktop development.
    """

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.picam2 = None
        self.cap = None
        self.is_picamera = False

        # Primary: Picamera2 for Raspberry Pi (IMX477 CSI Camera)
        try:
            # pyrefly: ignore [missing-import]
            from picamera2 import Picamera2
            try:
                self.picam2 = Picamera2(camera_num=camera_index)
            except Exception:
                self.picam2 = Picamera2()

            # Create preview configuration matching standard Raspberry Pi 4 setup
            try:
                config = self.picam2.create_preview_configuration(
                    main={"size": (width, height), "format": "RGB888"}
                )
                self.picam2.configure(config)
            except Exception:
                config = self.picam2.create_video_configuration(
                    main={"size": (width, height), "format": "RGB888"}
                )
                self.picam2.configure(config)

            self.picam2.start()
            time.sleep(2.0)  # Allow sensor & auto-exposure to warm up
            self.is_picamera = True
            print(f"[Camera] Raspberry Pi Picamera2 initialized successfully (Index #{camera_index})", flush=True)
        except Exception as err:
            print(f"[Camera Note] Picamera2 unavailable ({err}). Using OpenCV fallback for testing...", flush=True)
            self.picam2 = None
            self.is_picamera = False

        # Fallback: OpenCV VideoCapture for local desktop testing
        if not self.is_picamera:
            backend = _get_backend()
            self.cap = cv2.VideoCapture(camera_index, backend)
            if not self.cap or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(camera_index)

            if self.cap and self.cap.isOpened():
                try:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                except Exception:
                    pass
                print(f"[Camera] Initialized via OpenCV VideoCapture fallback (#{camera_index})", flush=True)

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
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    return True, frame_bgr
            except Exception:
                return False, None
        elif self.cap is not None and self.cap.isOpened():
            return self.cap.read()
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
    """Thread-safe singleton getter to manage camera hardware access."""
    global _CAMERA_SINGLETONS
    with _CAMERA_LOCK:
        cam = _CAMERA_SINGLETONS.get(camera_index)
        if cam is None or not cam.isOpened():
            cam = UnifiedCameraCapture(camera_index, width, height)
            _CAMERA_SINGLETONS[camera_index] = cam
        return cam


def _get_backend():
    """Return platform-appropriate OpenCV VideoCapture backend."""
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    elif sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


def capture_from_webcam(camera_index: int = 0, num_frames: int = 3) -> list:
    """
    Opens interactive preview window using Raspberry Pi Camera.
    When SPACE or 'c' is pressed, captures frames for instant OCR.
    """
    cam = get_unified_camera(camera_index, width=1280, height=720)
    if not cam.isOpened():
        raise RuntimeError("Could not open Raspberry Pi Camera. Verify CSI ribbon cable connection.")

    window_name = "Raspberry Pi Camera - [SPACE/C] Capture | [Q] Quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    print("\nRaspberry Pi Camera preview active.")
    print(" -> Press 'SPACE' or 'c' to CAPTURE burst frames for OCR.")
    print(" -> Press 'q' or 'ESC' to exit.\n")

    captured_frames = []
    try:
        while True:
            ret, live_frame = cam.read()
            if not ret or live_frame is None:
                time.sleep(0.1)
                continue

            display_frame = live_frame.copy()
            h, w = display_frame.shape[:2]

            cv2.rectangle(display_frame, (0, 0), (w, 40), (20, 20, 20), -1)
            cv2.putText(
                display_frame,
                "Raspberry Pi IMX477 Camera | Press 'SPACE' to CAPTURE | 'q' to QUIT",
                (15, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord('c'), ord('C'), 32):
                for idx in range(num_frames):
                    ret_b, burst_frame = cam.read()
                    if ret_b and burst_frame is not None:
                        captured_frames.append(burst_frame)
                        cv2.waitKey(200)
                print(f"Captured {len(captured_frames)} frame(s)!")
                break
            elif key in (ord('q'), ord('Q'), 27):
                break
    finally:
        cv2.destroyAllWindows()

    return captured_frames if captured_frames else []


def capture_headless(camera_index: int = 0, num_frames: int = 3, warmup_frames: int = 15) -> list:
    """Non-interactive headless frame capture for background tasks."""
    cam = get_unified_camera(camera_index, width=1280, height=720)
    if not cam.isOpened():
        raise RuntimeError("Could not open Raspberry Pi Camera.")

    frames = []
    for _ in range(warmup_frames):
        cam.read()
        time.sleep(0.03)

    for _ in range(num_frames):
        ret, frame = cam.read()
        if ret and frame is not None:
            frames.append(frame)
        time.sleep(0.3)

    return frames


def capture_live_stream(camera_index: int = 0, process_fn=None, frame_interval: float = 1.0):
    """Continuous live streaming and periodic OCR processing."""
    cam = get_unified_camera(camera_index, width=1280, height=720)
    if not cam.isOpened():
        raise RuntimeError("Could not open Raspberry Pi Camera.")

    window_name = "Raspberry Pi Camera Stream - [Q] Exit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    last_process_time = 0.0
    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            current_time = time.time()
            if process_fn and (current_time - last_process_time >= frame_interval):
                process_fn(frame)
                last_process_time = current_time

            display_frame = frame.copy()
            cv2.putText(
                display_frame,
                "Raspberry Pi Camera Live - Press 'q' to stop",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(20) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
    finally:
        cv2.destroyAllWindows()
