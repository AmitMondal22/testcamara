"""
capture.py
----------
Raspberry Pi 4 Model B - IMX477 CSI Camera Capture Module.

Uses Picamera2 for camera access on Raspberry Pi OS.
Reference initialisation (as specified by hardware):
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (1280, 720), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    frame_rgb = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

Desktop OpenCV fallback is active ONLY when picamera2 is not installed
(e.g., on a development Windows PC for UI/logic testing).
"""

import cv2
import numpy as np
import time
import sys
import threading

_CAMERA_SINGLETONS: dict = {}
_CAMERA_LOCK = threading.Lock()
_IS_RASPBERRY_PI = sys.platform.startswith("linux")

# Try importing Picamera2 once at module load
try:
    from picamera2 import Picamera2  # type: ignore[missing-import]
    _PICAMERA2_AVAILABLE = True
    print("[capture.py] Picamera2 library found — Raspberry Pi CSI camera mode active.", flush=True)
except Exception:
    _PICAMERA2_AVAILABLE = False
    Picamera2 = None  # type: ignore[assignment,misc]
    if _IS_RASPBERRY_PI:
        print(
            "[capture.py] WARNING: Running on Linux but picamera2 is NOT installed!\n"
            "             Install with:  sudo apt install python3-picamera2\n"
            "             Falling back to OpenCV V4L2 as a last resort.",
            flush=True,
        )
    else:
        print(
            "[capture.py] picamera2 not found (non-Pi host). Using OpenCV fallback for development.",
            flush=True,
        )


class UnifiedCameraCapture:
    """
    Raspberry Pi IMX477 IR-Cut CSI camera wrapper.

    Priority:
      1. Picamera2  — Raspberry Pi OS hardware (create_preview_configuration RGB888 → RGB→BGR)
      2. OpenCV V4L2 / CAP_DSHOW — fallback for dev machines only

    IMX477 IR-Cut specifics handled:
      - AeEnable=True / AwbEnable=True so IR-cut filter transitions work correctly
      - 2 s warm-up for sensor AE/AWB and IR-cut filter to stabilise
      - Startup test-capture to verify frame is non-empty before marking camera ready
      - 4-channel XRGB array safe conversion (some Picamera2 versions return XRGB)

    All frames returned from .read() are BGR (OpenCV standard).
    """

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.picam2 = None
        self.cap = None
        self.is_picamera = False

        if _PICAMERA2_AVAILABLE and Picamera2 is not None:
            self._init_picamera2(camera_index, width, height)

        if not self.is_picamera:
            self._init_opencv_fallback(camera_index, width, height)

    def _init_picamera2(self, camera_index: int, width: int, height: int):
        """
        Initialise Picamera2 with create_preview_configuration (RGB888).
        Exact reference pattern:
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(main={"size": (W, H), "format": "RGB888"})
            picam2.configure(config)
            picam2.start()
            time.sleep(2)
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        """
        try:
            # ── Step 1: open camera ──────────────────────────────────────
            try:
                self.picam2 = Picamera2(camera_num=camera_index)
                print(f"[Camera] Opened Picamera2(camera_num={camera_index})", flush=True)
            except Exception as e_idx:
                print(f"[Camera] camera_num={camera_index} failed ({e_idx}), trying Picamera2()...", flush=True)
                self.picam2 = Picamera2()

            # ── Step 2 & 3: configure with AeEnable/AwbEnable for IR-Cut ─
            # AeEnable and AwbEnable ensure the IR-cut filter auto-switches
            # correctly between visible light and IR modes.
            try:
                config = self.picam2.create_preview_configuration(
                    main={"size": (width, height), "format": "RGB888"},
                    controls={
                        "AeEnable": True,   # Auto-exposure ON — essential for IR-cut
                        "AwbEnable": True,  # Auto white-balance ON
                        "FrameRate": 30.0,  # Target 30 FPS
                    }
                )
                self.picam2.configure(config)
                print(
                    f"[Camera] create_preview_configuration({width}x{height} RGB888) "
                    f"+ AeEnable=True, AwbEnable=True applied.",
                    flush=True,
                )
            except Exception as e_cfg:
                # Retry without controls dict (older Picamera2 API)
                print(f"[Camera] Config with controls failed ({e_cfg}), retrying without controls...", flush=True)
                try:
                    config = self.picam2.create_preview_configuration(
                        main={"size": (width, height), "format": "RGB888"}
                    )
                    self.picam2.configure(config)
                    print(f"[Camera] create_preview_configuration({width}x{height} RGB888) applied.", flush=True)
                except Exception as e_cfg2:
                    print(f"[Camera] preview_configuration failed ({e_cfg2}) — trying video_configuration...", flush=True)
                    config = self.picam2.create_video_configuration(
                        main={"size": (width, height), "format": "RGB888"}
                    )
                    self.picam2.configure(config)

            # ── Step 4 & 5: start + warm-up ──────────────────────────────
            self.picam2.start()
            print(
                "[Camera] Picamera2 started. Waiting 2 s for IMX477 IR-Cut sensor, "
                "AE and AWB to stabilise...",
                flush=True,
            )
            time.sleep(2.0)

            # ── Step 6: startup test-capture to verify camera is working ─
            test_raw = self.picam2.capture_array()
            if test_raw is None or test_raw.size == 0:
                raise RuntimeError("IMX477 capture_array() returned empty frame on startup verification.")
            # Convert: handle both 3-ch RGB and 4-ch XRGB
            if test_raw.ndim == 3 and test_raw.shape[2] == 4:
                test_bgr = cv2.cvtColor(test_raw, cv2.COLOR_RGBA2BGR)
            else:
                test_bgr = cv2.cvtColor(test_raw, cv2.COLOR_RGB2BGR)
            h_t, w_t = test_bgr.shape[:2]

            self.is_picamera = True
            print(f"[Camera] ✓ IMX477 IR-Cut camera verified: first frame {w_t}x{h_t} BGR received.", flush=True)

        except Exception as err:
            print(
                f"[Camera] ✗ Picamera2 / IMX477 IR-Cut init failed: {err}\n"
                "  ── Troubleshooting checklist ────────────────────────────\n"
                "  1. Ribbon cable: IMX477 CSI cable must be seated in CAM0 or CAM1 port\n"
                "     (blue tab on cable faces the ethernet port).\n"
                "  2. /boot/firmware/config.txt must contain:\n"
                "       camera_auto_detect=1\n"
                "     or for explicit IMX477 overlay:\n"
                "       dtoverlay=imx477\n"
                "  3. Check no other process holds the camera:\n"
                "       sudo pkill -f python3 ; sudo pkill -f rpicam\n"
                "  4. Test camera detection from terminal:\n"
                "       rpicam-hello --list-cameras\n"
                "       libcamera-hello --list-cameras\n"
                "  ────────────────────────────────────────────────────────",
                flush=True,
            )
            if self.picam2 is not None:
                try:
                    self.picam2.close()
                except Exception:
                    pass
            self.picam2 = None
            self.is_picamera = False

    def _init_opencv_fallback(self, camera_index: int, width: int, height: int):
        """OpenCV VideoCapture fallback — used ONLY on non-Pi development hosts."""
        if _IS_RASPBERRY_PI:
            backend = cv2.CAP_V4L2
        elif sys.platform.startswith("win"):
            backend = cv2.CAP_DSHOW
        else:
            backend = cv2.CAP_ANY

        self.cap = cv2.VideoCapture(camera_index, backend)
        if not (self.cap and self.cap.isOpened()):
            self.cap = cv2.VideoCapture(camera_index)

        if self.cap and self.cap.isOpened():
            try:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            except Exception:
                pass
            print(f"[Camera] OpenCV VideoCapture fallback opened (index #{camera_index}).", flush=True)
        else:
            print("[Camera] ✗ OpenCV VideoCapture could not open any camera device.", flush=True)
            self.cap = None

    def isOpened(self) -> bool:
        if self.is_picamera:
            return self.picam2 is not None
        return self.cap is not None and self.cap.isOpened()

    def read(self):
        """
        Returns (True, bgr_frame) on success, (False, None) on failure.
        Picamera2: capture_array() → RGB888 → cv2.COLOR_RGB2BGR → BGR frame.
        """
        if self.is_picamera and self.picam2 is not None:
            try:
                frame_raw = self.picam2.capture_array()
                if frame_raw is not None and frame_raw.size > 0:
                    # Handle 4-channel XRGB/RGBA format if returned by sensor
                    if frame_raw.shape[2] == 4:
                        frame_bgr = cv2.cvtColor(frame_raw, cv2.COLOR_RGBA2BGR)
                    else:
                        frame_bgr = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2BGR)
                    return True, frame_bgr
                return False, None
            except Exception as e:
                print(f"[Camera] capture_array error: {e}", flush=True)
                return False, None

        if self.cap is not None and self.cap.isOpened():
            return self.cap.read()

        return False, None

    def release(self):
        """Cleanly stop and release all camera resources."""
        if self.is_picamera and self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
                print("[Camera] Picamera2 stopped and closed.", flush=True)
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
    """
    Thread-safe singleton camera factory.
    Re-initialises automatically if the camera was closed, lost, or previously failed.
    Calling this from multiple threads is safe.
    """
    with _CAMERA_LOCK:
        cam = _CAMERA_SINGLETONS.get(camera_index)
        if cam is None or not cam.isOpened():
            # Force-release broken instance before re-creating
            if cam is not None:
                try:
                    cam.release()
                except Exception:
                    pass
            cam = UnifiedCameraCapture(camera_index, width, height)
            _CAMERA_SINGLETONS[camera_index] = cam
        return cam


def release_all_cameras():
    """Release every camera singleton (call on application shutdown)."""
    with _CAMERA_LOCK:
        for cam in _CAMERA_SINGLETONS.values():
            try:
                cam.release()
            except Exception:
                pass
        _CAMERA_SINGLETONS.clear()
    print("[Camera] All camera singletons released.", flush=True)


def capture_from_webcam(camera_index: int = 0, num_frames: int = 3) -> list:
    """
    Opens an OpenCV preview window showing the live IMX477 IR-Cut camera feed.
    Press SPACE or 'c' to capture a burst of frames for OCR processing.
    Press 'q' or ESC to exit without capturing.
    Returns list of BGR frames (empty if user quits).
    """
    cam = get_unified_camera(camera_index, width=1280, height=720)
    if not cam.isOpened():
        raise RuntimeError(
            "Could not open IMX477 IR-Cut Camera.\n"
            "• Check ribbon cable → CAM0/CAM1 CSI port (blue tab faces ethernet)\n"
            "• /boot/firmware/config.txt must have: camera_auto_detect=1\n"
            "• Run: rpicam-hello --list-cameras  to verify detection"
        )

    window_name = "IMX477 IR-Cut Camera  |  [SPACE/C] Capture  |  [Q/ESC] Quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    print("\n[Preview] IMX477 IR-Cut Camera preview active.")
    print("  → Press SPACE or 'c'  to CAPTURE frames for OCR.")
    print("  → Press 'q' or ESC    to quit without capturing.\n")

    captured_frames: list = []
    try:
        while True:
            ret, live_frame = cam.read()
            if not ret or live_frame is None:
                time.sleep(0.05)
                continue

            display = live_frame.copy()
            h, w = display.shape[:2]

            # Top HUD bar
            cv2.rectangle(display, (0, 0), (w, 44), (18, 18, 18), -1)
            cv2.putText(
                display,
                "Raspberry Pi IMX477 IR-Cut  |  SPACE/C = Capture  |  Q/ESC = Quit",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 220),
                2,
                cv2.LINE_AA,
            )
            # Bottom status bar
            cv2.rectangle(display, (0, h - 28), (w, h), (18, 18, 18), -1)
            cv2.putText(
                display,
                f"Resolution: {w}x{h}  |  BGR  |  Press SPACE to capture burst",
                (12, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (150, 210, 150),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, display)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("c"), ord("C"), 32):  # 32 = SPACE
                print(f"[Capture] Taking {num_frames}-frame burst from IMX477 IR-Cut...", flush=True)
                for i in range(num_frames):
                    ret_b, burst_frame = cam.read()
                    if ret_b and burst_frame is not None:
                        captured_frames.append(burst_frame)
                        print(f"[Capture]   Frame {i+1}/{num_frames} ✓", flush=True)
                    cv2.waitKey(200)
                print(f"[Capture] ✓ {len(captured_frames)} frame(s) captured from IMX477 IR-Cut.", flush=True)
                break
            elif key in (ord("q"), ord("Q"), 27):
                print("[Preview] Quit by user.", flush=True)
                break
    finally:
        cv2.destroyAllWindows()

    return captured_frames


def capture_headless(camera_index: int = 0, num_frames: int = 3, warmup_frames: int = 20) -> list:
    """
    Non-interactive headless frame capture — no display required.
    Suitable for SSH sessions on Raspberry Pi.

    warmup_frames=20 discards initial frames so AE/AWB and the
    IMX477 IR-cut filter can stabilise before data frames are taken.

    Returns list of BGR frames.
    """
    cam = get_unified_camera(camera_index, width=1280, height=720)
    if not cam.isOpened():
        raise RuntimeError(
            "Could not open IMX477 IR-Cut Camera (headless mode).\n"
            "Ensure camera_auto_detect=1 in /boot/firmware/config.txt and "
            "ribbon cable is seated correctly."
        )

    print(f"[Headless] Warming up IMX477 IR-Cut camera ({warmup_frames} discard frames)...", flush=True)
    for _ in range(warmup_frames):
        cam.read()
        time.sleep(0.03)

    frames: list = []
    print(f"[Headless] Capturing {num_frames} frame(s) from IMX477 IR-Cut...", flush=True)
    for i in range(num_frames):
        ret, frame = cam.read()
        if ret and frame is not None:
            frames.append(frame)
            h_f, w_f = frame.shape[:2]
            print(f"[Headless] Frame {i + 1}/{num_frames} captured ({w_f}x{h_f} BGR) ✓", flush=True)
        else:
            print(f"[Headless] Frame {i + 1}/{num_frames} ✗ — skipping.", flush=True)
        time.sleep(0.3)

    return frames


def capture_live_stream(camera_index: int = 0, process_fn=None, frame_interval: float = 1.0):
    """
    Continuous live stream from the IMX477 IR-Cut camera.

    Args:
        camera_index:   Camera device index (default 0).
        process_fn:     Optional callback(frame_bgr) fired every `frame_interval` seconds.
        frame_interval: Seconds between OCR / processing callbacks (default 1.0 s).

    Press 'q' or ESC in the preview window to stop.
    """
    cam = get_unified_camera(camera_index, width=1280, height=720)
    if not cam.isOpened():
        raise RuntimeError(
            "Could not open IMX477 IR-Cut Camera (live stream mode).\n"
            "Ensure camera_auto_detect=1 in /boot/firmware/config.txt."
        )

    window_name = "Raspberry Pi IMX477 IR-Cut  |  Live Stream  |  [Q/ESC] Stop"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    last_process_time = 0.0
    frame_count = 0
    fps_last_time = time.time()
    fps_display = 0.0

    print("[Stream] IMX477 IR-Cut live stream started. Press Q or ESC to stop.", flush=True)

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            frame_count += 1
            now = time.time()

            # Update FPS counter every second
            elapsed = now - fps_last_time
            if elapsed >= 1.0:
                fps_display = frame_count / elapsed
                frame_count = 0
                fps_last_time = now

            # Periodic OCR / processing callback
            if process_fn is not None and (now - last_process_time) >= frame_interval:
                process_fn(frame)
                last_process_time = now

            # HUD overlay
            display = frame.copy()
            h, w = display.shape[:2]
            cv2.rectangle(display, (0, h - 32), (w, h), (15, 15, 15), -1)
            cv2.putText(
                display,
                f"IMX477 IR-Cut  |  Live Stream  |  {fps_display:.1f} FPS  |  Q/ESC = Stop",
                (12, h - 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (120, 230, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, display)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("[Stream] Stopped by user.", flush=True)
                break
    finally:
        cv2.destroyAllWindows()
