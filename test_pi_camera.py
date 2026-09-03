"""
test_pi_camera.py
------------------
Hardware & Software Diagnostic Tool for Raspberry Pi 4 Model B (IMX477 IR Camera).
Tests Picamera2, GStreamer, V4L2 device nodes, and saves a test image capture.

Usage on Raspberry Pi:
    python test_pi_camera.py
"""

import sys
import os
import time
import cv2

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print(" RASPBERRY PI 4 (IMX477 CAMERA) HARDWARE DIAGNOSTIC TOOL ")
print("=" * 65)

# 1. System Platform Info
print(f"[1/4] Platform: {sys.platform}")
print(f"[1/4] Python  : {sys.version.split()[0]}")
print(f"[1/4] OpenCV  : {cv2.__version__}")

# 2. Check Picamera2 Module
picam2_ok = False
if sys.platform.startswith("win"):
    print("[2/4] Note: 'picamera2' is a Raspberry Pi OS (Linux) specific library for IMX477/CSI cameras.")
    print("      Running on Windows — Picamera2 hardware test skipped. Proceeding to OpenCV webcam test...\n")
else:
    try:
        # pyrefly: ignore [missing-import]
        # type: ignore
        from picamera2 import Picamera2
        print("[2/4] Picamera2 python library is INSTALLED.")
        
        try:
            print(" -> Attempting to open Picamera2(camera_num=0)...")
            picam2 = Picamera2(camera_num=0)
        except Exception:
            picam2 = Picamera2()
        
        # Configure headlessly using create_video_configuration
        try:
            config = picam2.create_video_configuration(main={"size": (1280, 720), "format": "RGB888"})
            picam2.configure(config)
            print(" -> Configured with video_configuration (1280x720 RGB888).")
        except Exception as e_cfg:
            config = picam2.create_preview_configuration(main={"size": (1280, 720), "format": "RGB888"})
            picam2.configure(config)
            print(" -> Configured with preview_configuration.")

        picam2.start()
        print(" -> Picamera2 STARTED successfully! Warming up sensor...")
        time.sleep(1.5)

        frame_rgb = picam2.capture_array()
        if frame_rgb is not None and frame_rgb.size > 0:
            import numpy as np
            frame_bgr = cv2.cvtColor(np.ascontiguousarray(frame_rgb.copy()), cv2.COLOR_RGB2BGR)
            save_path = os.path.join(OUTPUT_DIR, "test_picamera2_capture.png")
            cv2.imwrite(save_path, frame_bgr)
            h, w = frame_bgr.shape[:2]
            print(f" -> SUCCESS! Captured {w}x{h} frame via Picamera2.")
            print(f" -> Saved test snapshot to: {save_path}")
            picam2_ok = True
        else:
            print(" -> WARNING: Picamera2 frame array was empty.")

        picam2.stop()
        picam2.close()
        print(" -> Picamera2 closed cleanly.\n")

    except Exception as err:
        print(f"[2/4] Picamera2 Error: {err}\n")

# 3. Check OpenCV GStreamer / V4L2 fallback
if not picam2_ok:
    print("[3/4] Testing OpenCV GStreamer & V4L2 Fallbacks...")
    pipelines = [
        ("GStreamer libcamerasrc", "libcamerasrc ! video/x-raw, width=1280, height=720 ! videoconvert ! appsink"),
        ("OpenCV V4L2 Device 0", 0),
        ("OpenCV V4L2 Device 1", 1),
    ]

    cv_ok = False
    for label, src in pipelines:
        print(f" -> Testing {label}...")
        try:
            if isinstance(src, str):
                cap = cv2.VideoCapture(src, cv2.CAP_GSTREAMER)
            else:
                backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
                cap = cv2.VideoCapture(src, backend)

            if cap and cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    h, w = frame.shape[:2]
                    save_path = os.path.join(OUTPUT_DIR, "test_opencv_capture.png")
                    cv2.imwrite(save_path, frame)
                    print(f" -> SUCCESS! {label} captured {w}x{h} frame.")
                    print(f" -> Saved test snapshot to: {save_path}")
                    cv_ok = True
                    cap.release()
                    break
                cap.release()
        except Exception as e:
            print(f" -> {label} failed: {e}")

# 4. Summary & Advice
print("\n" + "=" * 65)
print(" DIAGNOSTIC SUMMARY ")
print("=" * 65)
if picam2_ok or cv_ok:
    print(" [OK] CAMERA HARDWARE IS OPERATIONAL & ACCESSIBLE!")
    print("      You can now run 'python app.py' to launch the web application server.")
else:
    print(" [FAILED] NO CAMERA CAPTURE WORKED.")
    print("          Please check the following Raspberry Pi hardware & OS settings:")
    print("          1. Verify ribbon cable connection: Ensure the CSI cable from IMX477 is")
    print("             securely attached to the 'CAM/DISP' port with blue tab facing ethernet port.")
    print("          2. Check Raspberry Pi OS Camera Overlay in /boot/firmware/config.txt:")
    print("             Ensure the file contains:")
    print("                 camera_auto_detect=1")
    print("             or:")
    print("                 dtoverlay=imx477")
    print("          3. Test system camera detection via terminal:")
    print("             Run:  rpicam-hello --list-cameras  or  libcamera-hello --list-cameras")
    print("          4. If another process or previous python script is still running in background,")
    print("             kill existing camera processes with:")
    print("                 sudo pkill -f python3; sudo pkill -f rpicam")
print("=" * 65 + "\n")
