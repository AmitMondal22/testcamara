"""
test_camera.py
--------------
Raspberry Pi 4 USB Webcam Diagnostic Tool.
Scans and tests all connected video devices (/dev/video* or camera indices).

Usage:
    python test_camera.py
"""

import sys
import cv2
from src.extractor import find_available_cameras, open_camera

print("=" * 55)
print("  RASPBERRY PI 4 — CAMERA DIAGNOSTIC TOOL")
print("=" * 55)

print("\n1. Scanning for connected camera devices...")
available = find_available_cameras(max_tested=10)

if not available:
    print("\n[ERROR] No working cameras detected.")
    print("\nRaspberry Pi Troubleshooting Steps:")
    print("  1. Check USB connection:      lsusb")
    print("  2. Check video devices:       ls -l /dev/video*")
    print("  3. Grant user permission:     sudo usermod -a -G video $USER")
    print("     (Log out and log back in for group change to take effect)")
    print("  4. Install V4L2 utility:      sudo apt install v4l-utils")
    print("  5. List V4L2 hardware:        v4l2-ctl --list-devices")
    sys.exit(1)

print(f"\n[SUCCESS] Found {len(available)} working camera device(s): Indices {available}")

for idx in available:
    try:
        cap = open_camera(idx, (1280, 720))
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            print(f"  - Camera Index {idx}: OK (Captured frame size {w}x{h})")
        cap.release()
    except Exception as e:
        print(f"  - Camera Index {idx}: Failed ({e})")

print("\nRecommended .env setting:")
print(f"CAMERA_INDEX={available[0]}")
print("=" * 55)
