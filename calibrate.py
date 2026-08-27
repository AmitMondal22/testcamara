"""
calibrate.py
------------
If your camera angle/zoom is different from the sample photo, the
hardcoded SCREEN_ROI / FIELD_POSITIONS in src/screen_extractor.py
won't line up. This script helps you find the right numbers.

Usage:
    python calibrate.py path/to/new_photo.jpg

It saves two debug images to output/:
  - calibrate_boxes.png   : every detected value box outlined + numbered
  - calibrate_screen.png  : just the cropped screen region

Look at calibrate_boxes.png, note which numbered box is which field,
then update FIELD_POSITIONS in src/screen_extractor.py with each
box's (x, y) shown in the printed list below.
"""

import sys
import os
import cv2

from src.screen_extractor import crop_screen, detect_value_boxes, SCREEN_ROI

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def main():
    if len(sys.argv) < 2:
        print("Usage: python calibrate.py path/to/photo.jpg")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print("Could not read image.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Current SCREEN_ROI = {SCREEN_ROI}")
    print("If the screen crop below looks wrong, adjust SCREEN_ROI in")
    print("src/screen_extractor.py to (x1, y1, x2, y2) around your screen.\n")

    screen = crop_screen(img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "calibrate_screen.png"), screen)

    boxes = detect_value_boxes(screen)
    annotated = screen.copy()
    print(f"Detected {len(boxes)} value boxes:")
    for i, (x, y, w, h) in enumerate(sorted(boxes, key=lambda b: (b[1], b[0]))):
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated, str(i), (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        print(f"  box {i}: position=({x}, {y}) size={w}x{h}")

    out_path = os.path.join(OUTPUT_DIR, "calibrate_boxes.png")
    cv2.imwrite(out_path, annotated)
    print(f"\nSaved annotated boxes image to: {out_path}")
    print(f"Saved cropped screen to: {os.path.join(OUTPUT_DIR, 'calibrate_screen.png')}")


if __name__ == "__main__":
    main()
