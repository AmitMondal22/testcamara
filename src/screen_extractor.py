"""
screen_extractor.py
--------------------
Region-based value box detection and layout matching for structured LCD screens.
Includes automatic deskewing and perspective compensation before extraction.
"""

import cv2
import pytesseract
import numpy as np
import os
import json

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")

# Calibrated positions for Fresenius 4008S LCD layout
SCREEN_ROI = (430, 290, 1400, 990)

FIELD_POSITIONS = {
    "UF Volume":        (836, 128),
    "UF Time Left":     (830, 218),
    "UF Rate":          (824, 307),
    "UF Goal":          (819, 393),
    "Eff. Blood Flow":  (814, 479),
    "Cum. Blood Vol.":  (809, 563),
    "Kt/V":             (204, 418),
    "Plasma Na":        (496, 423),
    "Goal in":          (205, 516),
    "Clearance":        (493, 520),
}


def crop_screen(img: np.ndarray) -> np.ndarray:
    """Safely crop screen ROI based on image dimensions."""
    if img is None:
        return img
    h_img, w_img = img.shape[:2]
    x1, y1, x2, y2 = SCREEN_ROI
    if x2 <= w_img and y2 <= h_img:
        return img[y1:y2, x1:x2]
    return img


def detect_value_boxes(screen_img: np.ndarray):
    """Find dark rectangular value box regions on screen using contour detection."""
    gray = cv2.cvtColor(screen_img, cv2.COLOR_BGR2GRAY) if len(screen_img.shape) == 3 else screen_img
    _, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        aspect = w / float(h) if h else 0
        if area < 1000:
            continue
        if aspect < 1.1 or aspect > 7.0:
            continue
        if h < 15 or h > 150:
            continue
        boxes.append((x, y, w, h))
    return boxes


def extract_fields(img: np.ndarray, engine: str = "auto") -> dict:
    """
    Robust medical screen extraction pipeline:
    1. Unwarps perspective and deskews camera tilt.
    2. Enhances LCD contrast.
    3. Runs spatial label-to-value proximity matching.
    """
    from src.ocr_extract import extract_image_data
    from src.field_parser import parse_spatial_dialysis_fields

    lines_data = extract_image_data(img, engine=engine, unwarp=True)
    results = parse_spatial_dialysis_fields(lines_data)
    return results
