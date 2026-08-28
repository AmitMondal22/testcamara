"""
screen_extractor.py
--------------------
Region-based value box detection and layout matching for structured LCD screens.
Includes safe boundary checks so generic images don't fail when cropped.
"""

import cv2
 # pyrefly: ignore [missing-import]
import pytesseract
import numpy as np

# Calibration constants for Fresenius 4008S sample screen layout
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

UNITS = {
    "UF Volume": "ml", "UF Time Left": "h:min", "UF Rate": "ml/h",
    "UF Goal": "ml", "Eff. Blood Flow": "ml/min", "Cum. Blood Vol.": "l",
    "Kt/V": "", "Plasma Na": "mmol/l", "Goal in": "h:min", "Clearance": "ml/min",
}


def crop_screen(img: np.ndarray) -> np.ndarray:
    """
    Safely crop the screen ROI based on image dimensions.
    Returns the original image if ROI bounds exceed image size.
    """
    h_img, w_img = img.shape[:2]
    x1, y1, x2, y2 = SCREEN_ROI

    # If image matches the calibrated size, crop to ROI
    if x2 <= w_img and y2 <= h_img:
        return img[y1:y2, x1:x2]

    # Otherwise return full image
    return img


def detect_value_boxes(screen_img: np.ndarray):
    """
    Find dark rectangular "value box" regions on screen using contour detection.
    """
    gray = cv2.cvtColor(screen_img, cv2.COLOR_BGR2GRAY)
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


def ocr_box_digits(gray_screen: np.ndarray, box) -> str:
    """OCR a single detected value box for digits."""
    x, y, w, h = box
    crop = gray_screen[y:y + h, x:x + w]
    if crop.size == 0:
        return ""
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(crop, 100, 255, cv2.THRESH_BINARY)
    try:
        text = pytesseract.image_to_string(
            thresh, config="--psm 7 -c tessedit_char_whitelist=0123456789:."
        ).strip()
        return text
    except Exception:
        return ""


def match_box_to_field(box, used_fields: set):
    """Find the closest known field position to a detected box."""
    bx, by = box[0], box[1]
    best_field, best_dist = None, float("inf")
    for field, (fx, fy) in FIELD_POSITIONS.items():
        if field in used_fields:
            continue
        dist = (bx - fx) ** 2 + (by - fy) ** 2
        if dist < best_dist:
            best_dist, best_field = dist, field
    return best_field


def _reformat(field: str, raw: str) -> str:
    if not raw:
        return raw
    digits = "".join(ch for ch in raw if ch.isdigit())

    if field in ("UF Time Left", "Goal in") and ":" not in raw and len(digits) in (3, 4):
        return f"{digits[:-2]}:{digits[-2:]}"

    if field in ("Kt/V",) and "." not in raw and len(digits) == 3:
        return f"{digits[0]}.{digits[1:]}"

    if field in ("Cum. Blood Vol.",) and "." not in raw and len(digits) >= 3:
        return f"{digits[:-1]}.{digits[-1]}"

    return raw


def extract_fields(img: np.ndarray, engine: str = "auto") -> dict:
    """
    Robust medical screen extraction pipeline:
    1. Unwarps perspective if webcam is angled/tilted.
    2. Enhances LCD contrast.
    3. Runs spatial label-to-value proximity matching.
    """
    from src.ocr_extract import extract_image_data
    from src.field_parser import parse_spatial_dialysis_fields

    lines_data = extract_image_data(img, engine=engine, unwarp=True)
    results = parse_spatial_dialysis_fields(lines_data)
    return results
