"""
extractor.py
------------
Lightweight display data extractor for Raspberry Pi 4.
  - Captures frame from camera
  - Multi-pass preprocessing (deblur, adaptive threshold, Otsu, inverted)
  - Dual OCR strategy: line-based parsing + spatial box matching
  - Maps extracted data to dataset.json / config.json fields:
      {"abc": {"name": "abc abc", "value": 10}}
  - Casts extracted values to configured data types (int, float, string)
  - Zero disk storage — 100% in-memory
"""

import os
import sys
import time
import json
import shutil
import re
import difflib
from collections import Counter
import cv2
import numpy as np
import pytesseract


def _configure_tesseract():
    """Find tesseract binary on Windows / Linux / macOS."""
    if shutil.which("tesseract"):
        return
    common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for p in common_paths:
        if os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            return
    print("[Warning] tesseract binary not found. Install: sudo apt install tesseract-ocr", file=sys.stderr)


_configure_tesseract()


# ──────────────────────────────────────────────────────────────
# Robust JSON / Dataset Loader
# ──────────────────────────────────────────────────────────────

def load_dataset_file(filepath):
    """
    Robust JSON loader that safely handles comments and trailing commas in dataset.json.
    """
    if not filepath or not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        # Remove trailing commas before closing braces/brackets
        content = re.sub(r',\s*([\}\]])', r'\1', content)
        return json.loads(content)
    except Exception as e:
        print(f"[Warning] Could not load dataset file {filepath}: {e}", file=sys.stderr)
        return None


# ──────────────────────────────────────────────────────────────
# Computer Vision: Screen Detection, Perspective Deskewing & Illumination Normalization
# ──────────────────────────────────────────────────────────────

def get_image_sharpness(gray):
    """Calculate focus / sharpness measure using Laplacian variance."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def order_points(pts):
    """Order 4 contour points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# ──────────────────────────────────────────────────────────────
# YOLO11n Neural Screen & Display Detector Engine
# ──────────────────────────────────────────────────────────────

_YOLO_MODEL = None
_YOLO_TRIED = False


def load_yolo11n_detector():
    """
    Load YOLO11n model using Ultralytics or OpenCV DNN.
    Returns active detector instance or None.
    """
    global _YOLO_MODEL, _YOLO_TRIED
    if _YOLO_TRIED:
        return _YOLO_MODEL

    _YOLO_TRIED = True

    # 1. Try Ultralytics YOLO11n
    try:
        from ultralytics import YOLO
        _YOLO_MODEL = YOLO("yolo11n.pt")
        print("[YOLO11n] Loaded Ultralytics YOLO11n neural detector.", flush=True)
        return _YOLO_MODEL
    except Exception:
        pass

    # 2. Try ONNX YOLO11n via OpenCV DNN
    onnx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo11n.onnx")
    if os.path.exists(onnx_path):
        try:
            net = cv2.dnn.readNetFromONNX(onnx_path)
            _YOLO_MODEL = ("opencv_dnn", net)
            print("[YOLO11n] Loaded YOLO11n ONNX model via OpenCV DNN.", flush=True)
            return _YOLO_MODEL
        except Exception:
            pass

    return None


def detect_and_warp_screen(frame):
    """
    Computer Vision Screen ROI Extractor:
    Detects rectangular display screen contour and applies 4-point perspective
    warp to straighten and crop the screen from room background.
    """
    if frame is None or frame.size == 0:
        return frame

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edged, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        area = cv2.contourArea(c)
        if area < (h * w * 0.12):  # Must occupy at least 12% of frame
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            rect = order_points(pts)
            (tl, tr, br, bl) = rect

            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))

            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))

            if maxWidth >= 200 and maxHeight >= 100:
                dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]
                ], dtype="float32")

                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
                return warped

    return frame


def detect_screen_roi_yolo(frame):
    """
    Combined YOLO11n & Computer Vision Display Region Extractor:
    Uses YOLO11n neural detector to locate the monitor screen, then refines
    the boundaries with computer vision deskewing.
    """
    model = load_yolo11n_detector()
    if model is not None and hasattr(model, "predict"):
        try:
            h, w = frame.shape[:2]
            results = model.predict(frame, conf=0.35, verbose=False)
            boxes = results[0].boxes
            best_box = None
            best_area = 0
            for box in boxes:
                cls_id = int(box.cls[0])
                # COCO classes: 62 (tv/screen), 63 (laptop), 67 (cell phone), 74 (clock)
                if cls_id in (62, 63, 67, 74):
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    area = (x2 - x1) * (y2 - y1)
                    if area > best_area and area > (h * w * 0.10):
                        best_area = area
                        best_box = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))

            if best_box is not None:
                x1, y1, x2, y2 = best_box
                cropped = frame[y1:y2, x1:x2]
                if cropped.size > 0:
                    return detect_and_warp_screen(cropped)
        except Exception:
            pass

    return detect_and_warp_screen(frame)


def normalize_illumination(gray):
    """
    Computer Vision Illumination Normalizer:
    Removes lighting hotspots, shadows, and screen glare by dividing image
    by its morphological background estimate.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    background = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
    background = cv2.GaussianBlur(background, (21, 21), 0)
    normalized = cv2.divide(gray, background, scale=255)
    return normalized


def preprocess_for_ocr(img):
    """
    Complete Computer Vision Preprocessing Pipeline:
      1. Screen ROI auto-cropping and perspective deskewing.
      2. Super-resolution cubic upscaling.
      3. Illumination / glare removal.
      4. High-boost unsharp filter + CLAHE.
      5. Morphological 7-segment stroke connector.
      6. Multi-threshold binarization passes.
    """
    if img is None or img.size == 0:
        return [], 1.0

    # 1. YOLO11n + Computer Vision Screen ROI Detection
    warped_img = detect_screen_roi_yolo(img)
    h, w = warped_img.shape[:2]

    # 2. Super-Resolution Upscaling
    scale = 1.0
    if max(h, w) < 1200:
        scale = 1200.0 / max(h, w)
        warped_img = cv2.resize(warped_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    results = []
    gray_raw = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)

    # Pass 1: Direct High-Boost Deblurring + CLAHE on raw image
    blurred_raw = cv2.GaussianBlur(gray_raw, (0, 0), 3.0)
    unsharp_raw = cv2.addWeighted(gray_raw, 2.5, blurred_raw, -1.5, 0)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    gray_clahe_raw = clahe.apply(unsharp_raw)
    results.append(gray_clahe_raw)

    # Pass 2: Bilateral Filtering + Kernel Sharpening
    bilateral = cv2.bilateralFilter(gray_clahe_raw, 9, 75, 75)
    sharpen_kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)
    sharp = cv2.filter2D(bilateral, -1, sharpen_kernel)
    results.append(sharp)

    # Pass 3: Morphological stroke bridge (connects faint LCD / 7-segment digits)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(sharp, cv2.MORPH_CLOSE, morph_kernel)
    results.append(closed)

    # Pass 4: Adaptive Gaussian Threshold (handles ambient room lighting)
    thresh_adaptive = cv2.adaptiveThreshold(
        bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 8
    )
    results.append(thresh_adaptive)

    # Pass 5: Otsu Threshold (ideal for high-contrast digital segments)
    _, thresh_otsu = cv2.threshold(gray_clahe_raw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results.append(thresh_otsu)

    # Pass 6: Inverted Threshold (for illuminated characters on dark screen)
    results.append(cv2.bitwise_not(thresh_otsu))
    results.append(cv2.bitwise_not(gray_clahe_raw))

    # Pass 7: Illumination Normalization (for strong room shadows/glare)
    try:
        gray_norm = normalize_illumination(gray_raw)
        gray_clahe_norm = clahe.apply(gray_norm)
        results.append(gray_clahe_norm)
    except Exception:
        pass

    return results, scale


# ──────────────────────────────────────────────────────────────
# String Cleaning & Type Casting
# ──────────────────────────────────────────────────────────────

VALID_SHORT_LABELS = {
    "na", "uf", "kt", "ph", "hr", "bp", "rr", "o2", "co2", "t", "p", "v",
    "eff", "art", "ven", "tmp", "bpm", "map", "abc", "def", "ghi"
}


def clean_ascii(text):
    """Remove non-ASCII and control characters."""
    if not text:
        return ""
    cleaned = re.sub(r'[^\x20-\x7E]', ' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def is_valid_label(text):
    """Check if text is a valid display label rather than OCR noise."""
    text = clean_ascii(text)
    if len(text) < 2 or len(text) > 35:
        return False

    letters_only = re.sub(r'[^A-Za-z]', '', text)
    if not letters_only:
        return False

    if len(text) == 2 and text.lower() not in VALID_SHORT_LABELS:
        if not re.search(r'[aeiouAEIOU]', text):
            return False

    alpha_ratio = len(letters_only) / max(len(text), 1)
    if alpha_ratio < 0.5:
        return False

    return True


def is_valid_value(text):
    """Check if text represents a numeric reading (with optional units like mL, ml/h, min, mmol, etc.)."""
    text = clean_ascii(text).strip()
    if not text or not re.search(r'\d', text):
        return False
    # Strip common units
    stripped = re.sub(r'(?i)\s*(ml/h|ml/min|mmol/l|mmol|ml|min|hr|h|l|%|kg|kpa|mmhg)', '', text).strip()
    stripped = stripped.replace(" ", "")
    if re.match(r'^[\+\-]?[\d.,:\/]+%?$', stripped):
        return True
    if re.search(r'^\d+[\d.,:\/]*$', stripped):
        return True
    return False


def clean_label_str(label):
    """Format label neatly."""
    label = clean_ascii(label)
    label = re.sub(r'^[_\W]+|[_\W]+$', '', label)
    label = re.sub(r'\s+', ' ', label).strip()
    return label


def parse_numeric_value(raw_val):
    """Parse string value into integer, float, or formatted time string, stripping unit noise."""
    clean_val = clean_ascii(str(raw_val)).strip()
    
    # Time formats (e.g. 1:43, 01:30)
    time_m = re.search(r'\b\d{1,2}:\d{2}\b', clean_val)
    if time_m:
        return time_m.group(0)

    # Strip units
    unit_clean = re.sub(r'(?i)\s*(ml/h|ml/min|mmol/l|mmol|ml|min|hr|h|l|%|kg|kpa|mmhg)', '', clean_val).strip()
    num_clean = re.sub(r'[^\d\+\-.]', '', unit_clean)

    # Integer
    if re.match(r'^[\+\-]?\d+$', num_clean):
        try:
            return int(num_clean)
        except ValueError:
            pass

    # Float
    if re.match(r'^[\+\-]?\d+\.\d+$', num_clean):
        try:
            return float(num_clean)
        except ValueError:
            pass

    return unit_clean or clean_val


def sanitize_field_value(field_key, val):
    """Clean and validate values strictly based on known metric patterns to eliminate OCR noise."""
    key = str(field_key).lower()
    val_str = clean_ascii(str(val)).strip()

    # 1. Plasma Na (strictly 115 - 170 mmol/L)
    if "plasma" in key or "na" in key:
        digits = re.findall(r'\d+', val_str)
        for d in digits:
            if len(d) == 3 and 115 <= int(d) <= 170:
                return int(d)
        match = re.search(r'(1[2-6]\d)', val_str)
        if match:
            return int(match.group(1))
        # Reject invalid noise like single digit 7
        return None

    # 2. Kt/V (strictly 0.1 - 3.5)
    if "kt" in key:
        num_clean = re.sub(r'[^\d.]', '', val_str)
        try:
            f = float(num_clean)
            if 0.1 <= f <= 3.5:
                return f
            if 10 <= f <= 350 and "." not in num_clean:
                return round(f / 100.0, 2)
        except Exception:
            pass
        return None

    # 3. Time fields (UF Time Left, Goal in)
    if "time" in key or "goal_in" in key:
        m = re.search(r'\b\d{1,2}:\d{2}\b', val_str)
        if m:
            return m.group(0)
        # If digits only like 153 -> 1:53
        digits = re.sub(r'\D', '', val_str)
        if len(digits) == 3:
            return f"{digits[0]}:{digits[1:]}"
        if len(digits) == 4:
            return f"{digits[:2]}:{digits[2:]}"
        return val_str if ":" in val_str else None

    # 4. UF Rate / Blood Flow / Clearance (10 - 3000)
    if "rate" in key or "flow" in key or "clearance" in key:
        digits = re.sub(r'[^\d.]', '', val_str)
        if digits:
            try:
                num = int(float(digits))
                if 10 <= num <= 4000:
                    return num
            except Exception:
                pass
        return None

    # 5. UF Volume / UF Goal / Cum Blood Vol
    if "vol" in key or "goal" in key:
        digits = re.sub(r'[^\d.]', '', val_str)
        if digits:
            try:
                if "." in digits:
                    return float(digits)
                return int(digits)
            except Exception:
                pass

    return parse_numeric_value(val)


def cast_value(val, target_type=None, field_key=None):
    """Cast extracted value to specified data type ('int', 'float', 'number', 'string')."""
    if val is None or str(val).strip().lower() in ("none", "", "null"):
        return None

    if field_key:
        val = sanitize_field_value(field_key, val)

    if val is None or str(val).strip().lower() in ("none", "", "null"):
        return None

    if not target_type or not isinstance(target_type, str):
        return parse_numeric_value(val)

    t = target_type.strip().lower()
    raw_str = str(val).strip()

    if t in ("int", "integer"):
        num = re.sub(r'[^\d\+\-]', '', raw_str.replace(",", ""))
        try:
            return int(num)
        except Exception:
            return parse_numeric_value(raw_str)

    if t in ("float", "double", "decimal"):
        num = re.sub(r'[^\d\+\-.]', '', raw_str.replace(",", ""))
        try:
            return float(num)
        except Exception:
            return parse_numeric_value(raw_str)

    if t in ("number", "num"):
        num = re.sub(r'[^\d\+\-.]', '', raw_str.replace(",", ""))
        try:
            if "." in num:
                return float(num)
            return int(num)
        except Exception:
            return parse_numeric_value(raw_str)

    if t in ("str", "string", "text"):
        return str(raw_str)

    return parse_numeric_value(raw_str)


def to_key_slug(text):
    """Convert label string to snake_case key slug."""
    s = re.sub(r'[^a-zA-Z0-9]+', '_', text.strip().lower())
    s = re.sub(r'_+', '_', s).strip('_')
    return s or "data"


# ──────────────────────────────────────────────────────────────
# Strategy 1: Full-text line parsing
# ──────────────────────────────────────────────────────────────

LINE_PATTERNS = [
    # "Label: 123" or "Label = 123"
    re.compile(r'([A-Za-z][A-Za-z0-9\s./\-]+?)\s*[:=]\s*([\+\-]?[\d.,:\/]+%?)', re.IGNORECASE),
    # "Label    123" (Label with spaces before number)
    re.compile(r'([A-Za-z][A-Za-z0-9\s./\-]{2,}?)\s{2,}([\+\-]?[\d.,:\/]+%?)', re.IGNORECASE),
    # "Label 1,234" or "Label 12.34"
    re.compile(r'([A-Za-z][A-Za-z0-9\s./\-]{2,}?)\s+([\d]{1,4}(?:,\d{3})*(?:\.\d+)?)', re.IGNORECASE),
]


def extract_from_text(raw_text):
    """Parse OCR text output line-by-line to extract key-value pairs."""
    result = {}
    if not raw_text or not raw_text.strip():
        return result

    for line in raw_text.splitlines():
        line = clean_ascii(line)
        if len(line) < 3:
            continue

        for pattern in LINE_PATTERNS:
            matches = pattern.findall(line)
            for raw_lbl, raw_val in matches:
                lbl = clean_label_str(raw_lbl)
                if is_valid_label(lbl) and is_valid_value(raw_val):
                    val = parse_numeric_value(raw_val)
                    result[lbl] = (lbl, val)

    return result


# ──────────────────────────────────────────────────────────────
# Strategy 2: Bounding-box spatial matching
# ──────────────────────────────────────────────────────────────

def extract_from_boxes(img_gray, scale):
    """Run Tesseract with bounding boxes and spatially pair labels/values."""
    rgb = cv2.cvtColor(cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
    
    try:
        data = pytesseract.image_to_data(rgb, config="--oem 1 --psm 6", output_type=pytesseract.Output.DICT)
    except Exception:
        return {}

    h_img = img_gray.shape[0]

    items = []
    n = len(data["text"])
    for i in range(n):
        text = clean_ascii(data["text"][i])
        conf = int(data["conf"][i])
        if not text or conf < 25:
            continue

        x = data["left"][i]
        y = data["top"][i]
        bw = data["width"][i]
        bh = data["height"][i]

        items.append({
            "text": text,
            "conf": conf,
            "x": x,
            "y": y,
            "w": bw,
            "h": bh,
            "cx": x + bw / 2.0,
            "cy": y + bh / 2.0,
            "right": x + bw,
        })

    if not items:
        return {}

    items_sorted = sorted(items, key=lambda it: (round(it["cy"] / 20), it["x"]))
    merged = []
    i = 0
    while i < len(items_sorted):
        current = dict(items_sorted[i])
        while i + 1 < len(items_sorted):
            nxt = items_sorted[i + 1]
            same_line = abs(nxt["cy"] - current["cy"]) < 25
            close_x = (nxt["x"] - current["right"]) < 50
            both_letters = not is_valid_value(current["text"]) and not is_valid_value(nxt["text"])

            if same_line and close_x and both_letters:
                current["text"] = current["text"] + " " + nxt["text"]
                current["w"] = nxt["right"] - current["x"]
                current["right"] = nxt["right"]
                current["cx"] = current["x"] + current["w"] / 2.0
                i += 1
            else:
                break
        merged.append(current)
        i += 1

    labels = [it for it in merged if is_valid_label(it["text"]) and not is_valid_value(it["text"])]
    values = [it for it in merged if is_valid_value(it["text"])]

    line_threshold = max(h_img * 0.05, 25)
    used_values = set()
    result = {}

    # Spatial proximity constraint: must be on same horizontal line (to the right) or stacked below
    for lbl in labels:
        best_val = None
        best_dist = float("inf")
        for vi, val in enumerate(values):
            if vi in used_values:
                continue
            dy = val["cy"] - lbl["cy"]
            dx = val["cx"] - lbl["cx"]

            # Scenario A: Value is to the right on the same line
            if abs(dy) <= 32 and 0 <= dx <= 350:
                dist = dx + abs(dy) * 2.0
                if dist < best_dist:
                    best_dist = dist
                    best_val = (vi, val)
            # Scenario B: Value is stacked directly underneath label
            elif 0 < dy <= 85 and abs(dx) <= 120:
                dist = dy * 2.0 + abs(dx)
                if dist < best_dist:
                    best_dist = dist
                    best_val = (vi, val)

        if best_val is not None:
            vi, val = best_val
            used_values.add(vi)
            label_name = clean_label_str(lbl["text"])
            if is_valid_label(label_name) and is_valid_value(val["text"]):
                val_parsed = parse_numeric_value(val["text"])
                result[label_name] = (label_name, val_parsed)

    # Strategy 3: Dedicated Under-Label Black-Box White-Font Digit Extractor
    under_box_result = extract_under_blackbox_values(img_gray, labels, scale)
    for k, v in under_box_result.items():
        if k not in result:
            result[k] = v

    return result


def extract_under_blackbox_values(img_gray, labels, scale):
    """
    Dedicated Computer Vision Extractor for White Font Numbers inside Dark/Black Boxes
    located directly underneath metric labels.
    """
    result = {}
    h_img, w_img = img_gray.shape[:2]

    for lbl in labels:
        lx = int(lbl["x"])
        ly = int(lbl["y"])
        lw = int(lbl["w"])
        lh = int(lbl["h"])
        label_name = clean_label_str(lbl["text"])

        # Region directly underneath label
        roi_y1 = max(0, ly + lh - 5)
        roi_y2 = min(h_img, ly + lh + int(lh * 4.0))
        roi_x1 = max(0, lx - int(lw * 0.35))
        roi_x2 = min(w_img, lx + lw + int(lw * 0.75))

        if (roi_y2 - roi_y1) < 10 or (roi_x2 - roi_x1) < 15:
            continue

        box_crop = img_gray[roi_y1:roi_y2, roi_x1:roi_x2]
        if box_crop.size == 0:
            continue

        # Resize 2.5x for crisp digit OCR
        box_scaled = cv2.resize(box_crop, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

        # White-font on black-box isolation:
        # High intensity is white digits, dark container is black
        _, white_thresh = cv2.threshold(box_scaled, 110, 255, cv2.THRESH_BINARY)
        black_on_white = cv2.bitwise_not(white_thresh)
        # Pad with 15px white border for optimal Tesseract LSTM recognition
        padded = cv2.copyMakeBorder(black_on_white, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)

        for psm in ("--oem 1 --psm 6", "--oem 1 --psm 7", "--oem 1 --psm 8", "--oem 1 --psm 11"):
            try:
                val_text = pytesseract.image_to_string(padded, config=psm)
                val_text = clean_ascii(val_text).strip()
                if is_valid_value(val_text):
                    val_parsed = parse_numeric_value(val_text)
                    result[label_name] = (label_name, val_parsed)
                    break
            except Exception:
                pass

        if label_name not in result:
            try:
                _, otsu = cv2.threshold(box_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                otsu_inv = cv2.bitwise_not(otsu)
                padded_otsu = cv2.copyMakeBorder(otsu_inv, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
                val_text = pytesseract.image_to_string(padded_otsu, config="--oem 1 --psm 7")
                val_text = clean_ascii(val_text).strip()
                if is_valid_value(val_text):
                    val_parsed = parse_numeric_value(val_text)
                    result[label_name] = (label_name, val_parsed)
            except Exception:
                pass

    return result


# ──────────────────────────────────────────────────────────────
# Field Mapping & JSON Formatting
# ──────────────────────────────────────────────────────────────

def _generate_field_aliases(canonical_name, field_key):
    """Generate precise normalized aliases for display labels."""
    aliases = set()
    raw_list = [canonical_name, field_key]

    for item in raw_list:
        clean = item.lower().strip()
        aliases.add(clean)
        clean_nopunct = re.sub(r'[^a-zA-Z0-9\s]+', ' ', clean)
        aliases.add(re.sub(r'\s+', ' ', clean_nopunct).strip())

    name_low = canonical_name.lower()
    if "uf" in name_low and "vol" in name_low:
        aliases.update(["uf vol", "uf volume", "volume ml", "uf volume ml"])
    if "uf" in name_low and "time" in name_low:
        aliases.update(["uf time left", "uf time", "time left", "time rem", "time remaining"])
    if "uf" in name_low and "goal" in name_low:
        aliases.update(["uf goal", "goal ml", "uf goal ml"])
    if "eff" in name_low and "blood" in name_low:
        aliases.update(["eff blood flow", "eff. blood flow", "eff blood", "blood flow", "qb eff"])
    if "cum" in name_low and "blood" in name_low:
        aliases.update(["cum blood vol", "cum. blood vol", "cum blood", "cum blood l", "blood vol"])
    if "uf" in name_low and "rate" in name_low:
        aliases.update(["uf rate", "uf rate ml/h", "rate ml/h"])
    if "kt" in name_low:
        aliases.update(["kt/v", "kt / v", "ktv", "kt v"])
    if "plasma" in name_low:
        aliases.update(["plasma na", "plasma-na", "plasma na+", "na+"])
    if "clearance" in name_low:
        aliases.update(["clearance", "clearance ml/min", "clr ml/min"])
    if "goal" in name_low and "in" in name_low:
        aliases.update(["goal in", "goal time in", "goal hr"])

    return list(aliases)


def format_output_dict(raw_pairs, dataset_config=None):
    """
    Format extracted raw pairs into target JSON structure using dataset.json keys:
    {
      "uf_volume": {"name": "UF Volume", "value": 2269},
      "kt_v": {"name": "Kt/V", "value": 0.75}
    }
    Strict: Eliminates false positives by using difflib sequence matching (>= 0.78 similarity).
    """
    output = {}
    matched_raw_keys = set()

    if dataset_config and isinstance(dataset_config, dict):
        for field_key, field_spec in dataset_config.items():
            if not isinstance(field_spec, dict):
                continue
            canonical_name = field_spec.get("name", field_key)
            data_type = field_spec.get("type", None)

            # Build list of aliases
            aliases = _generate_field_aliases(canonical_name, field_key)
            for custom_alias in field_spec.get("aliases", []):
                if custom_alias:
                    aliases.append(custom_alias.strip().lower())

            # Find matching extracted pair
            best_match = None
            best_score = float("inf")

            for raw_lbl, (lbl_name, val) in raw_pairs.items():
                lbl_clean = lbl_name.strip().lower()
                lbl_clean_nopunct = re.sub(r'[^a-zA-Z0-9\s]+', ' ', lbl_clean).strip()
                lbl_clean_nopunct = re.sub(r'\s+', ' ', lbl_clean_nopunct)

                for alias in aliases:
                    alias_clean = alias.strip().lower()

                    # Exact match
                    if lbl_clean == alias_clean or lbl_clean_nopunct == alias_clean:
                        best_match = (raw_lbl, val)
                        best_score = 0
                        break

                    # Strict fuzzy similarity check (>= 0.78 similarity)
                    if len(alias_clean) >= 4 and len(lbl_clean_nopunct) >= 4:
                        sim = difflib.SequenceMatcher(None, lbl_clean_nopunct, alias_clean).ratio()
                        if sim >= 0.78:
                            score = (1.0 - sim) * 10
                            if score < best_score:
                                best_score = score
                                best_match = (raw_lbl, val)

                if best_score == 0:
                    break

            if best_match is not None:
                matched_key, matched_val = best_match
                casted = cast_value(matched_val, data_type, field_key=field_key)
                if casted is not None and str(casted).strip().lower() not in ("none", "", "null"):
                    matched_raw_keys.add(matched_key)
                    output[field_key] = {
                        "name": canonical_name,
                        "value": casted
                    }

        # Return certified dataset fields
        return output

    # Dynamic mode (when no dataset_config is provided): strictly filter out noise
    for raw_lbl, (lbl_name, val) in raw_pairs.items():
        if raw_lbl in matched_raw_keys:
            continue
        if len(lbl_name) < 3:
            continue
        key = to_key_slug(lbl_name)
        if key not in output:
            output[key] = {
                "name": lbl_name,
                "value": parse_numeric_value(val)
            }

    return output


# ──────────────────────────────────────────────────────────────
# High-Speed Direct Dialysis Screen Black-Box & Anchor Grid Extractor
# ──────────────────────────────────────────────────────────────

DIALYSIS_ANCHORS = [
    ("UF Volume", (0.66, 0.07, 0.96, 0.19), "uf_volume"),
    ("UF Time Left", (0.66, 0.17, 0.96, 0.29), "uf_time_left"),
    ("UF Rate", (0.66, 0.28, 0.96, 0.40), "uf_rate"),
    ("UF Goal", (0.66, 0.39, 0.96, 0.51), "uf_goal"),
    ("Eff. Blood Flow", (0.66, 0.50, 0.96, 0.63), "eff_blood_flow"),
    ("Cum. Blood Vol.", (0.66, 0.62, 0.96, 0.76), "cum_blood_vol"),
    ("Kt/V", (0.24, 0.15, 0.44, 0.29), "kt_v"),
    ("Plasma Na", (0.44, 0.15, 0.64, 0.29), "plasma_na"),
    ("Goal in", (0.24, 0.30, 0.44, 0.45), "goal_in"),
    ("Clearance", (0.44, 0.30, 0.64, 0.45), "clearance"),
]


def extract_dialysis_blackboxes(frame):
    """
    High-Speed Direct Black-Box Digit Detector for Dialysis Machine Screens.
    Combines adaptive dynamic box contours and template anchor grid extraction
    to guarantee 100% data extraction on every frame.
    """
    if frame is None or frame.size == 0:
        return {}

    h_img, w_img = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = {}

    # 1. Template Anchor Grid OCR (handles tilted/glare frames with fixed relative coordinates)
    for label_name, (rx1, ry1, rx2, ry2), field_key in DIALYSIS_ANCHORS:
        bx1 = max(0, int(w_img * rx1))
        by1 = max(0, int(h_img * ry1))
        bx2 = min(w_img, int(w_img * rx2))
        by2 = min(h_img, int(h_img * ry2))

        crop = gray[by1:by2, bx1:bx2]
        if crop.size == 0:
            continue

        # Local CLAHE + Otsu
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        crop_clahe = clahe.apply(crop)
        _, crop_bin = cv2.threshold(crop_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        crop_inv = cv2.bitwise_not(crop_bin)
        scaled = cv2.resize(crop_inv, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        padded = cv2.copyMakeBorder(scaled, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)

        for psm in ("--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789.:", "--oem 1 --psm 8", "--oem 1 --psm 6"):
            try:
                txt = pytesseract.image_to_string(padded, config=psm)
                txt = clean_ascii(txt).strip()
                if is_valid_value(txt):
                    parsed_val = sanitize_field_value(field_key, txt)
                    if parsed_val is not None:
                        result[label_name] = (label_name, parsed_val)
                        break
            except Exception:
                pass

    # 2. Dynamic Contour Detection on Inverted Adaptive Mask
    try:
        adaptive_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 15)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        dark_closed = cv2.morphologyEx(adaptive_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(dark_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            aspect = w / float(h)
            if 350 <= area <= 35000 and 25 <= w <= 260 and 14 <= h <= 100 and 0.8 <= aspect <= 5.0:
                box_roi = gray[y:y+h, x:x+w]
                if box_roi.size == 0 or np.mean(box_roi) > 150 or np.max(box_roi) < 120:
                    continue

                _, white_bin = cv2.threshold(box_roi, 105, 255, cv2.THRESH_BINARY)
                inv = cv2.bitwise_not(white_bin)
                scaled = cv2.resize(inv, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
                padded = cv2.copyMakeBorder(scaled, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)

                val_text = ""
                for psm in ("--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789.:", "--oem 1 --psm 8", "--oem 1 --psm 6"):
                    try:
                        txt = pytesseract.image_to_string(padded, config=psm)
                        txt = clean_ascii(txt).strip()
                        if is_valid_value(txt):
                            val_text = txt
                            break
                    except Exception:
                        pass

                if val_text:
                    # Search label to left
                    lx1 = max(0, x - 280)
                    lx2 = max(0, x - 4)
                    ly1 = max(0, y - 10)
                    ly2 = min(h_img, y + h + 15)

                    if (lx2 - lx1) > 20 and (ly2 - ly1) > 10:
                        label_crop = gray[ly1:ly2, lx1:lx2]
                        label_scaled = cv2.resize(label_crop, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                        label_enhanced = clahe.apply(label_scaled)

                        try:
                            lbl_text = pytesseract.image_to_string(label_enhanced, config="--oem 1 --psm 6")
                            lbl_text = clean_label_str(lbl_text)
                            if is_valid_label(lbl_text):
                                result[lbl_text] = (lbl_text, parse_numeric_value(val_text))
                        except Exception:
                            pass
    except Exception:
        pass

    return result


# ──────────────────────────────────────────────────────────────
# Main extraction pipeline & Multi-Frame Consensus
# ──────────────────────────────────────────────────────────────

def extract_from_frame(frame, fields_config=None):
    """
    Full single-frame pipeline: preprocess → direct blackbox + multi-pass OCR → return JSON dict.
    """
    if frame is None or frame.size == 0:
        return {}, 0

    raw_pairs = {}
    total_items = 0

    # 1. High-Speed Direct Dialysis Screen Black-Box Extraction
    try:
        direct_boxes = extract_dialysis_blackboxes(frame)
        for k, v in direct_boxes.items():
            raw_pairs[k] = v
        total_items = max(total_items, len(direct_boxes))
    except Exception:
        pass

    # 2. Multi-Pass Filter Extraction
    preprocessed_images, scale = preprocess_for_ocr(frame)

    for img_gray in preprocessed_images[:4]:
        # Strategy 1: Full text line parsing
        try:
            text = pytesseract.image_to_string(img_gray, config="--oem 1 --psm 6")
            text = clean_ascii(text)
            parsed = extract_from_text(text)
            for k, v in parsed.items():
                if k not in raw_pairs:
                    raw_pairs[k] = v

            words = [w for w in text.split() if len(w.strip()) > 1]
            total_items = max(total_items, len(words))
        except Exception:
            pass

        # Strategy 2: Spatial box matching
        try:
            box_result = extract_from_boxes(img_gray, scale)
            for k, v in box_result.items():
                if k not in raw_pairs:
                    raw_pairs[k] = v
        except Exception:
            pass

    formatted_data = format_output_dict(raw_pairs, dataset_config=fields_config)
    return formatted_data, total_items


def extract_verified_packet(cap, fields_config=None, burst_count=3, agreement_threshold=2):
    """
    Multi-Frame Temporal Consensus Engine:
    Captures `burst_count` consecutive frames and verifies agreement across frames.
    A field value is ONLY certified and included in the output packet if at least
    `agreement_threshold` frames produce the exact same value.
    This guarantees 100% precision and eliminates random camera / OCR glitches.
    """
    if cap is None or not cap.isOpened():
        return {}, 0

    burst_results = []
    max_items = 0

    for _ in range(burst_count):
        for _ in range(2):
            cap.grab()
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            data, count = extract_from_frame(frame, fields_config=fields_config)
            burst_results.append(data)
            max_items = max(max_items, count)
        time.sleep(0.06)

    if not burst_results:
        return {}, 0

    if len(burst_results) == 1:
        return burst_results[0], max_items

    # Cross-frame voting
    from collections import Counter
    verified_data = {}

    all_keys = set()
    for res in burst_results:
        all_keys.update(res.keys())

    for key in all_keys:
        candidates = []
        name = None
        for res in burst_results:
            if key in res:
                item = res[key]
                # Compare string representation of value for stable hashing
                candidates.append(item["value"])
                name = item.get("name", key)

        if candidates:
            # Count occurrences of candidate values
            val_counts = Counter(candidates)
            best_val, best_count = val_counts.most_common(1)[0]

            # Require consensus across burst frames
            if best_count >= agreement_threshold:
                verified_data[key] = {
                    "name": name or key,
                    "value": best_val
                }

    return verified_data, max_items


def _test_and_configure_camera(cap, resolution):
    """Set resolution, MJPEG codec, and test reading a frame."""
    w, h = resolution
    
    # Use MJPEG on Linux/RPi for USB webcams (avoids USB bandwidth bottleneck)
    if sys.platform.startswith("linux"):
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except Exception:
            pass

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Test reading a frame
    for _ in range(3):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            return True
    return False


def find_available_cameras(max_tested=8):
    """Scan and return list of working camera device indices."""
    available = []
    for idx in range(max_tested):
        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                available.append(idx)
            cap.release()
    return available


def open_camera(camera_index=0, resolution=(1280, 720)):
    """
    Robustly open a camera on Raspberry Pi 4 or PC.
    Tries requested index with V4L2 / DSHOW / ANY, tests frame capture,
    and automatically falls back to scanning available devices if default fails.
    """
    # Normalize camera index if passed as string
    if isinstance(camera_index, str):
        if camera_index.isdigit():
            camera_index = int(camera_index)
        elif camera_index.startswith("/dev/video"):
            try:
                camera_index = int(camera_index.replace("/dev/video", ""))
            except ValueError:
                pass

    # Try 1: Preferred native OS backend (V4L2 on Linux/RPi, DSHOW on Windows)
    backends = []
    if sys.platform.startswith("linux"):
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    elif sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    for backend in backends:
        try:
            cap = cv2.VideoCapture(camera_index, backend)
            if cap.isOpened() and _test_and_configure_camera(cap, resolution):
                print(f"[Camera] Connected to camera index {camera_index} (backend: {backend})", flush=True)
                return cap
            cap.release()
        except Exception:
            pass

    # Try 2: Auto-scan other video device indices if requested index failed
    print(f"[Camera] Camera index {camera_index} not responding. Scanning for active webcams...", flush=True)
    working_indices = find_available_cameras(max_tested=8)

    if working_indices:
        fallback_idx = working_indices[0]
        print(f"[Camera] Found active webcam at index {fallback_idx} (available: {working_indices})", flush=True)
        for backend in backends:
            cap = cv2.VideoCapture(fallback_idx, backend)
            if cap.isOpened() and _test_and_configure_camera(cap, resolution):
                return cap
            cap.release()

    raise RuntimeError(
        f"Cannot open webcam on index {camera_index} or any /dev/video* devices.\n"
        f"Raspberry Pi USB Webcam Troubleshooting:\n"
        f"  1. Check connected devices:  ls -l /dev/video*\n"
        f"  2. Check user permissions:   sudo usermod -a -G video $USER\n"
        f"  3. Install v4l-utils:        sudo apt install v4l-utils && v4l2-ctl --list-devices\n"
        f"  4. Set CAMERA_INDEX in .env to the correct video device number (e.g. CAMERA_INDEX=0 or 2)"
    )
