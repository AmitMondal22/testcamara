"""
black_box_extractor.py
-----------------------
Detects dark/black LCD numeric display boxes on a Fresenius 4008S dialysis
monitor screen, extracts numeric digits with OCR, and assigns parameters by
quadrant/column position with tilt and skew resilience.
"""

import cv2
import numpy as np
import re
import threading
import sys
import os
import json

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Central Config
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")


def _load_units():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {k: v.get("unit", "") for k, v in data.get("dialysis_fields", {}).items()}
        except Exception:
            pass
    return {
        "UF Volume": "ml", "UF Time Left": "h:min", "UF Rate": "ml/h", "UF Goal": "ml",
        "Eff. Blood Flow": "ml/min", "Cum. Blood Vol.": "l", "Kt/V": "",
        "Plasma Na": "mmol/l", "Goal in": "h:min", "Clearance": "ml/min",
    }


FIELD_UNITS = _load_units()
RIGHT_COL_FIELDS = ["UF Volume", "UF Time Left", "UF Rate", "UF Goal", "Eff. Blood Flow", "Cum. Blood Vol."]
CENTER_FIELDS = ["Kt/V", "Plasma Na", "Goal in", "Clearance"]

_READER = None
_READER_LOCK = threading.Lock()


def _get_reader():
    global _READER
    if _READER is not None:
        return _READER
    with _READER_LOCK:
        if _READER is None:
            try:
                import easyocr  # pyrefly: ignore [missing-import]
                _READER = easyocr.Reader(["en"], gpu=False, verbose=False)
            except Exception as e:
                print(f"[BlackBoxOCR] EasyOCR init note: {e}", flush=True)
                _READER = None
    return _READER


def _preprocess_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w > 1280:
        scale = 1280.0 / w
        frame = cv2.resize(frame, (1280, int(h * scale)), interpolation=cv2.INTER_AREA)
    elif w < 320:
        scale = 640.0 / w
        frame = cv2.resize(frame, (640, int(h * scale)), interpolation=cv2.INTER_LINEAR)
    return frame


def _iou(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(x1, x2)
    iy = max(y1, y2)
    iw = min(x1 + w1, x2 + w2) - ix
    ih = min(y1 + h1, y2 + h2) - iy
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = w1 * h1 + w2 * h2 - inter
    return inter / max(union, 1)


def detect_dark_boxes(frame: np.ndarray):
    """
    Finds dark/black LCD value boxes using multi-threshold contour sweep.
    Resilient against screen reflections and lighting variations.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    candidates = []
    for thresh_val in [40, 60, 80, 100, 120]:
        _, dark_mask = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
        closed = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            aspect = w / max(h, 1)
            if area < 800 or area > 0.35 * w_img * h_img:
                continue
            if aspect < 0.7 or aspect > 10.0:
                continue
            if w < 25 or h < 12:
                continue
            candidates.append((x, y, w, h))

    candidates.sort(key=lambda b: b[2] * b[3], reverse=True)
    filtered = []
    for box in candidates:
        if not any(_iou(box, kept) > 0.4 for kept in filtered):
            filtered.append(box)

    filtered.sort(key=lambda b: (b[1], b[0]))
    return filtered


def _ocr_box_value(frame: np.ndarray, x: int, y: int, w: int, h: int) -> str:
    """Crops box, inverts colors, upscales, and extracts digits."""
    pad = 3
    x1, y1 = max(0, x + pad), max(0, y + pad)
    x2, y2 = min(frame.shape[1], x + w - pad), min(frame.shape[0], y + h - pad)
    if x2 <= x1 or y2 <= y1:
        return ""

    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    big = cv2.resize(gray, (gray.shape[1] * 3, gray.shape[0] * 3), interpolation=cv2.INTER_CUBIC)
    inverted = cv2.bitwise_not(big)
    _, thresh = cv2.threshold(inverted, 140, 255, cv2.THRESH_BINARY)

    reader = _get_reader()
    if reader is None:
        return ""

    try:
        results = reader.readtext(thresh, detail=1, allowlist="0123456789.:")
        if not results:
            results = reader.readtext(thresh, detail=1)
        if not results:
            return ""

        text = re.sub(r"\s+", "", " ".join(r[1].strip() for r in results))

        if text.startswith("0.") or text.startswith("0,"):
            return text.replace(",", ".")

        m_time = re.match(r"^([1-9])[\.:](\d{2})$", text)
        if m_time:
            text = f"{m_time.group(1)}:{m_time.group(2)}"

        cleaned = re.sub(r"[^0-9\.:]", "", text)
        return cleaned if cleaned else ""
    except Exception:
        return ""


def extract_from_black_boxes(frame: np.ndarray) -> dict:
    """
    Detects dark LCD boxes and assigns field names by screen position and column.
    """
    if frame is None or frame.size == 0:
        return {}

    frame = _preprocess_frame(frame)
    h_img, w_img = frame.shape[:2]
    boxes = detect_dark_boxes(frame)

    box_values = []
    for (x, y, w, h) in boxes:
        val = _ocr_box_value(frame, x, y, w, h)
        if val:
            box_values.append((x, y, w, h, val))

    if not box_values:
        return {}

    assigned = {}
    units = _load_units()

    # Split into right-column (UF fields) and center (OCM-Data)
    right_boxes = sorted(
        [(x, y, w, h, v) for (x, y, w, h, v) in box_values if (x + w / 2) > w_img * 0.62],
        key=lambda b: b[1]
    )
    center_boxes = sorted(
        [(x, y, w, h, v) for (x, y, w, h, v) in box_values if (x + w / 2) <= w_img * 0.62],
        key=lambda b: (b[1], b[0])
    )

    # Right column: UF Volume -> Cum. Blood Vol.
    for i, (x, y, w, h, val) in enumerate(right_boxes):
        if i < len(RIGHT_COL_FIELDS):
            fname = RIGHT_COL_FIELDS[i]
            assigned[fname] = {
                "value": val,
                "unit": units.get(fname, ""),
                "confidence": 0.85
            }

    # Center OCM-Data: Quadrant grid
    n = len(center_boxes)
    if n == 4:
        ys = [b[1] for b in center_boxes]
        mid_y = (min(ys) + max(ys)) / 2.0
        top_row = sorted([b for b in center_boxes if b[1] <= mid_y], key=lambda b: b[0])
        bot_row = sorted([b for b in center_boxes if b[1] > mid_y], key=lambda b: b[0])
        quad = []
        if len(top_row) >= 1:
            quad.append(("Kt/V", top_row[0]))
        if len(top_row) >= 2:
            quad.append(("Plasma Na", top_row[1]))
        if len(bot_row) >= 1:
            quad.append(("Goal in", bot_row[0]))
        if len(bot_row) >= 2:
            quad.append(("Clearance", bot_row[1]))
        for fname, (x, y, w, h, val) in quad:
            if fname not in assigned:
                assigned[fname] = {
                    "value": val,
                    "unit": units.get(fname, ""),
                    "confidence": 0.85
                }
    elif n >= 1:
        for i, (x, y, w, h, val) in enumerate(center_boxes):
            if i < len(CENTER_FIELDS):
                fname = CENTER_FIELDS[i]
                if fname not in assigned:
                    assigned[fname] = {
                        "value": val,
                        "unit": units.get(fname, ""),
                        "confidence": 0.75
                    }

    return assigned
