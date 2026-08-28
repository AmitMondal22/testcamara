"""
black_box_extractor.py
-----------------------
Detects dark/black LCD numeric display boxes on a Fresenius 4008S dialysis
monitor screen captured via USB webcam, and extracts numeric values using OCR.

Approach:
  1. Multi-threshold sweep to find dark rectangular boxes in frame
  2. OCR each box using EasyOCR (digits only, inverted + upscaled crop)
  3. Assign field names by POSITION only (no label OCR - too unreliable at webcam distance)

Fresenius 4008S screen layout:
  Center-Left OCM-Data (2x2 grid):
    Top-Left  = Kt/V,      Top-Right = Plasma Na
    Bot-Left  = Goal in,   Bot-Right = Clearance
  Right column (top to bottom):
    UF Volume, UF Time Left, UF Rate, UF Goal, Eff. Blood Flow, Cum. Blood Vol.
"""

import cv2
import numpy as np
import re
import threading
import sys
import io

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────
# EasyOCR singleton (thread-safe)
# ─────────────────────────────────────────────────────────────
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
                print("[BlackBoxOCR] EasyOCR reader initialized.", flush=True)
            except Exception as e:
                print(f"[BlackBoxOCR] EasyOCR init error: {e}", flush=True)
                _READER = None
    return _READER


# ─────────────────────────────────────────────────────────────
# Field configuration
# ─────────────────────────────────────────────────────────────
FIELD_UNITS = {
    "UF Volume": "ml",
    "UF Time Left": "h:min",
    "UF Rate": "ml/h",
    "UF Goal": "ml",
    "Eff. Blood Flow": "ml/min",
    "Cum. Blood Vol.": "l",
    "Kt/V": "",
    "Plasma Na": "mmol/l",
    "Goal in": "h:min",
    "Clearance": "ml/min",
}

RIGHT_COL_FIELDS = ["UF Volume", "UF Time Left", "UF Rate", "UF Goal", "Eff. Blood Flow", "Cum. Blood Vol."]
CENTER_FIELDS = ["Kt/V", "Plasma Na", "Goal in", "Clearance"]


# ─────────────────────────────────────────────────────────────
# Frame preprocessing
# ─────────────────────────────────────────────────────────────

def _preprocess_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w > 1280:
        scale = 1280.0 / w
        frame = cv2.resize(frame, (1280, int(h * scale)), interpolation=cv2.INTER_AREA)
    elif w < 320:
        scale = 640.0 / w
        frame = cv2.resize(frame, (640, int(h * scale)), interpolation=cv2.INTER_LINEAR)
    return frame


# ─────────────────────────────────────────────────────────────
# Box detection
# ─────────────────────────────────────────────────────────────

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
    Find dark/black LCD value boxes using multi-threshold sweep.
    Returns list of (x, y, w, h) sorted top-to-bottom, left-to-right.
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

    # NMS: remove boxes with IoU > 0.4 with a larger box
    candidates.sort(key=lambda b: b[2] * b[3], reverse=True)
    filtered = []
    for box in candidates:
        dominated = any(_iou(box, kept) > 0.4 for kept in filtered)
        if not dominated:
            filtered.append(box)

    filtered.sort(key=lambda b: (b[1], b[0]))
    return filtered


# ─────────────────────────────────────────────────────────────
# Per-box OCR
# ─────────────────────────────────────────────────────────────

def _ocr_box_value(frame: np.ndarray, x: int, y: int, w: int, h: int) -> str:
    """
    Crop box, invert colors, upscale 3x, run EasyOCR for digits only.
    Fixes colon/period ambiguity:
      - D.DD (1 digit + dot + 2 digits) -> time h:mm  (e.g. 1.43 -> 1:43)
      - D:DD where right part > 59      -> decimal     (e.g. 0:84 -> 0.84)
    """
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

        # 1. Decimal values starting with 0 (e.g. Kt/V 0.68, 0.61, 0.84, 0.11)
        if text.startswith("0.") or text.startswith("0,"):
            return text.replace(",", ".")

        # 2. Time values starting with non-zero hour (e.g. 1.43 -> 1:43, 1.53 -> 1:53)
        m_time = re.match(r"^([1-9])[\.:](\d{2})$", text)
        if m_time:
            text = f"{m_time.group(1)}:{m_time.group(2)}"

        cleaned = re.sub(r"[^0-9\.:]" , "", text)
        return cleaned if cleaned else ""
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────
# Main extraction entry point
# ─────────────────────────────────────────────────────────────

def extract_from_black_boxes(frame: np.ndarray) -> dict:
    """
    Detect dark LCD boxes and assign field names by screen position.

    Returns dict: {field_name: {"value": str, "unit": str, "confidence": float}}
    """
    if frame is None or frame.size == 0:
        return {}

    frame = _preprocess_frame(frame)
    h_img, w_img = frame.shape[:2]
    boxes = detect_dark_boxes(frame)

    print(f"[BlackBoxOCR] Detected {len(boxes)} dark boxes ({w_img}x{h_img})", flush=True)

    # OCR all boxes
    box_values = []
    for (x, y, w, h) in boxes:
        val = _ocr_box_value(frame, x, y, w, h)
        print(f"[BlackBoxOCR]   box({x},{y},{w}x{h}) -> '{val}'", flush=True)
        if val:
            box_values.append((x, y, w, h, val))

    if not box_values:
        print("[BlackBoxOCR] No numeric values extracted.", flush=True)
        return {}

    assigned: dict = {}

    # Split into right-column (UF fields) and center (OCM-Data)
    # Threshold 0.65: OCM-Data section spans up to ~60% of frame width;
    # the UF column appears at 65%+ (only when full screen is in view)
    right_boxes = sorted(
        [(x, y, w, h, v) for (x, y, w, h, v) in box_values if (x + w / 2) > w_img * 0.65],
        key=lambda b: b[1]
    )
    center_boxes = sorted(
        [(x, y, w, h, v) for (x, y, w, h, v) in box_values if (x + w / 2) <= w_img * 0.65],
        key=lambda b: (b[1], b[0])
    )

    # Right column: UF Volume -> Cum. Blood Vol. (top to bottom)
    for i, (x, y, w, h, val) in enumerate(right_boxes):
        if i < len(RIGHT_COL_FIELDS):
            fname = RIGHT_COL_FIELDS[i]
            assigned[fname] = {
                "value": val,
                "unit": FIELD_UNITS.get(fname, ""),
                "confidence": round(0.80, 2)
            }

    # Center OCM-Data: 2x2 quadrant grid
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
                    "unit": FIELD_UNITS.get(fname, ""),
                    "confidence": round(0.82, 2)
                }

    elif n == 2:
        ys = [b[1] for b in center_boxes]
        lr = sorted(center_boxes, key=lambda b: b[0])
        if abs(ys[0] - ys[1]) < 25:
            # Same row - determine which row based on y position
            row_y = center_boxes[0][1] / h_img
            row_fields = ["Kt/V", "Plasma Na"] if row_y < 0.55 else ["Goal in", "Clearance"]
        else:
            row_fields = ["Kt/V", "Goal in"]
        for i, (x, y, w, h, val) in enumerate(lr):
            fname = row_fields[i] if i < len(row_fields) else None
            if fname and fname not in assigned:
                assigned[fname] = {
                    "value": val,
                    "unit": FIELD_UNITS.get(fname, ""),
                    "confidence": round(0.70, 2)
                }

    elif n == 1:
        x, y, w, h, val = center_boxes[0]
        fname = "Kt/V" if (x + w / 2) < w_img * 0.35 else "Plasma Na"
        if fname not in assigned:
            assigned[fname] = {
                "value": val,
                "unit": FIELD_UNITS.get(fname, ""),
                "confidence": round(0.60, 2)
            }

    elif n >= 3:
        # Assign sequentially
        for i, (x, y, w, h, val) in enumerate(center_boxes):
            if i < len(CENTER_FIELDS):
                fname = CENTER_FIELDS[i]
                if fname not in assigned:
                    assigned[fname] = {
                        "value": val,
                        "unit": FIELD_UNITS.get(fname, ""),
                        "confidence": round(0.65, 2)
                    }

    print(f"[BlackBoxOCR] Extracted {len(assigned)} fields: {list(assigned.keys())}", flush=True)
    return assigned
