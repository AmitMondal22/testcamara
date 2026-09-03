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

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
try:
    # pyrefly: ignore [missing-import]
    import torch
    torch.set_num_threads(1)
    torch.multiprocessing.set_sharing_strategy('file_system')
except Exception:
    pass

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
# Screen layout constants (Fresenius 4008S)
# Based on actual webcam capture:
#
#  ┌────────────────────────────────────────────────────────────┐
#  │ Dialysis                                     │  UF Volume  │
#  │────────────────────────────────────────────────────────────│
#  │ OCM-Data                                     │  2,650      │
#  │  Kt/V     [0.88]  Plasma Na [134]            │  UF Time L. │
#  │  Goal in  [1:53]  Clearance [157]            │  1:43       │
#  │                                              │  UF Rate    │
#  │                                              │  1,046      │
#  │                                              │  UF Goal    │
#  │                                              │  4,000      │
#  │                                              │  Eff. BF    │
#  │                                              │  231        │
#  │                                              │  Cum. BV    │
#  │                                              │  38.7       │
#  └────────────────────────────────────────────────────────────┘
#
# x-split thresholds (fraction of frame width):
#   Left zone:   0.00 – 0.38  →  Kt/V, Goal in
#   Middle zone: 0.38 – 0.68  →  Plasma Na, Clearance
#   Right zone:  0.68 – 1.00  →  UF Volume … Cum. Blood Vol.
# ─────────────────────────────────────────────────────────────
LEFT_SPLIT   = 0.38   # boundary between left-zone and middle-zone
RIGHT_SPLIT  = 0.68   # boundary between middle-zone and right-zone

FIELD_UNITS = {
    "UF Volume":       "ml",
    "UF Time Left":    "h:min",
    "UF Rate":         "ml/h",
    "UF Goal":         "ml",
    "Eff. Blood Flow": "ml/min",
    "Cum. Blood Vol.": "l",
    "Kt/V":            "",
    "Plasma Na":       "mmol/l",
    "Goal in":         "h:min",
    "Clearance":       "ml/min",
}

LEFT_FIELDS   = ["Kt/V", "Goal in"]                            # top→bottom in left zone
MIDDLE_FIELDS = ["Plasma Na", "Clearance"]                     # top→bottom in middle zone
RIGHT_COL_FIELDS = ["UF Volume", "UF Time Left", "UF Rate",    # top→bottom in right zone
                     "UF Goal", "Eff. Blood Flow", "Cum. Blood Vol."]
CENTER_FIELDS = ["Kt/V", "Plasma Na", "Goal in", "Clearance"]  # legacy alias


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
    Tuned for Fresenius 4008S display: near-pure-black boxes with
    bright white digits, captured via USB webcam or IMX477 IR camera.
    Returns list of (x, y, w, h) sorted top-to-bottom, left-to-right.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape

    # Mild blur to reduce webcam/IR noise without losing box edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    candidates = []
    # Sweep thresholds to catch dark boxes even under varying IR / ambient lighting
    for thresh_val in [20, 30, 45, 65, 90, 115]:
        _, dark_mask = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 6))
        closed = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            aspect = w / max(h, 1)
            # Filter out tiny noise and full-screen containers
            if area < 500 or area > 0.30 * w_img * h_img:
                continue
            if aspect < 0.5 or aspect > 12.0:
                continue
            if w < 18 or h < 10:
                continue

            # Filter out graph Y-axis tick mark zone (bottom-left graph region) to prevent reading scale label '-300'
            if x < 0.15 * w_img and y > 0.55 * h_img:
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
# Per-box OCR with Dynamic Margin Expansion
# ─────────────────────────────────────────────────────────────

def _ocr_box_value(frame: np.ndarray, x: int, y: int, w: int, h: int) -> str:
    """
    Crop a detected dark box with dynamic margin expansion, prepare it for OCR,
    and return the cleaned digit string without truncation.

    Pipeline:
      1. Dynamic margin expansion (adds 18% width & 10% height padding) to guarantee zero clipping
      2. Convert to grayscale, upscale 4x with bicubic interpolation
      3. Invert image (dark digits on white background)
      4. Otsu binarization (adaptive to camera light / IR glare)
      5. Add white margin border to protect edge characters
      6. Run EasyOCR / PyTesseract with digit whitelist
      7. Post-process: thousand-comma removal, time & decimal formatting
    """
    pad_w = max(10, int(w * 0.18))
    pad_h = max(4, int(h * 0.10))
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(frame.shape[1], x + w + pad_w)
    y2 = min(frame.shape[0], y + h + pad_h)

    if x2 <= x1 or y2 <= y1:
        return ""

    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # 4× upscale for sharper digit recognition
    big = cv2.resize(gray, (gray.shape[1] * 4, gray.shape[0] * 4), interpolation=cv2.INTER_CUBIC)

    # Invert: make digits dark, background white
    inverted = cv2.bitwise_not(big)

    # Otsu threshold (adaptive to lighting conditions)
    _, thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Add 15px white border padding to prevent character clipping at boundaries
    thresh = cv2.copyMakeBorder(thresh, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)

    # Convert to contiguous RGB image to avoid PyTorch tensor alignment SIGBUS
    thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    thresh_rgb = np.ascontiguousarray(thresh_rgb)

    reader = _get_reader()
    if reader is None:
        return ""

    try:
        results = None
        with _READER_LOCK:
            # Primary: digits + punctuation allowlist
            results = reader.readtext(thresh_rgb, detail=1, allowlist="0123456789.,:-")
            if not results:
                results = reader.readtext(thresh_rgb, detail=1)

        if not results:
            return ""

        raw = re.sub(r"\s+", "", " ".join(r[1].strip() for r in results))

        # Remove thousand-separator commas: "2,923" → "2923", "2,650" → "2650", "1,046" → "1046"
        text = re.sub(r"(\d+)[,\s](\d{3})", r"\1\2", raw)
        text = re.sub(r"(\d),(\d{3})", r"\1\2", text)

        # Decimal values starting with 0 (Kt/V: 0.88, Cum.Blood Vol.: 38.7)
        if text.startswith("0.") or text.startswith("0,"):
            return text.replace(",", ".")

        # Time values: D.DD or D:DD → D:DD (e.g. 1.43 → 1:43, 1.53 → 1:53)
        m_time = re.match(r"^([0-9]{1,2})[\.:](\d{2})$", text)
        if m_time:
            return f"{m_time.group(1)}:{m_time.group(2)}"

        # Strip anything that isn't a digit, dot, or colon
        cleaned = re.sub(r"[^0-9\.:]", "", text)
        return cleaned if cleaned else ""
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────
# Main extraction entry point
# ─────────────────────────────────────────────────────────────

def extract_from_black_boxes(frame: np.ndarray) -> dict:
    """
    Detect dark LCD boxes and assign field names by screen position.

    Uses a 3-zone layout matching the Fresenius 4008S display:
      Left zone   (x < 38% width) : Kt/V, Goal in      (top→bottom)
      Middle zone (38% – 68%)     : Plasma Na, Clearance (top→bottom)
      Right zone  (x > 68% width) : UF Volume, UF Time Left, UF Rate,
                                     UF Goal, Eff. Blood Flow, Cum. Blood Vol.

    Returns dict: {field_name: {"value": str, "unit": str, "confidence": float}}
    """
    assigned: dict = {
        fname: {"value": None, "unit": u, "confidence": 0.0}
        for fname, u in FIELD_UNITS.items()
    }

    if frame is None or frame.size == 0:
        return assigned

    frame = _preprocess_frame(frame)
    h_img, w_img = frame.shape[:2]
    boxes = detect_dark_boxes(frame)

    # OCR every detected dark box
    box_values = []
    for (x, y, w, h) in boxes:
        val = _ocr_box_value(frame, x, y, w, h)
        if val:
            box_values.append((x, y, w, h, val))

    if not box_values:
        return assigned

    # ── Per-field range validation ─────────────────────────────────────────────
    _RANGES = {
        "UF Volume":       (0,     9999),
        "UF Rate":         (0,     9999),
        "UF Goal":         (500,   9999),
        "Eff. Blood Flow": (100,   500),
        "Cum. Blood Vol.": (0,     200),
        "Kt/V":            (0.0,   3.0),
        "Plasma Na":       (120,   160),
        "Clearance":       (50,    350),
    }

    def _validated(fname: str, val: str) -> "str | None":
        rng = _RANGES.get(fname)
        if rng is None:
            return val  # time fields (UF Time Left, Goal in): accept as-is
        try:
            num = float(str(val).replace(":", ".").replace(",", ""))
            return val if rng[0] <= num <= rng[1] else None
        except (ValueError, TypeError):
            return val  # unparseable → pass through

    # ── Zone splitting ──────────────────────────────────────────────────────────
    # Use box centre-x for zone classification
    cx_threshold_left   = w_img * LEFT_SPLIT   # 38% of frame width
    cx_threshold_right  = w_img * RIGHT_SPLIT  # 68% of frame width

    right_boxes  = sorted(
        [b for b in box_values if (b[0] + b[2] / 2) >= cx_threshold_right],
        key=lambda b: b[1]   # top → bottom
    )
    middle_boxes = sorted(
        [b for b in box_values if cx_threshold_left <= (b[0] + b[2] / 2) < cx_threshold_right],
        key=lambda b: b[1]   # top → bottom
    )
    left_boxes   = sorted(
        [b for b in box_values if (b[0] + b[2] / 2) < cx_threshold_left],
        key=lambda b: b[1]   # top → bottom
    )

    # ── Right zone: UF Volume → Cum. Blood Vol. (top → bottom) ─────────────────
    # Confidence 0.92 — right column has the clearest separation from labels
    for i, (x, y, w, h, val) in enumerate(right_boxes):
        if i < len(RIGHT_COL_FIELDS):
            fname = RIGHT_COL_FIELDS[i]
            vv = _validated(fname, val)
            if vv:
                assigned[fname] = {
                    "value": vv,
                    "unit": FIELD_UNITS.get(fname, ""),
                    "confidence": 0.92
                }

    # ── Middle zone: Plasma Na (top), Clearance (bottom) ───────────────────────
    # Confidence 0.90 — 2-box middle column is unambiguous
    for i, (x, y, w, h, val) in enumerate(middle_boxes):
        if i < len(MIDDLE_FIELDS):
            fname = MIDDLE_FIELDS[i]
            vv = _validated(fname, val)
            if vv:
                assigned[fname] = {
                    "value": vv,
                    "unit": FIELD_UNITS.get(fname, ""),
                    "confidence": 0.90
                }

    # ── Left zone: Kt/V (top), Goal in (bottom) ────────────────────────────────
    # Confidence 0.88 — 2-box left column
    for i, (x, y, w, h, val) in enumerate(left_boxes):
        if i < len(LEFT_FIELDS):
            fname = LEFT_FIELDS[i]
            vv = _validated(fname, val)
            if vv:
                assigned[fname] = {
                    "value": vv,
                    "unit": FIELD_UNITS.get(fname, ""),
                    "confidence": 0.88
                }

    return assigned

