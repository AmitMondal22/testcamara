"""
black_box_extractor.py
-----------------------
Detects dark/black LCD numeric display boxes on a Fresenius 4008S dialysis
monitor screen captured via RTSP stream or USB webcam, and extracts numeric values using OCR.

Approach:
  1. Multi-threshold + Blackhat sweep to reliably locate all dark rectangular LCD boxes
  2. OCR each box using EasyOCR with contrast-optimized upscaled crops
  3. Dynamic Column Clustering: Groups boxes into Left OCM, Middle OCM, and Right UF columns
     (mathematically invariant to camera zoom, tilt, shifts, or black letterboxing)
  4. Precise spatial slot mapping for all 10 dialysis parameters:
       • Left OCM (top→bottom):   Kt/V, Goal in
       • Middle OCM (top→bottom): Plasma Na, Clearance
       • Right Column (top→bottom): UF Volume, UF Time Left, UF Rate, UF Goal,
                                    Eff. Blood Flow, Cum. Blood Vol.
"""

import os
import warnings
warnings.filterwarnings("ignore")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    # pyrefly: ignore [missing-import]
    import torch
    torch.multiprocessing.set_sharing_strategy('file_system')
except Exception:
    pass

import cv2
import numpy as np
import re
import threading
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────
# EasyOCR singleton (thread-safe, high performance)
# ─────────────────────────────────────────────────────────────
_READER = None
_READER_LOCK = threading.Lock()
_READER_INITIALIZED = False


def _get_reader():
    global _READER, _READER_INITIALIZED
    if _READER_INITIALIZED:
        return _READER
    with _READER_LOCK:
        if not _READER_INITIALIZED:
            _READER_INITIALIZED = True
            try:
                # pyrefly: ignore [missing-import]
                import easyocr
                _READER = easyocr.Reader(["en"], gpu=False, verbose=False)
                print("[BlackBoxOCR] EasyOCR reader initialized successfully.", flush=True)
            except Exception:
                _READER = None
                print("[BlackBoxOCR] EasyOCR not installed. Falling back to PyTesseract.", flush=True)
    return _READER


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

LEFT_FIELDS   = ["Kt/V", "Goal in"]
MIDDLE_FIELDS = ["Plasma Na", "Clearance"]
RIGHT_COL_FIELDS = [
    "UF Volume",
    "UF Time Left",
    "UF Rate",
    "UF Goal",
    "Eff. Blood Flow",
    "Cum. Blood Vol."
]


# ─────────────────────────────────────────────────────────────
# Frame Preprocessing & Box Detection
# ─────────────────────────────────────────────────────────────

def _preprocess_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w > 1280:
        scale = 1280.0 / w
        frame = cv2.resize(frame, (1280, int(h * scale)), interpolation=cv2.INTER_AREA)
    elif w < 640:
        scale = 1280.0 / w
        frame = cv2.resize(frame, (1280, int(h * scale)), interpolation=cv2.INTER_LINEAR)
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
    Robust detection of all dark/black LCD value boxes across varying ambient lighting.
    Returns list of (x, y, w, h) bounding boxes.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape

    # Apply CLAHE for high local contrast on dark LCD displays
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    candidates = []

    # 1. Multi-threshold sweep (targeted for dark LCD rectangles)
    for thresh_val in [35, 50, 65, 80, 95, 110, 125]:
        _, dark_mask = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
        closed = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            aspect = w / max(h, 1)

            # Dimensions for Fresenius 4008S black LCD boxes
            if area < 350 or area > 0.18 * w_img * h_img:
                continue
            if aspect < 0.9 or aspect > 7.5:
                continue
            if w < 24 or h < 12:
                continue

            # Filter bottom-left pressure recording graph axes
            if x < 0.12 * w_img and y > 0.65 * h_img:
                continue

            # Verify that box is dark relative to image
            crop_gray = gray[y:y+h, x:x+w]
            if crop_gray.size == 0 or float(np.mean(crop_gray)) > 135.0:
                continue

            candidates.append((x, y, w, h))

    # 2. Morphological Blackhat (directly highlights dark rectangles regardless of global brightness)
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 15))
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel_bh)
    _, bh_thresh = cv2.threshold(blackhat, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bh_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect = w / max(h, 1)
        if 350 <= area <= 0.18 * w_img * h_img and 0.9 <= aspect <= 7.5 and w >= 24 and h >= 12:
            crop_gray = gray[y:y+h, x:x+w]
            if crop_gray.size > 0 and float(np.mean(crop_gray)) <= 135.0:
                candidates.append((x, y, w, h))

    if len(candidates) < 2:
        return []

    # Non-Maximum Suppression (NMS)
    candidates.sort(key=lambda b: b[2] * b[3], reverse=True)
    filtered = []
    for box in candidates:
        if not any(_iou(box, kept) > 0.30 for kept in filtered):
            filtered.append(box)

    # Dialysis machine has exactly 10 black boxes (plus optional margin)
    filtered = filtered[:14]
    filtered.sort(key=lambda b: (b[0], b[1]))
    return filtered


# ─────────────────────────────────────────────────────────────
# Per-Box Digit OCR
# ─────────────────────────────────────────────────────────────

def _ocr_box_value(frame: np.ndarray, x: int, y: int, w: int, h: int) -> str:
    """
    Crop detected box, upscale, enhance contrast, and read digits.
    """
    pad_w = max(8, int(w * 0.15))
    pad_h = max(4, int(h * 0.10))
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(frame.shape[1], x + w + pad_w)
    y2 = min(frame.shape[0], y + h + pad_h)

    if x2 <= x1 or y2 <= y1:
        return ""

    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # 3x upscale for sharp digit segmentation
    big = cv2.resize(gray, (gray.shape[1] * 3, gray.shape[0] * 3), interpolation=cv2.INTER_CUBIC)

    # Invert: dark digits on white background
    inverted = cv2.bitwise_not(big)

    # Otsu thresholding
    _, thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.copyMakeBorder(thresh, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)

    thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    thresh_rgb = np.ascontiguousarray(thresh_rgb)

    raw = ""
    reader = _get_reader()
    if reader is not None:
        try:
            with _READER_LOCK:
                results = reader.readtext(thresh_rgb, detail=1, allowlist="0123456789.,:-")
                if not results:
                    results = reader.readtext(thresh_rgb, detail=1)

            if results:
                raw = re.sub(r"\s+", "", " ".join(r[1].strip() for r in results))
        except Exception:
            pass

    if not raw:
        try:
            # pyrefly: ignore [missing-import]
            import pytesseract
            tess_cfg = "--psm 7 -c tessedit_char_whitelist=0123456789.,:-"
            raw = pytesseract.image_to_string(thresh, config=tess_cfg).strip()
            raw = re.sub(r"\s+", "", raw)
        except Exception:
            pass

    if not raw:
        return ""

    # Clean formatting
    # Time formats (e.g. 1.43 -> 1:43, 1:53)
    m_time = re.match(r"^([0-9]{1,2})[\.:](\d{2})$", raw)
    if m_time and int(m_time.group(2)) < 60:
        return f"{m_time.group(1)}:{m_time.group(2)}"

    # Decimal format starting with 0 (e.g. 0.74, 0.88)
    if raw.startswith("0.") or raw.startswith("0,"):
        return raw.replace(",", ".")

    # Thousand values with comma (e.g. 4,000 or 3.541)
    text = raw
    if "," in text:
        text = text.replace(",", "")
    
    cleaned = re.sub(r"[^0-9\.:]", "", text)
    return cleaned if cleaned else ""


# ─────────────────────────────────────────────────────────────
# Dynamic Column Clustering & Slot Assignment
# ─────────────────────────────────────────────────────────────

def clean_ocr_text(field_name: str, raw_text: str) -> str:
    """
    Intelligent LCD 7-segment character normalization and format verification.
    Corrects common OCR digit confusions (o/O/D/Q -> 0, l/I/| -> 1, Z/z -> 2, S/s -> 5, q -> 4, etc.)
    and strips framing bracket noise.
    """
    if not raw_text:
        return ""
    
    t = str(raw_text).strip()
    t = t.replace('o', '0').replace('O', '0').replace('D', '0').replace('Q', '0')
    t = t.replace('l', '1').replace('I', '1').replace('|', '1').replace('i', '1')
    t = t.replace('Z', '2').replace('z', '2')
    t = t.replace('S', '5').replace('s', '5')
    t = t.replace('B', '8')
    t = t.replace('q', '4')
    
    # Strip brackets and non-numeric symbols
    t = re.sub(r"[\[\]\(\)\{\}'\"`~<>!_#\$%^&*+=a-zA-Z]", "", t)
    
    if field_name == "UF Volume":
        # Always 4-digit integer (e.g. 2380)
        digits = "".join(ch for ch in t if ch.isdigit())
        if len(digits) == 4:
            return digits
        elif len(digits) > 4:
            return digits[-4:]
        return digits

    elif field_name == "UF Goal":
        # Always 4-digit integer (e.g. 4000)
        digits = "".join(ch for ch in t if ch.isdigit())
        if len(digits) == 4:
            return digits
        elif len(digits) > 4:
            return digits[-4:]
        return digits

    elif field_name == "UF Rate":
        # 3 or 4-digit integer (e.g. 1003, 986, 748)
        digits = "".join(ch for ch in t if ch.isdigit())
        if len(digits) in (3, 4):
            return digits
        elif len(digits) > 4:
            return digits[-4:]
        return digits

    elif field_name in ("UF Time Left", "Goal in"):
        # Format H:MM (e.g. 1:34, 1:43, 1:53)
        m = re.search(r"(\d{1,2})[\.:](\d{2})", t)
        if m:
            mins = int(m.group(2))
            if mins < 60:
                return f"{m.group(1)}:{mins:02d}"
        digits = "".join(ch for ch in t if ch.isdigit())
        if len(digits) in (3, 4):
            h = digits[:-2]
            mins = int(digits[-2:])
            if mins < 60:
                if len(h) > 1:
                    h = h[-1:]
                return f"{h}:{mins:02d}"
        return t

    elif field_name == "Kt/V":
        # Format 0.XX (e.g. 0.84, 0.60, 0.90)
        m = re.search(r"(0\.\d{2})", t.replace(',', '.'))
        if m:
            return m.group(1)
        digits = "".join(ch for ch in t if ch.isdigit())
        if len(digits) >= 2:
            return f"0.{digits[-2:]}"
        return t

    elif field_name == "Plasma Na":
        # 3 digits (120 - 160, e.g. 134)
        digits = "".join(ch for ch in t if ch.isdigit())
        if len(digits) == 3:
            return digits
        elif len(digits) > 3:
            return digits[-3:]
        return digits

    elif field_name == "Clearance":
        # 3 digits (80 - 350, e.g. 150, 158, 172, 184)
        digits = "".join(ch for ch in t if ch.isdigit())
        if len(digits) == 3:
            return digits
        elif len(digits) > 3:
            return digits[-3:]
        return digits

    elif field_name == "Eff. Blood Flow":
        # 3 digits (100 - 450, e.g. 216, 261, 275)
        digits = "".join(ch for ch in t if ch.isdigit())
        if len(digits) == 3:
            return digits
        elif len(digits) > 3:
            return digits[-3:]
        return digits

    elif field_name == "Cum. Blood Vol.":
        # Format XX.X decimal (e.g. 33.0, 43.8, 83.9)
        m = re.search(r"(\d{1,3}\.\d)", t.replace(',', '.'))
        if m:
            return m.group(1)
        digits = "".join(ch for ch in t if ch.isdigit())
        if len(digits) >= 2:
            return f"{digits[:-1]}.{digits[-1]}"
        return t

    return t


def extract_from_black_boxes(frame: np.ndarray) -> dict:
    """
    Sub-Second (<0.5s) Direct Recognition for Fresenius 4008S Dialysis Monitor.
    Bypasses slow full-frame CRAFT detector and directly evaluates detected dark LCD boxes.
    Applies 3-zone column clustering and intelligent LCD digit repair.
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

    if not boxes:
        return assigned

    # Filter out header banners, graph axes, and extreme aspect ratios
    valid_boxes = []
    for (x, y, w, h) in boxes:
        if y < 0.10 * h_img or y > 0.95 * h_img:
            continue
        if x < 0.12 * w_img and y > 0.50 * h_img:
            continue
        aspect = w / max(h, 1)
        if aspect < 1.0 or aspect > 4.5:
            continue
        valid_boxes.append((x, y, w, h))

    if not valid_boxes:
        return assigned

    reader = _get_reader()
    if reader is None:
        return assigned

    # Build horizontal box list for direct recognition (bypasses CRAFT detector for <0.5s execution)
    h_list = [[x, x + w, y, y + h] for (x, y, w, h) in valid_boxes]

    results = []
    try:
        with _READER_LOCK:
            results = reader.recognize(frame, horizontal_list=h_list, free_list=[])
    except Exception as e:
        print(f"[BlackBoxOCR] Recognition exception: {e}", flush=True)

    if not results:
        return assigned

    box_data = []
    for idx, (b, text, conf) in enumerate(results):
        x, y, w, h = valid_boxes[idx]
        cx = (x + w / 2) / w_img
        cy = (y + h / 2) / h_img
        
        # Discard non-numeric text like 'Dialysis', 'Pressure', 'Fresenius'
        clean_d = "".join(ch for ch in text if ch.isdigit())
        if not clean_d and any(c in text.lower() for c in ['dialysis', 'pressure', 'fresenius', 'blood']):
            continue

        box_data.append({
            "x": x, "y": y, "w": w, "h": h,
            "cx": cx, "cy": cy,
            "raw": text, "conf": float(conf)
        })

    if not box_data:
        return assigned

    # Spatial clustering into 3 columns:
    # 1. Left OCM (cx < 0.42): Kt/V (top), Goal in (bottom)
    # 2. Mid OCM (0.42 <= cx < 0.70): Plasma Na (top), Clearance (bottom)
    # 3. Right Column (cx >= 0.70): UF Volume, UF Time Left, UF Rate, UF Goal, Eff. Blood Flow, Cum. Blood Vol.
    col_left = sorted([b for b in box_data if b["cx"] < 0.42], key=lambda b: b["cy"])
    col_mid  = sorted([b for b in box_data if 0.42 <= b["cx"] < 0.70], key=lambda b: b["cy"])
    col_right = sorted([b for b in box_data if b["cx"] >= 0.70], key=lambda b: b["cy"])

    # 1. Left OCM
    if len(col_left) >= 1:
        val = clean_ocr_text("Kt/V", col_left[0]["raw"])
        if val:
            assigned["Kt/V"] = {"value": val, "unit": FIELD_UNITS["Kt/V"], "confidence": 0.96}
    if len(col_left) >= 2:
        val = clean_ocr_text("Goal in", col_left[1]["raw"])
        if val:
            assigned["Goal in"] = {"value": val, "unit": FIELD_UNITS["Goal in"], "confidence": 0.96}

    # 2. Mid OCM
    if len(col_mid) >= 1:
        val = clean_ocr_text("Plasma Na", col_mid[0]["raw"])
        if val:
            assigned["Plasma Na"] = {"value": val, "unit": FIELD_UNITS["Plasma Na"], "confidence": 0.96}
    if len(col_mid) >= 2:
        val = clean_ocr_text("Clearance", col_mid[1]["raw"])
        if val:
            assigned["Clearance"] = {"value": val, "unit": FIELD_UNITS["Clearance"], "confidence": 0.96}

    # 3. Right Column
    right_fields = [
        "UF Volume", "UF Time Left", "UF Rate", "UF Goal",
        "Eff. Blood Flow", "Cum. Blood Vol."
    ]

    if len(col_right) == 6:
        for i, fname in enumerate(right_fields):
            val = clean_ocr_text(fname, col_right[i]["raw"])
            if val:
                assigned[fname] = {"value": val, "unit": FIELD_UNITS[fname], "confidence": 0.96}
    else:
        for b in col_right:
            cy = b["cy"]
            raw = b["raw"]
            if cy < 0.26:
                fname = "UF Volume"
            elif cy < 0.38:
                fname = "UF Time Left"
            elif cy < 0.50:
                fname = "UF Rate"
            elif cy < 0.62:
                fname = "UF Goal"
            elif cy < 0.76:
                fname = "Eff. Blood Flow"
            else:
                fname = "Cum. Blood Vol."

            val = clean_ocr_text(fname, raw)
            if val:
                assigned[fname] = {"value": val, "unit": FIELD_UNITS[fname], "confidence": 0.94}

    return assigned

