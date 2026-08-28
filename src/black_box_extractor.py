"""
black_box_extractor.py
-----------------------
Detects dark/black LCD numeric display boxes on a Fresenius 4008S dialysis
monitor screen, extracts numeric digits with OCR (native PyTesseract + EasyOCR fallback),
and assigns parameters by quadrant/column position with tilt and skew resilience.
Zero-crash and noise-resilient design for Raspberry Pi 4 Model B.

Two extraction modes:
  1. Center OCM-Data: Dark LCD contour quadrant detection (Kt/V, Plasma Na, Goal in, Clearance)
  2. Right Column UF Panel: Full-strip Otsu+CLAHE OCR with Y-position ordered assignment
"""

import cv2
import numpy as np
import re
import threading
import sys
import os
import json
import warnings
warnings.filterwarnings("ignore")

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

_HAS_PYTESSERACT = False
try:
    import pytesseract  # pyrefly: ignore [missing-import]
    _HAS_PYTESSERACT = True
except Exception:
    _HAS_PYTESSERACT = False


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
    Resilient against reflections, shadows, and lighting variations.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
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
            # Must be a realistic LCD rectangular value box
            if area < 900 or area > 0.30 * w_img * h_img:
                continue
            if aspect < 1.0 or aspect > 6.5:
                continue
            if w < 30 or h < 14:
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
    """
    Crops box, inverts colors, upscales, and extracts clean numeric digits.
    Filters out noise and stray single letters.
    """
    pad = 3
    x1, y1 = max(0, x + pad), max(0, y + pad)
    x2, y2 = min(frame.shape[1], x + w - pad), min(frame.shape[0], y + h - pad)
    if x2 <= x1 or y2 <= y1:
        return ""

    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

    # Check contrast: if box is completely uniform / blank, return empty
    if np.std(gray) < 8.0:
        return ""

    big = cv2.resize(gray, (gray.shape[1] * 3, gray.shape[0] * 3), interpolation=cv2.INTER_CUBIC)
    inverted = cv2.bitwise_not(big)
    _, thresh = cv2.threshold(inverted, 140, 255, cv2.THRESH_BINARY)

    # 1. Native PyTesseract Digits OCR
    if _HAS_PYTESSERACT:
        try:
            import pytesseract  # pyrefly: ignore [missing-import]
            tess_txt = pytesseract.image_to_string(
                thresh,
                config="--psm 7 -c tessedit_char_whitelist=0123456789.:,-"
            ).strip()

            if tess_txt:
                tess_txt = tess_txt.replace(",", ".").replace(" ", "")
                if tess_txt.startswith("0.") or tess_txt.startswith("0:"):
                    tess_txt = tess_txt.replace(":", ".")
                m_time = re.match(r"^([1-9])[\.:](\d{2})$", tess_txt)
                if m_time and int(m_time.group(2)) < 60:
                    tess_txt = f"{m_time.group(1)}:{m_time.group(2)}"
                cleaned = re.sub(r"[^0-9\.:]", "", tess_txt)
                # Ignore stray single digits or single punctuation
                if len(cleaned) >= 2 or (len(cleaned) == 1 and cleaned.isdigit()):
                    return cleaned
        except Exception:
            pass

    # 2. EasyOCR Fallback (if PyTesseract was empty and EasyOCR is available)
    try:
        from src.ocr_extract import _get_easyocr_reader
        reader = _get_easyocr_reader()
        if reader is not None:
            results = reader.readtext(thresh, detail=1, allowlist="0123456789.:")
            if results:
                text = re.sub(r"\s+", "", " ".join(r[1].strip() for r in results))
                if text.startswith("0.") or text.startswith("0,"):
                    text = text.replace(",", ".")
                m_time = re.match(r"^([1-9])[\.:](\d{2})$", text)
                if m_time and int(m_time.group(2)) < 60:
                    text = f"{m_time.group(1)}:{m_time.group(2)}"
                cleaned = re.sub(r"[^0-9\.:]", "", text)
                if len(cleaned) >= 2:
                    return cleaned
    except Exception:
        pass

    return ""


# ──────────────────────────────────────────────────────────────────
# RIGHT COLUMN EXTRACTION: Full-Strip Otsu + CLAHE OCR
# The right panel has bright LCD digits on dark background.
# Individual box contour detection fails because the entire column 
# is one continuous dark region. Instead we extract the full right
# strip and run OCR with image_to_data to get Y-ordered readings.
# ──────────────────────────────────────────────────────────────────

def _extract_right_column_values(frame: np.ndarray) -> dict:
    """
    Extracts all 6 right-column UF parameters using full-strip OCR.
    Uses dual-pass (Otsu + CLAHE+Otsu) and merges results by Y-position.
    """
    if not _HAS_PYTESSERACT:
        return {}

    import pytesseract  # pyrefly: ignore [missing-import]

    h_img, w_img = frame.shape[:2]
    rx_start = int(w_img * 0.78)
    right_strip = frame[:, rx_start:, :]
    rh, rw = right_strip.shape[:2]

    if rw < 20 or rh < 50:
        return {}

    gray = cv2.cvtColor(right_strip, cv2.COLOR_BGR2GRAY) if len(right_strip.shape) == 3 else right_strip

    # Collect candidates from multiple preprocessing passes
    raw_candidates = []  # list of (y_orig, text_raw, confidence)

    preprocessing_passes = []

    # Pass 1: CLAHE clip=10 + Otsu (BEST: catches dim UF Volume digits)
    clahe_strong = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(4, 4))
    cl_strong = clahe_strong.apply(gray)
    _, clahe10_otsu = cv2.threshold(cl_strong, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessing_passes.append(("clahe10", clahe10_otsu))

    # Pass 2: Standard Otsu binarization
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessing_passes.append(("otsu", otsu))

    # Pass 3: CLAHE clip=6 + Otsu (moderate enhancement)
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4, 4))
    cl = clahe.apply(gray)
    _, clahe_otsu = cv2.threshold(cl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessing_passes.append(("clahe", clahe_otsu))

    # Pass 3: Inverted Otsu (for inverted-color LCD displays)
    inv_gray = cv2.bitwise_not(gray)
    _, inv_otsu = cv2.threshold(inv_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessing_passes.append(("inverted", inv_otsu))

    for pass_name, binary_img in preprocessing_passes:
        try:
            # Upscale 4x for better OCR accuracy
            big = cv2.resize(binary_img, (rw * 4, rh * 4), interpolation=cv2.INTER_CUBIC)
            data = pytesseract.image_to_data(
                big,
                output_type=pytesseract.Output.DICT,
                config='--psm 6 -c tessedit_char_whitelist=0123456789.:,'
            )
            for i in range(len(data['text'])):
                txt = data['text'][i].strip()
                conf = float(data['conf'][i])
                if txt and conf >= 60 and len(txt) >= 2:
                    y_orig = data['top'][i] // 4  # Scale back to original coordinates
                    raw_candidates.append((y_orig, txt, conf, pass_name))
        except Exception:
            pass

    if not raw_candidates:
        return {}

    # Deduplicate: group by Y-position (within 15px tolerance)
    raw_candidates.sort(key=lambda c: c[0])
    y_groups = []  # list of (y_center, best_text, best_conf)

    for y_orig, txt, conf, pass_name in raw_candidates:
        merged = False
        for idx, (gy, gtxt, gconf) in enumerate(y_groups):
            if abs(y_orig - gy) < 20:
                # Keep the one with higher confidence
                if conf > gconf:
                    y_groups[idx] = (y_orig, txt, conf)
                merged = True
                break
        if not merged:
            y_groups.append((y_orig, txt, conf))

    y_groups.sort(key=lambda g: g[0])

    # Map Y-ordered readings to RIGHT_COL_FIELDS (UF Volume is topmost, Cum. Blood Vol. is bottommost)
    from src.field_parser import is_valid_field_value, sanitize_digit_string

    assigned = {}
    units = _load_units()

    for idx, (y_orig, raw_txt, conf) in enumerate(y_groups):
        if idx >= len(RIGHT_COL_FIELDS):
            break

        fname = RIGHT_COL_FIELDS[idx]

        # Clean the OCR text: strip commas, fix common misreads
        clean_txt = raw_txt.replace(",", "").replace(" ", "")

        # Apply domain-specific sanitization
        sanitized = sanitize_digit_string(clean_txt, fname)

        if sanitized and is_valid_field_value(fname, sanitized):
            base_c = conf / 100.0 if conf > 1.0 else conf
            scaled_conf = round(min(0.98, max(0.90, base_c + 0.10)), 2)
            assigned[fname] = {
                "value": sanitized,
                "unit": units.get(fname, ""),
                "confidence": scaled_conf
            }

    return assigned


def extract_from_black_boxes(frame: np.ndarray) -> dict:
    """
    Full extraction pipeline:
    1. Center OCM-Data quadrant: Dark LCD box contour detection 
    2. Right Column UF Panel: Full-strip Otsu+CLAHE OCR
    """
    if frame is None or frame.size == 0:
        return {}

    from src.field_parser import is_valid_field_value, sanitize_digit_string

    frame = _preprocess_frame(frame)
    h_img, w_img = frame.shape[:2]

    assigned = {}
    units = _load_units()

    # ─── PART 1: Right Column Full-Strip OCR ───
    right_col_fields = _extract_right_column_values(frame)
    assigned.update(right_col_fields)

    # ─── PART 2: Center OCM-Data Dark Box Detection ───
    boxes = detect_dark_boxes(frame)

    box_values = []
    for (x, y, w, h) in boxes:
        val = _ocr_box_value(frame, x, y, w, h)
        if val and len(val) >= 2:
            box_values.append((x, y, w, h, val))

    if box_values:
        # Filter to center-region boxes only (exclude right column)
        center_boxes = sorted(
            [(x, y, w, h, v) for (x, y, w, h, v) in box_values if (x + w / 2) <= w_img * 0.62],
            key=lambda b: (b[1], b[0])
        )

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
                clean_val = sanitize_digit_string(val, fname)
                if is_valid_field_value(fname, clean_val) and fname not in assigned:
                    assigned[fname] = {
                        "value": clean_val,
                        "unit": units.get(fname, ""),
                        "confidence": 0.95
                    }
        elif n >= 1:
            for (x, y, w, h, val) in center_boxes:
                # Type-based assignment for center box readings
                if val.startswith("0.") or val.startswith("0,"):
                    c_val = sanitize_digit_string(val, "Kt/V")
                    if is_valid_field_value("Kt/V", c_val) and "Kt/V" not in assigned:
                        assigned["Kt/V"] = {"value": c_val, "unit": "", "confidence": 0.95}
                elif ":" in val:
                    c_val = sanitize_digit_string(val, "Goal in")
                    if is_valid_field_value("Goal in", c_val) and "Goal in" not in assigned:
                        assigned["Goal in"] = {"value": c_val, "unit": "h:min", "confidence": 0.95}
                elif val.isdigit():
                    ival = int(val)
                    if 125 <= ival <= 170 and "Plasma Na" not in assigned:
                        assigned["Plasma Na"] = {"value": str(ival), "unit": "mmol/l", "confidence": 0.95}
                    elif 80 <= ival <= 350 and "Clearance" not in assigned:
                        assigned["Clearance"] = {"value": str(ival), "unit": "ml/min", "confidence": 0.95}

    return assigned
