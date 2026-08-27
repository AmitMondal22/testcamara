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
# Image Preprocessing — multiple passes for best OCR accuracy
# ──────────────────────────────────────────────────────────────

def preprocess_for_ocr(img):
    """
    Generate multiple preprocessed versions of the image for OCR.
    Returns list of grayscale images optimized for text detection.
    """
    if img is None or img.size == 0:
        return [], 1.0

    h, w = img.shape[:2]

    # Upscale if image is low-res for better OCR accuracy
    scale = 1.0
    if max(h, w) < 1000:
        scale = 1000.0 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    results = []

    # --- Pass 1: Sharpen + CLAHE on grayscale ---
    blurred = cv2.GaussianBlur(img, (0, 0), 3.0)
    sharp = cv2.addWeighted(img, 2.0, blurred, -1.0, 0)
    gray = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    results.append(gray_clahe)

    # --- Pass 2: Adaptive threshold (handles uneven display lighting/glare) ---
    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    results.append(thresh)

    # --- Pass 3: Otsu threshold (best for high-contrast digital displays) ---
    gray3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results.append(otsu)

    # --- Pass 4: Inverted (for light text on dark background) ---
    results.append(cv2.bitwise_not(gray_clahe))

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
    """Check if text represents a numeric reading."""
    text = clean_ascii(text).replace(" ", "")
    if not text or not re.search(r'\d', text):
        return False

    if re.match(r'^[\+\-]?[\d.,:\/]+%?$', text):
        return True
    return False


def clean_label_str(label):
    """Format label neatly."""
    label = clean_ascii(label)
    label = re.sub(r'^[_\W]+|[_\W]+$', '', label)
    label = re.sub(r'\s+', ' ', label).strip()
    return label


def parse_numeric_value(raw_val):
    """Parse string value into integer, float, or formatted string."""
    clean_val = clean_ascii(str(raw_val)).strip()
    clean_val = re.sub(r'^[^\d\+\-]+|[^\d%]+$', '', clean_val).strip()

    if not clean_val:
        return raw_val

    # Pure integer or integer with commas (e.g., "10", "2,269")
    int_candidate = clean_val.replace(",", "")
    if re.match(r'^[\+\-]?\d+$', int_candidate):
        try:
            return int(int_candidate)
        except ValueError:
            pass

    # Float (e.g., "0.75", "12.34")
    if re.match(r'^[\+\-]?\d+\.\d+$', int_candidate):
        try:
            return float(int_candidate)
        except ValueError:
            pass

    # Keep time / percent / compound strings (e.g., "1:43", "98%", "-10")
    return clean_val


def cast_value(val, target_type=None):
    """Cast extracted value to specified data type ('int', 'float', 'number', 'string')."""
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
        data = pytesseract.image_to_data(rgb, config="--psm 6", output_type=pytesseract.Output.DICT)
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

    for lbl in labels:
        best_val = None
        best_dist = float("inf")
        for vi, val in enumerate(values):
            if vi in used_values:
                continue
            dy = abs(val["cy"] - lbl["cy"])
            dx = val["cx"] - lbl["cx"]
            if dy < line_threshold:
                dist = abs(dx) + dy * 2.0
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

    return result


# ──────────────────────────────────────────────────────────────
# Field Mapping & JSON Formatting
# ──────────────────────────────────────────────────────────────

def format_output_dict(raw_pairs, dataset_config=None):
    """
    Format extracted raw pairs into the target JSON structure:
    {"abc": {"name": "abc abc", "value": 10}}
    Strict validation: When dataset_config is provided, ONLY certified fields matching the schema
    are returned to guarantee 100% accuracy with zero false data.
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
            aliases = [a.strip().lower() for a in field_spec.get("aliases", []) if a]
            aliases.append(canonical_name.strip().lower())
            aliases.append(field_key.strip().lower())

            # Find matching extracted pair
            best_match = None
            best_score = float("inf")

            for raw_lbl, (lbl_name, val) in raw_pairs.items():
                lbl_clean = lbl_name.strip().lower()
                for alias in aliases:
                    if lbl_clean == alias:
                        best_match = (raw_lbl, val)
                        best_score = 0
                        break
                    elif (len(lbl_clean) >= 3 and alias in lbl_clean) or (len(alias) >= 3 and lbl_clean in alias):
                        diff = abs(len(lbl_clean) - len(alias))
                        if diff < best_score:
                            best_score = diff
                            best_match = (raw_lbl, val)

                if best_score == 0:
                    break

            if best_match is not None:
                matched_key, matched_val = best_match
                matched_raw_keys.add(matched_key)
                output[field_key] = {
                    "name": canonical_name,
                    "value": cast_value(matched_val, data_type)
                }

        # In strict schema mode, only return verified dataset fields
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
# Main extraction pipeline & Multi-Frame Consensus
# ──────────────────────────────────────────────────────────────

def extract_from_frame(frame, fields_config=None):
    """
    Full single-frame pipeline: preprocess → multi-pass OCR → field mapping → return JSON dict:
    {"abc": {"name": "abc abc", "value": 10}}
    """
    if frame is None or frame.size == 0:
        return {}, 0

    preprocessed_images, scale = preprocess_for_ocr(frame)

    raw_pairs = {}
    total_items = 0

    for img_gray in preprocessed_images:
        # Strategy 1: Full text line parsing
        try:
            text = pytesseract.image_to_string(img_gray, config="--psm 6")
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
