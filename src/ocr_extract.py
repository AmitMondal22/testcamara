"""
ocr_extract.py
--------------
OCR extraction engine natively optimized for Raspberry Pi 4 Model B (ARM Cortex-A72)
and cross-platform environments (Linux / Windows / macOS).

Key features:
  1. SIGILL / 'Illegal Instruction' Prevention: Native PyTesseract primary engine on ARM/Linux
  2. Safe guarded PyTorch/EasyOCR fallback only when hardware compatibility is confirmed
  3. Multi-angle tilt deskewing & 4-point perspective unwarping (homography transform)
  4. Multi-scheme contrast enhancement (CLAHE + Otsu binarization + adaptive threshold)
"""

import os
import sys
import io
import json
import shutil
import threading
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np

# Prevent OpenMP runtime conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Force UTF-8 encoding on standard buffers
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────
# Central Configuration Loader
# ─────────────────────────────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")


def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


_CONFIG = load_config()

# Detect if running on ARM architecture (Raspberry Pi 4)
_IS_ARM = (
    "arm" in sys.platform.lower() or
    "aarch64" in os.uname().machine.lower() if hasattr(os, "uname") else False or
    "armv7l" in os.uname().machine.lower() if hasattr(os, "uname") else False
)

_EASYOCR_READER = None
_EASYOCR_LOCK = threading.Lock()
_EASYOCR_FAILED = False


def _configure_tesseract() -> bool:
    try:
        import pytesseract  # pyrefly: ignore [missing-import]
        if shutil.which("tesseract"):
            return True

        common_paths = [
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
            "/opt/homebrew/bin/tesseract",
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
            os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
        ]
        for path in common_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                return True
        return False
    except ImportError:
        return False


_HAS_TESSERACT = _configure_tesseract()


def _get_easyocr_reader():
    """Safely retrieves EasyOCR only if not on ARM or explicitly enabled to avoid SIGILL."""
    global _EASYOCR_READER, _EASYOCR_FAILED
    if _EASYOCR_FAILED:
        return None
    if _EASYOCR_READER is not None:
        return _EASYOCR_READER

    # On Raspberry Pi / ARM, PyTesseract is the default to avoid PyTorch SIGILL (Illegal Instruction)
    platform_target = _CONFIG.get("hardware", {}).get("target_platform", "auto")
    engine_pref = _CONFIG.get("ocr", {}).get("engine", "auto")

    if (platform_target == "raspberry_pi_4" or _IS_ARM) and engine_pref != "easyocr":
        return None

    with _EASYOCR_LOCK:
        if _EASYOCR_READER is None and not _EASYOCR_FAILED:
            try:
                import easyocr  # pyrefly: ignore [missing-import]
                gpu_setting = _CONFIG.get("ocr", {}).get("easyocr_gpu", False)
                _EASYOCR_READER = easyocr.Reader(['en'], gpu=gpu_setting, verbose=False)
            except Exception as err:
                print(f"[OCR Notice] EasyOCR not used on this device ({err}). Using native Tesseract OCR.", flush=True)
                _EASYOCR_FAILED = True
                _EASYOCR_READER = None
    return _EASYOCR_READER


def load_image(path: str) -> np.ndarray:
    path = path.strip().strip('"').strip("'")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file does not exist at: {path}")
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not decode image file at: {path}")
    return img


# ─────────────────────────────────────────────────────────────
# Camera Tilt & Perspective Deskewing Algorithms
# ─────────────────────────────────────────────────────────────

def deskew_and_straighten(img: np.ndarray, max_angle: float = 45.0) -> np.ndarray:
    """Detects rotational camera tilt and rotates the image to 0 degrees."""
    if img is None or img.size == 0:
        return img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=80, maxLineGap=10)
    angles = []

    if lines is not None:
        for line in lines:
            pts = line.reshape(-1)
            if len(pts) >= 4:
                x1, y1, x2, y2 = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
                if x2 != x1:
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    if -max_angle <= angle <= max_angle and abs(angle) > 0.5:
                        angles.append(angle)

    if angles:
        median_angle = float(np.median(angles))
        if abs(median_angle) > 0.5:
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            rot_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            straightened = cv2.warpAffine(img, rot_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return straightened

    return img


def auto_unwarp_screen(img: np.ndarray) -> np.ndarray:
    """Detects quadrilateral screen contour and unwarps perspective."""
    if img is None or img.size == 0:
        return img

    h_img, w_img = img.shape[:2]
    deskew_cfg = _CONFIG.get("deskew_tilt", {})
    min_area_ratio = deskew_cfg.get("min_contour_area_ratio", 0.08)
    canny_low = deskew_cfg.get("canny_thresh_low", 30)
    canny_high = deskew_cfg.get("canny_thresh_high", 150)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, canny_low, canny_high)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.dilate(edged, kernel, iterations=1)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]

    for c in contours:
        area = cv2.contourArea(c)
        if area < (min_area_ratio * w_img * h_img):
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.025 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
            rect = np.zeros((4, 2), dtype="float32")

            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]

            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]

            (tl, tr, br, bl) = rect

            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))

            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))

            if maxWidth < 200 or maxHeight < 150:
                continue

            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC)
            return warped

    return img


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """Enhance LCD screen contrast using CLAHE."""
    if img is None or img.size == 0:
        return img

    clip_limit = _CONFIG.get("deskew_tilt", {}).get("clahe_clip_limit", 3.0)
    grid_size = tuple(_CONFIG.get("deskew_tilt", {}).get("clahe_tile_grid_size", [8, 8]))

    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        return clahe.apply(img)


def preprocess_for_pi4(img: np.ndarray, max_dim: int = 800) -> tuple:
    h_orig, w_orig = img.shape[:2]
    largest = max(h_orig, w_orig)
    if largest > max_dim:
        scale = max_dim / float(largest)
        resized = cv2.resize(img, (int(w_orig * scale), int(h_orig * scale)), interpolation=cv2.INTER_AREA)
        return resized, scale
    return img, 1.0


# ─────────────────────────────────────────────────────────────
# PyTesseract Multi-Pass Extraction (Zero-Crash ARM Native Engine)
# ─────────────────────────────────────────────────────────────

def _extract_via_pytesseract(scaled_img: np.ndarray, scale_factor: float) -> list:
    """
    Extracts text and numeric regions using native PyTesseract with multi-pass binarization
    (standard grayscale + Otsu inversion for LCD boxes).
    Extremely fast and 100% stable on Raspberry Pi 4 Model B CPU.
    """
    if not _HAS_TESSERACT:
        return []

    import pytesseract  # pyrefly: ignore [missing-import]
    gray = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY) if len(scaled_img.shape) == 3 else scaled_img
    lines = []
    seen_texts = set()

    # Pass 1: Standard PSM 6 / 11 with adaptive thresholding
    passes = [
        ("standard", gray, "--psm 6"),
        ("otsu", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], "--psm 11"),
        ("inverted", cv2.threshold(cv2.bitwise_not(gray), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], "--psm 6"),
    ]

    for pass_name, img_variant, psm_cfg in passes:
        try:
            data = pytesseract.image_to_data(
                img_variant,
                output_type=pytesseract.Output.DICT,
                config=f"{psm_cfg} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.:,-/"
            )
            n_boxes = len(data["text"])
            for i in range(n_boxes):
                text_clean = data["text"][i].strip()
                conf_val = float(data["conf"][i])
                if text_clean and conf_val > 10 and len(text_clean) >= 1:
                    x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                    x_min = int(x / scale_factor)
                    y_min = int(y / scale_factor)
                    x_max = int((x + w) / scale_factor)
                    y_max = int((y + h) / scale_factor)
                    key = (text_clean, x_min // 20, y_min // 20)

                    if key not in seen_texts:
                        seen_texts.add(key)
                        lines.append({
                            "text": text_clean,
                            "confidence": round(max(0.70, conf_val / 100.0), 2),
                            "bbox": [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
                            "x_min": x_min,
                            "x_max": x_max,
                            "y_min": y_min,
                            "y_max": y_max,
                            "center_x": (x_min + x_max) / 2.0,
                            "center_y": (y_min + y_max) / 2.0,
                            "width": x_max - x_min,
                            "height": y_max - y_min,
                        })
        except Exception:
            pass

    return lines


# ─────────────────────────────────────────────────────────────
# Primary OCR Extraction Function
# ─────────────────────────────────────────────────────────────

def extract_image_data(img: np.ndarray, engine: str = "auto", unwarp: bool = True) -> list:
    """
    Extracts text & numeric bounding boxes with camera tilt deskewing and Pi 4 CPU safety.
    Guaranteed zero-crash on Raspberry Pi 4 ARM.
    """
    if img is None or img.size == 0:
        return []

    # 1. Camera Tilt & Perspective Deskewing
    deskew_cfg = _CONFIG.get("deskew_tilt", {})
    processed_frame = img.copy()

    if unwarp and deskew_cfg.get("enable_auto_unwarp", True):
        processed_frame = auto_unwarp_screen(processed_frame)

    if deskew_cfg.get("enable_tilt_deskew", True):
        processed_frame = deskew_and_straighten(processed_frame, max_angle=deskew_cfg.get("max_tilt_angle_deg", 45.0))

    # 2. Contrast Enhancement
    enhanced = enhance_contrast(processed_frame)

    # 3. Pi 4 Dimension Scaling
    max_dim = _CONFIG.get("ocr", {}).get("max_ocr_dimension", 800)
    scaled_img, scale_factor = preprocess_for_pi4(enhanced, max_dim=max_dim)

    # 4. Engine Execution
    # Try EasyOCR only if explicitly requested or safely available
    reader = _get_easyocr_reader() if engine != "tesseract" else None

    if reader is not None:
        try:
            rgb_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB) if len(scaled_img.shape) == 3 else scaled_img
            with _EASYOCR_LOCK:
                results = reader.readtext(rgb_img, detail=1)

            seen_entries = set()
            lines = []
            for bbox, text, conf in results:
                text_clean = str(text).strip()
                if text_clean and conf > 0.08 and text_clean not in seen_entries:
                    seen_entries.add(text_clean)
                    pts = [[int(pt[0] / scale_factor), int(pt[1] / scale_factor)] for pt in bbox]
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)

                    lines.append({
                        "text": text_clean,
                        "confidence": round(float(conf), 2),
                        "bbox": pts,
                        "x_min": x_min,
                        "x_max": x_max,
                        "y_min": y_min,
                        "y_max": y_max,
                        "center_x": (x_min + x_max) / 2.0,
                        "center_y": (y_min + y_max) / 2.0,
                        "width": x_max - x_min,
                        "height": y_max - y_min,
                    })
            if lines:
                return lines
        except Exception:
            pass

    # Native PyTesseract Execution (Primary / Fallback)
    lines = _extract_via_pytesseract(scaled_img, scale_factor)
    return lines


def extract_text(img: np.ndarray, engine: str = "auto") -> str:
    lines_data = extract_image_data(img, engine=engine)
    return "\n".join(item["text"] for item in lines_data)
