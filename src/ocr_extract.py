"""
ocr_extract.py
--------------
Dual-engine OCR processing module for image text & data extraction.
Supports:
  1. EasyOCR (PyTorch-based, runs completely in Python on Windows/Linux/macOS)
  2. PyTesseract (with automatic Windows installation path discovery)

Auto-selects the available engine so execution never crashes due to missing binaries.
"""

import os
import sys
import io
import shutil

# Ensure Windows OpenMP DLL compatibility for PyTorch/EasyOCR before OpenCV initialization
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
try:
     # pyrefly: ignore [missing-import]
    import torch
    # Set PyTorch sharing strategy to file_system to prevent /dev/shm shared memory Bus Errors (SIGBUS)
    torch.multiprocessing.set_sharing_strategy('file_system')
except Exception:
    pass

import cv2
import numpy as np

# Force UTF-8 encoding on standard output/error buffers to prevent Windows CP1252 charmap errors
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

import threading

# Global EasyOCR reader cache & thread safety lock
_EASYOCR_READER = None
_EASYOCR_LOCK = threading.Lock()


def _configure_tesseract():
    try:
         # pyrefly: ignore [missing-import]
        import pytesseract
        if shutil.which("tesseract"):
            return True

        common_paths = [
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


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as a BGR numpy array."""
    path = path.strip().strip('"').strip("'")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file does not exist at: {path}")
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not decode image file at: {path}")
    return img


def preprocess(img: np.ndarray, upscale: bool = True) -> np.ndarray:
    """
    Clean up the image so OCR engines can read text/digits cleanly.
    Optimized for IMX477 IR-CUT Infrared camera input and LCD monitor feeds.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    if upscale:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Adaptive CLAHE contrast enhancement for Infrared light variations
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    denoised = cv2.fastNlMeansDenoising(gray, h=8)

    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15
    )
    return thresh


def _get_easyocr_reader():
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        with _EASYOCR_LOCK:
            if _EASYOCR_READER is None:
                # pyrefly: ignore [missing-import]
                import easyocr
                _EASYOCR_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _EASYOCR_READER


def auto_unwarp_screen(img: np.ndarray) -> np.ndarray:
    """
    Detects rectangular display screen contours and un-warps perspective
    so tilted webcam images are flattened and deskewed before OCR.
    """
    if img is None or img.size == 0:
        return img

    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > (0.15 * w_img * h_img):
            pts = approx.reshape(4, 2)
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
            warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
            return warped

    return img


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """Enhance contrast for LCD screens using CLAHE (supports BGR & Grayscale/IR)."""
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        return clahe.apply(img)


def extract_image_data(img: np.ndarray, engine: str = "auto", unwarp: bool = False) -> list:
    """
    Scrapes text & numeric entries with double-pass OCR (Raw RGB + Otsu Binarized) for LCD displays.
    Optimized for high-speed sub-second execution on Raspberry Pi 4 and PC platforms.
    """
    if img is None or img.size == 0:
        return []

    lines = []
    h_orig, w_orig = img.shape[:2]
    max_dim = max(h_orig, w_orig)
    if max_dim > 1400:
        scale_factor = 1400.0 / max_dim
        scaled_img = cv2.resize(img, (int(w_orig * scale_factor), int(h_orig * scale_factor)), interpolation=cv2.INTER_AREA)
    else:
        scale_factor = 1.0
        scaled_img = img

    if unwarp:
        scaled_img = auto_unwarp_screen(scaled_img)

    use_tesseract = (engine == "tesseract") or (engine == "auto" and _HAS_TESSERACT)

    if use_tesseract:
        try:
            # pyrefly: ignore [missing-import]
            import pytesseract
            gray_proc = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY) if len(scaled_img.shape) == 3 else scaled_img
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray_proc = clahe.apply(gray_proc)

            tess_data = pytesseract.image_to_data(
                gray_proc,
                output_type=pytesseract.Output.DICT,
                config="--psm 6"
            )
            n_boxes = len(tess_data["text"])
            for i in range(n_boxes):
                text_clean = tess_data["text"][i].strip()
                conf = float(tess_data["conf"][i])
                if text_clean and conf > 10:
                    x = int(tess_data["left"][i] / scale_factor)
                    y = int(tess_data["top"][i] / scale_factor)
                    w = int(tess_data["width"][i] / scale_factor)
                    h = int(tess_data["height"][i] / scale_factor)
                    lines.append({
                        "text": text_clean,
                        "confidence": round(conf / 100.0, 2),
                        "bbox": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                        "x_min": x,
                        "x_max": x + w,
                        "y_min": y,
                        "y_max": y + h,
                        "center_x": x + w / 2.0,
                        "center_y": y + h / 2.0,
                        "width": w,
                        "height": h,
                    })
            if lines:
                return lines
        except Exception as err:
            pass

    # Fallback to EasyOCR
    try:
        reader = _get_easyocr_reader()
        rgb_raw = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB) if (len(scaled_img.shape) == 3 and scaled_img.shape[2] == 3) else scaled_img
        rgb_raw = np.ascontiguousarray(rgb_raw)
        results = reader.readtext(rgb_raw, canvas_size=960, detail=1)

        seen_entries = set()
        for bbox, text, conf in results:
            text_clean = str(text).strip()
            if text_clean and conf > 0.10 and text_clean not in seen_entries:
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
        return lines
    except Exception as err:
        print(f"EasyOCR Note: {err}")

    return lines

    return lines


def extract_text(img: np.ndarray, debug_save_path: str = None, engine: str = "auto") -> str:
    if debug_save_path:
        processed = preprocess(img)
        cv2.imwrite(debug_save_path, processed)

    lines_data = extract_image_data(img, engine=engine)
    raw_text = "\n".join(item["text"] for item in lines_data)
    return raw_text
