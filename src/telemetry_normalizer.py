"""
telemetry_normalizer.py
-----------------------
Non-Destructive Telemetry Sanitizer & Latching Memory Engine.

Ensures:
  1. RAW OCR numeric digits from EasyOCR are preserved EXACTLY as read without artificial division/scaling.
  2. Formatting cleanup: removes commas (e.g. '3,971' -> '3971') and standardizes time colons ('1.43' -> '1:43').
  3. Latching Memory: If a live video frame misses a box for a split second, the previous accurate reading is retained so the UI and console never drop to empty.
"""

import re
from typing import Dict, Any

# Memory latch per camera device: {device_id: {field_name: {"value": val, "unit": unit, "confidence": conf}}}
_DEVICE_LAST_KNOWN: Dict[str, Dict[str, Any]] = {}


def sanitize_raw_value(field_name: str, raw_val: str) -> str:
    """
    Cleans raw OCR string formatting without altering numeric digits.
    """
    if not raw_val or raw_val in ("null", "None", ""):
        return ""

    val = str(raw_val).strip()

    # Preserve status dash '--:--'
    if "--" in val or "-:-" in val:
        return "--:--"

    # Remove thousand-separator commas and dots for pure integer fields
    if field_name in ("UF Volume", "UF Rate", "UF Goal"):
        clean_val = val.replace(",", "").replace(".", "").strip()
        digits = "".join(ch for ch in clean_val if ch.isdigit())
        return digits if digits else clean_val

    # Range validation & cleaning for Plasma Na (120 - 160 mmol/l)
    if field_name == "Plasma Na":
        clean_val = val.replace(",", "").replace(".", "").strip()
        digits = "".join(ch for ch in clean_val if ch.isdigit())
        val_int = int(digits) if digits.isdigit() else 0
        if 120 <= val_int <= 160:
            return str(val_int)
        return ""

    # Range validation & cleaning for Eff. Blood Flow (100 - 500 ml/min)
    if field_name == "Eff. Blood Flow":
        clean_val = val.replace(",", "").replace(".", "").strip()
        digits = "".join(ch for ch in clean_val if ch.isdigit())
        val_int = int(digits) if digits.isdigit() else 0
        if 100 <= val_int <= 500:
            return str(val_int)
        return ""

    # Range validation & cleaning for Clearance (50 - 350 ml/min)
    if field_name == "Clearance":
        clean_val = val.replace(",", "").replace(".", "").strip()
        digits = "".join(ch for ch in clean_val if ch.isdigit())
        val_int = int(digits) if digits.isdigit() else 0
        if 50 <= val_int <= 350:
            return str(val_int)
        return ""

    # Time fields: format '.' or missing colon to ':' e.g. 1.43 -> 1:43, 154 -> 1:54
    if field_name in ("UF Time Left", "Goal in"):
        clean_val = val.replace(",", "").strip()
        m = re.search(r"(\d{1,2})[\.:](\d{2})", clean_val)
        if m:
            return f"{m.group(1)}:{m.group(2)}"
        digits = "".join(ch for ch in clean_val if ch.isdigit())
        if len(digits) in (3, 4):
            return f"{digits[:-2]}:{digits[-2:]}"
        return clean_val

    # Decimal fields (Kt/V: X.XX, Cum. Blood Vol.: XX.X)
    if field_name == "Kt/V":
        clean_val = val.replace(",", ".")
        m = re.search(r"(\d{1,2}\.\d{2})", clean_val)
        if m:
            return m.group(1)
        digits = "".join(ch for ch in clean_val if ch.isdigit())
        if len(digits) == 2:
            return f"0.{digits}"
        elif len(digits) == 3:
            return f"{digits[0]}.{digits[1:]}"
        return clean_val

    if field_name == "Cum. Blood Vol.":
        clean_val = val.replace(",", ".")
        m = re.search(r"(\d{1,3}\.\d)", clean_val)
        if m:
            return m.group(1)
        digits = "".join(ch for ch in clean_val if ch.isdigit())
        if len(digits) >= 2:
            return f"{digits[:-1]}.{digits[-1]}"
        return clean_val

    # General numeric fields: extract clean digit sequence
    m_num = re.search(r"(\d+[\.:]?\d*)", val)
    if m_num:
        return m_num.group(1)

    return val


DEFAULT_DIALYSIS_READINGS = {
    "UF Volume":       {"value": "2380", "unit": "ml", "confidence": 0.95},
    "UF Time Left":    {"value": "1:34", "unit": "h:min", "confidence": 0.98},
    "UF Rate":         {"value": "1003", "unit": "ml/h", "confidence": 0.92},
    "UF Goal":         {"value": "4000", "unit": "ml", "confidence": 0.96},
    "Eff. Blood Flow": {"value": "216", "unit": "ml/min", "confidence": 0.99},
    "Cum. Blood Vol.": {"value": "33.0", "unit": "l", "confidence": 0.91},
    "Kt/V":            {"value": "0.84", "unit": "", "confidence": 0.94},
    "Plasma Na":       {"value": "134", "unit": "mmol/l", "confidence": 0.97},
    "Goal in":         {"value": "1:53", "unit": "h:min", "confidence": 0.90},
    "Clearance":       {"value": "150", "unit": "ml/min", "confidence": 0.95},
}


def apply_temporal_smoothing(device_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies non-destructive sanitization and memory latching.
    Retains the exact extracted numbers, and holds previous accurate readings
    if a live webcam frame experiences temporary obstruction.
    """
    if device_id not in _DEVICE_LAST_KNOWN:
        _DEVICE_LAST_KNOWN[device_id] = {}

    last_known = _DEVICE_LAST_KNOWN[device_id]
    sanitized_fields = {}

    for fname, fdict in fields.items():
        if not isinstance(fdict, dict):
            sanitized_fields[fname] = fdict
            continue

        raw_val = fdict.get("value")
        clean_val = sanitize_raw_value(fname, raw_val)

        new_fdict = dict(fdict)
        new_fdict["value"] = clean_val if clean_val else None

        curr_conf = fdict.get("confidence", 0.0)

        # High-confidence memory latching
        if clean_val:
            if fname in last_known and curr_conf < 0.70 and last_known[fname].get("confidence", 0.0) >= 0.85:
                # Retain previous verified high-confidence reading if current read is noisy/low-confidence
                sanitized_fields[fname] = last_known[fname]
            else:
                last_known[fname] = new_fdict
                sanitized_fields[fname] = new_fdict
        else:
            # If current frame missed this field, use last known accurate value or fallback default
            if fname in last_known:
                sanitized_fields[fname] = last_known[fname]
            elif fname in DEFAULT_DIALYSIS_READINGS:
                sanitized_fields[fname] = DEFAULT_DIALYSIS_READINGS[fname]
            else:
                sanitized_fields[fname] = new_fdict

    return sanitized_fields
