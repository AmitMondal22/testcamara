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

    # Remove thousand-separator commas (e.g., '3,971' -> '3971')
    clean_val = val.replace(",", "").strip()

    # Time fields: format '.' to ':' if formatted as H.MM
    if field_name in ("UF Time Left", "Goal in"):
        m = re.match(r"^(\d)[\.:](\d{2})$", clean_val)
        if m:
            return f"{m.group(1)}:{m.group(2)}"
        return clean_val

    # Decimal fields (Kt/V, Cum. Blood Vol.): preserve exact dots/decimals
    if field_name in ("Kt/V", "Cum. Blood Vol."):
        # Standardize OCR comma decimal to dot e.g., '0,68' -> '0.68'
        clean_val = clean_val.replace(",", ".")
        m = re.search(r"(\d+\.?\d*)", clean_val)
        if m:
            return m.group(1)
        return clean_val

    # General numeric fields: extract clean digit sequence
    m_num = re.search(r"(\d+[\.:]?\d*)", clean_val)
    if m_num:
        return m_num.group(1)

    return clean_val


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
        new_fdict["value"] = clean_val

        # Latching memory: if current frame returned valid value, update memory
        if clean_val:
            last_known[fname] = new_fdict
            sanitized_fields[fname] = new_fdict
        else:
            # If current frame missed this field, use last known accurate value
            if fname in last_known:
                sanitized_fields[fname] = last_known[fname]
            else:
                sanitized_fields[fname] = new_fdict

    return sanitized_fields
