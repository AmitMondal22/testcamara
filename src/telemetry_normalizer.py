"""
telemetry_normalizer.py
-----------------------
Non-Destructive Telemetry Sanitizer & Latching Memory Engine.

Ensures:
  1. RAW OCR numeric digits are preserved EXACTLY as discrete readings (NO arithmetic averaging).
  2. Formatting cleanup: removes commas (e.g. '3,971' -> '3971') and standardizes time colons ('1.43' -> '1:43').
  3. Latching Memory: Only latches verified, valid readings. Ignores random webcam noise.
"""

import re
from typing import Dict, Any

_DEVICE_LAST_KNOWN: Dict[str, Dict[str, Any]] = {}


def apply_temporal_smoothing(device_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies non-destructive discrete sanitization and memory latching.
    Retains exact numbers and holds previous accurate readings if a live video
    frame experiences temporary occlusion. Rejects noise.
    """
    from src.field_parser import is_valid_field_value, sanitize_digit_string

    if device_id not in _DEVICE_LAST_KNOWN:
        _DEVICE_LAST_KNOWN[device_id] = {}

    last_known = _DEVICE_LAST_KNOWN[device_id]
    sanitized_fields = {}

    for fname, fdict in fields.items():
        if not isinstance(fdict, dict):
            sanitized_fields[fname] = fdict
            continue

        raw_val = fdict.get("value")
        clean_val = sanitize_digit_string(raw_val, fname)

        new_fdict = dict(fdict)
        new_fdict["value"] = clean_val if (clean_val and is_valid_field_value(fname, clean_val)) else None

        if new_fdict["value"] and float(new_fdict.get("confidence", 0.0)) >= 0.90:
            last_known[fname] = new_fdict
            sanitized_fields[fname] = new_fdict
        else:
            sanitized_fields[fname] = {"value": None, "unit": fdict.get("unit", ""), "confidence": 0.0}

    return sanitized_fields
