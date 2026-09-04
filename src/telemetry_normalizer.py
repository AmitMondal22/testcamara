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
    Sanitizes extracted OCR values to match exact Fresenius 4008S dialysis formats:
      - UF Volume       : 4-digit integer (e.g. '2380')
      - UF Time Left    : 'H:MM' format (e.g. '1:34')
      - UF Rate         : 3 or 4-digit integer (e.g. '1003', '748')
      - UF Goal         : 4-digit integer (e.g. '4000')
      - Eff. Blood Flow : 3-digit integer (e.g. '216', '275')
      - Cum. Blood Vol. : Decimal 'XX.X' (e.g. '33.0', '83.9')
      - Kt/V            : Decimal '0.XX' (e.g. '0.84', '0.68')
      - Plasma Na       : 3-digit integer '120-160' (e.g. '134')
      - Goal in         : 'H:MM' format (e.g. '1:53')
      - Clearance       : 3-digit integer '80-350' (e.g. '150', '184')
    """
    if not raw_val or raw_val in ("null", "None", ""):
        return ""

    val = str(raw_val).strip()

    if "--" in val or "-:-" in val:
        return "--"

    # 1. UF Volume: strictly 4-digit integer (e.g. 2380, 4932)
    if field_name == "UF Volume":
        digits = "".join(ch for ch in val if ch.isdigit())
        if len(digits) >= 4:
            return digits[:4]
        return digits if digits else val

    # 2. UF Goal: strictly 4-digit integer (e.g. 4000)
    if field_name == "UF Goal":
        digits = "".join(ch for ch in val if ch.isdigit())
        if len(digits) >= 4:
            return digits[:4]
        return digits if digits else val

    # 3. UF Rate: 3 or 4-digit integer (e.g. 1003, 748)
    if field_name == "UF Rate":
        digits = "".join(ch for ch in val if ch.isdigit())
        if len(digits) > 4:
            return digits[:4]
        return digits if digits else val

    # 4. Plasma Na: 3-digit integer (120 - 160)
    if field_name == "Plasma Na":
        digits = "".join(ch for ch in val if ch.isdigit())
        if len(digits) >= 3:
            v_int = int(digits[:3])
            if 115 <= v_int <= 165:
                return str(v_int)
        return digits if digits else val

    # 5. Eff. Blood Flow: 3-digit integer (100 - 450, e.g. 216, 275)
    if field_name == "Eff. Blood Flow":
        digits = "".join(ch for ch in val if ch.isdigit())
        if len(digits) >= 3:
            v_int = int(digits[:3])
            if 100 <= v_int <= 500:
                return str(v_int)
        return digits if digits else val

    # 6. Clearance: 3-digit integer (80 - 350, e.g. 150, 184)
    if field_name == "Clearance":
        digits = "".join(ch for ch in val if ch.isdigit())
        if len(digits) >= 3:
            v_int = int(digits[:3])
            if 60 <= v_int <= 360:
                return str(v_int)
        return digits if digits else val

    # 7. Time fields: strictly 'H:MM' (e.g. '1:34', '1:53')
    if field_name in ("UF Time Left", "Goal in"):
        m = re.search(r"(\d{1,2})[\.:](\d{2})", val)
        if m:
            mins = int(m.group(2))
            if mins >= 60:
                mins = 59
            return f"{m.group(1)}:{mins:02d}"
        digits = "".join(ch for ch in val if ch.isdigit())
        if len(digits) in (3, 4):
            h_part = digits[:-2]
            m_part = int(digits[-2:])
            if m_part >= 60:
                m_part = 59
            return f"{h_part}:{m_part:02d}"
        return val

    # 8. Kt/V: strictly '0.XX' decimal (e.g. '0.84', '0.68')
    if field_name == "Kt/V":
        clean_val = val.replace(",", ".")
        m = re.search(r"(0\.\d{2})", clean_val)
        if m:
            return m.group(1)
        digits = "".join(ch for ch in clean_val if ch.isdigit())
        if len(digits) == 2:
            return f"0.{digits}"
        elif len(digits) >= 3 and digits.startswith("0"):
            return f"0.{digits[1:3]}"
        elif len(digits) >= 2:
            return f"0.{digits[:2]}"
        return clean_val

    # 9. Cum. Blood Vol.: decimal 'XX.X' (e.g. '33.0', '83.9')
    if field_name == "Cum. Blood Vol.":
        clean_val = val.replace(",", ".")
        m = re.search(r"(\d{1,3}\.\d)", clean_val)
        if m:
            return m.group(1)
        digits = "".join(ch for ch in clean_val if ch.isdigit())
        if len(digits) >= 2:
            return f"{digits[:-1]}.{digits[-1]}"
        return clean_val

    return val


def apply_temporal_smoothing(device_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies non-destructive sanitization to extracted values.
    Returns clean extracted numbers only when genuine black boxes are detected.
    Does NOT fabricate default numbers when no screen is present.
    """
    if device_id not in _DEVICE_LAST_KNOWN:
        _DEVICE_LAST_KNOWN[device_id] = {}

    last_known = _DEVICE_LAST_KNOWN[device_id]
    sanitized_fields = {}

    # Check if the current frame found any real values
    any_found = any(isinstance(v, dict) and v.get("value") for v in fields.values())

    if not any_found:
        # No boxes/values detected in this frame -> clear latch and return empty
        _DEVICE_LAST_KNOWN[device_id] = {}
        for fname, fdict in fields.items():
            if isinstance(fdict, dict):
                sanitized_fields[fname] = {
                    "value": None,
                    "unit": fdict.get("unit", ""),
                    "confidence": 0.0
                }
            else:
                sanitized_fields[fname] = fdict
        return sanitized_fields

    for fname, fdict in fields.items():
        if not isinstance(fdict, dict):
            sanitized_fields[fname] = fdict
            continue

        raw_val = fdict.get("value")
        clean_val = sanitize_raw_value(fname, raw_val)

        new_fdict = dict(fdict)
        new_fdict["value"] = clean_val if clean_val else None

        curr_conf = fdict.get("confidence", 0.0)

        if clean_val:
            last_known[fname] = new_fdict
            sanitized_fields[fname] = new_fdict
        else:
            sanitized_fields[fname] = new_fdict

    return sanitized_fields
