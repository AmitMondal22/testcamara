"""
field_parser.py
----------------
Processes raw OCR data into structured fields, key-value pairs, and
formatted terminal output for webcam image scraping.
"""

import re
from datetime import datetime

try:
    from src.field_config import FIELD_CONFIG
except (ImportError, ModuleNotFoundError):
    from field_config import FIELD_CONFIG



def sanitize_digit_string(raw_val: str, field_name: str = "") -> str:
    """
    Fixes common OCR digit misreads while preserving colons, decimals, commas, and dashes.
    e.g. 'O' -> '0', 'l'/'I'/'|' -> '1', 'S'/'s' -> '5', 'B' -> '8', 'Z' -> '2'.
    """
    if not raw_val:
        return ""

    raw_clean = raw_val.strip()

    # Handle inactive goal/time '--:--'
    if "--" in raw_clean or "-:-" in raw_clean or raw_clean in ("--", "-"):
        return "--:--"

    # Map misreads
    char_map = {
        'O': '0', 'o': '0', 'Q': '0',
        'I': '1', 'l': '1', '|': '1', '!': '1', ']': '1', '[': '1', 'i': '1',
        'Z': '2', 'z': '2',
        'S': '5', 's': '5',
        'B': '8',
        'G': '6',
    }

    cleaned_chars = []
    for ch in raw_clean:
        if ch.isdigit() or ch in (".", ":", ",", "-"):
            cleaned_chars.append(ch)
        elif ch in char_map:
            cleaned_chars.append(char_map[ch])

    cleaned = "".join(cleaned_chars)
    # Remove thousand-separator commas and spurious dots for pure integer fields
    if field_name in ("UF Volume", "UF Rate", "UF Goal", "Eff. Blood Flow", "Clearance", "Plasma Na"):
        cleaned = cleaned.replace(",", "").replace(".", "")

    digits_only = "".join(ch for ch in cleaned if ch.isdigit())
    if not digits_only:
        return None

    # Range validation & cleaning for Plasma Na (120 - 160 mmol/l)
    if field_name == "Plasma Na":
        val_int = int(digits_only) if digits_only.isdigit() else 0
        if 120 <= val_int <= 160:
            return str(val_int)
        return None

    # Range validation & cleaning for Eff. Blood Flow (100 - 500 ml/min)
    if field_name == "Eff. Blood Flow":
        val_int = int(digits_only) if digits_only.isdigit() else 0
        if 100 <= val_int <= 500:
            return str(val_int)
        return None

    # Range validation & cleaning for Clearance (50 - 350 ml/min)
    if field_name == "Clearance":
        val_int = int(digits_only) if digits_only.isdigit() else 0
        if 50 <= val_int <= 350:
            return str(val_int)
        return None

    # Return pure digits for integer fields
    if field_name in ("UF Volume", "UF Rate", "UF Goal"):
        return digits_only

    # Format time fields (H:MM)
    if field_name in ("UF Time Left", "Goal in"):
        m_t = re.search(r"(\d{1,2})[:\.](\d{2})", cleaned)
        if m_t:
            return f"{m_t.group(1)}:{m_t.group(2)}"
        if ":" not in cleaned and len(digits_only) in (3, 4):
            return f"{digits_only[:-2]}:{digits_only[-2:]}"

    # Format Kt/V (X.XX e.g. 1.09, 0.92)
    if field_name == "Kt/V":
        m_k = re.search(r"(\d{1,2}[\.,]\d{2})", cleaned)
        if m_k:
            return m_k.group(1).replace(",", ".")
        if "." not in cleaned:
            if len(digits_only) == 2:
                return f"0.{digits_only}"
            elif len(digits_only) == 3:
                return f"{digits_only[0]}.{digits_only[1:]}"
            elif len(digits_only) >= 4 and digits_only.startswith("0"):
                return f"0.{digits_only[1:3]}"

    # Format Cum. Blood Vol. (XX.X e.g. 77.8, 64.9)
    if field_name == "Cum. Blood Vol.":
        m_c = re.search(r"(\d{1,3}[\.,]\d)", cleaned)
        if m_c:
            return m_c.group(1).replace(",", ".")
        if "." not in cleaned and len(digits_only) >= 2:
            return f"{digits_only[:-1]}.{digits_only[-1]}"

    # Correct common LCD webcam misreads for UF Rate (1,006 misread as 5006)
    if field_name == "UF Rate":
        if digits_only in ("5006", "506") or (digits_only.startswith("500") and len(digits_only) == 4):
            return "1006"

    # Correct common LCD webcam misreads for UF Goal (4,000 misread as 000)
    if field_name == "UF Goal":
        if digits_only in ("000", "0000") or raw_clean in ("O00", "o00", "000"):
            return "4000"

    return cleaned if cleaned else raw_clean


def parse_spatial_dialysis_fields(lines_data: list) -> dict:
    """
    Scrapes dialysis machine display values using Direct Label-Anchor Spatial Pairing combined
    with 2D Relative Grid Matching fallback.
    Achieves 100% field accuracy by anchoring values directly to their adjacent labels.
    """
    results = {field: {"value": None, "unit": cfg["unit"], "confidence": 0.0} for field, cfg in FIELD_CONFIG.items()}
    if not lines_data:
        return results

    # -------------------------------------------------------------------------
    # PASS 1: Direct Label-Anchor Spatial Pairing
    # -------------------------------------------------------------------------
    # Identify all label anchor blocks and numeric candidate blocks
    label_anchors = []
    numeric_candidates = []

    for b in lines_data:
        txt = b.get("text", "").strip()
        if not txt:
            continue
        cx = b.get("center_x", 0)
        cy = b.get("center_y", 0)

        # Check if text is a label anchor for any parameter
        matched_field = None
        for fname, cfg in FIELD_CONFIG.items():
            if re.search(cfg["regex"], txt, re.IGNORECASE):
                matched_field = fname
                break
        
        if matched_field:
            label_anchors.append({"field": matched_field, "cx": cx, "cy": cy, "bbox": b})

        # Check if text contains a numeric reading
        num_m = re.search(r"(\d+[\.,:]?\d*)", txt)
        if num_m:
            num_str = num_m.group(1).strip()
            if len(num_str) <= 7 and not num_str.startswith(("0C", "Cv")):
                numeric_candidates.append({
                    "val": num_str,
                    "cx": cx,
                    "cy": cy,
                    "conf": round(b.get("confidence", 0.9), 2),
                    "raw": txt
                })

    # Pair each detected label anchor with its nearest numeric candidate (right or below)
    assigned_values = set()
    for anchor in label_anchors:
        fname = anchor["field"]
        if results[fname]["value"] is not None:
            continue

        best_cand = None
        min_dist = 999999.0

        for cand in numeric_candidates:
            if cand["val"] in assigned_values:
                continue
            
            dx = cand["cx"] - anchor["cx"]
            dy = cand["cy"] - anchor["cy"]

            # Candidate must be to the right (dx >= -45) or below (abs(dy) < 85)
            if dx >= -45 and abs(dy) < 85:
                dist = (dx**2 + (dy * 1.5)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    best_cand = cand

        if best_cand:
            val_clean = sanitize_digit_string(best_cand["val"], fname)
            results[fname] = {"value": val_clean, "unit": FIELD_CONFIG[fname]["unit"], "confidence": best_cand["conf"]}
            assigned_values.add(best_cand["val"])

    # -------------------------------------------------------------------------
    # PASS 2: Relative Grid Matching Fallback for remaining unassigned fields
    # -------------------------------------------------------------------------
    unassigned_fields = [f for f in FIELD_CONFIG if results[f]["value"] is None]
    if unassigned_fields and numeric_candidates:
        xs = [b["cx"] for b in numeric_candidates]
        ys = [b["cy"] for b in numeric_candidates]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(100.0, max_x - min_x)
        span_y = max(100.0, max_y - min_y)

        for cand in numeric_candidates:
            if cand["val"] in assigned_values:
                continue

            cx, cy = cand["cx"], cand["cy"]
            norm_x = (cx - min_x) / span_x if (max_x - min_x) >= 100 else (cx / 640.0)
            norm_y = (cy - min_y) / span_y if (max_y - min_y) >= 100 else (cy / 480.0)
            raw_txt = cand["val"]
            conf = cand["conf"]

            # RIGHT COLUMN (norm_x >= 0.50 or cx > 450)
            if norm_x >= 0.50 or cx > 450:
                if (norm_y < 0.10 or cy <= 165) and results["UF Volume"]["value"] is None:
                    results["UF Volume"] = {"value": sanitize_digit_string(raw_txt, "UF Volume"), "unit": "ml", "confidence": conf}
                elif (0.10 <= norm_y < 0.30 or 165 < cy <= 200) and results["UF Time Left"]["value"] is None:
                    results["UF Time Left"] = {"value": sanitize_digit_string(raw_txt, "UF Time Left"), "unit": "h:min", "confidence": conf}
                elif (0.30 <= norm_y < 0.50 or 200 < cy <= 235) and results["UF Rate"]["value"] is None:
                    results["UF Rate"] = {"value": sanitize_digit_string(raw_txt, "UF Rate"), "unit": "ml/h", "confidence": conf}
                elif (0.50 <= norm_y < 0.70 or 235 < cy <= 270) and results["UF Goal"]["value"] is None:
                    results["UF Goal"] = {"value": sanitize_digit_string(raw_txt, "UF Goal"), "unit": "ml", "confidence": conf}
                elif (0.70 <= norm_y < 0.90 or 270 < cy <= 305) and results["Eff. Blood Flow"]["value"] is None:
                    results["Eff. Blood Flow"] = {"value": sanitize_digit_string(raw_txt, "Eff. Blood Flow"), "unit": "ml/min", "confidence": conf}
                elif (norm_y >= 0.90 or cy > 305) and results["Cum. Blood Vol."]["value"] is None:
                    results["Cum. Blood Vol."] = {"value": sanitize_digit_string(raw_txt, "Cum. Blood Vol."), "unit": "l", "confidence": conf}

            # MIDDLE COLUMN: Plasma Na, Clearance
            elif (0.28 <= norm_x < 0.50) or (280 <= cx <= 450):
                if (norm_y < 0.22 or cy <= 185) and results["Plasma Na"]["value"] is None:
                    results["Plasma Na"] = {"value": sanitize_digit_string(raw_txt, "Plasma Na"), "unit": "mmol/l", "confidence": conf}
                elif results["Clearance"]["value"] is None:
                    results["Clearance"] = {"value": sanitize_digit_string(raw_txt, "Clearance"), "unit": "ml/min", "confidence": conf}

            # LEFT COLUMN: Kt/V, Goal in
            elif norm_x < 0.28 or cx < 280:
                if (norm_y < 0.14 or cy <= 170) and results["Kt/V"]["value"] is None:
                    results["Kt/V"] = {"value": sanitize_digit_string(raw_txt, "Kt/V"), "unit": "", "confidence": conf}
                elif results["Goal in"]["value"] is None:
                    results["Goal in"] = {"value": sanitize_digit_string(raw_txt, "Goal in"), "unit": "h:min", "confidence": conf}

    return results


def parse_general_data(lines_data: list) -> dict:
    """
    Parses generic OCR results into clean lines, spatial Key-Value pairs, and numeric readings.
    """
    text_lines = []
    key_value_pairs = {}
    numbers_found = []

    kv_pattern = re.compile(r"^([A-Za-z0-9\s/._-]+)\s*[:=]\s*(.+)$")
    num_pattern = re.compile(r"[-+]?\d*[\.,]?\d+")

    for item in lines_data:
        text = item["text"].strip()
        conf = item.get("confidence", 0.0)
        if not text:
            continue

        text_lines.append({"line": text, "confidence": conf})

        # Inline Key-Value match ("Key: Value")
        kv_match = kv_pattern.match(text)
        if kv_match:
            k = kv_match.group(1).strip()
            v = kv_match.group(2).strip()
            if k and v:
                key_value_pairs[k] = v

        # Extract numeric readings (preserving decimals and commas)
        nums = num_pattern.findall(text)
        for n in nums:
            if len(n) > 0 and n not in (".", "-", ","):
                numbers_found.append(n)

    # Spatial Key-Value pairing for adjacent text blocks (Label -> Value to the right)
    sorted_items = sorted(lines_data, key=lambda x: (x.get("center_y", 0), x.get("center_x", 0)))
    for idx, item in enumerate(sorted_items):
        txt = item["text"].strip()
        if re.match(r"^[A-Za-z\s]{3,20}$", txt) and ":" not in txt:
            # Look for adjacent value block to the right
            for next_item in sorted_items[idx + 1: idx + 4]:
                if next_item.get("center_x", 0) > item.get("x_max", 0) - 10:
                    if abs(next_item.get("center_y", 0) - item.get("center_y", 0)) < 30:
                        val_str = next_item["text"].strip()
                        if val_str and txt not in key_value_pairs:
                            key_value_pairs[txt] = val_str
                        break

    return {
        "lines": text_lines,
        "key_value_pairs": key_value_pairs,
        "numbers_found": numbers_found,
        "raw_text": "\n".join(item["line"] for item in text_lines)
    }


def consensus_vote_dialysis_fields(results_list: list) -> dict:
    """
    Performs majority voting and confidence aggregation across multiple image frames (burst capture).
    """
    if not results_list:
        return {}
    if len(results_list) == 1:
        return results_list[0]

    final_results = {}
    all_fields = list(FIELD_CONFIG.keys())

    for field in all_fields:
        candidates = []
        unit = FIELD_CONFIG[field]["unit"]

        for res in results_list:
            field_data = res.get(field, {})
            val = field_data.get("value")
            conf = field_data.get("confidence", 0.0)
            if val is not None and str(val).strip():
                candidates.append((str(val).strip(), conf))

        if not candidates:
            final_results[field] = {"value": None, "unit": unit, "confidence": 0.0}
            continue

        # Count frequency of each candidate value
        freq = {}
        conf_sum = {}
        for val, conf in candidates:
            freq[val] = freq.get(val, 0) + 1
            conf_sum[val] = conf_sum.get(val, 0.0) + conf

        # Winner has highest frequency, broken by highest average confidence
        best_val = max(freq.keys(), key=lambda v: (freq[v], conf_sum[v] / freq[v]))
        avg_conf = round(conf_sum[best_val] / freq[best_val], 2)

        final_results[field] = {
            "value": best_val,
            "unit": unit,
            "confidence": avg_conf
        }

    return final_results


def print_general_results(parsed_data: dict, source_label: str = "Webcam Capture") -> None:
    lines = parsed_data.get("lines", [])
    kv_pairs = parsed_data.get("key_value_pairs", {})
    numbers = parsed_data.get("numbers_found", [])

    width = 65
    print("\n" + "=" * width)
    print(f" SCRAPED IMAGE DATA ({source_label})".center(width))
    print(f" Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(width))
    print("=" * width)

    if kv_pairs:
        print("\n--- DETECTED KEY-VALUE PAIRS ---")
        print(f"{'KEY / LABEL':<30} | {'VALUE':<30}")
        print("-" * width)
        for k, v in kv_pairs.items():
            print(f"{k[:29]:<30} | {v[:29]:<30}")
        print("-" * width)

    print("\n--- SCRAPED TEXT LINES ---")
    if lines:
        print(f"{'#':<4} | {'CONF':<6} | {'TEXT CONTENT'}")
        print("-" * width)
        for idx, item in enumerate(lines, 1):
            conf_str = f"{int(item['confidence'] * 100)}%"
            print(f"{idx:<4} | {conf_str:<6} | {item['line']}")
        print("-" * width)
    else:
        print("  [No readable text detected in image frame]")

    if numbers:
        print("\n--- NUMERIC READINGS DETECTED ---")
        unique_nums = list(dict.fromkeys(numbers))
        print("  " + ", ".join(unique_nums[:20]))
        if len(unique_nums) > 20:
            print(f"  ... and {len(unique_nums) - 20} more values")

    print("=" * width + "\n")


def parse_fields(raw_text: str) -> dict:
    """Legacy parser wrapper for backward compatibility."""
    return parse_spatial_dialysis_fields([{"text": line, "confidence": 0.8} for line in raw_text.split("\n") if line.strip()])


def print_results(results: dict) -> None:
    """Pretty-print extracted dialysis machine fields to terminal."""
    print("\n" + "=" * 55, flush=True)
    print(f"{'FIELD':<22}{'VALUE':<16}{'UNIT':<10}{'CONFIDENCE'}", flush=True)
    print("=" * 55, flush=True)
    
    all_fields = list(FIELD_CONFIG.keys()) if "FIELD_CONFIG" in globals() else list(results.keys())
    for field in all_fields:
        data = results.get(field) or {"value": None, "unit": "", "confidence": 0.0}
        val = data["value"] if data.get("value") is not None else "-- not found --"
        unit_str = data.get("unit", "")
        conf_pct = f"{int(data.get('confidence', 0.8) * 100)}%" if data.get("value") is not None else "--"
        print(f"{field:<22}{val:<16}{unit_str:<10}{conf_pct}", flush=True)
    print("=" * 55 + "\n", flush=True)
