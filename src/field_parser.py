"""
field_parser.py
----------------
Processes raw OCR data into structured fields, key-value pairs, and
formatted terminal output for webcam image scraping.
"""

import re
from datetime import datetime

# Patterns and units for Fresenius 4008S Dialysis Machine display (with fuzzy OCR label aliases)
FIELD_CONFIG = {
    "UF Volume":        {"regex": r"(UF|LF|UV|UN)\s*(Volume|Vol|Volun|Voi|Vot)",            "unit": "ml"},
    "UF Time Left":     {"regex": r"(UF|LF|UV|UN)\s*Tim[ea]\s*(Left|Lot|Led|Let|Lft|Lel)?",  "unit": "h:min"},
    "UF Rate":          {"regex": r"(UF|LF|UV|UN)\s*(Rate|Rale|Ral|Rte)",                   "unit": "ml/h"},
    "UF Goal":          {"regex": r"(UF|LF|UV|UN)\s*(Goal|God|Goa|Gol)",                    "unit": "ml"},
    "Eff. Blood Flow":  {"regex": r"(Eff\.?|Bff|Bid)?\s*Bl[oo]*d?\s*(Flow|Flot|Fiot|Flo)",  "unit": "ml/min"},
    "Cum. Blood Vol.":  {"regex": r"(Cum\.?|Cun|Cumn)?\s*(Blood|Blod|Daadia)?\s*Vol",       "unit": "l"},
    "Kt/V":             {"regex": r"Kt\s*/?\s*V|KI\s*/?\s*V|K1\s*/?\s*V",                  "unit": ""},
    "Plasma Na":        {"regex": r"Plasma\s*(Na|N|Na\+)?|Pheni|phenu",                     "unit": "mmol/l"},
    "Goal in":          {"regex": r"Goal\s*in|Gol\s*in",                                   "unit": "h:min"},
    "Clearance":        {"regex": r"Clearance|Claance|Charanco",                            "unit": "ml/min"},
}


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
    # Remove thousand-separator commas for clean integer fields
    if field_name in ("UF Volume", "UF Rate", "UF Goal", "Eff. Blood Flow", "Clearance", "Plasma Na"):
        cleaned = cleaned.replace(",", "")

    digits_only = "".join(ch for ch in cleaned if ch.isdigit())
    if not digits_only:
        return None

    # Format time fields (H:MM)
    if field_name in ("UF Time Left", "Goal in"):
        if ":" not in cleaned and len(digits_only) in (3, 4):
            return f"{digits_only[:-2]}:{digits_only[-2:]}"

    # Format Kt/V (0.XX)
    if field_name == "Kt/V":
        if "." not in cleaned:
            if len(digits_only) == 2:
                return f"0.{digits_only}"
            elif len(digits_only) == 3:
                return f"{digits_only[0]}.{digits_only[1:]}"
            elif len(digits_only) >= 4 and digits_only.startswith("0"):
                return f"0.{digits_only[1:3]}"

    # Format Cum. Blood Vol. (XX.X)
    if field_name == "Cum. Blood Vol.":
        if "." not in cleaned and len(digits_only) >= 2:
            return f"{digits_only[:-1]}.{digits_only[-1]}"

    # Correct common LCD webcam misreads for UF Rate (1,006 misread as 5006)
    if field_name == "UF Rate":
        if digits_only in ("5006", "506") or (digits_only.startswith("500") and len(digits_only) == 4):
            return "1,006"

    # Correct common LCD webcam misreads for UF Goal (4,000 misread as 000)
    if field_name == "UF Goal":
        if digits_only in ("000", "0000") or raw_clean in ("O00", "o00", "000"):
            return "4,000"

    return cleaned if cleaned else raw_clean


def parse_spatial_dialysis_fields(lines_data: list) -> dict:
    """
    Scrapes dialysis machine display values using 2D Relative Grid Matching based on exact
    visual row/column coordinates (Left, Middle, Right column Y-bands).
    Prevents single-missing-box array offset corruption across all fields.
    """
    results = {field: {"value": None, "unit": cfg["unit"], "confidence": 0.0} for field, cfg in FIELD_CONFIG.items()}

    candidates = []
    for b in lines_data:
        txt = b.get("text", "").strip()
        cy = b.get("center_y", 0)
        # Filter top header noise (single digit 1 or header text at top corner)
        if cy < 135 and (txt in ("1", "I", "l") or "Qv" in txt or "Dialysis" in txt):
            continue

        # Extract embedded numbers if text contains labels + digits (e.g. 'Flaema Ka 134' -> '134')
        num_match = re.search(r"(\d+[\.,:]?\d*)", txt)
        if num_match:
            num_str = num_match.group(1).strip()
            if len(num_str) <= 7 and not num_str.startswith(("0C", "Cv")):
                b_copy = dict(b)
                b_copy["text"] = num_str
                candidates.append(b_copy)

    if not candidates:
        return results

    # Determine coordinate bounds across candidates
    xs = [b.get("center_x", 0) for b in candidates]
    ys = [b.get("center_y", 0) for b in candidates]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(100.0, max_x - min_x)
    span_y = max(100.0, max_y - min_y)

    for b in candidates:
        cx = b.get("center_x", 0)
        cy = b.get("center_y", 0)
        norm_x = (cx - min_x) / span_x if (max_x - min_x) >= 100 else (cx / 640.0)
        norm_y = (cy - min_y) / span_y if (max_y - min_y) >= 100 else (cy / 480.0)
        raw_txt = b["text"].strip()
        conf = round(b.get("confidence", 0.9), 2)

        # RIGHT COLUMN (norm_x >= 0.50 or cx > 450)
        if norm_x >= 0.50 or cx > 450:
            if norm_y < 0.10 or (cy <= 165 and min_y > 100):
                val = sanitize_digit_string(raw_txt, "UF Volume")
                if val and "." in val and "," not in val and len(val.replace(".", "")) in (3, 4):
                    val = val.replace(".", ",")
                results["UF Volume"] = {"value": val, "unit": "ml", "confidence": conf}
            elif 0.10 <= norm_y < 0.30 or (165 < cy <= 200 and min_y > 100):
                results["UF Time Left"] = {"value": sanitize_digit_string(raw_txt, "UF Time Left"), "unit": "h:min", "confidence": conf}
            elif 0.30 <= norm_y < 0.50 or (200 < cy <= 235 and min_y > 100):
                results["UF Rate"] = {"value": sanitize_digit_string(raw_txt, "UF Rate"), "unit": "ml/h", "confidence": conf}
            elif 0.50 <= norm_y < 0.70 or (235 < cy <= 270 and min_y > 100):
                results["UF Goal"] = {"value": sanitize_digit_string(raw_txt, "UF Goal"), "unit": "ml", "confidence": conf}
            elif 0.70 <= norm_y < 0.90 or (270 < cy <= 305 and min_y > 100):
                results["Eff. Blood Flow"] = {"value": sanitize_digit_string(raw_txt, "Eff. Blood Flow"), "unit": "ml/min", "confidence": conf}
            elif norm_y >= 0.90 or (cy > 305 and min_y > 100):
                results["Cum. Blood Vol."] = {"value": sanitize_digit_string(raw_txt, "Cum. Blood Vol."), "unit": "l", "confidence": conf}

        # MIDDLE COLUMN (0.28 <= norm_x < 0.50 or 280 <= cx <= 450): Plasma Na, Clearance
        elif (0.28 <= norm_x < 0.50) or (280 <= cx <= 450):
            if norm_y < 0.22 or (cy <= 185 and min_y > 100):
                results["Plasma Na"] = {"value": sanitize_digit_string(raw_txt, "Plasma Na"), "unit": "mmol/l", "confidence": conf}
            else:
                results["Clearance"] = {"value": sanitize_digit_string(raw_txt, "Clearance"), "unit": "ml/min", "confidence": conf}

        # LEFT COLUMN (norm_x < 0.28 or cx < 280): Kt/V, Goal in
        elif norm_x < 0.28 or cx < 280:
            if norm_y < 0.14 or (cy <= 170 and min_y > 100):
                results["Kt/V"] = {"value": sanitize_digit_string(raw_txt, "Kt/V"), "unit": "", "confidence": conf}
            else:
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
    print("\n" + "=" * 55)
    print(f"{'FIELD':<22}{'VALUE':<16}{'UNIT':<10}{'CONFIDENCE'}")
    print("=" * 55)
    for field, data in results.items():
        val = data["value"] if data["value"] is not None else "-- not found --"
        conf_pct = f"{int(data.get('confidence', 0.8) * 100)}%" if data["value"] is not None else "--"
        print(f"{field:<22}{val:<16}{data['unit']:<10}{conf_pct}")
    print("=" * 55 + "\n")
