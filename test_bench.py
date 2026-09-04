import cv2, os, re, numpy as np, easyocr, time

reader = easyocr.Reader(['en'], gpu=False, verbose=False)

def clean_ocr_text(field_name: str, raw_text: str) -> str:
    if not raw_text:
        return ""
    
    # Common OCR character substitutions for LCD digits
    t = raw_text.strip()
    t = t.replace('o', '0').replace('O', '0').replace('D', '0').replace('Q', '0')
    t = t.replace('l', '1').replace('I', '1').replace('|', '1').replace('i', '1')
    t = t.replace('Z', '2').replace('z', '2')
    t = t.replace('S', '5').replace('s', '5')
    t = t.replace('B', '8')
    t = t.replace('q', '4')
    
    # Clean brackets and noise
    t = re.sub(r"[\[\]\(\)\{\}'\"`~<>!_#\$%^&*+=a-zA-Z]", "", t)
    
    # Handle specific field formats
    if field_name == "UF Volume":
        # Always 4 digits (e.g. 2380)
        digits = "".join(ch for ch in t if ch.isdigit())
        if len(digits) == 4:
            return digits
        elif len(digits) > 4:
            # If leading noise digit like 42380 -> 2380 or trailing
            return digits[-4:]
        return digits

    elif field_name == "UF Goal":
        # Always 4 digits (e.g. 4000)
        digits = "".join(ch for ch in t if ch.isdigit())
        if len(digits) == 4:
            return digits
        elif len(digits) > 4:
            return digits[-4:]
        return digits

    elif field_name == "UF Rate":
        # 3 or 4 digits (e.g. 1003 or 748 or 986)
        digits = "".join(ch for ch in t if ch.isdigit())
        if len(digits) in (3, 4):
            return digits
        elif len(digits) > 4:
            return digits[-4:]
        return digits

    elif field_name in ("UF Time Left", "Goal in"):
        # Format H:MM (e.g. 1:34, 1:43, 1:53)
        m = re.search(r"(\d{1,2})[\.:](\d{2})", t)
        if m:
            mins = int(m.group(2))
            if mins < 60:
                return f"{m.group(1)}:{mins:02d}"
        digits = "".join(ch for ch in t if ch.isdigit())
        if len(digits) in (3, 4):
            # e.g. 143 -> 1:43, 1243 -> 1:43 or 2:43
            h = digits[:-2]
            mins = int(digits[-2:])
            if mins < 60:
                # If h has extra leading digit like 0143 or 1243 -> take last digit of h
                if len(h) > 1:
                    h = h[-1:]
                return f"{h}:{mins:02d}"
        return t

    elif field_name == "Kt/V":
        # Format 0.XX (e.g. 0.84, 0.60, 0.90)
        m = re.search(r"(0\.\d{2})", t.replace(',', '.'))
        if m:
            return m.group(1)
        digits = "".join(ch for ch in t if ch.isdigit())
        if len(digits) >= 2:
            return f"0.{digits[-2:]}"
        return t

    elif field_name == "Plasma Na":
        # 3 digits (120 - 160, e.g. 134)
        digits = "".join(ch for ch in t if ch.isdigit())
        if len(digits) == 3:
            return digits
        elif len(digits) > 3:
            # e.g. 0134 -> 134
            return digits[-3:]
        return digits

    elif field_name == "Clearance":
        # 3 digits (80 - 350, e.g. 150, 158, 172, 184)
        digits = "".join(ch for ch in t if ch.isdigit())
        if len(digits) == 3:
            return digits
        elif len(digits) > 3:
            return digits[-3:]
        return digits

    elif field_name == "Eff. Blood Flow":
        # 3 digits (100 - 450, e.g. 216, 261, 275)
        digits = "".join(ch for ch in t if ch.isdigit())
        if len(digits) == 3:
            return digits
        elif len(digits) > 3:
            return digits[-3:]
        return digits

    elif field_name == "Cum. Blood Vol.":
        # Format XX.X (e.g. 33.0, 43.8, 83.9)
        m = re.search(r"(\d{1,3}\.\d)", t.replace(',', '.'))
        if m:
            return m.group(1)
        digits = "".join(ch for ch in t if ch.isdigit())
        if len(digits) >= 2:
            return f"{digits[:-1]}.{digits[-1]}"
        return t

    return t


def test_full_pipeline(fpath):
    img = cv2.imread(fpath)
    if img is None:
        return
    h_img, w_img = img.shape[:2]
    print(f"\n=======================================================")
    print(f"PIPELINE TEST: {fpath} ({w_img}x{h_img})")
    print(f"=======================================================")

    from src.black_box_extractor import detect_dark_boxes
    boxes = detect_dark_boxes(img)

    # Filter boxes that are not in the main monitor area
    valid_boxes = []
    for (x, y, w, h) in boxes:
        # Ignore top banner (y < 10% height) or bottom bar (y > 95% height) or left graph (x < 12% width)
        if y < 0.10 * h_img or y > 0.95 * h_img or (x < 0.12 * w_img and y > 0.50 * h_img):
            continue
        # Check aspect ratio
        aspect = w / max(h, 1)
        if aspect < 1.0 or aspect > 4.5:
            continue
        valid_boxes.append((x, y, w, h))

    # Recognize
    h_list = [[x, x + w, y, y + h] for (x, y, w, h) in valid_boxes]
    results = reader.recognize(img, horizontal_list=h_list, free_list=[])

    box_data = []
    for idx, (b, text, conf) in enumerate(results):
        x, y, w, h = valid_boxes[idx]
        cx = (x + w / 2) / w_img
        cy = (y + h / 2) / h_img
        # Ignore text that is purely alphabetic like 'Dialysis' or 'Pressure'
        clean_d = "".join(ch for ch in text if ch.isdigit())
        if not clean_d and any(c in text.lower() for c in ['dialysis', 'pressure', 'fresenius']):
            continue
        box_data.append({"x": x, "y": y, "w": w, "h": h, "cx": cx, "cy": cy, "raw": text, "conf": conf})

    # Cluster columns: Left OCM (cx < 0.42), Mid OCM (0.42 <= cx < 0.70), Right Column (cx >= 0.70)
    col_left = sorted([b for b in box_data if b["cx"] < 0.42], key=lambda b: b["cy"])
    col_mid  = sorted([b for b in box_data if 0.42 <= b["cx"] < 0.70], key=lambda b: b["cy"])
    col_right = sorted([b for b in box_data if b["cx"] >= 0.70], key=lambda b: b["cy"])

    extracted = {}

    # Left OCM: Kt/V (top), Goal in (bottom)
    if len(col_left) >= 1:
        extracted["Kt/V"] = clean_ocr_text("Kt/V", col_left[0]["raw"])
    if len(col_left) >= 2:
        extracted["Goal in"] = clean_ocr_text("Goal in", col_left[1]["raw"])

    # Mid OCM: Plasma Na (top), Clearance (bottom)
    if len(col_mid) >= 1:
        extracted["Plasma Na"] = clean_ocr_text("Plasma Na", col_mid[0]["raw"])
    if len(col_mid) >= 2:
        extracted["Clearance"] = clean_ocr_text("Clearance", col_mid[1]["raw"])

    # Right Column: 6 fields in order:
    # 1. UF Volume
    # 2. UF Time Left
    # 3. UF Rate
    # 4. UF Goal
    # 5. Eff. Blood Flow
    # 6. Cum. Blood Vol.
    right_fields = ["UF Volume", "UF Time Left", "UF Rate", "UF Goal", "Eff. Blood Flow", "Cum. Blood Vol."]
    
    if len(col_right) == 6:
        for i, fname in enumerate(right_fields):
            extracted[fname] = clean_ocr_text(fname, col_right[i]["raw"])
    else:
        # Match by relative Y position
        for b in col_right:
            cy = b["cy"]
            raw = b["raw"]
            if cy < 0.26:
                extracted["UF Volume"] = clean_ocr_text("UF Volume", raw)
            elif cy < 0.38:
                extracted["UF Time Left"] = clean_ocr_text("UF Time Left", raw)
            elif cy < 0.50:
                extracted["UF Rate"] = clean_ocr_text("UF Rate", raw)
            elif cy < 0.62:
                extracted["UF Goal"] = clean_ocr_text("UF Goal", raw)
            elif cy < 0.76:
                extracted["Eff. Blood Flow"] = clean_ocr_text("Eff. Blood Flow", raw)
            else:
                extracted["Cum. Blood Vol."] = clean_ocr_text("Cum. Blood Vol.", raw)

    all_10_keys = [
        "UF Volume", "UF Time Left", "UF Rate", "UF Goal",
        "Eff. Blood Flow", "Cum. Blood Vol.", "Kt/V", "Plasma Na",
        "Goal in", "Clearance"
    ]
    for k in all_10_keys:
        val = extracted.get(k, "MISSING")
        print(f"  • {k:<16}: {val}")

test_full_pipeline('output/capture_20260827_141849.png')
test_full_pipeline('output/capture_20260828_160930.png')
test_full_pipeline('output/capture_20260827_134421.png')
