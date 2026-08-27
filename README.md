# RTSP Multi-Camera Vision Scraper & Live Medical Data Extractor

Real-time automated vision scraper and web application server for monitoring **multiple RTSP IP cameras** and **USB webcams** facing medical monitors (such as **Fresenius Medical Care 4008 S** dialysis displays).

Extracts LCD numerical parameters, structured key-value pairs, and OCR telemetry in real time, serving a live FastAPI dashboard and writing telemetry logs directly to the **terminal console** and **JSON files**.

---

## 🌟 Key Features

1. **Multi-Camera RTSP & Webcam Management**:
   - Streams live video from multiple connected RTSP IP cameras (`rtsp://username:password@ip:554/stream`) and USB webcams simultaneously.
   - Built-in hardware scanner to auto-discover all USB webcams connected to the host PC.
   - Live camera switcher in the browser header (`ACTIVE STREAM: [Dropdown]`).

2. **2D Relative Spatial Grid Extraction Engine**:
   - Specifically engineered for **Fresenius Medical Care 4008 S** dialysis LCD displays.
   - Uses 2D spatial coordinate bands to map each detected bounding box to its exact screen location (Left, Middle, Right column Y-bands).
   - **Zero Offset Corruption**: Single missing or obscured numbers never shift other fields.

3. **Extracted Parameters**:
   - `UF Volume` (e.g. `2380 ml`)
   - `UF Time Left` (e.g. `1:34 h:min`)
   - `UF Rate` (e.g. `1003 ml/h`)
   - `UF Goal` (e.g. `4000 ml`)
   - `Eff. Blood Flow` (e.g. `216 ml/min`)
   - `Cum. Blood Vol.` (e.g. `33.0 l`)
   - `Kt/V` (e.g. `0.84`)
   - `Plasma Na` (e.g. `134 mmol/l`)
   - `Goal in` (e.g. `1:53 h:min`)
   - `Clearance` (e.g. `150 ml/min`)

4. **Real-Time Console Logging & JSON Telemetry Persistence**:
   - Prints formatted extracted parameters to the Python server console in real time for every live camera pass.
   - Automatically saves live telemetry JSON files (`output/live_telemetry_{device_id}.json` and `output/scraped_data_{timestamp}.json`).

5. **Responsive Dashboard with Collapsible Sidebars**:
   - Independent left (`◀ Devices`) and right (`Extracted Data ▶`) sidebar toggles.
   - Allows expanding the live camera video stream to **100% full screen**.

---

## 📁 File Structure

```text
image-to-extraction/
├── app.py                   # FastAPI Web Application & RTSP Stream Server
├── main.py                  # CLI entry point (webcam / upload / live / menu)
├── calibrate.py             # Calibration tool for dialysis screen box positions
├── requirements.txt         # Python package dependencies
├── src/
│   ├── __init__.py
│   ├── ocr_extract.py       # Dual-pass OCR engine & PyTorch thread-safe locks
│   ├── field_parser.py      # 2D Spatial Grid Parser & Digit Sanitizer
│   ├── rtsp_manager.py      # Multi-Camera Worker Manager & Live Telemetry Logger
│   ├── screen_extractor.py  # Box detection & visual boundary cropping
│   └── capture.py           # Multi-camera discovery & opencv video loop
├── static/                  # CSS styles, icons, and frontend JavaScript engine
│   ├── css/style.css
│   └── js/app.js
├── templates/               # Jinja2 HTML Dashboard Templates
│   └── index.html
└── output/                  # Created automatically; live JSON logs & captured PNG images land here
```

---

## ⚙️ Setup & Installation

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

*(Note: EasyOCR runs out of the box in Python. No external C++ software downloads are required!)*

---

## 🚀 Running the Web Application

Launch the FastAPI Multi-Camera Server from terminal:

```powershell
& c:/Users/USER/Desktop/web-cam-project/image-to-extraction/.venv/Scripts/python.exe app.py
```

Then open your browser at:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🖥️ Live Terminal Console Output Example

Whenever an RTSP camera stream pass finishes extraction, the Python terminal prints a clean telemetry log:

```text
=================================================================
[RTSP LIVE OCR] Camera: 'Dialysis Bed 01' (ID: 009)
[RTSP LIVE OCR] Time  : 2026-08-27 16:40:15
[RTSP LIVE OCR] Extracted Parameters:
   • UF Volume       : 2380 ml
   • UF Time Left    : 1:34 h:min
   • UF Rate         : 1003 ml/h
   • UF Goal         : 4000 ml
   • Eff. Blood Flow : 216 ml/min
   • Cum. Blood Vol. : 33.0 l
   • Kt/V            : 0.84
   • Plasma Na       : 134 mmol/l
   • Goal in         : 1:53 h:min
   • Clearance       : 150 ml/min
=================================================================
```

---

## 📄 Sample Saved JSON Telemetry

```json
{
  "device_id": "009",
  "device_name": "Dialysis Bed 01",
  "rtsp_url": "rtsp://admin:pass@192.168.1.101:554/stream1",
  "timestamp": "2026-08-27 16:40:15",
  "mode": "dialysis",
  "fields": {
    "UF Volume": { "value": "2380", "unit": "ml", "confidence": 0.95 },
    "UF Time Left": { "value": "1:34", "unit": "h:min", "confidence": 0.98 },
    "UF Rate": { "value": "1003", "unit": "ml/h", "confidence": 0.92 },
    "UF Goal": { "value": "4000", "unit": "ml", "confidence": 0.96 },
    "Eff. Blood Flow": { "value": "216", "unit": "ml/min", "confidence": 0.99 },
    "Cum. Blood Vol.": { "value": "33.0", "unit": "l", "confidence": 0.91 },
    "Kt/V": { "value": "0.84", "unit": "", "confidence": 0.94 },
    "Plasma Na": { "value": "134", "unit": "mmol/l", "confidence": 0.97 },
    "Goal in": { "value": "1:53", "unit": "h:min", "confidence": 0.90 },
    "Clearance": { "value": "150", "unit": "ml/min", "confidence": 0.95 }
  },
  "key_value_pairs": {
    "UF Volume": "2380 ml",
    "UF Time Left": "1:34 h:min",
    "UF Rate": "1003 ml/h",
    "UF Goal": "4000 ml",
    "Eff. Blood Flow": "216 ml/min",
    "Cum. Blood Vol.": "33.0 l",
    "Kt/V": "0.84",
    "Plasma Na": "134 mmol/l",
    "Goal in": "1:53 h:min",
    "Clearance": "150 ml/min"
  }
}
```
