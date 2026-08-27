# Display Data Extractor — Raspberry Pi 4

Extract structured text/numeric data from **any** LCD/LED display screen using a camera.  
Runs **both** the Web Server (for camera view/alignment) and the Continuous JSON Extraction Process simultaneously. Zero disk storage. Lightweight.

## Requirements

- **Raspberry Pi 4 Model B** (or any Linux/Windows PC)
- **USB Webcam** or Pi Camera
- **Tesseract OCR** (`sudo apt install tesseract-ocr` on Linux)
- Python 3.9+

---

## Quick Start (Runs Server & Extractor Simultaneously)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run All-in-One Command
python main.py
```

Running `python main.py` automatically:
1. **Starts the Web Server** on port 5000 (`http://localhost:5000` & `http://<rpi-ip>:5000`).
2. **Starts the Continuous Extraction Process**, printing structured JSON every 5 seconds to terminal.
3. **Shares the Camera in RAM** with zero conflict between live streaming and OCR.

---

## Schema Configuration (`dataset.json`)

Define target fields and data types:
```json
{
  "uf_volume": {
    "name": "UF Volume",
    "type": "number"
  },
  "uf_time_left": {
    "name": "UF Time Left",
    "type": "string"
  },
  "kt_v": {
    "name": "Kt/V",
    "type": "number"
  },
  "plasma_na": {
    "name": "Plasma Na",
    "type": "number"
  }
}
```

### JSON Output Format
```json
{
  "reading": 1,
  "timestamp": "2026-08-27T16:32:00",
  "items_detected": 12,
  "data": {
    "uf_volume": {
      "name": "UF Volume",
      "value": 2269
    },
    "kt_v": {
      "name": "Kt/V",
      "value": 0.75
    }
  }
}
```

---

## Web Dashboard & Camera Alignment

Open **`http://localhost:5000`** (or `http://<rpi-ip>:5000`):
- **Left**: Real-time camera video stream (30 FPS) for physical positioning & lens focus.
- **Right**: Live extracted table of detected metrics.
- **Top banner**: Camera alignment warning if no readable text is detected.

---

## Environment Variables (`.env`)
```env
SERVER_PORT=5000
SERVER_HOST=0.0.0.0
CAMERA_INDEX=0
INTERVAL_SECONDS=5
```
