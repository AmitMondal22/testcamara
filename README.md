# Display Data Extractor — Raspberry Pi 4

Extract structured text/numeric data from **any** LCD/LED display screen using a camera.  
Prints clean JSON to stdout every 5 seconds. Zero disk storage. Lightweight.

## Requirements

- **Raspberry Pi 4 Model B** (or any Linux/Windows PC)
- **USB Webcam** or Pi Camera
- **Tesseract OCR** (`sudo apt install tesseract-ocr` on Linux)
- Python 3.9+

## Install

```bash
python -m venv env
source env/bin/activate          # Linux/RPi
# .\env\Scripts\activate         # Windows

pip install -r requirements.txt
```

## Schema Configuration (`dataset.json`)

Define your target display fields and data types in `dataset.json`:

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

### Supported Data Types
- `"number"`: Automatically preserves integers (`2269`) or decimals (`0.75`)
- `"int"`: Integer type
- `"float"`: Floating-point decimal
- `"string"`: String / time text (`"1:43"`)

---

## Output JSON Format

```json
{
  "uf_volume": {
    "name": "UF Volume",
    "value": 2269
  },
  "uf_time_left": {
    "name": "UF Time Left",
    "value": "1:43"
  },
  "kt_v": {
    "name": "Kt/V",
    "value": 0.75
  },
  "plasma_na": {
    "name": "Plasma Na",
    "value": 134
  }
}
```

---

## Usage

### 1. Terminal Data Extractor (JSON stdout every 5s)

```bash
python main.py
```

### 2. Camera Adjustment Web Server

```bash
python server.py
```

Open `http://localhost:5000` (or `http://<rpi-ip>:5000`) in your browser:
- **Left**: Live camera stream for positioning and focusing.
- **Right**: Real-time extracted table of detected metrics.
- **Top banner**: Camera alignment warning if no readable text is detected.

## Project Structure

```
├── dataset.json        # Target field schema & data types
├── config.json         # Runtime settings (interval, camera index, port)
├── .env                # Environment variables (SERVER_PORT, CAMERA_INDEX)
├── main.py             # Continuous extraction loop → JSON stdout
├── server.py           # Flask web server (camera view + data sidebar)
├── requirements.txt    # opencv-python, pytesseract, flask, python-dotenv
└── src/
    ├── __init__.py
    └── extractor.py    # OCR extraction engine + dataset.json schema matcher
```
