# Display Data Extractor — Raspberry Pi 4

Extract structured text/numeric data from **any** LCD/LED display screen using a camera.  
Prints clean JSON to stdout every 5 seconds. Zero disk storage. Lightweight.

## Requirements

- **Raspberry Pi 4 Model B** (or any Linux/Windows PC)
- **USB Webcam** or Pi Camera
- **Tesseract OCR** (`sudo apt install tesseract-ocr` on Linux)
- Python 3.9+

---

## Raspberry Pi USB Webcam Setup

If your USB webcam is not opening on Raspberry Pi 4, follow these steps:

### 1. Grant Camera Permissions to your user
```bash
sudo usermod -a -G video $USER
```
*(Log out and log back in for changes to take effect)*

### 2. Install V4L2 Tools & Check Connected Devices
```bash
sudo apt update
sudo apt install v4l-utils tesseract-ocr -y

# Check video device files:
ls -l /dev/video*

# Check recognized webcam hardware:
v4l2-ctl --list-devices
```

### 3. Run the Camera Diagnostic Tool
```bash
python test_camera.py
```
This tool automatically scans indices `0..8` and tells you which camera index works (e.g. `CAMERA_INDEX=0` or `CAMERA_INDEX=2`).

### 4. Set Camera Index in `.env`
Edit `.env`:
```env
CAMERA_INDEX=0
SERVER_PORT=5000
SERVER_HOST=0.0.0.0
INTERVAL_SECONDS=5
```

---

## Install & Run

```bash
# Virtual environment setup
python -m venv env
source env/bin/activate          # Linux/RPi

pip install -r requirements.txt
```

### Option A: Camera Adjustment Web Server
```bash
python server.py
```
Open **`http://<rpi-ip>:5000`** in any browser on your local network:
- **Left**: Live MJPEG video stream to physically aim & focus your camera.
- **Right**: Real-time extracted table of detected metrics.
- **Top banner**: Camera alignment warning if no readable text is detected.

### Option B: Terminal JSON Data Extractor
```bash
python main.py
```

---

## Schema Configuration (`dataset.json`)

Define target fields and types:
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

Output format:
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
  }
}
```
