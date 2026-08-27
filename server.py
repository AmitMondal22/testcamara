"""
server.py
---------
Flask web server for camera adjustment & live data monitoring on Raspberry Pi 4.

Features:
  - Accessible from any device on local network (http://<rpi-ip>:5000)
  - Live camera MJPEG stream with auto-reconnection & placeholder when camera is busy
  - Real-time extracted data table side-by-side with camera feed
  - Camera alignment alert (red warning when no text detected)
  - Clean shutdown

Usage:
    python server.py                    # Start server on port 5000 (0.0.0.0)
    python server.py --port 8080        # Custom port
"""

import json
import os
import sys
import time
import socket
import argparse
import threading
from datetime import datetime

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template_string

from src.extractor import extract_from_frame, open_camera, load_dataset_file

# Load .env file
ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(ENV_FILE)

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
DATASET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.json")

app = Flask(__name__)

# Shared state between camera thread and web requests
_lock = threading.Lock()
_latest_frame = None
_latest_data = {}
_latest_item_count = 0
_latest_timestamp = ""
_camera_ok = False
_camera_status_msg = "Connecting to camera..."
_running = True


def get_local_ip():
    """Get the primary local network IP address of this device."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def load_config():
    cfg = {
        "interval_seconds": 5,
        "camera_index": 0,
        "camera_resolution": [1280, 720],
        "server_port": 5000,
        "server_host": "0.0.0.0",
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                cfg.update(json.load(f))
        except Exception:
            pass

    # dataset.json overrides fields definition if present
    dataset = load_dataset_file(DATASET_FILE)
    if dataset:
        cfg["fields"] = dataset

    # .env overrides config.json
    if os.getenv("SERVER_PORT"):
        cfg["server_port"] = int(os.getenv("SERVER_PORT"))
    if os.getenv("SERVER_HOST"):
        cfg["server_host"] = os.getenv("SERVER_HOST")
    if os.getenv("INTERVAL_SECONDS"):
        cfg["interval_seconds"] = int(os.getenv("INTERVAL_SECONDS"))
    if os.getenv("CAMERA_INDEX"):
        cfg["camera_index"] = int(os.getenv("CAMERA_INDEX"))

    return cfg


def create_placeholder_frame(text="Camera Not Connected", subtext="Check USB cable or permissions"):
    """Create a dark placeholder image with informative text."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (25, 20, 20)  # Dark navy

    cv2.putText(img, text, (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
    cv2.putText(img, subtext, (40, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    return img


def camera_extraction_loop(config):
    """Background thread with auto-reconnection and continuous frame capture."""
    global _latest_frame, _latest_data, _latest_item_count, _latest_timestamp, _camera_ok, _camera_status_msg, _running

    cam_idx = config.get("camera_index", 0)
    resolution = tuple(config.get("camera_resolution", [1280, 720]))
    interval = config.get("interval_seconds", 5)

    last_extract_time = 0
    cap = None

    while _running:
        # 1. Connect or reconnect to camera if needed
        if cap is None or not cap.isOpened():
            with _lock:
                _camera_ok = False
                _camera_status_msg = f"Connecting to camera index {cam_idx}..."
            try:
                cap = open_camera(cam_idx, resolution)
                with _lock:
                    _camera_ok = True
                    _camera_status_msg = "Camera connected"
                print(f"[Server] Camera connected successfully on index {cam_idx}", flush=True)
            except Exception as e:
                with _lock:
                    _camera_ok = False
                    _camera_status_msg = f"Camera error: {e}"
                    _latest_frame = create_placeholder_frame("Camera Not Detected", "Reconnecting...")
                time.sleep(2.0)
                continue

        # 2. Grab frame
        try:
            for _ in range(2):
                cap.grab()
            ret, frame = cap.read()
        except Exception:
            ret, frame = False, None

        if not ret or frame is None:
            with _lock:
                _camera_ok = False
                _camera_status_msg = "Frame capture failed"
                _latest_frame = create_placeholder_frame("Camera Read Failed", "Re-initializing...")
            if cap:
                cap.release()
            cap = None
            time.sleep(1.0)
            continue

        with _lock:
            _latest_frame = frame.copy()
            _camera_ok = True
            _camera_status_msg = "OK"

        # 3. OCR extraction at interval
        now = time.time()
        if now - last_extract_time >= interval:
            data, count = extract_from_frame(frame, fields_config=config.get("fields"))
            with _lock:
                _latest_data = data
                _latest_item_count = count
                _latest_timestamp = datetime.now().isoformat()

            output = {
                "timestamp": _latest_timestamp,
                "items_detected": count,
                "data": data,
            }
            print(json.dumps(output, indent=2, ensure_ascii=False), flush=True)
            last_extract_time = now

        time.sleep(0.03)  # ~30fps smooth streaming

    if cap:
        cap.release()


# ─── Web Routes ──────────────────────────────────────────────

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Display Data Extractor — Camera Adjustment</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0f1117;
    color: #e4e4e7;
    min-height: 100vh;
  }
  header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    padding: 16px 24px;
    border-bottom: 1px solid #2a2a3e;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  header h1 {
    font-size: 20px;
    font-weight: 600;
    color: #60a5fa;
  }
  header .status {
    font-size: 13px;
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: 500;
  }
  .status-ok { background: #064e3b; color: #34d399; }
  .status-warn { background: #7c2d12; color: #fb923c; animation: pulse 1.5s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.6; } }

  .alert-banner {
    background: linear-gradient(90deg, #991b1b, #b91c1c);
    color: #fecaca;
    padding: 12px 24px;
    font-size: 14px;
    font-weight: 600;
    text-align: center;
    display: none;
  }
  .alert-banner.show { display: block; }

  .container {
    display: flex;
    gap: 20px;
    padding: 20px;
    max-height: calc(100vh - 80px);
  }

  .camera-section {
    flex: 1.5;
    background: #1a1a2e;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #2a2a3e;
    display: flex;
    flex-direction: column;
  }
  .camera-section .title-bar {
    padding: 12px 16px;
    background: #16213e;
    font-size: 14px;
    font-weight: 600;
    color: #93c5fd;
    border-bottom: 1px solid #2a2a3e;
  }
  .camera-section img {
    width: 100%;
    height: auto;
    display: block;
    background: #000;
    object-fit: contain;
  }

  .data-section {
    flex: 1;
    background: #1a1a2e;
    border-radius: 12px;
    border: 1px solid #2a2a3e;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .data-section .title-bar {
    padding: 12px 16px;
    background: #16213e;
    font-size: 14px;
    font-weight: 600;
    color: #93c5fd;
    border-bottom: 1px solid #2a2a3e;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .data-section .title-bar span { font-size: 11px; color: #64748b; font-weight: 400; }
  .data-body {
    flex: 1;
    overflow-y: auto;
    padding: 12px;
  }

  table {
    width: 100%;
    border-collapse: collapse;
  }
  table th {
    text-align: left;
    padding: 8px 10px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #64748b;
    border-bottom: 1px solid #2a2a3e;
  }
  table td {
    padding: 10px;
    font-size: 14px;
    border-bottom: 1px solid #1e1e30;
  }
  table tr:hover { background: #1e1e30; }
  td.field-name { color: #94a3b8; font-weight: 500; }
  td.field-value {
    color: #34d399;
    font-weight: 700;
    font-size: 16px;
    font-variant-numeric: tabular-nums;
  }

  .empty-state {
    text-align: center;
    padding: 40px 20px;
    color: #64748b;
    font-size: 14px;
  }
  .empty-state .icon { font-size: 48px; margin-bottom: 12px; }

  @media (max-width: 768px) {
    .container { flex-direction: column; max-height: none; }
  }
</style>
</head>
<body>

<header>
  <h1>📺 Display Data Extractor</h1>
  <div class="status" id="statusBadge">Connecting...</div>
</header>

<div class="alert-banner" id="alertBanner">
  ⚠️ CAMERA NOT ALIGNED — Point camera at the display to read values.
</div>

<div class="container">
  <div class="camera-section">
    <div class="title-bar">📷 Live Camera Feed — Position & Focus Camera</div>
    <img id="cameraFeed" src="/video_feed" alt="Camera Feed">
  </div>

  <div class="data-section">
    <div class="title-bar">
      📊 Extracted Display Data
      <span id="lastUpdate">Waiting...</span>
    </div>
    <div class="data-body" id="dataBody">
      <div class="empty-state">
        <div class="icon">📡</div>
        <div>Waiting for first extraction...</div>
      </div>
    </div>
  </div>
</div>

<script>
function fetchData() {
  fetch('/data')
    .then(r => r.json())
    .then(d => {
      const badge = document.getElementById('statusBadge');
      const alert = document.getElementById('alertBanner');

      if (!d.camera_ok) {
        badge.className = 'status status-warn';
        badge.textContent = '⚠ ' + (d.status_msg || 'Camera Error');
        alert.classList.add('show');
        alert.textContent = '⚠️ ' + (d.status_msg || 'Camera Not Detected. Check USB connection.');
        return;
      }

      if (d.items_detected === 0) {
        badge.className = 'status status-warn';
        badge.textContent = '⚠ No Text Detected';
        alert.classList.add('show');
        alert.textContent = '⚠️ CAMERA NOT ALIGNED — No readable text detected. Please aim camera at display.';
      } else if (Object.keys(d.data || {}).length === 0) {
        badge.className = 'status status-warn';
        badge.textContent = '● Scanning (' + d.items_detected + ' words)';
        alert.classList.remove('show');
      } else {
        badge.className = 'status status-ok';
        badge.textContent = '● Reading OK (' + Object.keys(d.data).length + ' fields)';
        alert.classList.remove('show');
      }

      if (d.timestamp) {
        const t = new Date(d.timestamp);
        document.getElementById('lastUpdate').textContent =
          'Updated: ' + t.toLocaleTimeString();
      }

      const data = d.data || {};
      const keys = Object.keys(data);
      const body = document.getElementById('dataBody');

      if (keys.length === 0) {
        body.innerHTML = '<div class="empty-state"><div class="icon">📡</div><div>Scanning for display numbers...</div></div>';
        return;
      }

      let html = '<table><thead><tr><th>Field</th><th>Value</th></tr></thead><tbody>';
      for (const k of keys) {
        const item = data[k];
        html += '<tr><td class="field-name">' + escHtml(item.name || k) + '</td>';
        html += '<td class="field-value">' + escHtml(String(item.value)) + '</td></tr>';
      }
      html += '</tbody></table>';
      body.innerHTML = html;
    })
    .catch(() => {
      document.getElementById('statusBadge').className = 'status status-warn';
      document.getElementById('statusBadge').textContent = '⚠ Disconnected';
    });
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

// Poll data endpoint every 1.5 seconds
setInterval(fetchData, 1500);
fetchData();
</script>
</body>
</html>
"""


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/data")
def data_api():
    with _lock:
        return jsonify({
            "camera_ok": _camera_ok,
            "status_msg": _camera_status_msg,
            "items_detected": _latest_item_count,
            "timestamp": _latest_timestamp,
            "data": _latest_data,
        })


def generate_mjpeg():
    """Yields continuous MJPEG frames for live camera stream without blocking."""
    while _running:
        with _lock:
            frame = _latest_frame

        if frame is None:
            frame = create_placeholder_frame("Connecting...", "Waiting for camera frame")

        try:
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        except Exception:
            pass

        time.sleep(0.04)  # ~25fps


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def main():
    global _running
    config = load_config()

    parser = argparse.ArgumentParser(description="Camera Adjustment Web Server")
    parser.add_argument("-p", "--port", type=int, help="Server port")
    parser.add_argument("-c", "--camera", type=int, help="Camera index")
    args = parser.parse_args()

    if args.camera is not None:
        config["camera_index"] = args.camera

    port = args.port or config.get("server_port", 5000)
    host = "0.0.0.0"  # Always bind to 0.0.0.0 for LAN access

    local_ip = get_local_ip()

    print("=" * 60)
    print("  DISPLAY DATA EXTRACTOR — CAMERA ADJUSTMENT SERVER")
    print("=" * 60)
    print(f"  Access from this Raspberry Pi:   http://localhost:{port}")
    print(f"  Access from other PC / Phone:     http://{local_ip}:{port}")
    print("=" * 60)
    print("  Press Ctrl+C to stop\n", flush=True)

    # Start camera worker thread
    cam_thread = threading.Thread(target=camera_extraction_loop, args=(config,), daemon=True)
    cam_thread.start()

    try:
        app.run(host=host, port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        _running = False


if __name__ == "__main__":
    main()
