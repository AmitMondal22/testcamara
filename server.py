"""
server.py
---------
Flask web server for camera adjustment & live data monitoring.

Features:
  - Live camera MJPEG stream for adjusting camera position
  - Real-time extracted data table alongside camera feed
  - Camera alignment alert (red warning when no text detected)
  - Clean shutdown

Usage:
    python server.py                    # Start server on port 5000
    python server.py --port 8080        # Custom port

Open browser: http://<raspberry-pi-ip>:5000
"""

import json
import os
import sys
import time
import argparse
import threading
from datetime import datetime

import cv2
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
_running = True


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


def camera_extraction_loop(config):
    """Background thread: captures frames + runs OCR extraction at intervals."""
    global _latest_frame, _latest_data, _latest_item_count, _latest_timestamp, _camera_ok, _running

    cam_idx = config.get("camera_index", 0)
    resolution = tuple(config.get("camera_resolution", [1280, 720]))
    interval = config.get("interval_seconds", 5)

    try:
        cap = open_camera(cam_idx, resolution)
    except RuntimeError as e:
        print(f"[ERROR] Cannot open camera: {e}", file=sys.stderr)
        _camera_ok = False
        return

    last_extract_time = 0

    try:
        while _running:
            # Always grab frames for live feed
            for _ in range(2):
                cap.grab()
            ret, frame = cap.read()

            if not ret or frame is None:
                with _lock:
                    _camera_ok = False
                time.sleep(0.1)
                continue

            with _lock:
                _latest_frame = frame.copy()
                _camera_ok = True

            # Run extraction at configured interval
            now = time.time()
            if now - last_extract_time >= interval:
                data, count = extract_from_frame(frame, fields_config=config.get("fields"))
                with _lock:
                    _latest_data = data
                    _latest_item_count = count
                    _latest_timestamp = datetime.now().isoformat()

                # Also print to stdout
                output = {
                    "timestamp": _latest_timestamp,
                    "items_detected": count,
                    "data": data,
                }
                print(json.dumps(output, indent=2, ensure_ascii=False), flush=True)
                last_extract_time = now

            time.sleep(0.03)  # ~30fps for smooth live view

    finally:
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
    display: block;
    background: #000;
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
    color: #4a5568;
    font-size: 14px;
  }
  .empty-state .icon { font-size: 48px; margin-bottom: 12px; }
</style>
</head>
<body>

<header>
  <h1>📺 Display Data Extractor</h1>
  <div class="status" id="statusBadge">Connecting...</div>
</header>

<div class="alert-banner" id="alertBanner">
  ⚠️ CAMERA NOT ALIGNED — No readable text detected. Please adjust camera to point at the display.
</div>

<div class="container">
  <div class="camera-section">
    <div class="title-bar">📷 Live Camera Feed — Adjust Position</div>
    <img id="cameraFeed" src="/video_feed" alt="Camera Feed">
  </div>

  <div class="data-section">
    <div class="title-bar">
      📊 Extracted Data
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
      // Update status badge
      const badge = document.getElementById('statusBadge');
      const alert = document.getElementById('alertBanner');

      if (!d.camera_ok) {
        badge.className = 'status status-warn';
        badge.textContent = '⚠ Camera Error';
        alert.classList.add('show');
        return;
      }

      if (d.items_detected === 0) {
        badge.className = 'status status-warn';
        badge.textContent = '⚠ No Text Detected';
        alert.classList.add('show');
      } else if (Object.keys(d.data || {}).length === 0) {
        badge.className = 'status status-warn';
        badge.textContent = '⚠ No Data Pairs';
        alert.classList.remove('show');
      } else {
        badge.className = 'status status-ok';
        badge.textContent = '● Reading OK (' + Object.keys(d.data).length + ' fields)';
        alert.classList.remove('show');
      }

      // Update timestamp
      if (d.timestamp) {
        const t = new Date(d.timestamp);
        document.getElementById('lastUpdate').textContent =
          'Updated: ' + t.toLocaleTimeString();
      }

      // Build data table
      const data = d.data || {};
      const keys = Object.keys(data);
      const body = document.getElementById('dataBody');

      if (keys.length === 0) {
        body.innerHTML = '<div class="empty-state"><div class="icon">📡</div><div>No data extracted yet. Point camera at display.</div></div>';
        return;
      }

      let html = '<table><thead><tr><th>Field</th><th>Value</th></tr></thead><tbody>';
      for (const k of keys) {
        const item = data[k];
        html += '<tr><td class="field-name">' + escHtml(item.name) + '</td>';
        html += '<td class="field-value">' + escHtml(item.value) + '</td></tr>';
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

// Poll every 2 seconds for data updates
setInterval(fetchData, 2000);
fetchData();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/data")
def data_api():
    with _lock:
        return jsonify({
            "camera_ok": _camera_ok,
            "items_detected": _latest_item_count,
            "timestamp": _latest_timestamp,
            "data": _latest_data,
        })


def generate_mjpeg():
    """Yields MJPEG frames for the live camera feed."""
    while _running:
        with _lock:
            frame = _latest_frame

        if frame is not None:
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

        time.sleep(0.05)  # ~20fps


@app.route("/video_feed")
def video_feed():
    return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


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
    host = config.get("server_host", "0.0.0.0")

    # Start camera thread
    cam_thread = threading.Thread(target=camera_extraction_loop, args=(config,), daemon=True)
    cam_thread.start()

    print(f"\n  Server starting at http://{host}:{port}")
    print(f"  Open browser to adjust camera and view live data")
    print(f"  Press Ctrl+C to stop\n", flush=True)

    try:
        app.run(host=host, port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        _running = False


if __name__ == "__main__":
    main()
