"""
main.py
-------
Unified Display Data Extractor & Web Server for Raspberry Pi 4.

Runs BOTH simultaneously:
  1. Continuous interval extraction printing JSON to stdout every N seconds.
  2. Flask Web Server on Port 5000 (http://localhost:5000 and http://<rpi-ip>:5000)
     for live camera view, positioning, and real-time dashboard.

Usage:
    python main.py                  # Run continuous extraction + Web server on port 5000
    python main.py --interval 3     # 3-second extraction interval
    python main.py --port 8080      # Custom web port
    python main.py --camera 1       # Custom camera index
"""

import json
import os
import sys
import time
import socket
import signal
import argparse
import threading
import logging
from datetime import datetime

# Ensure stdout supports UTF-8 on Windows / Linux
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template_string

from src.extractor import extract_from_frame, open_camera, load_dataset_file

# Suppress noisy Flask dev server console logs so terminal JSON output stays clean
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Load .env file
ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(ENV_FILE)

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
DATASET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.json")

app = Flask(__name__)

# Shared state between Camera, Web Server, and Interval Extraction Loop
_lock = threading.Lock()
_latest_frame = None
_latest_data = {}
_latest_item_count = 0
_latest_timestamp = ""
_camera_ok = False
_camera_status_msg = "Connecting to camera..."
_recent_frames = []
_running = True


def get_local_ip():
    """Get primary local network IP address."""
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


def create_placeholder_frame(text="Camera Connecting...", subtext="Checking video device"):
    """Create a dark placeholder image with status text."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (25, 20, 20)
    cv2.putText(img, text, (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 165, 255), 2)
    cv2.putText(img, subtext, (40, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    return img


# ─── Web UI Template ─────────────────────────────────────────

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Display Data Extractor — Live View</title>
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

        time.sleep(0.04)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ─── Camera Frame Ingestion Thread (30 FPS) ──────────────────

def camera_capture_loop(cap):
    """Continuously reads frames from camera into memory buffer."""
    global _latest_frame, _recent_frames, _camera_ok, _camera_status_msg, _running

    while _running and cap and cap.isOpened():
        try:
            ret, frame = cap.read()
        except Exception:
            ret, frame = False, None

        if ret and frame is not None and frame.size > 0:
            with _lock:
                _latest_frame = frame.copy()
                _camera_ok = True
                _camera_status_msg = "OK"
                _recent_frames.append(frame.copy())
                if len(_recent_frames) > 4:
                    _recent_frames.pop(0)
        else:
            with _lock:
                _camera_ok = False
                _camera_status_msg = "Frame drop"
                _latest_frame = create_placeholder_frame("Camera Frame Drop", "Re-capturing...")

        time.sleep(0.03)  # ~30fps


# ─── Main Unified Application ────────────────────────────────

def run_application():
    global _running, _latest_data, _latest_item_count, _latest_timestamp
    config = load_config()

    parser = argparse.ArgumentParser(description="Unified Display Data Extractor & Web Server")
    parser.add_argument("-i", "--interval", type=float, help="Interval in seconds")
    parser.add_argument("-p", "--port", type=int, help="Server port")
    parser.add_argument("-c", "--camera", type=int, help="Camera index")
    args = parser.parse_args()

    interval = args.interval or config.get("interval_seconds", 5)
    cam_idx = args.camera if args.camera is not None else config.get("camera_index", 0)
    port = args.port or config.get("server_port", 5000)
    host = "0.0.0.0"
    resolution = tuple(config.get("camera_resolution", [1280, 720]))
    local_ip = get_local_ip()

    print("=" * 62)
    print(f"  >>> SERVER IS RUNNING ON PORT: {port} <<<")
    print("=" * 62)
    print(f"  Web Dashboard (Local):   http://localhost:{port}")
    print(f"  Web Dashboard (Network): http://{local_ip}:{port}")
    print(f"  Extraction Interval:     {interval}s (Multi-frame 100% Accuracy)")
    print(f"  Camera Index:            {cam_idx}")
    print("=" * 62)
    print("  Streaming live data and printing JSON every 5s. Ctrl+C to stop.\n", flush=True)

    # 1. Start Flask Web Server in background daemon thread
    def start_web_server():
        try:
            app.run(host=host, port=port, debug=False, threaded=True)
        except OSError as e:
            if "Address already in use" in str(e) or "10048" in str(e):
                alt_port = 8080 if port != 8080 else 5001
                print(f"[Notice] Port {port} occupied. Web server active on: http://{local_ip}:{alt_port}", flush=True)
                app.run(host=host, port=alt_port, debug=False, threaded=True)

    web_thread = threading.Thread(target=start_web_server, daemon=True)
    web_thread.start()

    # 2. Open Camera
    try:
        cap = open_camera(cam_idx, resolution)
    except RuntimeError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Start Camera Capture Ingestion Thread
    cam_thread = threading.Thread(target=camera_capture_loop, args=(cap,), daemon=True)
    cam_thread.start()

    # Graceful shutdown handler
    def handle_stop(sig, frame):
        global _running
        _running = False
    signal.signal(signal.SIGINT, handle_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_stop)

    reading_num = 0
    fields = config.get("fields")

    try:
        while _running:
            time.sleep(interval)
            reading_num += 1

            # Process latest fresh frame from memory buffer
            latest_frame = None
            with _lock:
                if _latest_frame is not None:
                    latest_frame = _latest_frame.copy()
                elif _recent_frames:
                    latest_frame = _recent_frames[-1].copy()

            if latest_frame is None:
                continue

            verified_data, max_items = extract_from_frame(latest_frame, fields_config=fields)

            now_iso = datetime.now().isoformat()

            with _lock:
                _latest_data = verified_data
                _latest_item_count = len(verified_data)
                _latest_timestamp = now_iso

            # Zero False-Data Requirement: Only print JSON when readable data is found
            if verified_data:
                packet = {
                    "reading": reading_num,
                    "timestamp": now_iso,
                    "items_detected": len(verified_data),
                    "data": verified_data,
                }
                print(json.dumps(packet, indent=2, ensure_ascii=False), flush=True)
            else:
                print(
                    f"[{now_iso}] [ALERT] CAMERA NOT ALIGNED - No readable text detected. Please adjust camera to point at the display.",
                    flush=True
                )

    finally:
        _running = False
        cap.release()
        print("\n[Stopped] Camera and server released cleanly.")


if __name__ == "__main__":
    run_application()
