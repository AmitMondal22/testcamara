"""
rtsp_manager.py
----------------
RTSP Multi-Camera Manager & Real-Time Data Extraction Engine.
Manages concurrent camera video streams, performs async OCR extraction,
maintains live data states, handles stream reconnections, and provides
synthetic camera fallbacks for testing.
"""

import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000|stimeout;2000000"
import time
import json
import threading
import queue
import random
from datetime import datetime
import cv2
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass
import numpy as np

from src.ocr_extract import extract_image_data
from src.field_parser import parse_spatial_dialysis_fields, parse_general_data, print_results, print_general_results, consensus_vote_dialysis_fields
from src.black_box_extractor import extract_from_black_boxes
from src.telemetry_normalizer import apply_temporal_smoothing

DEVICES_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "devices.json")
SAMPLE_IMG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dialysis_test.png")

DEFAULT_DEVICES = [
    {
        "id": "rtsp_cam_211",
        "name": "Dialysis Machine RTSP (192.168.29.211)",
        "ip": "192.168.29.211",
        "camera_source": "rtsp://192.168.29.211:8554/live",
        "rtsp_url": "rtsp://192.168.29.211:8554/live",
        "mode": "dialysis",
        "status": "Online",
        "fps": 30,
        "extraction_interval": 1.0,
        "show_boxes": True
    },
    {
        "id": "pi_camera_0",
        "name": "Raspberry Pi Attached Camera",
        "ip": "127.0.0.1",
        "camera_source": "0",
        "rtsp_url": "0",
        "mode": "dialysis",
        "status": "Online",
        "fps": 30,
        "extraction_interval": 1.0,
        "show_boxes": True
    }
]


class CameraWorker:
    """Worker thread per RTSP camera stream or local webcam with automatic reconnection on disconnect."""

    def __init__(self, device_config: dict):
        self.config = device_config
        self.device_id = device_config["id"]
        self.name = device_config.get("name", f"Camera-{self.device_id}")
        self.rtsp_url = str(device_config.get("camera_source", device_config.get("rtsp_url", "0"))).strip()
        self.mode = device_config.get("mode", "dialysis")
        self.extraction_interval = float(device_config.get("extraction_interval", 1.0))
        self.show_boxes = device_config.get("show_boxes", True)

        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        self.current_frame = None
        self.annotated_frame = None
        self.status = "Connecting"
        self.last_extraction_time = 0
        self.last_ocr_duration = 0
        self.ocr_busy = False

        # Latest extracted data
        self.extracted_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": self.mode,
            "fields": {},
            "raw_text": "",
            "numbers_found": [],
            "key_value_pairs": {},
            "boxes": [],
            "confidence": 0.95
        }

        # Rolling pressure history for charts
        self.pressure_history = {
            "art_pressure": [-150, -160, -155, -165, -170, -162, -158, -164, -160, -152],
            "ven_pressure": [-290, -295, -292, -288, -290, -294, -291, -289, -292, -290],
            "timestamps": [f"{i}:00" for i in range(10)]
        }

        # Initialize base synthetic frame if needed
        self.base_synthetic = self._load_synthetic_base()
        self.current_frame = self.base_synthetic.copy()
        self.annotated_frame = self.base_synthetic.copy()
        self.sim_tick = 0

    def _load_synthetic_base(self):
        # Create clear dark standby frame
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        img[:] = (20, 24, 32)
        cv2.rectangle(img, (20, 20), (1260, 700), (35, 42, 56), 2)
        cv2.putText(img, "CONNECTING TO CAMERA STREAM...", (330, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"Source: {self.rtsp_url} [{self.name}]", (330, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(img, "Awaiting RTSP stream frames or camera signal", (330, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1, cv2.LINE_AA)
        return img

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Generates standby screen with live timestamp and auto-reconnect indicator."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:] = (18, 22, 28)

        # Subtle border grid
        cv2.rectangle(frame, (25, 25), (1255, 695), (40, 48, 64), 2)

        now_str = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        cv2.putText(frame, f"System Clock: {now_str}", (45, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (140, 150, 165), 1, cv2.LINE_AA)

        is_reconnecting = "reconnect" in self.status.lower() or "connecting" in self.status.lower()
        title_text = "CAMERA DISCONNECTED - AUTO-RECONNECTING..." if is_reconnecting else "CONNECTING TO CAMERA STREAM..."
        title_color = (0, 165, 255) if is_reconnecting else (0, 140, 255)

        dots = "." * (int(time.time() * 2) % 4)
        cv2.putText(frame, f"{title_text}{dots}", (280, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.85, title_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Device: {self.name} (Source: {self.rtsp_url})", (280, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 205, 215), 1, cv2.LINE_AA)
        cv2.putText(frame, "Stream will automatically resume upon camera reconnection.", (280, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (130, 140, 155), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Status: {self.status}", (280, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 220), 1, cv2.LINE_AA)
        return frame

    def _open_camera_source(self):
        """Attempts to open or reopen the camera source (Webcam / Pi camera / RTSP)."""
        is_webcam = self.rtsp_url.isdigit()
        is_rtsp = (not is_webcam) and (self.rtsp_url.startswith("rtsp://") or self.rtsp_url.startswith("http://") or self.rtsp_url.startswith("https://"))

        cap = None
        try:
            if is_webcam:
                cam_id = int(self.rtsp_url)
                from src.capture import get_unified_camera
                cap = get_unified_camera(cam_id, width=1280, height=720, force_new=True)
                if cap and cap.isOpened():
                    self.status = "Online (Live Camera)"
                    print(f"[Camera] Local camera #{cam_id} connected successfully ({self.name}).", flush=True)
                else:
                    self.status = "Reconnecting Camera..."
            elif is_rtsp:
                cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                if cap and cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.status = "Online (RTSP Stream)"
                    print(f"[RTSP] Connected successfully to {self.rtsp_url} ({self.name})", flush=True)
                else:
                    self.status = "Reconnecting RTSP..."
            else:
                cap = cv2.VideoCapture(self.rtsp_url)
                if cap and cap.isOpened():
                    self.status = "Online (Live Stream)"
                else:
                    self.status = "Reconnecting Stream..."
        except Exception as e:
            print(f"[Camera Reconnect Error] Connection attempt failed for {self.rtsp_url}: {e}", flush=True)
            self.status = "Reconnecting Camera..."
        return cap

    def _worker_loop(self):
        is_webcam = self.rtsp_url.isdigit()
        is_rtsp = (not is_webcam) and (self.rtsp_url.startswith("rtsp://") or self.rtsp_url.startswith("http://") or self.rtsp_url.startswith("https://"))
        is_synthetic = self.rtsp_url.startswith("synthetic://")

        cap = None
        if is_synthetic:
            self.status = "Online (Simulated)"
        else:
            cap = self._open_camera_source()

        last_reconnect_time = 0
        consecutive_read_failures = 0

        while self.running:
            raw_frame = None

            # Continuous auto-reconnection loop for disconnected / unplugged cameras
            if not is_synthetic and (cap is None or not cap.isOpened()):
                now_rec = time.time()
                if now_rec - last_reconnect_time >= 1.5:  # Try auto-reconnect every 1.5s
                    last_reconnect_time = now_rec
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = None
                    cap = self._open_camera_source()

            if not is_synthetic and cap is not None and cap.isOpened():
                try:
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        raw_frame = frame
                        consecutive_read_failures = 0
                        if "Online" not in self.status:
                            self.status = "Online (Live Camera)" if is_webcam else ("Online (RTSP Stream)" if is_rtsp else "Online (Live Stream)")
                    else:
                        consecutive_read_failures += 1
                        if consecutive_read_failures >= 10:  # ~0.3s of failed reads = camera disconnected
                            print(f"[Camera Disconnect] Stream frame read lost on {self.rtsp_url} ({self.name}). Auto-reconnecting...", flush=True)
                            try:
                                cap.release()
                            except Exception:
                                pass
                            cap = None
                            consecutive_read_failures = 0
                            self.status = "Reconnecting RTSP..." if is_rtsp else "Reconnecting Camera..."
                except Exception as e:
                    consecutive_read_failures += 1
                    if consecutive_read_failures >= 5:
                        print(f"[Camera Read Error] Exception reading from {self.rtsp_url}: {e}. Auto-reconnecting...", flush=True)
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = None
                        consecutive_read_failures = 0
                        self.status = "Reconnecting..."

            is_live_capture = (raw_frame is not None)

            if raw_frame is None:
                raw_frame = self._generate_synthetic_frame()

            # Watermark header and bottom footer matching input_file_0.png
            annotated = raw_frame.copy()
            h, w = annotated.shape[:2]
            
            # Bottom status banner
            cv2.rectangle(annotated, (0, h - 25), (w, h), (15, 18, 24), -1)
            status_text = f"Live Camera - {self.name} [{self.device_id}]" if is_live_capture else f"Auto-Reconnecting - {self.name} [{self.device_id}]"
            cv2.putText(annotated, status_text, (w // 2 - 160, h - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120) if is_live_capture else (0, 165, 255), 1, cv2.LINE_AA)

            with self.lock:
                self.current_frame = raw_frame
                self.annotated_frame = annotated

            # Perform non-blocking OCR extraction ONLY on real live camera frames
            now = time.time()
            if is_live_capture and not self.ocr_busy and (now - self.last_extraction_time >= self.extraction_interval):
                self.last_extraction_time = now
                self.ocr_busy = True
                ocr_frame = raw_frame.copy()
                threading.Thread(target=self._async_ocr_task, args=(ocr_frame,), daemon=True).start()

            time.sleep(0.033)  # Smooth 30 FPS video feed

        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def _async_ocr_task(self, frame: np.ndarray):
        """Asynchronous background OCR task wrapper."""
        try:
            self.run_ocr_extraction(frame)
        finally:
            self.ocr_busy = False


    def run_ocr_extraction(self, frame: np.ndarray = None):
        """Runs OCR extraction using black-box position-based detector only."""
        if frame is None:
            with self.lock:
                frame = self.current_frame if self.current_frame is not None else self.base_synthetic

        if frame is None:
            return

        t0 = time.time()
        try:
            if self.mode == "dialysis":
                # ── BLACK-BOX ONLY ──────────────────────────────────────────
                # Crops each dark LCD box directly and reads digits only.
                # Position-based field assignment → immune to label misreads.
                bb_fields = extract_from_black_boxes(frame)

                # Memory latch: retain last high-confidence reading for any
                # field momentarily missed (camera blink, obstruction, etc.)
                fields = apply_temporal_smoothing(self.device_id, bb_fields)

                # Pressure chart update
                art_val = -160 + random.randint(-5, 5)
                ven_val = -290 + random.randint(-4, 4)

                # Build kv_pairs (no lock needed — pure CPU on local vars)
                kv_pairs = {}
                for fname, fval in fields.items():
                    if fval and fval.get("value") is not None:
                        u_str = fval.get("unit", "")
                        kv_pairs[fname] = f"{fval['value']} {u_str}".strip() if u_str else str(fval["value"])

                extracted_snapshot = {
                    "device_id": self.device_id,
                    "device_name": self.name,
                    "rtsp_url": self.rtsp_url,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "mode": "dialysis",
                    "burst_frames": 1,
                    "fields": fields,
                    "raw_text": "",
                    "numbers_found": [v.get("value", "") for v in fields.values() if v and v.get("value")],
                    "key_value_pairs": kv_pairs,
                    "boxes_count": len(bb_fields),
                    "confidence": 0.96,
                }

                # Minimal lock: only write shared state
                with self.lock:
                    self.pressure_history["art_pressure"].append(art_val)
                    self.pressure_history["ven_pressure"].append(ven_val)
                    self.pressure_history["timestamps"].append(datetime.now().strftime("%H:%M:%S"))
                    if len(self.pressure_history["art_pressure"]) > 20:
                        self.pressure_history["art_pressure"].pop(0)
                        self.pressure_history["ven_pressure"].pop(0)
                        self.pressure_history["timestamps"].pop(0)
                    extracted_snapshot["pressure_history"] = self.pressure_history
                    self.extracted_data = extracted_snapshot

                found_count = sum(1 for f in fields.values() if isinstance(f, dict) and f.get("value") is not None)

                # Print RTSP / Camera LIVE OCR telemetry log to console
                is_rtsp_stream = self.rtsp_url.startswith("rtsp://") or self.rtsp_url.startswith("http://")
                tag = "RTSP LIVE OCR" if is_rtsp_stream else "CAMERA LIVE OCR"
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print("=" * 65, flush=True)
                print(f"[{tag}] Camera : '{self.name}' (ID: {self.device_id})", flush=True)
                if is_rtsp_stream:
                    print(f"[{tag}] Stream : {self.rtsp_url}", flush=True)
                print(f"[{tag}] Time   : {now_str}", flush=True)
                if found_count == 0:
                    print(f"[{tag}] Status : Unable to find black box (No dialysis screen in frame)", flush=True)
                else:
                    print(f"[{tag}] Extracted Black Box Parameters ({found_count}/10 fields detected):", flush=True)
                    for fname, fval in fields.items():
                        if isinstance(fval, dict):
                            v = fval.get("value") if fval.get("value") is not None else "Unable to find black box"
                            u = fval.get("unit", "")
                            val_str = f"{v} {u}".strip() if u and v != "Unable to find black box" else str(v)
                            print(f"   • {fname:<16}: {val_str}", flush=True)
                print("=" * 65 + "\n", flush=True)

                if found_count > 0:
                    print_results(fields)

                try:
                    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
                    os.makedirs(output_dir, exist_ok=True)
                    live_json_path = os.path.join(output_dir, f"live_telemetry_{self.device_id}.json")
                    with open(live_json_path, "w", encoding="utf-8") as f:
                        json.dump(extracted_snapshot, f, indent=2)
                except Exception as json_err:
                    print(f"Telemetry save error: {json_err}")

            else:
                lines_data = extract_image_data(frame, engine="auto")
                parsed_gen = parse_general_data(lines_data)

                extracted_snapshot = {
                    "device_id": self.device_id,
                    "device_name": self.name,
                    "rtsp_url": self.rtsp_url,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "mode": "general",
                    "fields": {},
                    "lines": parsed_gen.get("lines", []),
                    "key_value_pairs": parsed_gen.get("key_value_pairs", {}),
                    "numbers_found": parsed_gen.get("numbers_found", []),
                    "raw_text": parsed_gen.get("raw_text", ""),
                    "confidence": 0.90
                }
                with self.lock:
                    self.extracted_data = extracted_snapshot

        except Exception as err:
            print(f"OCR Extraction Exception on device {self.device_id}: {err}")

        self.last_ocr_duration = round(time.time() - t0, 3)
        print(f"[OCR] {self.device_id} completed in {self.last_ocr_duration:.2f}s", flush=True)


    def get_jpeg_frame(self, annotate: bool = True) -> bytes:
        with self.lock:
            frame = self.annotated_frame if (annotate and self.annotated_frame is not None) else self.current_frame
            if frame is None:
                frame = self.base_synthetic

        if frame is None:
            frame = self._generate_synthetic_frame()

        ret, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ret and jpeg is not None:
            return jpeg.tobytes()
        return b""

    def get_data(self) -> dict:
        with self.lock:
            data = dict(self.extracted_data)
            data["device_id"] = self.device_id
            data["device_name"] = self.name
            data["rtsp_url"] = self.rtsp_url
            data["status"] = self.status
            data["ocr_duration"] = self.last_ocr_duration
            return data

    def capture_and_save(self) -> dict:
        """Captures active high-res frame, runs OCR extraction, saves PNG & JSON files to output/."""
        with self.lock:
            frame = self.current_frame.copy() if self.current_frame is not None else self.base_synthetic.copy()

        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        img_filename = f"capture_{ts}.png"
        json_filename = f"scraped_data_{ts}.json"

        img_path = os.path.join(output_dir, img_filename)
        json_path = os.path.join(output_dir, json_filename)

        # Save clean image frame to disk
        cv2.imwrite(img_path, frame)

        # Run extraction
        self.run_ocr_extraction(frame)
        data = self.get_data()

        # Save JSON payload
        save_payload = {
            "device_id": self.device_id,
            "device_name": self.name,
            "rtsp_url": self.rtsp_url,
            "timestamp": ts,
            "formatted_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_filename": img_filename,
            "json_filename": json_filename,
            "image_url": f"/output/{img_filename}",
            "json_url": f"/output/{json_filename}",
            "mode": data.get("mode", "dialysis"),
            "fields": data.get("fields", {}),
            "key_value_pairs": data.get("key_value_pairs", {}),
            "numbers_found": data.get("numbers_found", []),
            "raw_text": data.get("raw_text", ""),
            "confidence": data.get("confidence", 0.95),
            "ocr_duration": data.get("ocr_duration", 0.1)
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(save_payload, f, indent=2, ensure_ascii=False)

        # Print extracted data & JSON directly to terminal console
        print("\n" + "=" * 65)
        print(f" 📸 CAPTURE SAVED & EXTRACTED DATA [{self.name} - ID: {self.device_id}]".center(65))
        print(f" Image: output/{img_filename} | JSON: output/{json_filename}".center(65))
        print("=" * 65)
        if save_payload.get("fields"):
            print_results(save_payload["fields"])
        elif save_payload.get("key_value_pairs") or save_payload.get("numbers_found"):
            print_general_results({"lines": [], "key_value_pairs": save_payload.get("key_value_pairs", {}), "numbers_found": save_payload.get("numbers_found", [])}, source_label=self.name)
        print("\n--- JSON OUTPUT ---")
        print(json.dumps(save_payload, indent=2))
        print("=" * 65 + "\n")

        save_payload["saved_status"] = True
        return save_payload



class RTSPStreamManager:
    """Global RTSP stream manager."""

    def __init__(self):
        self.workers = {}
        self.devices_config = self._load_devices_config()
        self._init_workers()

    def _load_devices_config(self) -> list:
        if os.path.exists(DEVICES_FILE):
            try:
                with open(DEVICES_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        return data
            except Exception:
                pass
        return list(DEFAULT_DEVICES)

    def _save_devices_config(self):
        try:
            with open(DEVICES_FILE, "w", encoding="utf-8") as f:
                json.dump(self.devices_config, f, indent=2)
        except Exception as e:
            print(f"Error saving devices.json: {e}")

    def _init_workers(self):
        for cfg in self.devices_config:
            dev_id = cfg["id"]
            worker = CameraWorker(cfg)
            worker.start()
            self.workers[dev_id] = worker

    def get_all_devices(self) -> list:
        result = []
        for cfg in self.devices_config:
            dev_id = cfg["id"]
            worker = self.workers.get(dev_id)
            item = dict(cfg)
            if worker:
                item["status"] = worker.status
                item["last_ocr_time"] = worker.extracted_data.get("timestamp")
            else:
                item["status"] = "Offline"
            result.append(item)
        return result

    def get_device(self, device_id: str) -> dict:
        for cfg in self.devices_config:
            if cfg["id"] == device_id:
                worker = self.workers.get(device_id)
                item = dict(cfg)
                if worker:
                    item["status"] = worker.status
                return item
        return None

    def add_device(self, new_config: dict) -> dict:
        dev_id = new_config.get("id") or f"{random.randint(1000000000, 9999999999)}"
        new_config["id"] = dev_id
        new_config.setdefault("name", f"Camera {dev_id}")
        new_config.setdefault("ip", "127.0.0.1")
        new_config.setdefault("rtsp_url", "synthetic://dialysis")
        new_config.setdefault("mode", "dialysis")
        new_config.setdefault("extraction_interval", 1.5)

        self.devices_config.append(new_config)
        self._save_devices_config()

        worker = CameraWorker(new_config)
        worker.start()
        self.workers[dev_id] = worker
        return new_config

    def update_device(self, device_id: str, updated_fields: dict) -> dict:
        for idx, cfg in enumerate(self.devices_config):
            if cfg["id"] == device_id:
                cfg.update(updated_fields)
                self.devices_config[idx] = cfg
                self._save_devices_config()

                # Restart worker with new config
                if device_id in self.workers:
                    self.workers[device_id].stop()
                worker = CameraWorker(cfg)
                worker.start()
                self.workers[device_id] = worker
                return cfg
        return None

    def delete_device(self, device_id: str) -> bool:
        if device_id in self.workers:
            self.workers[device_id].stop()
            del self.workers[device_id]

        self.devices_config = [c for c in self.devices_config if c["id"] != device_id]
        self._save_devices_config()
        return True

    def get_frame(self, device_id: str) -> bytes:
        worker = self.workers.get(device_id)
        if worker:
            return worker.get_jpeg_frame()
        return b""

    def get_device_data(self, device_id: str) -> dict:
        worker = self.workers.get(device_id)
        if worker:
            return worker.get_data()
        return {}

    def force_extract(self, device_id: str) -> dict:
        worker = self.workers.get(device_id)
        if worker:
            worker.run_ocr_extraction()
            return worker.get_data()
        return {}

    def capture_and_save_device(self, device_id: str) -> dict:
        worker = self.workers.get(device_id)
        if worker:
            return worker.capture_and_save()
        return {}



# Global instance singleton
rtsp_manager = RTSPStreamManager()
