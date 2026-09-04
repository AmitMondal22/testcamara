"""
app.py
------
FastAPI Web Application Server for RTSP Multi-Camera Surveillance & Live Data Extraction.
Serves Jinja2 templates, MJPEG stream endpoints, and REST API.
"""

import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000|stimeout;2000000"
import asyncio
import json
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, Request, HTTPException, Response
# pyrefly: ignore [missing-import]
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
# pyrefly: ignore [missing-import]
from fastapi.staticfiles import StaticFiles
# pyrefly: ignore [missing-import]
from fastapi.templating import Jinja2Templates
# pyrefly: ignore [missing-import]
from pydantic import BaseModel

from typing import Optional, List

from src.rtsp_manager import rtsp_manager
from src.capture import discover_camera_details

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(
    title="Vision Scraper - Dialysis Monitor Live Telemetry & OCR Extraction",
    description="FastAPI Web UI for Dialysis machine streaming and real-time screen OCR extraction.",
    version="1.0.0"
)

# Mount Static assets, Output files, and Jinja2 Templates
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))



# Pydantic Request Models
class DeviceConfigModel(BaseModel):
    id: str
    name: str
    ip: Optional[str] = "127.0.0.1"
    camera_source: Optional[str] = "0"
    mode: Optional[str] = "dialysis"
    extraction_interval: Optional[float] = 1.0
    show_boxes: Optional[bool] = True


# ============================================================================
# JINJA2 FRONTEND DASHBOARD
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """Renders the main Jinja2 Surveillance & Data Extraction Control Center."""
    devices = rtsp_manager.get_all_devices()
    active_device = devices[0] if devices else None
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "devices": devices,
            "active_device": active_device
        }
    )


# ============================================================================
# RTSP MJPEG LIVE STREAMING ENDPOINTS
# ============================================================================

async def generate_mjpeg_frames(device_id: str):
    """Async generator for high-speed MJPEG video stream to browser."""
    while True:
        frame_bytes = rtsp_manager.get_frame(device_id)
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        await asyncio.sleep(0.015)  # Fast 60 FPS video stream loop



@app.get("/api/stream/{device_id}")
async def get_mjpeg_stream(device_id: str):
    """Returns continuous MJPEG video stream feed for target RTSP device."""
    return StreamingResponse(
        generate_mjpeg_frames(device_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/stream/{device_id}/snapshot")
async def get_stream_snapshot(device_id: str):
    """Returns current captured frame snapshot as downloadable/viewable JPEG image."""
    frame_bytes = rtsp_manager.get_frame(device_id)
    if not frame_bytes:
        raise HTTPException(status_code=404, detail="Device stream frame unavailable")
    return Response(content=frame_bytes, media_type="image/jpeg")


# ============================================================================
# REST API FOR CAMERA DEVICES & DATA EXTRACTION
# ============================================================================

@app.get("/api/devices")
async def list_devices():
    """Returns list of registered RTSP camera devices."""
    return rtsp_manager.get_all_devices()


@app.post("/api/devices")
async def create_device(dev: DeviceConfigModel):
    """Registers a new RTSP camera device."""
    created = rtsp_manager.add_device(dev.dict())
    return created


@app.get("/api/devices/{device_id}")
async def get_device_info(device_id: str):
    """Gets details for a single camera device."""
    dev = rtsp_manager.get_device(device_id)
    if not dev:
        raise HTTPException(status_code=404, detail="Device not found")
    return dev


@app.put("/api/devices/{device_id}")
async def update_device(device_id: str, dev: DeviceConfigModel):
    """Updates device settings (RTSP URL, Name, Extraction Mode)."""
    updated = rtsp_manager.update_device(device_id, dev.dict())
    if not updated:
        raise HTTPException(status_code=404, detail="Device not found")
    return updated


@app.delete("/api/devices/{device_id}")
async def delete_device(device_id: str):
    """Removes a camera device."""
    success = rtsp_manager.delete_device(device_id)
    return {"status": "success", "deleted": success}


@app.get("/api/devices/{device_id}/data")
async def get_device_extracted_data(device_id: str):
    """Gets real-time extracted data readings, OCR fields, and chart metrics."""
    data = rtsp_manager.get_device_data(device_id)
    if not data:
        raise HTTPException(status_code=404, detail="Device data unavailable")
    return data


@app.post("/api/devices/{device_id}/extract")
async def trigger_manual_extraction(device_id: str):
    """Forces instant OCR extraction burst on active camera frame."""
    data = rtsp_manager.force_extract(device_id)
    return data


@app.get("/api/cameras/discover")
async def discover_hardware_cameras():
    """Scans and discovers connected local webcam hardware (Laptop built-in & attached USB cameras)."""
    cams = discover_camera_details(max_tested=4)
    return {"cameras": cams, "count": len(cams)}


@app.post("/api/devices/{device_id}/capture-and-save")
async def capture_and_save_frame(device_id: str):
    """Captures a frame on demand, performs OCR extraction, and saves image PNG and JSON data to output/."""
    result = rtsp_manager.capture_and_save_device(device_id)
    if not result:
        raise HTTPException(status_code=404, detail="Device frame capture failed")
    return result


@app.get("/api/captures")
async def list_saved_captures():
    """Lists saved image captures and JSON data extraction files in output/ directory."""
    files = []
    if os.path.exists(OUTPUT_DIR):
        for fname in sorted(os.listdir(OUTPUT_DIR), reverse=True):
            if fname.startswith("scraped_data_") and fname.endswith(".json"):
                fpath = os.path.join(OUTPUT_DIR, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        files.append(meta)
                except Exception:
                    pass
    return {"captures": files}


@app.get("/api/captures/download/{filename}")
async def download_capture_file(filename: str):
    """Allows downloading of saved JSON or PNG capture files."""
    fpath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="File not found")
    media_type = "image/png" if filename.endswith(".png") else "application/json"
    return FileResponse(path=fpath, filename=filename, media_type=media_type)



# Direct script runner
if __name__ == "__main__":
    # pyrefly: ignore [missing-import]
    import uvicorn
    print("=" * 65)
    print(" Starting Vision Scraper - RTSP Multi-Camera FastAPI Server...")
    print(" Click or Open in Browser: http://127.0.0.1:8000")
    print("                           or: http://localhost:8000")
    print("=" * 65)
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
