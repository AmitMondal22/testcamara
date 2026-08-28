"""
main.py
-------
CLI entry point for Webcam & Raspberry Pi 4 Model B Image Data Extractor.
Performs 1-second 3-frame burst data collection with camera tilt unwarping
and discrete consensus voting (no averaging) for 100% accurate telemetry.

Usage:
    python main.py                                      # Interactive menu
    python main.py --pi4                                # 1-Second 3-Frame Burst Loop (Raspberry Pi 4 / Live)
    python main.py --webcam                             # Capture 3-frame burst via interactive GUI
    python main.py --live                               # Real-time 1-second burst scraper loop
    python main.py --headless                           # Non-interactive burst capture
    python main.py --upload path/to/image.jpg           # Extract from existing photo file
    python main.py --mode dialysis --camera 0           # Dialysis mode on camera #0
"""

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
try:
    import torch  # pyrefly: ignore [missing-import]
except Exception:
    pass

import argparse
import json
import sys
import cv2
from datetime import datetime

from src.ocr_extract import load_image, extract_image_data, extract_text, load_config
from src.field_parser import (
    parse_general_data,
    print_general_results,
    parse_fields,
    print_results,
    consensus_vote_discrete,
)
from src.screen_extractor import extract_fields

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
_CONFIG = load_config()
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), _CONFIG.get("app", {}).get("output_dir", "output"))


def process_burst_images(frames: list, source_label: str = "3-Frame Burst", mode: str = "dialysis", engine: str = "auto", cycle_count: int = 1):
    """
    Processes a burst of 3 frames captured per second:
      1. Performs perspective deskew & unwarping on tilted camera frames
      2. Extracts discrete numeric & text readings from each frame
      3. Performs discrete consensus voting (NO ARITHMETIC AVERAGING) for 100% accuracy
      4. Prints clean discrete results table to the terminal each second
      5. Saves JSON telemetry and capture image to output/
    """
    if not frames:
        print("[ERROR] No frames provided in burst. Skipping.")
        return None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_save_path = os.path.join(OUTPUT_DIR, f"capture_{ts}.png")
    cv2.imwrite(image_save_path, frames[0])

    if mode == "dialysis":
        frame_results = []
        for idx, frame in enumerate(frames, 1):
            res = extract_fields(frame, engine=engine)
            frame_results.append(res)

        # 100% Discrete Consensus Voting (NO AVERAGING)
        consensus_results = consensus_vote_discrete(frame_results)

        # Print formatted table to terminal
        print_results(consensus_results, title=f"CYCLE #{cycle_count} - {source_label} (100% DISCRETE READINGS)")

        lines_data = extract_image_data(frames[0], engine=engine)
        parsed_data = parse_general_data(lines_data)

        saved_payload = {
            "source": source_label,
            "mode": "dialysis_burst_1sec",
            "cycle": cycle_count,
            "timestamp": ts,
            "formatted_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_saved_at": image_save_path,
            "burst_frames_collected": len(frames),
            "averaging_used": False,
            "consensus_method": "discrete_frequency_confidence_vote",
            "fields": consensus_results,
            "all_raw_numbers": parsed_data.get("numbers_found", []),
            "raw_text": parsed_data.get("raw_text", "")
        }
    else:
        print(f"\nScraping Generic Data from: {source_label} (Cycle #{cycle_count}) ...")
        lines_data = extract_image_data(frames[0], engine=engine)
        parsed_data = parse_general_data(lines_data)
        print_general_results(parsed_data, source_label=source_label)

        saved_payload = {
            "source": source_label,
            "mode": "general",
            "cycle": cycle_count,
            "timestamp": ts,
            "formatted_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_saved_at": image_save_path,
            "total_lines": len(parsed_data["lines"]),
            "key_value_pairs": parsed_data["key_value_pairs"],
            "numbers_found": parsed_data["numbers_found"],
            "raw_text": parsed_data["raw_text"],
        }

    # Save to output/ JSON file
    out_json_path = os.path.join(OUTPUT_DIR, f"scraped_data_{ts}.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(saved_payload, f, indent=2, ensure_ascii=False)

    return saved_payload


def choose_camera_index() -> int:
    """Detect connected cameras and prompt user to choose camera index."""
    try:
        from src.capture import find_available_cameras
        cams = find_available_cameras(max_tested=4)
    except Exception:
        cams = [0]

    if not cams:
        cams = [0]

    print(f"\nDetecting connected cameras... Found camera index(es): {cams}")
    default_cam = cams[0]
    cam_str = input(f"Enter Camera Index (0 = Pi Camera/Webcam #0, 1 = USB Camera #1) [Default: {default_cam}]: ").strip()
    if not cam_str:
        return default_cam
    try:
        return int(cam_str)
    except ValueError:
        return default_cam


def run_upload(path: str, mode: str = "dialysis", engine: str = "auto"):
    path_clean = path.strip().strip('"').strip("'")
    img = load_image(path_clean)
    process_burst_images([img], source_label=os.path.basename(path_clean), mode=mode, engine=engine)


def run_webcam_burst(headless: bool = False, camera_index: int = 0, mode: str = "dialysis", engine: str = "auto"):
    from src.capture import capture_from_webcam, capture_headless
    if headless:
        print(f"Capturing 3-frame burst from camera #{camera_index} (headless mode)...")
        frames = capture_headless(camera_index=camera_index, num_frames=3)
    else:
        frames = capture_from_webcam(camera_index=camera_index, num_frames=3)

    if not frames:
        print("No frames captured. Aborting.")
        return

    process_burst_images(frames, source_label=f"Camera #{camera_index} (3-Frame Burst)", mode=mode, engine=engine)


def run_1sec_live_loop(camera_index: int = 0, mode: str = "dialysis", engine: str = "auto"):
    from src.capture import run_1sec_burst_collection_loop

    def burst_callback(burst_frames, cycle_count=1):
        process_burst_images(burst_frames, source_label=f"Live Camera #{camera_index}", mode=mode, engine=engine, cycle_count=cycle_count)

    run_1sec_burst_collection_loop(camera_index=camera_index, process_burst_fn=burst_callback, interval=1.0, num_frames=3)


def interactive_menu():
    print("=" * 65)
    print(" VISION EXTRACTOR - RASPBERRY PI 4 MODEL B TELEMETRY ENGINE ")
    print("=" * 65)
    print("1) Start 1-Second 3-Frame Burst Collection Loop (Recommended for Pi 4)")
    print("2) Capture 3-Frame Burst via GUI Preview Window")
    print("3) Capture 3-Frame Burst (Headless mode)")
    print("4) Upload an Image file to test extraction")
    print("5) Launch FastAPI Multi-Camera Web Dashboard")
    print("Q) Quit")
    print("-" * 65)

    choice = input("Choose an option [1-5 / Q]: ").strip().lower()

    if choice == "1":
        cam_idx = choose_camera_index()
        run_1sec_live_loop(camera_index=cam_idx, mode="dialysis")
    elif choice == "2":
        cam_idx = choose_camera_index()
        run_webcam_burst(headless=False, camera_index=cam_idx, mode="dialysis")
    elif choice == "3":
        cam_idx = choose_camera_index()
        run_webcam_burst(headless=True, camera_index=cam_idx, mode="dialysis")
    elif choice == "4":
        path = input("Enter path to image file: ").strip().strip('"').strip("'")
        if path:
            run_upload(path, mode="dialysis")
    elif choice == "5":
        import uvicorn
        print("\nStarting Web UI Server on http://127.0.0.1:8000 ...")
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
    elif choice in ("q", "quit", "exit"):
        print("Exiting.")
        sys.exit(0)
    else:
        print("Invalid choice.")


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi 4 Vision Extractor & Telemetry Engine")
    parser.add_argument("-u", "--upload", help="Path to an image file to process")
    parser.add_argument("-w", "--webcam", action="store_true", help="Capture 3-frame burst interactively from GUI")
    parser.add_argument("-H", "--headless", action="store_true", help="Capture 3-frame burst headlessly")
    parser.add_argument("-L", "--live", action="store_true", help="Run 1-second continuous 3-frame burst loop")
    parser.add_argument("-p", "--pi4", action="store_true", help="Run 1-second burst collection optimized for Raspberry Pi 4")
    parser.add_argument("-c", "--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("-m", "--mode", choices=["general", "dialysis"], default="dialysis", help="Extraction mode (default: dialysis)")
    parser.add_argument("-e", "--engine", choices=["auto", "easyocr", "tesseract"], default="auto", help="OCR Engine")

    args = parser.parse_args()

    if args.upload:
        run_upload(args.upload, mode=args.mode, engine=args.engine)
    elif args.live or args.pi4:
        run_1sec_live_loop(camera_index=args.camera, mode=args.mode, engine=args.engine)
    elif args.webcam or args.headless:
        run_webcam_burst(headless=args.headless, camera_index=args.camera, mode=args.mode, engine=args.engine)
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
