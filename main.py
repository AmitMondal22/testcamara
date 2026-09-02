"""
main.py
-------
CLI entry point for Webcam Image Data Extractor & Terminal Scraper.

Usage:
    python main.py                              # Interactive menu
    python main.py --webcam                     # Webcam GUI capture -> scrape & print to terminal
    python main.py --live                       # Live real-time webcam text scraper
    python main.py --headless                   # Non-interactive webcam capture
    python main.py --upload path/to/image.jpg   # Extract image data from existing file
    python main.py --mode dialysis              # Use specialized dialysis screen parser

Options:
    --engine {auto,easyocr,tesseract}
    --mode {general,dialysis}
    --camera 0
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
try:
    # pyrefly: ignore [missing-import]
    import torch
except Exception:
    pass

import argparse
import json
import sys
import cv2
from datetime import datetime

from src.ocr_extract import load_image, extract_image_data, extract_text
from src.field_parser import (
    parse_general_data,
    print_general_results,
    parse_fields,
    print_results,
)
from src.screen_extractor import extract_fields

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# Warm up OCR engine at startup so camera capture runs instantly
try:
    from src.ocr_extract import _get_easyocr_reader
    _get_easyocr_reader()
except Exception:
    pass


def process_burst_images(frames: list, source_label: str = "Webcam Burst", mode: str = "dialysis", engine: str = "auto"):
    """
    Processes a burst of 3 frames captured over 1.5s:
      1. Runs unwarping + spatial label-value extraction on each frame
      2. Performs majority consensus voting across all frames for maximum accuracy
      3. Prints clean results table to terminal
      4. Saves JSON results and primary PNG image to output/ folder
    """
    if not frames:
        print("No frames provided to process. Aborting.")
        return None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_save_path = os.path.join(OUTPUT_DIR, f"capture_{ts}.png")
    cv2.imwrite(image_save_path, frames[0])

    if mode == "dialysis":
        print(f"\nProcessing {len(frames)}-Frame Burst with Consensus Voting on: {source_label} ...")
        frame_results = []
        for idx, frame in enumerate(frames, 1):
            print(f" -> Processing Frame {idx}/{len(frames)} (Perspective Unwarp & Spatial OCR)...")
            res = extract_fields(frame, engine=engine)
            frame_results.append(res)

        from src.field_parser import consensus_vote_dialysis_fields
        consensus_results = consensus_vote_dialysis_fields(frame_results)

        print_results(consensus_results)

        lines_data = extract_image_data(frames[0], engine=engine)
        parsed_data = parse_general_data(lines_data)
        if parsed_data.get("numbers_found"):
            print("--- ALL RAW NUMERIC READINGS DETECTED IN PHOTO ---")
            unique_nums = list(dict.fromkeys(parsed_data["numbers_found"]))
            print("  " + ", ".join(unique_nums))
            print("=" * 55 + "\n")

        saved_payload = {
            "source": source_label,
            "mode": "dialysis_burst",
            "timestamp": ts,
            "image_saved_at": image_save_path,
            "total_frames_analyzed": len(frames),
            "fields": consensus_results,
            "all_raw_numbers": parsed_data.get("numbers_found", []),
            "raw_text": parsed_data.get("raw_text", "")
        }
    else:
        print(f"\nScraping Image Data from: {source_label} ...")
        lines_data = extract_image_data(frames[0], engine=engine)
        parsed_data = parse_general_data(lines_data)
        print_general_results(parsed_data, source_label=source_label)

        saved_payload = {
            "source": source_label,
            "mode": "general",
            "timestamp": ts,
            "image_saved_at": image_save_path,
            "total_lines": len(parsed_data["lines"]),
            "key_value_pairs": parsed_data["key_value_pairs"],
            "numbers_found": parsed_data["numbers_found"],
            "raw_text": parsed_data["raw_text"],
        }

    out_json_path = os.path.join(OUTPUT_DIR, f"scraped_data_{ts}.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(saved_payload, f, indent=2, ensure_ascii=False)

    print(f"Saved consensus JSON to: {out_json_path}")
    print(f"Saved captured frame image to: {image_save_path}\n")

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
    default_cam = cams[-1] if len(cams) > 1 else cams[0]
    cam_str = input(f"Enter Camera Index to use (0 = Laptop Camera, 1 = USB Webcam) [Default: {default_cam}]: ").strip()
    if not cam_str:
        return default_cam
    try:
        return int(cam_str)
    except ValueError:
        print(f"Invalid input. Using camera index {default_cam}.")
        return default_cam


def run_upload(path: str, mode: str = "dialysis", engine: str = "auto"):
    path_clean = path.strip().strip('"').strip("'")
    img = load_image(path_clean)
    process_burst_images([img], source_label=os.path.basename(path_clean), mode=mode, engine=engine)


def run_webcam(headless: bool = False, camera_index: int = 0, mode: str = "dialysis", engine: str = "auto"):
    from src.capture import capture_from_webcam, capture_headless
    if headless:
        print("Capturing 3-frame burst from webcam (headless mode)...")
        frames = capture_headless(camera_index=camera_index, num_frames=3)
    else:
        frames = capture_from_webcam(camera_index=camera_index, num_frames=3)

    if not frames:
        print("No frames captured. Aborting.")
        return

    process_burst_images(frames, source_label=f"Webcam (Camera #{camera_index})", mode=mode, engine=engine)


def run_live(camera_index: int = 0, mode: str = "dialysis", engine: str = "auto"):
    from src.capture import capture_live_stream

    def live_callback(frame_or_frames):
        frames = frame_or_frames if isinstance(frame_or_frames, list) else [frame_or_frames]
        process_burst_images(frames, source_label=f"Live Webcam #{camera_index}", mode=mode, engine=engine)

    capture_live_stream(camera_index=camera_index, process_fn=live_callback, frame_interval=1.0)


def interactive_menu():
    print("=" * 55)
    print(" WEBCAM IMAGE DATA EXTRACTOR & TERMINAL SCRAPER ")
    print("=" * 55)
    print("1) Capture from Webcam (GUI preview window - Dialysis Mode)")
    print("2) Live Webcam Stream Scraping (Continuous real-time OCR)")
    print("3) Capture from Webcam (Headless mode)")
    print("4) Upload an Image file")
    print("5) Dialysis Screen Mode (Specialized Medical Display)")
    print("Q) Quit")
    print("-" * 55)

    choice = input("Choose an option [1-5 / Q]: ").strip().lower()

    if choice == "1":
        cam_idx = choose_camera_index()
        run_webcam(headless=False, camera_index=cam_idx, mode="dialysis")
    elif choice == "2":
        cam_idx = choose_camera_index()
        run_live(camera_index=cam_idx, mode="dialysis")
    elif choice == "3":
        cam_idx = choose_camera_index()
        run_webcam(headless=True, camera_index=cam_idx, mode="dialysis")
    elif choice == "4":
        path = input("Enter path to image file: ").strip().strip('"').strip("'")
        if path:
            run_upload(path, mode="dialysis")
    elif choice == "5":
        path = input("Enter image path (or press Enter for webcam capture): ").strip().strip('"').strip("'")
        if path:
            run_upload(path, mode="dialysis")
        else:
            cam_idx = choose_camera_index()
            run_webcam(mode="dialysis", camera_index=cam_idx)
    elif choice in ("q", "quit", "exit"):
        print("Exiting.")
        sys.exit(0)
    else:
        print("Invalid choice.")


def main():
    parser = argparse.ArgumentParser(description="Webcam Image Data Extractor & Terminal Scraper")
    parser.add_argument("-u", "--upload", help="Path to an image file to process")
    parser.add_argument("-w", "--webcam", action="store_true", help="Capture frame interactively from webcam GUI")
    parser.add_argument("-H", "--headless", action="store_true", help="Capture webcam frame headlessly")
    parser.add_argument("-L", "--live", action="store_true", help="Run live webcam text scraper stream")
    parser.add_argument("-c", "--camera", type=int, default=0, help="Webcam camera index (default: 0)")
    parser.add_argument("-m", "--mode", choices=["general", "dialysis"], default="dialysis", help="Extraction mode (default: dialysis)")
    parser.add_argument("-e", "--engine", choices=["auto", "easyocr", "tesseract"], default="auto", help="OCR Engine")

    args = parser.parse_args()

    if args.upload:
        run_upload(args.upload, mode=args.mode, engine=args.engine)
    elif args.live:
        run_live(camera_index=args.camera, mode=args.mode, engine=args.engine)
    elif args.webcam or args.headless:
        run_webcam(headless=args.headless, camera_index=args.camera, mode=args.mode, engine=args.engine)
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
