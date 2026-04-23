"""
Real-Time Face Swap — Virtual Camera
=====================================
Swaps your webcam face with a target photo face in real time.
Output goes to a virtual camera readable by Zoom / Meet / Teams.

Requirements: see requirements.txt
Run:  python face_swap.py --target path/to/photo.jpg
"""

import cv2
import insightface
import numpy as np
import pyvirtualcam
import argparse
import sys
import time
from pathlib import Path


def load_models():
    print("[*] Loading InsightFace models...")
    # Auto-detect GPU
    providers = ["CPUExecutionProvider"]
    try:
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            print("[*] Using GPU (CUDA)")
        else:
            print("[*] Using CPU")
    except ImportError:
        print("[*] onnxruntime not fully available, using CPU")

    # Face analyser — detects & embeds faces
    app = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)

    # Face swapper model (inswapper_128)
    model_path = Path("inswapper_128.onnx")
    if not model_path.exists():
        print("\n[!] inswapper_128.onnx not found.")
        print("    Download it from:")
        print("    https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx")
        print("    and place it in the same folder as face_swap.py\n")
        sys.exit(1)

    swapper = insightface.model_zoo.get_model(
        str(model_path), providers=providers
    )
    print("[*] Models loaded.\n")
    return app, swapper


def get_target_face(app, target_path: str):
    img = cv2.imread(target_path)
    if img is None:
        print(f"[!] Could not load image: {target_path}")
        sys.exit(1)
    faces = app.get(img)
    if not faces:
        print("[!] No face detected in target photo. Try a clearer, front-facing photo.")
        sys.exit(1)
    print(f"[*] Target face loaded from: {target_path}")
    return faces[0]


def run(target_path: str, cam_index: int = 0, fps: int = 20, det_size: tuple = (320, 320), skip_frames: int = 1):
    app, swapper = load_models()
    app.prepare(ctx_id=0, det_size=det_size)
    target_face = get_target_face(app, target_path)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[!] Cannot open webcam index {cam_index}")
        sys.exit(1)

    # Limit webcam resolution for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[*] Webcam resized to: {width}x{height} @ {fps}fps (det_size={det_size}, skip={skip_frames})")

    frame_count = 0
    avg_fps = 0
    perf_interval = 30  # Print perf every 30 frames

    print("[*] Starting virtual camera... (press Ctrl+C to stop)\n")

    with pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR) as vcam:
        print(f"[*] Virtual camera: {vcam.device}")
        print("    Select this device in Zoom / Meet / Teams as your camera.\n")

        loop_start = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] Failed to read webcam frame.")
                break

            frame_count += 1

            # Frame skipping for speed
            if frame_count % skip_frames == 0:
                frame_start = time.time()
                # Resize frame if too large
                h, w = frame.shape[:2]
                if max(h, w) > 640:
                    scale = 640 / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h))

                # Detect faces in live frame
                det_start = time.time()
                faces = app.get(frame)
                det_time = time.time() - det_start

                if faces:
                    swap_start = time.time()
                    # Swap each detected face with target
                    for face in faces:
                        frame = swapper.get(frame, face, target_face, paste_back=True)
                    swap_time = time.time() - swap_start
                else:
                    swap_time = 0

                process_time = time.time() - frame_start

            # Send to virtual camera
            vcam.send(frame)
            vcam.sleep_until_next_frame()

            # Performance monitoring
            loop_time = time.time() - loop_start
            if frame_count % perf_interval == 0:
                current_fps = perf_interval / loop_time
                avg_fps = 0.9 * avg_fps + 0.1 * current_fps if avg_fps else current_fps
                print(f"[*] Perf: FPS={avg_fps:.1f}, process={process_time*1000:.0f}ms, detect={det_time*1000:.0f}ms, swap={swap_time*1000:.0f}ms")
                loop_start = time.time()

    cap.release()
    print("[*] Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time face swap virtual camera")
    parser.add_argument("--target", required=True, help="Path to target face photo (JPG/PNG)")
    parser.add_argument("--cam",    type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--fps",    type=int, default=20, help="Virtual camera FPS (default: 20)")
    parser.add_argument("--det-size", nargs=2, type=int, default=[320, 320], help="Detection size e.g. 320 320 (lower=faster)")
    parser.add_argument("--skip-frames", type=int, default=1, help="Process every Nth frame (higher=faster, default 1)")
    args = parser.parse_args()

    run(target_path=args.target, cam_index=args.cam, fps=args.fps, det_size=tuple(args.det_size), skip_frames=args.skip_frames)

