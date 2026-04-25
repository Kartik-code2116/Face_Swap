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
import threading
from pathlib import Path


class LatestFrameCamera:
    def __init__(self, cam_index: int, width: int, height: int, fps: int):
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(cam_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open webcam index {cam_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
        self.lock = threading.Lock()
        self.latest_frame = None
        self.stopped = False
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.latest_frame = frame

    def read(self):
        with self.lock:
            if self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()

    def release(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()


def load_models():
    print("[*] Loading InsightFace models...")
    # Auto-detect GPU
    providers = ["CPUExecutionProvider"]
    use_cuda = False
    try:
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            use_cuda = True
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
    return app, swapper, use_cuda


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


def run(
    target_path: str,
    cam_index: int = 0,
    fps: int = 20,
    det_size: tuple = (256, 256),
    skip_frames: int = 2,
    width: int = 640,
    height: int = 480,
    process_width: int = 480,
):
    app, swapper, use_cuda = load_models()
    app.prepare(ctx_id=0 if use_cuda else -1, det_size=det_size)
    target_face = get_target_face(app, target_path)

    try:
        cap = LatestFrameCamera(cam_index, width=width, height=height, fps=fps)
    except RuntimeError as exc:
        print(f"[!] {exc}")
        sys.exit(1)

    width = cap.width
    height = cap.height
    print(f"[*] Webcam resized to: {width}x{height} @ {fps}fps (det_size={det_size}, skip={skip_frames})")

    frame_count = 0
    avg_fps = 0
    perf_interval = 30  # Print perf every 30 frames
    last_output = np.zeros((height, width, 3), dtype=np.uint8)
    process_time = 0.0
    det_time = 0.0
    swap_time = 0.0

    print("[*] Starting virtual camera... (press Ctrl+C to stop)\n")

    with pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR) as vcam:
        print(f"[*] Virtual camera: {vcam.device}")
        print("    Select this device in Zoom / Meet / Teams as your camera.\n")

        loop_start = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                vcam.send(last_output)
                vcam.sleep_until_next_frame()
                continue

            frame_count += 1

            # Frame skipping for speed. When skipping, keep sending the last processed
            # frame so video-call apps see a stable feed instead of stale queued frames.
            if frame_count == 1 or frame_count % max(skip_frames, 1) == 0:
                frame_start = time.time()
                processed_frame = frame.copy()

                # Resize frame if too large
                h, w = processed_frame.shape[:2]
                scale = 1.0
                if process_width > 0 and max(h, w) > process_width:
                    scale = process_width / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    processed_frame = cv2.resize(processed_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # Detect faces in live frame
                det_start = time.time()
                faces = app.get(processed_frame)
                det_time = time.time() - det_start

                if faces:
                    swap_start = time.time()
                    # Swap each detected face with target
                    for face in faces:
                        processed_frame = swapper.get(processed_frame, face, target_face, paste_back=True)
                    swap_time = time.time() - swap_start
                else:
                    swap_time = 0

                if scale != 1.0:
                    processed_frame = cv2.resize(processed_frame, (width, height), interpolation=cv2.INTER_LINEAR)

                last_output = processed_frame
                process_time = time.time() - frame_start

            # Send to virtual camera
            vcam.send(last_output)
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
    parser.add_argument("--det-size", nargs=2, type=int, default=[256, 256], help="Detection size e.g. 256 256 (lower=faster)")
    parser.add_argument("--skip-frames", type=int, default=2, help="Process every Nth frame (higher=faster, default 2)")
    parser.add_argument("--width", type=int, default=640, help="Capture width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Capture height (default: 480)")
    parser.add_argument("--process-width", type=int, default=480, help="Max width used for face processing (default: 480)")
    args = parser.parse_args()

    run(
        target_path=args.target,
        cam_index=args.cam,
        fps=args.fps,
        det_size=tuple(args.det_size),
        skip_frames=args.skip_frames,
        width=args.width,
        height=args.height,
        process_width=args.process_width,
    )
