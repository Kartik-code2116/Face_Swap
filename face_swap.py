"""
Real-Time Face Swap — Virtual Camera
=====================================
Swaps your webcam face with a target photo face in real time.
Output goes to a virtual camera readable by Zoom / Meet / Teams.

Requirements: see requirements.txt
Run:  python face_swap.py --target path/to/photo.jpg

For use WITH a streaming app open (Zoom/Meet/OBS):
      python face_swap.py --target photo.jpg --low-cpu
"""

import cv2
import insightface
import numpy as np
import pyvirtualcam
import argparse
import sys
import time
import threading
import signal
import queue
from pathlib import Path

import os
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'   # Prevents MSMF stutter on Windows

# ── Windows thread-priority helper ──────────────────────────────────────────
def _boost_thread_priority():
    """Raise the current thread to ABOVE_NORMAL priority on Windows."""
    try:
        import ctypes
        THREAD_SET_INFORMATION = 0x20
        ABOVE_NORMAL = 1
        handle = ctypes.windll.kernel32.GetCurrentThread()
        ctypes.windll.kernel32.SetThreadPriority(handle, ABOVE_NORMAL)
    except Exception:
        pass   # Non-Windows or no permission — silently skip


# Global shutdown flag
shutdown_requested = False

def _signal_handler(sig, frame):
    global shutdown_requested
    print("\n[!] Shutdown requested, cleaning up...")
    shutdown_requested = True

signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ── Webcam reader ────────────────────────────────────────────────────────────
class LatestFrameCamera:
    """
    Background-thread webcam reader.
    Always exposes the LATEST frame via a zero-copy atomic pointer swap.
    The main thread never blocks waiting for a lock held by the reader.
    """

    def __init__(self, cam_index: int, width: int, height: int, fps: int):
        self.cam_index = cam_index
        self._open_cap(cam_index, width, height, fps)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width  = actual_w if actual_w > 0 else width
        self.height = actual_h if actual_h > 0 else height

        self._frame   = None          # atomic pointer (GIL makes this safe)
        self._lock    = threading.Lock()
        self._ready   = threading.Event()

        self.stopped  = False
        self._thread  = threading.Thread(target=self._reader, daemon=True,
                                         name="cam-reader")
        self._thread.start()
        self._ready.wait(timeout=3.0)

    def _open_cap(self, cam_index, width, height, fps):
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open webcam index {cam_index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS,          fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # discard stale frames
        self.cap = cap

    def _reader(self):
        fails = 0
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                fails += 1
                if fails > 30:
                    print("[!] Camera read failures — re-opening...")
                    self.cap.release()
                    time.sleep(0.4)
                    self._open_cap(self.cam_index, self.width, self.height, 30)
                    fails = 0
                time.sleep(0.005)
                continue
            fails = 0
            # Atomic write — no copy needed here; the main thread copies on read
            with self._lock:
                self._frame = frame
            self._ready.set()

    def read(self):
        """Return (True, frame) or (False, None).  Copies only when called."""
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def release(self):
        self.stopped = True
        self._thread.join(timeout=1.0)
        self.cap.release()


# ── AI processing thread ─────────────────────────────────────────────────────
class FaceSwapProcessor:
    """
    Dedicated thread for face detection + swap so the send loop is never stalled
    by AI inference (which can take 100–800 ms on CPU).

    Design:
    • Input  slot (size-1): holds the NEWEST raw frame to process.
      If a newer frame arrives before the worker finishes the previous one,
      the old one is silently dropped — we always process the freshest data.
    • Output slot (size-1): holds the latest processed frame for the main loop.
    """

    def __init__(self, app, swapper, target_face,
                 process_width: int, low_cpu: bool = False):
        self.app           = app
        self.swapper       = swapper
        self.target_face   = target_face
        self.process_width = process_width
        self.low_cpu       = low_cpu   # insert a small sleep to yield CPU

        self._pending  = None          # raw frame waiting to be processed
        self._result   = None          # latest processed frame
        self._plock    = threading.Lock()
        self._rlock    = threading.Lock()
        self._work_evt = threading.Event()

        self.det_ms  = 0.0
        self.swap_ms = 0.0
        self.proc_ms = 0.0

        self.stopped = False
        self._thread = threading.Thread(target=self._worker, daemon=True,
                                        name="ai-worker")
        self._thread.start()

    def _worker(self):
        while not self.stopped:
            # Wait for work (100 ms timeout so we check stopped flag)
            if not self._work_evt.wait(timeout=0.1):
                continue
            self._work_evt.clear()

            with self._plock:
                frame = self._pending
                self._pending = None
            if frame is None:
                continue

            t0 = time.perf_counter()

            # ── Downscale for speed ──────────────────────────────────────
            h, w  = frame.shape[:2]
            scale = 1.0
            if self.process_width > 0 and max(h, w) > self.process_width:
                scale = self.process_width / max(h, w)
                proc  = cv2.resize(frame,
                                   (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_AREA)
            else:
                proc = frame

            # ── Detect faces ─────────────────────────────────────────────
            t1    = time.perf_counter()
            faces = self.app.get(proc)
            self.det_ms = (time.perf_counter() - t1) * 1000

            # ── Swap each face ───────────────────────────────────────────
            t2 = time.perf_counter()
            if faces:
                for face in faces:
                    proc = self.swapper.get(proc, face, self.target_face,
                                            paste_back=True)
            self.swap_ms = (time.perf_counter() - t2) * 1000

            # ── Upscale back to original resolution ──────────────────────
            if scale != 1.0:
                proc = cv2.resize(proc, (w, h), interpolation=cv2.INTER_LINEAR)

            self.proc_ms = (time.perf_counter() - t0) * 1000

            with self._rlock:
                self._result = proc

            # In low-cpu mode yield a little so Zoom/Meet get CPU time
            if self.low_cpu:
                time.sleep(0.005)

    def submit(self, frame: np.ndarray):
        """
        Non-blocking: store frame as pending work and signal the worker.
        If a frame is already pending (worker still busy), replace it so we
        always process the freshest one and never build a backlog.
        """
        with self._plock:
            self._pending = frame
        self._work_evt.set()

    def get_result(self):
        """Return the latest processed frame (or None if not ready yet)."""
        with self._rlock:
            return self._result

    def stop(self):
        self.stopped = True
        self._work_evt.set()          # unblock the worker so it can exit
        self._thread.join(timeout=2.0)


# ── Model helpers ─────────────────────────────────────────────────────────────
def load_models():
    print("[*] Loading InsightFace models...")
    providers = ["CPUExecutionProvider"]
    use_cuda  = False
    try:
        import onnxruntime as ort
        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" in avail:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            use_cuda  = True
            print("[*] NVIDIA GPU detected — using CUDA")
        else:
            print("[*] No CUDA GPU — using CPU")
    except ImportError:
        print("[*] onnxruntime import failed — using CPU")

    app = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)

    model_path = Path("inswapper_128.onnx")
    if not model_path.exists():
        print("\n[!] inswapper_128.onnx not found.")
        print("    Download: https://huggingface.co/ezioruan/inswapper_128.onnx/"
              "resolve/main/inswapper_128.onnx")
        print("    Place it in the same folder as face_swap.py\n")
        sys.exit(1)

    swapper = insightface.model_zoo.get_model(str(model_path),
                                               providers=providers)
    print("[*] Models loaded.\n")
    return app, swapper, use_cuda


def get_target_face(app, target_path: str):
    img = cv2.imread(target_path)
    if img is None:
        print(f"[!] Cannot read image: {target_path}")
        sys.exit(1)
    faces = app.get(img)
    if not faces:
        print("[!] No face found in target image — use a clear, front-facing photo.")
        sys.exit(1)
    print(f"[*] Target face loaded: {target_path}")
    return faces[0]


# ── Virtual-camera helpers ────────────────────────────────────────────────────
def _open_vcam(width, height, fps):
    """Try RGB then BGR pixel format.  Return (vcam, fmt) or exit."""
    for fmt in (pyvirtualcam.PixelFormat.RGB, pyvirtualcam.PixelFormat.BGR):
        for attempt in range(3):
            try:
                vcam = pyvirtualcam.Camera(width=width, height=height,
                                           fps=fps, fmt=fmt)
                print(f"[*] Virtual camera opened  format={fmt.name}")
                return vcam, fmt
            except Exception as exc:
                print(f"    [{fmt.name}] attempt {attempt+1}/3 failed: {exc}")
                time.sleep(2.0)
    print("\n[!] Cannot open virtual camera.")
    print("    • Install OBS Studio + start OBS Virtual Camera once")
    print("    • Close any app holding the virtual camera and retry")
    sys.exit(1)


def _to_vcam(frame_bgr: np.ndarray,
             fmt: pyvirtualcam.PixelFormat) -> np.ndarray:
    if fmt == pyvirtualcam.PixelFormat.RGB:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_bgr


# ── Main send loop ────────────────────────────────────────────────────────────
def run(
    target_path  : str,
    cam_index    : int   = 0,
    fps          : int   = 30,
    det_size     : tuple = (256, 256),
    skip_frames  : int   = 2,
    width        : int   = 640,
    height       : int   = 480,
    process_width: int   = 480,
    low_cpu      : bool  = False,
):
    app, swapper, use_cuda = load_models()
    app.prepare(ctx_id=0 if use_cuda else -1, det_size=det_size)
    target_face = get_target_face(app, target_path)

    try:
        cap = LatestFrameCamera(cam_index, width=width, height=height, fps=fps)
    except RuntimeError as exc:
        print(f"[!] {exc}")
        sys.exit(1)

    width  = cap.width
    height = cap.height
    print(f"[*] Webcam      : {width}x{height} @ {fps} fps")
    print(f"[*] Detection   : {det_size}  skip={skip_frames}  process_width={process_width}")
    print(f"[*] Low-CPU mode: {'ON  (yields extra CPU to streaming apps)' if low_cpu else 'OFF'}\n")

    processor = FaceSwapProcessor(app, swapper, target_face,
                                  process_width, low_cpu=low_cpu)

    vcam, fmt = _open_vcam(width, height, fps)
    print(f"[*] Virtual cam : {vcam.device}")
    print("[*] Select this device in Zoom / Meet / Teams / OBS\n")
    print("[*] Running — Ctrl+C to stop\n")

    # Boost send-loop thread priority so streaming apps don't starve it
    _boost_thread_priority()

    frame_count  = 0
    last_output  = np.zeros((height, width, 3), dtype=np.uint8)

    # ── Manual frame timer (replaces vcam.sleep_until_next_frame) ────────────
    # vcam.sleep_until_next_frame() can stall when other apps compete for the
    # scheduler.  We manage our own deadline instead.
    frame_interval = 1.0 / fps
    next_deadline  = time.perf_counter() + frame_interval

    log_every = max(fps, 30)   # log once per second (approx)
    log_count = 0
    log_t0    = time.perf_counter()

    with vcam:
        while not shutdown_requested:

            # ── 1. Read latest webcam frame (non-blocking) ─────────────────
            ret, frame = cap.read()

            if ret:
                frame_count += 1
                # Submit every Nth frame to the AI worker
                if frame_count % max(skip_frames, 1) == 0:
                    processor.submit(frame)

            # ── 2. Pick up latest AI result ────────────────────────────────
            result = processor.get_result()
            if result is not None:
                last_output = result

            # ── 3. Send to virtual camera ──────────────────────────────────
            vcam.send(_to_vcam(last_output, fmt))

            # ── 4. Sleep exactly until the next frame deadline ─────────────
            # This is the key fix: we compute the exact remaining sleep time
            # ourselves so scheduler jitter from Zoom/Meet doesn't accumulate.
            now   = time.perf_counter()
            sleep = next_deadline - now
            if sleep > 0.001:
                time.sleep(sleep)
            elif sleep < -frame_interval:
                # We're running more than one full frame behind — re-sync
                next_deadline = time.perf_counter()
            next_deadline += frame_interval

            # ── 5. Perf log ────────────────────────────────────────────────
            log_count += 1
            if log_count >= log_every:
                elapsed = time.perf_counter() - log_t0
                actual_fps = log_count / elapsed if elapsed > 0 else 0
                print(f"[perf] send={actual_fps:.1f}fps  "
                      f"detect={processor.det_ms:.0f}ms  "
                      f"swap={processor.swap_ms:.0f}ms  "
                      f"total_ai={processor.proc_ms:.0f}ms")
                log_count = 0
                log_t0    = time.perf_counter()

    processor.stop()
    cap.release()
    print("[*] Stopped cleanly.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time face swap to virtual camera",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic:
    python face_swap.py --target photo.jpg

  With Zoom/Meet/OBS also open (recommended):
    python face_swap.py --target photo.jpg --low-cpu

  Fastest (weaker CPU):
    python face_swap.py --target photo.jpg --low-cpu --skip-frames 3 --process-width 320 --det-size 128 128
""",
    )
    parser.add_argument("--target",        required=True,
                        help="Path to target face photo (JPG/PNG)")
    parser.add_argument("--cam",           type=int, default=0,
                        help="Webcam index (default 0)")
    parser.add_argument("--fps",           type=int, default=30,
                        help="Virtual camera FPS (default 30)")
    parser.add_argument("--det-size",      nargs=2, type=int, default=[256, 256],
                        metavar=("W", "H"),
                        help="Face detection resolution (default 256 256, lower=faster)")
    parser.add_argument("--skip-frames",   type=int, default=2,
                        help="Send every Nth frame to AI (default 2; higher=faster/less accurate)")
    parser.add_argument("--width",         type=int, default=640,
                        help="Capture width  (default 640)")
    parser.add_argument("--height",        type=int, default=480,
                        help="Capture height (default 480)")
    parser.add_argument("--process-width", type=int, default=480,
                        help="Max width for AI processing (default 480, lower=faster)")
    parser.add_argument("--low-cpu",       action="store_true",
                        help="Yield extra CPU to streaming apps (Zoom/Meet/OBS)")
    args = parser.parse_args()

    run(
        target_path   = args.target,
        cam_index     = args.cam,
        fps           = args.fps,
        det_size      = tuple(args.det_size),
        skip_frames   = args.skip_frames,
        width         = args.width,
        height        = args.height,
        process_width = args.process_width,
        low_cpu       = args.low_cpu,
    )
