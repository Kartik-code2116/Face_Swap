# Real-Time Face Swap Setup Guide

## What it does
Your webcam video runs through Python, your face is detected and swapped
with any person's photo you provide — then the result is sent to a
**virtual camera** that Zoom / Google Meet / Teams sees as a normal webcam.

---

## Step 1 — Install a Virtual Camera driver

### Windows
Install **OBS Virtual Camera** (comes with OBS Studio):
https://obsproject.com/

Or install **Unity Capture** (lighter):
https://github.com/schellingb/UnityCapture


```bash
sudo apt install v4l2loopback-dkms
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="VirtualCam" exclusive_caps=1
```

---

## Step 2 — Install Python dependencies

Python 3.9 or 3.10 recommended.

```bash
pip install -r requirements.txt
```

---

## Step 3 — Download the face swap AI model

Download `inswapper_128.onnx` (~500MB) from:
https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx

Place it in the **same folder** as `face_swap.py`.

---

## Step 4 — Run the app

```bash
python face_swap.py --target photo.jpg
```

Replace `photo.jpg` with the path to your target person's photo.

Performance flags (new):
- `--det-size 320 320` → Detection size (default 320x320, lower = faster)
- `--skip-frames 2` → Process every Nth frame (1 = no skip, 2+ = faster)

Optional flags:
- `--cam 1` → different webcam (default 0)
- `--fps 25` → output FPS (default 20)

Perf example:
```bash
python face_swap.py --target elon.jpg --det-size 320 320 --skip-frames 2 --fps 30
```

**GPU**: `pip install onnxruntime-gpu` (auto-detected)

---

## Step 5 — Select virtual camera in Zoom / Meet / Teams

- **Zoom**: Settings → Video → Camera → select "OBS Virtual Camera" or "VirtualCam"
- **Google Meet**: Settings (gear icon) → Video → Camera → select virtual camera
- **Teams**: Settings → Devices → Camera → select virtual camera

---

## Tips for best results

- Use a **clear, front-facing photo** of the target person (good lighting)
- Your own face should also be **well lit and front-facing**
- For better performance on CPU, lower resolution: edit `det_size=(320, 320)` in the script
- For **GPU acceleration** (NVIDIA): replace `onnxruntime` with `onnxruntime-gpu`
  and change `CPUExecutionProvider` to `CUDAExecutionProvider` in the script

---

## Folder structure

```
faceswap_app/
├── face_swap.py          ← main script
├── requirements.txt      ← Python dependencies
├── inswapper_128.onnx    ← AI model (you download this)
└── photo.jpg             ← your target face photo
```
