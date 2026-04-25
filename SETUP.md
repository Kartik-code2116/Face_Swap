# Real-Time Face Swap Setup Guide

## What it does
Your webcam video runs through Python, your face is detected and swapped
with any person's photo you provide, then the result is sent to a
**virtual camera** that Zoom / Google Meet / Teams sees as a normal webcam.

---

## Step 1 - Install a Virtual Camera driver

### Windows
Install **OBS Virtual Camera** (comes with OBS Studio):
https://obsproject.com/

Or install **Unity Capture** (lighter):
https://github.com/schellingb/UnityCapture

### Linux
```bash
sudo apt install v4l2loopback-dkms
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="VirtualCam" exclusive_caps=1
```

---

## Step 2 - Install Python dependencies

Python 3.9 or 3.10 recommended.

```bash
pip install -r requirements.txt
```

---

## Step 3 - Download the face swap AI model

Download `inswapper_128.onnx` (~500MB) from:
https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx

Place it in the **same folder** as `face_swap.py`.

---

## Step 4 - Run the app

```bash
python face_swap.py --target photo.jpg
```

Replace `photo.jpg` with the path to your target person's photo.

Performance flags:
- `--det-size 256 256` -> detection size (lower = faster)
- `--skip-frames 2` -> process every 2nd frame for less CPU load
- `--process-width 480` -> downscale live frames before detection/swap

Optional flags:
- `--cam 1` -> use a different webcam
- `--fps 20` -> virtual camera output FPS
- `--width 640 --height 480` -> capture size sent to the virtual camera

Low-latency example:
```bash
python face_swap.py --target elon.jpg --det-size 256 256 --skip-frames 2 --process-width 480 --fps 20
```

**GPU**: `pip install onnxruntime-gpu` (auto-detected)

---

## Step 5 - Select virtual camera in Zoom / Meet / Teams

- **Zoom**: Settings -> Video -> Camera -> select "OBS Virtual Camera" or "VirtualCam"
- **Google Meet**: Settings -> Video -> Camera -> select virtual camera
- **Teams**: Settings -> Devices -> Camera -> select virtual camera

---

## Tips for best results

- Use a **clear, front-facing photo** of the target person
- Keep your own face well lit and front-facing
- On CPU, start with `--det-size 256 256 --skip-frames 2 --process-width 480`
- If latency is still high, try `--skip-frames 3 --process-width 360`
- For **GPU acceleration** (NVIDIA), install `onnxruntime-gpu` and the script will pick CUDA automatically

---

## Folder structure

```text
faceswap_app/
|-- face_swap.py
|-- requirements.txt
|-- inswapper_128.onnx
`-- photo.jpg
```
