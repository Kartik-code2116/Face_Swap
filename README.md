# Real-Time Face Swap
Real-time face swapping using your webcam and an AI model. Swap your face with any photo and output to a virtual camera for Zoom/Meet/Teams.

## Features
- Live webcam input
- Face swap with any target photo
- Output to virtual camera
- CPU/GPU support with auto-detection

## Prerequisites

### Windows
- **Python 3.9 or 3.10**
- **Virtual Camera**: OBS Virtual Camera (from OBS Studio) or Unity Capture.
- **Microsoft C++ Build Tools 14.0+** (required for insightface compilation):
  1. Download from https://visualstudio.microsoft.com/visual-cpp-build-tools/
  2. Install with 'C++ build tools' workload (includes MSVC v143, CMake, Windows SDK).

**Alternative (recommended, avoids build errors)**: Use Conda/Mamba:
  ```
  conda install -c conda-forge insightface opencv numpy
  pip install onnxruntime pyvirtualcam
  ```

### macOS/Linux
- Python 3.9+
- Virtual camera setup (v4l2loopback on Linux)

## Quick Start

1. **Clone/Download** this repo.

2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```
   If insightface fails on Windows:
   ```
   # Option 1: Install VS Build Tools (see Prerequisites)
   pip install insightface --no-cache-dir
   
   # Option 2: Conda (pre-built wheels)
   conda create -n faceswap python=3.10
   conda activate faceswap
   conda install -c conda-forge insightface opencv numpy
   pip install -r requirements.txt --upgrade
   ```

3. **Download model** (~500MB):
   Download `inswapper_128.onnx` from https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx
   Place in project folder.

4. **Run**:
   ```
   python face_swap.py --target path/to/photo.jpg
   ```

5. **Select virtual camera** in Zoom/Meet/Teams.

## Performance Optimizations (NEW)
- **GPU Auto-Detection**: Install `pip install onnxruntime-gpu` (NVIDIA CUDA), automatically used.
- `--det-size 320 320`: Lower detection res for speed (default 320x320).
- `--skip-frames 2`: Skip processing every 2nd frame for higher FPS.
- Real-time FPS/process timings printed.

**Fast example**:
```
python face_swap.py --target photo.jpg --det-size 320 320 --skip-frames 2 --fps 30
```

## Troubleshooting

### insightface build error on Windows
```
error: Microsoft Visual C++ 14.0 or greater is required
```
**Fixes**:
1. **Install Microsoft C++ Build Tools**:
   - Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Select: C++ build tools, latest MSVC, Windows 10/11 SDK.
   - Restart terminal, retry `pip install insightface`.

2. **Use pre-built wheels (Conda/Mamba)**:
   ```
   conda install -c conda-forge insightface==0.7.3
   ```

3. **Upgrade pip/setuptools/wheel**:
   ```
   pip install --upgrade pip setuptools wheel
   ```

### Other issues
- Model not found: Ensure `inswapper_128.onnx` in folder.
- No virtual cam: Install OBS VirtualCam.
- Poor perf: Use new flags, GPU, or lower webcam res.

## Project Structure
```
.
├── face_swap.py
├── requirements.txt
├── SETUP.md
├── inswapper_128.onnx  # Download this
├── README.md
└── target_photo.jpg    # Your swap target
```

## Credits
- insightface for face detection/swapping
- ONNX Runtime for inference

Enjoy your face swap! 🎭

