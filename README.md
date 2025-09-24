# 👻 Hello Ghost

**Hello Ghost** is a spooky interactive camera app that detects faces and applies creepy effects when no one is in view.  
When no faces are detected, the screen flickers like an old TV and ghost sprites appear randomly with eerie visual distortions.  

Built with **OpenCV**, **DeGirum AI SDK**, and a **Hailo-8L accelerator**, this project brings computer vision and horror aesthetics together.  

---

## ✨ Features

- 🎭 **Face Detection** — Uses YOLOv8 face detection running on Hailo hardware.  
- 🖤 **Creepy Visual Effects** — Black & white creepy filter, vignetting, jitter, and noise.  
- 📺 **Weak Signal Flicker** — Old-TV–style noise, tearing, rolling, desync, blackout/whiteout, and scanline effects.  
- 👻 **Ghost Sprites** — Random flickering PNG ghosts (with transparency) that appear on screen.  
- 🔁 **Mirror Mode** — Flips the camera feed horizontally for a natural “mirror” feel.  
- 🐛 **Debug Options** — Optional bounding boxes and debug text overlays.  

---

## 📂 Project Structure

```
hello_ghost/
├── main.py            # Main entry point (runs the ghost camera)
├── util.py            # Visual effects and rendering utilities
├── config.py          # Configuration (camera + model parameters)
├── requirements.txt   # Python dependencies
├── ghost.png          # Ghost sprite (transparent PNG)
├── ghost2.png         # Second ghost sprite
├── ghost6.png         # Extra ghost sprite (optional)
├── back.txt           # (Not used by code, placeholder file)
├── hailort.log        # Log output from Hailo runtime
├── .vscode/           # VSCode settings
├── venv/              # Python virtual environment (ignored by git)
└── .gitignore
```

---

## ⚙️ Installation

### 1. Clone this repo
```bash
git clone https://github.com/yourname/hello_ghost.git
cd hello_ghost
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Hardware & Drivers
- Requires **Hailo-8L** AI accelerator + HailoRT runtime (`hailort`).  
- Make sure your DeGirum/Hailo SDK is installed and working.  
- Webcam or Raspberry Pi camera required.  

---

## 🚀 Usage

Run the app:

```bash
python main.py
```

Controls:
- Press **`q`** to quit.  
- Ghosts appear when **no face is detected**.  
- Debug overlays can be toggled in `main.py`.  

---

## 🔧 Configuration

Edit [`config.py`](config.py) to adjust:
- **Model & Inference**
  - `MODEL_NAME`, `INFERENCE_HOST`, `DEVICE_TYPE`, etc.  
- **Camera**
  - `CAM_INDEX`, `FRAME_W`, `FRAME_H`.  

Additional runtime flags in `main.py`:
- `MIRROR = True` — Flip the camera feed.  
- `FLICKER_ENABLED = True` — Enable weak-signal flicker.  
- `GHOST_ENABLED = True` — Enable ghost sprites.  
- `DRAW_BOX = True` — Draw face detection bounding boxes.  

---

## 📸 Example Ghosts

Place your ghost PNGs (with transparent background) in the project root:  
- `ghost.png`  
- `ghost2.png`  
- (Optional) `ghost6.png`, etc.  

---

## 🛠 Dependencies

See [`requirements.txt`](requirements.txt). Main libraries:
- `opencv-python`
- `numpy`
- `degirum`
- `degirum_tools`

---

## 🧪 Known Issues

- If no ghost images are found, app runs without ghosts (warning printed).  
- Requires GPU/accelerator hardware; may not run purely on CPU.  
- Ghosts sometimes overlap with flicker effects (intended spooky chaos).  

---

## 📜 License

MIT License. Feel free to hack it and make your own haunted camera!  
