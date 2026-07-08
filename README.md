# Festival Cruïlla - Tal Cara, Tal Beat

Fork of [AgustinaLazzati/Festival-Cruilla](https://github.com/AgustinaLazzati/Festival-Cruilla) that includes a Gradio app.

## Requirements

- Python (tested with 3.10 and 3.11)
- [ffmpeg](https://ffmpeg.org/) available on PATH
- NVIDIA GPU

## Setup

1. **Get the artist data.** Download [Fake_Artists.zip](https://github.com/edgarGracia/26-Festival-Cruilla-Fork/releases/download/v0.0.1/Fake_Artists.zip) from the releases page and extract it into the project root, so the images end up under `Fake_Artists/`.

2. **Install dependencies.** The install script creates a `.venv` and installs a CUDA build of PyTorch (default `cu128`, required for RTX 50-series GPUs) plus the requirements.

   Linux:

   ```bash
   ./install.sh
   ```

   Defaults can be overridden with env vars, e.g. `PYTHON=python3.11 CUDA=cu124 ./install.sh`.

   Windows:

   ```powershell
   powershell -ExecutionPolicy Bypass -File install.ps1
   ```

3. **Configure uploads (optional).** Copy `.env.example` to `.env` and fill in the SSH credentials if you want generated results uploaded to a server and served via QR code. The app works without it.

## Run

On Linux:

```bash
./run.sh
```

On Windows use `run.bat`. Both activate the venv and launch `demo.py`.

This starts the Gradio kiosk UI on `http://localhost:7860` (bound to `0.0.0.0`). ACE-Step 1.5 is loaded in-process, so no separate API server is needed.


Set `gsettings set org.gnome.desktop.a11y.applications screen-keyboard-enabled false` to disable on screen keyboard