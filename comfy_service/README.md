# Comfy service

Runs the `3ingredients API.json` workflow behind a ComfyUI server and talks to
it from Python with [client.py](client.py). The app never imports ComfyUI, so
its dependencies stay separate.

ComfyUI itself is **not** vendored in this repo. It is cloned per machine into
`../Comfy` (git-ignored). Pick a run mode below.

## Run modes

### Docker (Windows / Linux / DGX) - primary

Prerequisites on Windows: Docker Desktop with the WSL2 backend, and a recent
NVIDIA driver on the Windows host (that is what exposes the GPU to WSL - no
toolkit install needed inside WSL). Verify the GPU reaches containers:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

Then, from the repo root:

```bash
docker compose build                 # build the image (clones ComfyUI + nodes)
docker compose run --rm download     # fetch the ~22 GB weights (one time)
docker compose up                    # start the server on :8188
```

Models land in `./Comfy/models` on the host (override with `COMFY_MODELS_DIR`)
and persist across rebuilds. The Hugging Face cache lives in the `hf-cache`
volume, so re-running `download` never re-fetches.

### Bare host server (no Docker) - fallback

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1   # clones Comfy + weights
.\.venv\Scripts\python.exe Comfy\main.py --port 8188
```

Both expose `http://localhost:8188`. The client code is identical for both.

## Models

Models are large (~22 GB unet alone) so they are never committed or baked into
the image. Download them with the script - it fetches from Hugging Face and
places each file in the right ComfyUI subfolder:

```bash
python comfy_service/download_models.py
# custom target: --models-dir /path/to/models
# gated repos:   --token hf_xxx   (or export HF_TOKEN)
# preview only:  --list
```

What it fetches (see [download_models.py](download_models.py) `MANIFEST`):

| File | Folder | Repo |
|------|--------|------|
| `Qwen-Image-Edit-2509-Q8_0.gguf` | `unet/` | QuantStack/Qwen-Image-Edit-2509-GGUF |
| `Qwen2.5-VL-7B-Instruct-Q4_K_S.gguf` | `clip/` | unsloth/Qwen2.5-VL-7B-Instruct-GGUF |
| `qwen_image_vae.safetensors` | `vae/` | Comfy-Org/Qwen-Image_ComfyUI |
| `Qwen-Image-Lightning-4steps-V1.0.safetensors` | `loras/` | lightx2v/Qwen-Image-Lightning |

The workflow originally named the CLIP `...UD-Q4_K_S.gguf`, which does not exist
upstream (404). Both workflow JSONs now point at the real `Q4_K_S` build.

The extra loras in the workflow (graffiti, realcomic, photo-to-anime - all off
by default) are custom/community weights not on Hugging Face. Drop them in
`loras/QWEN/` by hand if you enable them.

## Usage

```python
from comfy_service import ComfyClient, GenerationRequest

client = ComfyClient("3ingredients API.json")
image = client.generate(GenerationRequest(
    base_image="inputs/ca/bg_urban.png",
    person_image="user_image.png",
    object_image="inputs/blue_sunglasses.png",
    prompt="Place the person into the scene wearing the sunglasses.",
))
image.save("outputs/result.png")
```

Point the client at a remote server:

```python
from comfy_service import ComfyClient, ComfyServerConfig

client = ComfyClient(
    "3ingredients API.json",
    config=ComfyServerConfig(host="192.168.1.50", port=8188),
)
```

## Changing the workflow

Edit the graph in the ComfyUI UI, then export **API format**
(Workflow -> Export API) over `3ingredients API.json`. If you add or renumber
the input nodes, update `NodeMap` in [client.py](client.py) - no other code
changes needed.
