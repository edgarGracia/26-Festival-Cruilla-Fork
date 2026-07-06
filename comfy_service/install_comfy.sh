#!/bin/bash
# Clone ComfyUI + the custom nodes this workflow needs into a target dir.
# Reused by the Dockerfile and by bare Linux installs.
#
# Usage: ./install_comfy.sh [TARGET_DIR]
#   TARGET_DIR defaults to ../Comfy relative to this script.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-$SCRIPT_DIR/../Comfy}"
COMFY_REF="${COMFY_REF:-master}"

# Custom nodes required by 3ingredients API.json.
NODES=(
    "https://github.com/yolain/ComfyUI-Easy-Use"        # easy cleanGpuUsed / clearCacheAll
    "https://github.com/city96/ComfyUI-GGUF"            # UnetLoaderGGUF / CLIPLoaderGGUF
    "https://github.com/rgthree/rgthree-comfy"          # Power Lora Loader / Image Comparer
    "https://github.com/Fannovel16/comfyui_controlnet_aux"  # Openpose / DepthAnything
    "https://github.com/crystian/ComfyUI-Crystools"     # Switch latent / Switch any
)

if [ ! -d "$TARGET/.git" ]; then
    echo "Cloning ComfyUI into $TARGET"
    git clone https://github.com/comfyanonymous/ComfyUI.git "$TARGET"
fi

git -C "$TARGET" fetch --depth 1 origin "$COMFY_REF"
git -C "$TARGET" checkout "$COMFY_REF"

echo "Installing ComfyUI custom nodes"
mkdir -p "$TARGET/custom_nodes"
for repo in "${NODES[@]}"; do
    name="$(basename "$repo")"
    dest="$TARGET/custom_nodes/$name"
    if [ -d "$dest/.git" ]; then
        echo "  $name already present, pulling"
        git -C "$dest" pull --ff-only || true
    else
        echo "  cloning $name"
        git clone --depth 1 "$repo" "$dest"
    fi
    if [ -f "$dest/requirements.txt" ]; then
        ${PIP:-pip} install -r "$dest/requirements.txt"
    fi
done

echo "Installing ComfyUI core requirements"
${PIP:-pip} install -r "$TARGET/requirements.txt"

echo "Done. Models are NOT downloaded - see comfy_service/README.md."
