#!/bin/bash
# Activates the ComfyUI venv and launches its server listening on all interfaces.
set -e

COMFY_ROOT="${COMFY_ROOT:-$HOME/ComfyUI}"

if [ ! -f "$COMFY_ROOT/venv/bin/activate" ]; then
    echo "ComfyUI venv not found at $COMFY_ROOT/venv"
    exit 1
fi

source "$COMFY_ROOT/venv/bin/activate"
exec python3 "$COMFY_ROOT/main.py" --listen 0.0.0.0
