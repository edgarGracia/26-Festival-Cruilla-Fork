#!/bin/bash
# Creates a Python venv and installs torch + requirements.txt
# Usage: ./install.sh
# Override defaults with env vars, e.g.: PYTHON=python3.11 CUDA=cu124 ./install.sh
set -e

PYTHON="${PYTHON:-python}"
CUDA="${CUDA:-cu128}"   # e.g. cu118, cu121, cu124, cu128, cu130, or cpu
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"

if [ ! -d "$VENV" ]; then
    echo "Creating venv at $VENV"
    "$PYTHON" -m venv "$VENV"
else
    echo "venv already exists at $VENV"
fi

echo "Upgrading pip"
"$VENV/bin/python" -m pip install --upgrade pip

echo "Installing torch ($CUDA)"
"$VENV/bin/python" -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url "https://download.pytorch.org/whl/$CUDA"

echo "Installing requirements"
"$VENV/bin/python" -m pip install -r "$ROOT/requirements.txt"

# ── ACE-Step 1.5 (cloned into models/, installed with uv) ──────────────────
MODELS_DIR="$ROOT/models"
ACE_DIR="$MODELS_DIR/ACE-Step-1.5"

if [ ! -d "$ACE_DIR" ] || [ -z "$(ls -A "$ACE_DIR" 2>/dev/null)" ]; then
    echo "Cloning ACE-Step-1.5 into $MODELS_DIR"
    mkdir -p "$MODELS_DIR"
    git clone https://github.com/ace-step/ACE-Step-1.5.git "$ACE_DIR"
else
    echo "ACE-Step-1.5 already exists at $ACE_DIR"
fi

# Install uv if missing
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv lands in ~/.local/bin; add to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Running uv sync in ACE-Step-1.5"
(cd "$ACE_DIR" && uv sync)

echo "Done. Activate with: source .venv/bin/activate"
