#!/bin/bash
# Activates the project venv, launches demo.py, and opens it in Chromium fullscreen.
# demo.py loads ACE-Step 1.5 in-process, so no separate API server is needed.
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URL="http://127.0.0.1:7860"

if [ ! -f "$ROOT/.venv/bin/activate" ]; then
    echo ".venv not found - run install.sh first"
    exit 1
fi

source "$ROOT/.venv/bin/activate"

python "$ROOT/demo.py" &
DEMO_PID=$!

cleanup() {
    kill "$DEMO_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for the app to come up at $URL..."
until curl -s -o /dev/null "$URL"; do
    sleep 1
done

CHROMIUM_BIN="$(command -v chromium-browser || command -v chromium || command -v google-chrome)"
"$CHROMIUM_BIN" --start-fullscreen --new-window "$URL"

wait "$DEMO_PID"
