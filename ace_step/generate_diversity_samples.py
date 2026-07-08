"""
generate_diversity_samples.py
──────────────────────────────
Generates 4 deliberately contrasting 20-second audio clips using
build_ace_prompt() from music_generator.py, then saves them to:
  /home/cvcadmin/cruilla/Festival-Cruilla/outputs/music/metrics/

The 4 prompts are chosen to be maximally different from each other
so the cosine matrix has a wide spread to inspect:

  A — Hype / Urban Reggaeton / futuristic  (club, bass, dembow)
  B — Chill / Jazz Soul / 90s              (piano, late-night, warm)
  C — Happy / Folk Rock / medieval         (acoustic, bright, organic)
  D — Hype / Electronic Dance / actual     (synth, four-on-the-floor, festival drop)

Run from your project root:
  cd /home/cvcadmin/cruilla/Festival-Cruilla
  python ace_step/generate_diversity_samples.py
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

# ── make sure the project modules are importable ──────────────────────────────
PROJECT_ROOT = Path("/home/cvcadmin/cruilla/Festival-Cruilla")
sys.path.insert(0, str(PROJECT_ROOT / "api"))

from music_generator import build_ace_prompt, _call_ace_step

# ── output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "music" / "metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DURATION = 20   # seconds — matches the existing pipeline default

# ──────────────────────────────────────────────────────────────────────────────
# 4 maximally contrasting prompt configs
# ──────────────────────────────────────────────────────────────────────────────
SAMPLES = [
    {
        "id":          "A",
        "label":       "hype_reggaeton_futuristic",
        "mood":        "hype",
        "instrument":  "drums",
        "era":         "futuristic",
        "genre":       "Urban Pop Reggaeton",
        "tribe":       "La Calle",
        "confidence":  85,
        "why":         "High-energy dembow / trap bass / glitchy future production",
    },
    {
        "id":          "B",
        "label":       "chill_jazz_soul_90s",
        "mood":        "chill",
        "instrument":  "piano",
        "era":         "90s",
        "genre":       "Jazz / Soul",
        "tribe":       "Los Nómadas",
        "confidence":  70,
        "why":         "Late-night piano jazz, analogue warmth, slow groove",
    },
    {
        "id":          "C",
        "label":       "happy_folk_rock_medieval",
        "mood":        "happy",
        "instrument":  "guitar",
        "era":         "medieval",
        "genre":       "Folk Rock",
        "tribe":       "Los Románticos",
        "confidence":  60,
        "why":         "Bright acoustic guitar, natural reverb, troubadour energy",
    },
    {
        "id":          "D",
        "label":       "hype_electronic_dance_actual",
        "mood":        "hype",
        "instrument":  "trumpet",
        "era":         "actual",
        "genre":       "Electronic / Dance",
        "tribe":       "Los Soñadores",
        "confidence":  80,
        "why":         "Four-on-the-floor festival drop, brass stabs, modern master",
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Generate
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    generated = []

    print("\n" + "=" * 65)
    print("  Diversity Sample Generator — Festival Cruilla / ACE-Step 1.5")
    print("=" * 65)
    print(f"  Output dir : {OUTPUT_DIR}")
    print(f"  Duration   : {DURATION}s per clip")
    print(f"  Clips      : {len(SAMPLES)}")
    print("=" * 65 + "\n")

    for cfg in SAMPLES:
        print(f"[{cfg['id']}] {cfg['label']}")
        print(f"     Why distinct : {cfg['why']}")
        print(f"     mood={cfg['mood']}  instrument={cfg['instrument']}  "
              f"era={cfg['era']}  genre={cfg['genre']}")

        # Build prompt using your existing music_generator logic
        prompt_data = build_ace_prompt(
            mood=cfg["mood"],
            instrument=cfg["instrument"],
            era=cfg["era"],
            genre=cfg["genre"],
            tribe=cfg["tribe"],
            artist_confidence=cfg["confidence"],
            duration_seconds=DURATION,
        )

        # Unique filename so reruns never overwrite
        uid       = uuid.uuid4().hex[:6]
        filename  = f"diversity_{cfg['id']}_{cfg['label']}_{uid}.wav"
        out_path  = OUTPUT_DIR / filename

        print(f"\n     TAGS (first 120 chars):")
        print(f"     {prompt_data['tags'][:120]} …")
        print(f"\n     Calling ACE-Step …")

        t0     = time.time()
        result = _call_ace_step(
            tags=prompt_data["tags"],
            description=prompt_data["description"],
            duration=DURATION,
            output_path=out_path,
        )
        elapsed = time.time() - t0

        if result["success"]:
            size_kb = out_path.stat().st_size // 1024
            print(f"     ✓  Saved → {out_path.name}  ({size_kb} KB, {elapsed:.1f}s)\n")
            generated.append({
                "id":    cfg["id"],
                "label": cfg["label"],
                "path":  str(out_path),
            })
        else:
            print(f"     ✗  FAILED: {result['error']}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 65)
    print(f"  Done. {len(generated)}/{len(SAMPLES)} clips generated.")
    print("=" * 65)

    if generated:
        print("\n  Files ready for cosine matrix:\n")
        for g in generated:
            print(f"    [{g['id']}] {g['path']}")

    if len(generated) < len(SAMPLES):
        failed = [s["id"] for s in SAMPLES if s["id"] not in {g["id"] for g in generated}]
        print(f"\n  ⚠  Clips {failed} failed.")
        print("     Check that ACE-Step is running:")
        print("     cd /home/cvcadmin/cruilla/Festival-Cruilla")
        print("     uv run acestep   (or python -m acestep)")
        print("     Then re-run this script.\n")

    print(f"\n  Output dir : {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()