#!/usr/bin/env python3
"""
simple_generate_4samples.py
Simple script to generate 4 diverse audio samples using ACE-Step API
"""

import sys
import time
import requests
import shutil
from pathlib import Path

# API Configuration
API_URL = "http://localhost:8001"
OUTPUT_DIR = Path("/home/cvcadmin/cruilla/Festival-Cruilla/outputs/music/metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 4 Diverse samples
SAMPLES = [
    {
        "id": "A",
        "label": "hype_reggaeton_futuristic",
        "prompt": "reggaeton, dembow rhythm, 808 bass thud, urban Latino, perreo groove, Latin trap hi-hat, high-energy, adrenaline, electric, festival anthem, peak-time, explosive, live drum kit, tight snare, punchy kick, driving rhythm, polyrhythmic percussion, futuristic, hyperpop, glitchy, experimental electronic, AI-generated textures, avant-garde",
        "duration": 20,
    },
    {
        "id": "B",
        "label": "chill_jazz_soul_90s",
        "prompt": "jazz soul, warm piano chord, upright bass, soulful melody, late-night groove, harmonic richness, serene, meditative, flowing, peaceful, airy, spacious, 1990s, lo-fi grunge warmth, boom-bap hip-hop, alternative rock, Britpop, shoegaze",
        "duration": 20,
    },
    {
        "id": "C",
        "label": "happy_folk_rock_medieval",
        "prompt": "folk rock, acoustic guitar, organic warmth, rootsy storytelling, gentle drum groove, euphoric, upbeat, feel-good, bright, triumphant, joyful energy, electric guitar lead, clean Stratocaster tone, fingerpicked arpeggios, overdriven riff, medieval era, acoustic resonance, natural reverb, troubadour folk, Gregorian chant",
        "duration": 20,
    },
    {
        "id": "D",
        "label": "hype_electronic_dance_actual",
        "prompt": "electronic dance music, four-on-the-floor kick, supersaw lead, dancefloor energy, festival drop, build-up, high-energy, adrenaline, electric, festival anthem, peak-time, explosive, solo trumpet, brass section stabs, muted trumpet, jazz brass, fanfare horn, contemporary 2024 production, hi-fi master, modern mix, streaming-optimized, crystal-clear audio",
        "duration": 20,
    },
]


def generate_audio(prompt: str, duration: int, output_path: Path) -> bool:
    """Generate audio and save to output path"""
    try:
        # Submit task
        print(f"  Submitting task...")
        resp = requests.post(
            f"{API_URL}/release_task",
            json={"prompt": prompt, "duration": duration, "audio_format": "wav"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != 200:
            print(f"  ✗ API error: {data.get('error')}")
            return False

        task_id = data["data"]["task_id"]
        print(f"  Task ID: {task_id}")

        # Poll for completion
        max_polls = 300
        for poll in range(max_polls):
            time.sleep(1)
            query_resp = requests.post(
                f"{API_URL}/query_result",
                json={"task_ids": [task_id]},
                timeout=30,
            )
            query_resp.raise_for_status()
            query_data = query_resp.json()

            if not query_data.get("data"):
                continue

            tasks = query_data["data"]
            if isinstance(tasks, list) and tasks:
                task_info = tasks[0]
            else:
                continue

            task_status = task_info.get("status") if isinstance(task_info, dict) else task_info
            
            if task_status == 1:  # Success
                media_path = task_info.get("media_path", "") if isinstance(task_info, dict) else ""
                print(f"  ✓ Complete! Media: {media_path}")

                # Download the audio file
                if media_path.startswith("/"):
                    download_url = f"{API_URL}/v1/audio?path={media_path}"
                else:
                    download_url = media_path

                print(f"  Downloading...")
                audio_resp = requests.get(download_url, timeout=60)
                audio_resp.raise_for_status()

                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(audio_resp.content)
                print(f"  ✓ Saved → {output_path.name}")
                return True
            elif task_status == 2:  # Failed
                print(f"  ✗ Task failed")
                return False
            elif poll % 20 == 0:
                print(f"  In progress... ({poll}s)")

        print(f"  ✗ Timeout")
        return False

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    print("\n" + "=" * 65)
    print("  Festival Cruilla — 4 Diverse Audio Samples")
    print("=" * 65)
    print(f"  Output: {OUTPUT_DIR}\n")

    generated = 0
    for sample in SAMPLES:
        print(f"[{sample['id']}] {sample['label']}")
        output_file = OUTPUT_DIR / f"diversity_{sample['id']}_{sample['label']}.wav"
        
        if generate_audio(sample["prompt"], sample["duration"], output_file):
            generated += 1
        print()

    print("=" * 65)
    print(f"  Done: {generated}/4 clips generated")
    print("=" * 65)
    print(f"  Output dir: {OUTPUT_DIR}\n")

    # List files
    files = sorted(OUTPUT_DIR.glob("diversity_*.wav"))
    if files:
        print("  Generated files:")
        for f in files:
            size_kb = f.stat().st_size // 1024
            print(f"    {f.name}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
