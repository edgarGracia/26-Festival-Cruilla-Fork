#!/usr/bin/env python3
"""
direct_generate_4samples.py
Generate 4 diverse audio samples by directly calling the ACE-Step generation function
"""

import sys
import os
from pathlib import Path

# Add ACE-Step to path
ACE_STEP_PATH = Path("/home/cvcadmin/cruilla/Festival-Cruilla/models/ACE-Step-1.5")
sys.path.insert(0, str(ACE_STEP_PATH))

os.chdir(str(ACE_STEP_PATH))

from acestep import generate_music

OUTPUT_DIR = Path("/home/cvcadmin/cruilla/Festival-Cruilla/outputs/music/metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 4 Diverse samples
SAMPLES = [
    {
        "id": "A",
        "label": "hype_reggaeton_futuristic",
        "prompt": "reggaeton, dembow rhythm, 808 bass thud, urban Latino, perreo groove, Latin trap hi-hat, high-energy, adrenaline, electric, festival anthem, peak-time, explosive, live drum kit, tight snare, punchy kick, driving rhythm, polyrhythmic percussion, futuristic, hyperpop, glitchy, experimental electronic, AI-generated textures, avant-garde",
    },
    {
        "id": "B",
        "label": "chill_jazz_soul_90s",
        "prompt": "jazz soul, warm piano chord, upright bass, soulful melody, late-night groove, harmonic richness, serene, meditative, flowing, peaceful, airy, spacious, 1990s, lo-fi grunge warmth, boom-bap hip-hop, alternative rock, Britpop, shoegaze",
    },
    {
        "id": "C",
        "label": "happy_folk_rock_medieval",
        "prompt": "folk rock, acoustic guitar, organic warmth, rootsy storytelling, gentle drum groove, euphoric, upbeat, feel-good, bright, triumphant, joyful energy, electric guitar lead, clean Stratocaster tone, fingerpicked arpeggios, overdriven riff, medieval era, acoustic resonance, natural reverb, troubadour folk, Gregorian chant",
    },
    {
        "id": "D",
        "label": "hype_electronic_dance_actual",
        "prompt": "electronic dance music, four-on-the-floor kick, supersaw lead, dancefloor energy, festival drop, build-up, high-energy, adrenaline, electric, festival anthem, peak-time, explosive, solo trumpet, brass section stabs, muted trumpet, jazz brass, fanfare horn, contemporary 2024 production, hi-fi master, modern mix, streaming-optimized, crystal-clear audio",
    },
]


def main():
    print("\n" + "=" * 65)
    print("  Festival Cruilla — 4 Diverse Audio Samples (Direct)")
    print("=" * 65)
    print(f"  Output: {OUTPUT_DIR}\n")

    generated = 0
    for i, sample in enumerate(SAMPLES, 1):
        print(f"[{sample['id']}] {sample['label']}")
        print(f"  Prompt: {sample['prompt'][:80]}...")
        
        output_file = OUTPUT_DIR / f"diversity_{sample['id']}_{sample['label']}.wav"
        
        try:
            result = generate_music(
                prompt=sample["prompt"],
                audio_duration=20,
                audio_format="wav",
                seed=1000 + i,
            )
            
            # Save the generated audio
            if result is not None and isinstance(result, bytes):
                output_file.write_bytes(result)
                size_kb = output_file.stat().st_size // 1024
                print(f"  ✓ Generated: {output_file.name} ({size_kb} KB)\n")
                generated += 1
            else:
                print(f"  ✗ Generation returned unexpected type: {type(result)}\n")
                
        except Exception as e:
            print(f"  ✗ Error: {e}\n")

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
