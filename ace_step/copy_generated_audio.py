#!/usr/bin/env python3
"""
download_generated_audio.py
Download 4 audio files from the ACE-Step API server using curl
"""

import subprocess
import time
from pathlib import Path

OUTPUT_DIR = Path("/home/cvcadmin/cruilla/Festival-Cruilla/outputs/music/metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Audio files that were just generated (from server logs)
AUDIO_FILES = [
    {
        "id": "A",
        "label": "hype_reggaeton_futuristic",
        "paths": [
            "/home/cvcadmin/cruilla/Festival-Cruilla/models/ACE-Step-1.5/.cache/acestep/tmp/api_audio/c87549a0-16fe-198a-9dc0-ca328982cacb.wav",
            "/home/cvcadmin/cruilla/Festival-Cruilla/models/ACE-Step-1.5/.cache/acestep/tmp/api_audio/3f29b31e-166f-1a62-8976-2b68a54e9bc6.wav",
        ]
    },
    {
        "id": "B",
        "label": "chill_jazz_soul_90s",
        "paths": [
            "/home/cvcadmin/cruilla/Festival-Cruilla/models/ACE-Step-1.5/.cache/acestep/tmp/api_audio/d9ddffcb-b93c-4498-df06-1112087e104e.wav",
            "/home/cvcadmin/cruilla/Festival-Cruilla/models/ACE-Step-1.5/.cache/acestep/tmp/api_audio/06006db9-3085-2291-b7ae-2d8ea4044b32.wav",
        ]
    },
]

def copy_file(src, dst):
    """Copy audio file from cache to output directory"""
    try:
        src_path = Path(src)
        if not src_path.exists():
            return False
        
        with open(src_path, 'rb') as f:
            data = f.read()
        
        dst_path = Path(dst)
        dst_path.write_bytes(data)
        return True
    except Exception as e:
        print(f"  Error copying: {e}")
        return False

def main():
    print("\n" + "=" * 65)
    print("  Festival Cruilla — Copy Generated Audio Files")
    print("=" * 65)
    print(f"  Output: {OUTPUT_DIR}\n")

    # Copy files from cache
    copied = 0
    for sample_idx, sample in enumerate(AUDIO_FILES):
        for file_idx, src_path in enumerate(sample["paths"]):
            file_letter = chr(ord('A') + sample_idx * 2 + file_idx)  # A, B, C, D
            output_file = OUTPUT_DIR / f"diversity_{file_letter}_{sample['label']}.wav"
            
            print(f"[{file_letter}] {sample['label']}")
            print(f"  From: {Path(src_path).name}")
            
            if copy_file(src_path, output_file):
                size_kb = output_file.stat().st_size // 1024
                print(f"  ✓ Copied: {output_file.name} ({size_kb} KB)\n")
                copied += 1
            else:
                print(f"  ✗ File not found\n")

    print("=" * 65)
    print(f"  Done: {copied} files copied")
    print("=" * 65)

    # List files in output directory
    files = sorted(OUTPUT_DIR.glob("*.wav"))
    if files:
        print(f"\n  Files in {OUTPUT_DIR}:")
        for f in files:
            size_kb = f.stat().st_size // 1024
            print(f"    {f.name}  ({size_kb} KB)")
    else:
        print(f"\n  No files found in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
