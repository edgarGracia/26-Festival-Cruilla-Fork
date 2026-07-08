"""
Tal Cara, Tal Beat — main pipeline.

Usage:
    python main.py --image inputs/user.png --language ca
    python main.py --image inputs/user.png --mood happy --instrument guitar --era actual --with-music --language en
"""

import argparse
import os
import sys
import time
import unicodedata
from pathlib import Path

# ── Repo root & module paths ───────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "face2label" / "models"))
sys.path.insert(0, str(REPO_ROOT / "models" / "ACE-Step-1.5"))

# ── Default artifact paths ─────────────────────────────────────────────────
MODEL_PATH       = REPO_ROOT / "face2label" / "logs" / "artists_mlp.pth"
LABELS_PATH      = REPO_ROOT / "face2label" / "logs" / "labels.json"
METADATA_PATH    = REPO_ROOT / "Fake_Artist.csv"
ASSET_DIR        = REPO_ROOT / "inputs"
OUTPUT_DIR       = REPO_ROOT / "outputs"
OUTPUT_IMAGES_DIR = OUTPUT_DIR / "images"
OUTPUT_MUSIC_DIR  = OUTPUT_DIR / "music"
OUTPUT_VIDEO_DIR  = OUTPUT_DIR / "final_video"

# ── Canonical tribes: urban · indie · rock · pop · tecno ───────────────────
# Background files per language live at inputs/{lang}/bg_{tribe}.png
# Note: the file on disk is bg_rockstar.png but the canonical key is "rock".
TRIBE_BACKGROUNDS: dict[str, str] = {
    "urban": str(ASSET_DIR / "backgrounds" / "bg_urban.png"),
    "indie": str(ASSET_DIR / "backgrounds" / "bg_indie.png"),
    "rock":  str(ASSET_DIR / "backgrounds" / "bg_rockstar.png"),
    "pop":   str(ASSET_DIR / "backgrounds" / "bg_pop.png"),
    "tecno": str(ASSET_DIR / "backgrounds" / "bg_tecno.png"),
}
TEXT_BAND_FRACTION = 0.22
SUBJECT_HEIGHT_FRACTION = 0.72


# ── Helper ─────────────────────────────────────────────────────────────────

def _normalise_tribe(raw: str) -> str:
    nfkd = unicodedata.normalize("NFKD", raw.strip())
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()

# "rock" is normalised to "rockstars" to match the accessory folder name.
_TRIBE_ACCESSORY_FOLDER = {"rock": "rockstars"}

def _pick_random_accessory(tribe_key: str, asset_dir: str) -> str | None:
    import random
    folder_name = _TRIBE_ACCESSORY_FOLDER.get(tribe_key, tribe_key)
    for acc_root_name in ("accesories", "accessories"):
        tribe_dir = Path(asset_dir) / acc_root_name / folder_name
        if tribe_dir.is_dir():
            pngs = [str(p) for p in tribe_dir.rglob("*.png") if p.is_file()]
            return random.choice(pngs) if pngs else None
    return None


# ==============================================================================
# STEP 1 — FACE -> ARTIST LABEL
# ==============================================================================

def step_face2label(image_path: str) -> dict | None:

    from predictor import ArtistPredictor

    predictor = ArtistPredictor(
        model_path=str(MODEL_PATH),
        labels_path=str(LABELS_PATH),
        metadata_path=str(METADATA_PATH),
    )

    result = predictor.predict(image_path)

    if result is None:
        print("[face2label] No face detected.")
        return None

    print(f"[face2label] Matched: {result['name']}  "
          f"({result['confidence']}%)  "
          f"genre: {result['genre']}  "
          f"tribe: {result.get('tribe', 'unknown')}")
    return result

# ==============================================================================
# STEP 2 — CLOTHING / ACCESSORY OVERLAY
# ==============================================================================

def step_clothing(user_image_path: str, artist_match: dict, output_path: str) -> str | None:
    from clothing.Clothing import apply_look

    return apply_look(
        user_image_path=user_image_path,
        artist_name=artist_match["name"],
        output_path=output_path,
        csv_path=str(METADATA_PATH),
        asset_dir=str(ASSET_DIR),
    )

# ==============================================================================
# STEP 3 — MUSIC GENERATION
# ==============================================================================

def step_music(
    artist_match: dict,
    mood: str,
    instrument: str,
    era: str,
) -> dict | None:
    from api.music_generator import build_ace_prompt
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    save_dir = str(OUTPUT_MUSIC_DIR)
    os.makedirs(save_dir, exist_ok=True)

    prompt_data = build_ace_prompt(
        mood=mood,
        instrument=instrument,
        era=era,
        casa=casa,
        duration_seconds=20,
    )

    print(f" ------------ RESULTS ------------")
    print(f"Genre:      {artist_match.get('genre', 'pop')}")
    print(f"Tribe:      {artist_match.get('tribe', 'unknown')}")
    print(f"Confidence: {artist_match.get('confidence', 0)}")
    print(f"[music] Tags: {prompt_data['tags'][:120]}...")
    print(f"[music] Initializing ACE-Step 1.5...")
    t0 = time.perf_counter()

    handler = AceStepHandler()
    handler.initialize_service(
        project_root=None,
        config_path="acestep-v15-turbo",
        device="cuda",
    )
    print(f"[music] Model loaded in {time.perf_counter() - t0:.2f}s")

    params = GenerationParams(
        caption=prompt_data["tags"] + ". " + prompt_data["description"],
        lyrics="",
        duration=20,
        bpm=80,
    )
    gen_config = GenerationConfig(batch_size=1, audio_format="wav")

    print("[music] Generating...")
    t1 = time.perf_counter()
    result = generate_music(handler, None, params, gen_config, save_dir=save_dir)
    print(f"[music] Inference done in {time.perf_counter() - t1:.2f}s")

    if result.success:
        audio_path = result.audios[0]["path"] if result.audios else None
        print(f"[music] Saved: {audio_path}")
        return {"success": True, "audio_path": audio_path}
    else:
        print(f"[music] Error: {result.error}")
        return {"success": False, "error": result.error}


# ==============================================================================
# STEP 4 — TRIBE BACKGROUND COMPOSITE
# ==============================================================================
# ==============================================================================
# STEP 4 — COMPOSITE DE FONDO POR TRIBU (polaroid)
# ==============================================================================
def step_background(user_image_path: str, artist_match: dict, output_path: str,
                     language: str = "ca") -> str | None:
    try:
        from PIL import Image
        from person_segmentation import remove_background_center_person
    except ImportError as e:
        print(f"[background] Falta dependencia: {e}")
        return None

    raw_tribe = artist_match.get("tribe", "")
    tribe_key = _normalise_tribe(raw_tribe)

    # --- MODIFIED: Directly fetch the universal background for the tribe ---
    bg_path = TRIBE_BACKGROUNDS.get(tribe_key)
    
    if bg_path is None or not Path(bg_path).exists():
        print(f"[background] Tribu desconocida o falta imagen '{raw_tribe}' (key='{tribe_key}')")
        return None

    background = Image.open(bg_path).convert("RGBA")
    bg_w, bg_h = background.size

    import numpy as np
    from PIL import ImageFilter

    user_img = Image.open(user_image_path)
    subject = remove_background_center_person(user_img)

    alpha_arr = np.array(subject.split()[3], dtype=np.uint8)
    alpha_binary = np.where(alpha_arr >= 100, 255, 0).astype(np.uint8)
    alpha_edge = Image.fromarray(alpha_binary).filter(ImageFilter.GaussianBlur(1.5))
    r, g, b, _ = subject.split()
    subject = Image.merge("RGBA", (r, g, b, alpha_edge))

    text_band_px = int(bg_h * TEXT_BAND_FRACTION)
    usable_h = bg_h - text_band_px

    target_h = int(usable_h * SUBJECT_HEIGHT_FRACTION)
    scale = target_h / subject.height
    target_w = int(subject.width * scale)
    subject = subject.resize((target_w, target_h), Image.LANCZOS)

    paste_x = (bg_w - target_w) // 2
    paste_y = usable_h - target_h + 13

    composite = background.copy()
    composite.paste(subject, (paste_x, paste_y), subject)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    composite.convert("RGB").save(output_path, quality=95)
    print(f"[background] Poster guardado → {output_path}")
    return output_path

# ==============================================================================
# STEP 4b — COMFYUI POLAROID (AI compositing via remote ComfyUI server)
# ==============================================================================

def step_comfy_polaroid(
    person_image: str,
    artist_match: dict,
    output_path: str,
) -> str:
    from comfy_client import run_3ingredients_workflow

    raw_tribe = artist_match.get("tribe", "")
    tribe_key = _normalise_tribe(raw_tribe)

    bg_path = TRIBE_BACKGROUNDS.get(tribe_key)
    if not bg_path or not Path(bg_path).exists():
        raise RuntimeError(f"No background image found for tribe '{raw_tribe}'")

    accessory_path = _pick_random_accessory(tribe_key, str(ASSET_DIR))
    if not accessory_path:
        raise RuntimeError(f"No accessories found for tribe '{tribe_key}'")

    print(f"[comfy] bg={Path(bg_path).name}  "
          f"person={Path(person_image).name}  "
          f"accessory={Path(accessory_path).name}")

    image_bytes = run_3ingredients_workflow(
        base_image=bg_path,
        person_image=person_image,
        object_image=accessory_path,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(image_bytes)
    print(f"[comfy] Polaroid saved → {output_path}")
    return output_path

# ==============================================================================
# STEP 5 — VIDEO GENERATION
# ==============================================================================

def step_video(image_path: str, audio_path: str, output_path: str) -> str | None:
    try:
        from moviepy import ImageClip, AudioFileClip
    except ImportError as e:
        print(f"[video] Missing dependency: {e}  →  pip install moviepy")
        return None

    try:
        audio = AudioFileClip(audio_path)
        video = ImageClip(image_path, duration=audio.duration).with_audio(audio)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        video.write_videofile(
            output_path,
            fps=1,
            codec="libx264",
            audio_codec="aac",
            logger=None,
        )
        video.close()
        audio.close()

        print(f"[video] ✓ Video saved → {output_path}")
        return output_path
    except Exception as e:
        print(f"[video] ✗ Error generating video: {e}")
        return None


# ==============================================================================
# PIPELINE ORCHESTRATOR
# ==============================================================================

def run_pipeline(
    image_path: str,
    output_path: str | None = None,
    mood: str = "happy",
    instrument: str = "synth",
    era: str = "actual",
    language: str = "ca",  # <--- Added language to pipeline orchestrator
    skip_music: bool = True,
) -> dict:
    OUTPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_IMAGES_DIR.mkdir(exist_ok=True)
    OUTPUT_MUSIC_DIR.mkdir(exist_ok=True)
    OUTPUT_VIDEO_DIR.mkdir(exist_ok=True)

    stem = Path(image_path).stem

    if output_path is None:
        # Append the language to the filename to avoid overwriting different versions
        output_path = str(OUTPUT_IMAGES_DIR / f"{stem}_styled_{language}.png")

    timings = {}

    # ── Step 1 — face → artist ────────────────────────────────────────────
    t_start = time.perf_counter()
    artist_match = step_face2label(image_path)
    timings["step_face2label"] = time.perf_counter() - t_start

    if artist_match is None:
        return {"success": False, "error": "No face detected in image", "timings": timings}

    # ── Step 2 — segment user image (disabled) ───────────────────────────
    # Segmentation disabled — passing the original image directly to ComfyUI.
    # Uncomment the block below to re-enable background removal before ComfyUI.
    # t_start = time.perf_counter()
    # segmented_output = str(OUTPUT_IMAGES_DIR / f"{stem}_segmented.png")
    # try:
    #     from PIL import Image as _PILImage
    #     from person_segmentation import remove_background_center_person
    #     _orig = _PILImage.open(image_path).convert("RGB")
    #     _seg  = remove_background_center_person(_orig)
    #     _seg.save(segmented_output)
    #     person_for_comfy = segmented_output
    #     print(f"[segmentation] Saved → {segmented_output}")
    # except Exception as e:
    #     print(f"[segmentation] Failed ({e}), using original image")
    #     person_for_comfy = image_path
    # timings["step_segmentation"] = time.perf_counter() - t_start
    person_for_comfy = image_path

    # ── Step 3 — music ────────────────────────────────────────────────────
    music_result = None
    if not skip_music:
        t_start = time.perf_counter()
        music_result = step_music(artist_match, mood, instrument, era)
        timings["step_music"] = time.perf_counter() - t_start
    else:
        timings["step_music"] = 0.0

    # ── Step 4 — ComfyUI polaroid ─────────────────────────────────────────
    poster_output = str(OUTPUT_IMAGES_DIR / f"{stem}_tribe_poster_{language}.png")
    t_start = time.perf_counter()
    try:
        tribe_poster = step_comfy_polaroid(
            person_image=person_for_comfy,
            artist_match=artist_match,
            output_path=poster_output,
        )
    except Exception as e:
        timings["step_comfy_polaroid"] = time.perf_counter() - t_start
        return {"success": False, "error": f"[comfy] {e}", "timings": timings}
    timings["step_comfy_polaroid"] = time.perf_counter() - t_start

    # ── Step 5 — video generation ─────────────────────────────────────────
    final_video = None
    audio_path = music_result.get("audio_path") if music_result and music_result.get("success") else None
    if tribe_poster and audio_path:
        t_start = time.perf_counter()
        video_output = str(OUTPUT_VIDEO_DIR / f"{stem}_final_{language}.mp4")
        final_video = step_video(tribe_poster, audio_path, video_output)
        timings["step_video"] = time.perf_counter() - t_start
    else:
        timings["step_video"] = 0.0

    return {
        "success":      True,
        "artist_match": artist_match,
        "styled_image": None,
        "tribe_poster": tribe_poster,
        "music":        music_result,
        "final_video":  final_video,
        "timings":      timings,
    }


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Festival Cruilla — Tal Cara, Tal Beat")
    parser.add_argument("--image",      required=True,  help="Path to user image")
    parser.add_argument("--output",     default=None,   help="Output styled image path")
    parser.add_argument("--mood",       default="happy")
    parser.add_argument("--instrument", default="synth")
    parser.add_argument("--era",        default="actual")
    parser.add_argument("--language",   default="ca", choices=["en", "es", "ca"], 
                        help="Language context for backgrounds (ca, es, en)")
    parser.add_argument("--with-music", action="store_true",
                        help="Also generate music (requires ACE-Step API)")
    args = parser.parse_args()

    total_start = time.perf_counter()
    
    result = run_pipeline(
        image_path=args.image,
        output_path=args.output,
        mood=args.mood,
        instrument=args.instrument,
        era=args.era,
        language=args.language,  # <--- Catch argument from CLI
        skip_music=not args.with_music,
    )
    
    total_duration = time.perf_counter() - total_start

    if result["success"]:
        print(f"\nArtist       : {result['artist_match']['name']} "
              f"({result['artist_match']['confidence']}%)")
        print(f"Tribe        : {result['artist_match'].get('tribe', 'unknown')}")
        if result.get("styled_image"):
            print(f"Styled image : {result['styled_image']}")
        if result["tribe_poster"]:
            print(f"Tribe poster : {result['tribe_poster']}")
        if result["music"] and result["music"].get("audio_path"):
            print(f"Music        : {result['music']['audio_path']}")
        if result["final_video"]:
            print(f"Final video  : {result['final_video']}")
            
        print("\n" + "="*40)
        print("         PERFORMANCE TIMINGS          ")
        print("="*40)
        for step, duration in result["timings"].items():
            print(f"{step:<20} : {duration:.2f} seconds")
        print("-"*40)
        print(f"{'Total Pipeline Time':<20} : {total_duration:.2f} seconds")
        print("="*40)
        
    else:
        print(f"\nPipeline failed: {result['error']}")
        
        if "timings" in result:
            print("\n--- Timings up to failure ---")
            for step, duration in result["timings"].items():
                print(f"{step}: {duration:.2f}s")
                
        sys.exit(1)