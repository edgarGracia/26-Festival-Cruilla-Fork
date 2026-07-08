"""
Tal Cara, Tal Beat — main pipeline (Parallel Version, 2 GPUs reales).

Diferencias clave frente a la versión anterior:
  - La música y la cara/vestuario corren en DOS PROCESOS de sistema operativo
    separados (multiprocessing con start method "spawn"), cada uno con
    CUDA_VISIBLE_DEVICES fijado ANTES de importar torch/acestep/instantid.
    Esto asegura que cada pipeline usa su propia GPU física, no solo un
    "device" lógico dentro del mismo proceso/contexto CUDA.
  - Los modelos se cargan en cada request (nada de servicio persistente
    todavía) — es la versión "simple" para medir tiempos reales.
  - Se simula la llegada escalonada de los dos inputs: los 4 parámetros
    (mood/instrument/era/casa) lanzan la GPU de música inmediatamente;
    la imagen "llega" --image-delay segundos después y entonces se lanza
    la GPU de cara.
  - El vídeo final se genera en local (CPU/ffmpeg) y se queda guardado en
    disco tal cual — sin QR, sin servidor HTTP, nada más que abrir la
    carpeta de salida.

Uso (con 1 GPU o simulando, para pruebas locales):
    python main_parallel.py --image inputs/es/test.jpg \
        --mood happy --instrument guitar --era actual --casa pop \
        --language es --with-music --gpu-music 0 --gpu-face 0

Uso (con 2 GPUs reales):
    python main_parallel.py --image inputs/es/test.jpg \
        --mood happy --instrument guitar --era actual --casa pop \
        --language es --with-music --gpu-music 0 --gpu-face 1 --image-delay 7
"""

import argparse
import os
import sys
import time
import unicodedata
import multiprocessing as mp
from pathlib import Path

# ── Repo root & module paths ───────────────────────────────────────────────
# NOTA: con start method "spawn", cada proceso hijo re-ejecuta este script
# como módulo __main__ hasta el guard `if __name__ == "__main__":`, así que
# estos sys.path.insert se repiten automáticamente en cada hijo. No hace
# falta pasarlos "a mano".
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "face2label" / "models"))
sys.path.insert(0, str(REPO_ROOT / "models" / "ACE-Step-1.5"))
sys.path.insert(0, str(REPO_ROOT / "final_video"))

# ── Default artifact paths ─────────────────────────────────────────────────
MODEL_PATH           = REPO_ROOT / "face2label" / "logs" / "artists_mlp.pth"
LABELS_PATH          = REPO_ROOT / "face2label" / "logs" / "labels.json"
METADATA_PATH        = REPO_ROOT / "Fake_Artist.csv"
DATASET_DIR          = REPO_ROOT / "Fake_Artists"
ASSET_DIR            = REPO_ROOT / "inputs"
OUTPUT_DIR           = REPO_ROOT / "outputs"
OUTPUT_IMAGES_DIR    = OUTPUT_DIR / "images"
OUTPUT_MUSIC_DIR     = OUTPUT_DIR / "music"
OUTPUT_VIDEO_DIR     = OUTPUT_DIR / "final_video"
OUTPUT_LANDMARKS_DIR = OUTPUT_DIR / "landmarks"
CASAS_DIR            = REPO_ROOT / "final_video" / "casas"
TEMPLATES_DIR        = REPO_ROOT / "final_video" / "templates"
FONDO_DERECHA        = Path("/home/cvcadmin/cruilla/Festival-Cruilla/final_video/img/fondo.png")

CASA_STICKERS = {
    "indie": str(CASAS_DIR / "Casa_Indie.png"),
    "pop":   str(CASAS_DIR / "Casa_Pop.png"),
    "rock":  str(CASAS_DIR / "Casa_Rock.png"),
    "tecno": str(CASAS_DIR / "Casa_Techno.png"),
    "urban": str(CASAS_DIR / "Casa_Urban.png"),
}

# ── Canonical tribes: urban · indie · rock · pop · tecno ───────────────────
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

# accessory type (inputs/accesories/{casa}/{type}/) -> (name used in the
# prompt, where on the body ComfyUI should place it).
_ACCESSORY_PLACEMENT = {
    "glasses": (
        "a pair of glasses",
        "on the person's face, resting on the bridge of the nose with the temples reaching to the sides of the head. "
        "Scale the glasses so the frame spans only the width of the person's eyes — no wider than the face. "
        "They must sit flush against the face as if physically worn, not floating or oversized.",
    ),
    "hats": (
        "a hat",
        "on top of the person's head, sitting naturally on the crown of the hair. "
        "Size it proportionally to the head — the brim should not extend beyond shoulder width. "
        "The hat must look like it is resting on the head, not floating above it.",
    ),
    "necklaces": (
        "a necklace",
        "around the person's neck, hanging naturally at chest level against the clothing or skin. "
        "Scale it proportionally to the person's neck and torso — the pendant should be no larger than a fist. "
        "The chain should follow the curvature of the neck and chest, not float in front of the body.",
    ),
}

def _pick_random_accessory(tribe_key: str, asset_dir: str) -> tuple[str, str] | None:
    """Returns (accessory_image_path, accessory_type) or None.

    accessory_type is the name of the immediate parent folder (e.g. "hats",
    "glasses", "necklaces"), used to tell ComfyUI what the object is and
    where on the body to place it.
    """
    import random
    folder_name = _TRIBE_ACCESSORY_FOLDER.get(tribe_key, tribe_key)
    for acc_root_name in ("accesories", "accessories"):
        tribe_dir = Path(asset_dir) / acc_root_name / folder_name
        if tribe_dir.is_dir():
            pngs = [p for p in tribe_dir.rglob("*.png") if p.is_file()]
            if not pngs:
                return None
            chosen = random.choice(pngs)
            return str(chosen), chosen.parent.name
    return None


def _build_polaroid_prompt(accessory_type: str) -> str:
    label, placement = _ACCESSORY_PLACEMENT.get(
        accessory_type,
        (
            "the accessory",
            "on the person in a natural, proportionate position. "
            "Scale it to a realistic, wearable size relative to the person's body — do not make it oversized.",
        ),
    )
    return (
        "Use Image 1 as the base photo. Preserve its composition, framing, lighting style, "
        "colors, design elements, text, logos, borders, and overall layout exactly as they are.\n\n"
        "Take the person from Image 2 and place them naturally into the scene of Image 1. "
        "Remove the original environment from Image 2 completely. Preserve the person's identity, "
        "face, expression, hairstyle, body proportions, clothing, pose, and natural appearance.\n\n"
        f"Image 3 shows {label}. Place it {placement} "
        "Match the person's pose, scale, and perspective. "
        "The accessory must be correctly sized for a real human body — if it appears too large relative "
        "to the person, scale it down until it looks naturally wearable. "
        "Ensure correct contact points, shadows, occlusion, and lighting.\n\n"
        "Blend everything seamlessly into Image 1. Match the lighting, color temperature, "
        "contrast, sharpness, and perspective of the base photo. The final result should look "
        "like a single real photograph, not a collage."
    )


def _pin_gpu(gpu_id: int | None) -> None:
    """
    Fija la GPU visible para ESTE proceso. Tiene que llamarse antes de
    cualquier `import torch` / `import acestep` / `import predictor`
    (todos ellos son imports diferidos dentro de las funciones step_*),
    y por eso se llama al principio de cada función "worker" de proceso,
    nunca a nivel de módulo.
    """
    if gpu_id is None:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def _log_gpu_binding(tag: str) -> None:
    """Debug opcional: confirma qué GPU física está usando el proceso."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[{tag}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
                  f"→ torch ve {torch.cuda.device_count()} GPU(s), "
                  f"actual: {torch.cuda.get_device_name(0)}")
        else:
            print(f"[{tag}] CUDA no disponible en este proceso.")
    except Exception as e:
        print(f"[{tag}] No se pudo comprobar el binding de GPU: {e}")


# ==============================================================================
# STEP 1 — FACE -> ARTIST LABEL (top-3 con imágenes)
# ==============================================================================
def step_face2label(image_path: str) -> dict | None:
    from predictor import ArtistPredictor

    predictor = ArtistPredictor(
        model_path=str(MODEL_PATH),
        labels_path=str(LABELS_PATH),
        metadata_path=str(METADATA_PATH),
        dataset_dir=str(DATASET_DIR) if DATASET_DIR.exists() else None,
    )

    top3 = predictor.predict_topk(image_path, k=3)
    if top3 is None:
        print("[face2label] No se detectó ninguna cara.")
        return None

    best = top3[0]
    result = {
        "name":        best["name"],
        "confidence":  best["confidence"],
        "genre":       predictor.genre_map.get(best["name"], "Unknown"),
        "tribe":       predictor.tribe_map.get(best["name"], "Unknown"),
        "top_artists": top3,
    }
    print(f"[face2label] Match: {result['name']} ({result['confidence']}%) "
          f"genre: {result['genre']} tribe: {result.get('tribe', 'unknown')}")
    return result


# ==============================================================================
# STEP 2 — CLOTHING / ACCESORIOS
# ==============================================================================
def step_clothing(user_image_path: str, artist_match: dict, output_path: str,
                   landmarks_path: str | None = None) -> str | None:
    from clothing.Clothing import apply_look

    return apply_look(
        user_image_path=user_image_path,
        tribe=artist_match["tribe"],
        output_path=output_path,
        asset_dir=str(ASSET_DIR),
        landmarks_path=landmarks_path,
    )


# ==============================================================================
# STEP 2b — LANDMARK EXTRACTION (face mesh + pose skeleton visualisation)
# ==============================================================================
def step_landmarks(user_image_path: str, landmarks_path: str) -> bool:
    import cv2
    from clothing.Clothing import _get_landmarks, _save_landmarks

    img = cv2.imread(user_image_path)
    if img is None:
        print(f"[landmarks] Cannot open image: {user_image_path}")
        return False

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_lm, pose_lm = _get_landmarks(rgb)
    _save_landmarks(img, face_lm, pose_lm, landmarks_path)
    return True


# ==============================================================================
# STEP 3 — GENERACIÓN DE MÚSICA (carga el modelo en cada request)
# ==============================================================================
def step_music(mood: str, instrument: str, era: str, casa: str) -> dict | None:
    from api.music_generator import build_ace_prompt
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    save_dir = str(OUTPUT_MUSIC_DIR)
    os.makedirs(save_dir, exist_ok=True)

    prompt_data = build_ace_prompt(
        mood=mood, instrument=instrument, era=era, genre=casa, duration_seconds=25,
    )

    print(f"[music] Generando pista instrumental para Casa: {casa.upper()}...")
    t0 = time.perf_counter()

    handler = AceStepHandler()
    handler.initialize_service(
        project_root=None, config_path="acestep-v15-turbo", device="cuda",
    )
    print(f"[music] Modelo cargado en {time.perf_counter() - t0:.2f}s")

    params = GenerationParams(
        caption=prompt_data["tags"] + ". " + prompt_data["description"],
        lyrics="", duration=25, bpm=80,
    )
    gen_config = GenerationConfig(batch_size=1, audio_format="wav")

    t1 = time.perf_counter()
    result = generate_music(handler, None, params, gen_config, save_dir=save_dir)
    print(f"[music] Inferencia terminada en {time.perf_counter() - t1:.2f}s")

    if result.success:
        audio_path = result.audios[0]["path"] if result.audios else None
        return {"success": True, "audio_path": audio_path,
                "model_load_s": time.perf_counter() - t0}
    else:
        return {"success": False, "error": result.error}

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

    accessory = _pick_random_accessory(tribe_key, str(ASSET_DIR))
    if not accessory:
        raise RuntimeError(f"No accessories found for tribe '{tribe_key}'")
    accessory_path, accessory_type = accessory

    print(f"[comfy] bg={Path(bg_path).name}  "
          f"person={Path(person_image).name}  "
          f"accessory={Path(accessory_path).name} (type={accessory_type})")

    image_bytes = run_3ingredients_workflow(
        base_image=bg_path,
        person_image=person_image,
        object_image=accessory_path,
        prompt=_build_polaroid_prompt(accessory_type),
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(image_bytes)
    print(f"[comfy] Polaroid saved → {output_path}")
    return output_path

# ==============================================================================
# STEP 5 — VÍDEO FINAL (CPU/ffmpeg, en local)
# ==============================================================================
def step_rich_video(
    polaroid_path: str,
    landmarks_path: str,
    audio_path: str,
    artist_match: dict,
    casa: str,
    language: str,
    output_path: str,
) -> str | None:
    from videoFERNANDO import generar_video

    casa_key = _normalise_tribe(casa)
    cfg = {
        "polaroid_path":       polaroid_path,
        "fondo_derecha_path":  FONDO_DERECHA,
        "landmarks_path":      landmarks_path,
        "music_path":          audio_path,
        "casa_sticker_path":   CASA_STICKERS.get(casa_key, ""),
        "casa_nombre":         casa.capitalize(),
        "artistas":            artist_match.get("top_artists", []),
        "output_path":         output_path,
        "resolucion":          (1920, 1080),
        "fps":                 30,
        "duracion_total":      20,
        "duracion_bloque":     5,
        "duracion_transicion": 0.6,
        "usar_gpu":            True,
        "ffmpeg_preset":       "ultrafast",
        "crf":                 23,
        "threads":             0,
        "split_min_frac":      0.36,
        "split_max_frac":      0.52,
        "card_w_frac":         0.58,
        "card_aspect":         1.1875,
        "texto_fade_dur":      0.30,
        "foto_delay":          0.45,
        "foto_fade_dur":       0.40,
        "language":            language,
    }
    try:
        return generar_video(cfg)
    except Exception as e:
        print(f"[video] Error: {e}")
        return None


# ==============================================================================
# WORKFLOW DE IMAGEN (face → clothing → background)
# ==============================================================================
def workflow_crea_polaroid(image_path: str, output_path: str, language: str) -> dict:
    stem = Path(image_path).stem
    timings = {}

    t0 = time.perf_counter()
    artist_match = step_face2label(image_path)
    timings["step_face2label"] = time.perf_counter() - t0

    if not artist_match:
        return {"success": False, "error": "No face detected", "timings": timings}

    landmarks_path = str(OUTPUT_LANDMARKS_DIR / f"{stem}_landmarks.png")
    t0 = time.perf_counter()
    try:
        step_landmarks(image_path, landmarks_path)
    except Exception as e:
        print(f"[landmarks] Failed: {e}")
    timings["step_landmarks"] = time.perf_counter() - t0

    # Segmentation disabled — passing the original image directly to ComfyUI.
    # Uncomment the block below to re-enable background removal before ComfyUI.
    # t0 = time.perf_counter()
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
    # timings["step_segmentation"] = time.perf_counter() - t0
    person_for_comfy = image_path

    poster_output = str(OUTPUT_IMAGES_DIR / f"{stem}_tribe_poster_{language}.png")
    t0 = time.perf_counter()
    try:
        tribe_poster = step_comfy_polaroid(
            person_image=person_for_comfy,
            artist_match=artist_match,
            output_path=poster_output,
        )
    except Exception as e:
        timings["step_comfy_polaroid"] = time.perf_counter() - t0
        return {"success": False, "error": f"[comfy] {e}", "timings": timings}
    timings["step_comfy_polaroid"] = time.perf_counter() - t0

    return {
        "success": True,
        "artist_match": artist_match,
        "tribe_poster": tribe_poster,
        "landmarks_path": landmarks_path,
        "timings": timings,
    }


# ==============================================================================
# PROCESOS HIJOS (uno por GPU)
# ==============================================================================
def _music_process(mood, instrument, era, casa, gpu_id, queue):
    """Corre entero en un proceso de SO dedicado a `gpu_id`."""
    _pin_gpu(gpu_id)
    t0 = time.perf_counter()
    _log_gpu_binding("music-proc")
    try:
        result = step_music(mood, instrument, era, casa)
    except Exception as e:
        result = {"success": False, "error": str(e)}
    result["wall_time_s"] = time.perf_counter() - t0
    queue.put(("music", result))


def _image_process(image_path, output_path, language, gpu_id, queue):
    """Corre entero en un proceso de SO dedicado a `gpu_id`."""
    _pin_gpu(gpu_id)
    t0 = time.perf_counter()
    _log_gpu_binding("face-proc")
    try:
        result = workflow_crea_polaroid(image_path, output_path, language)
    except Exception as e:
        result = {"success": False, "error": str(e)}
    result["wall_time_s"] = time.perf_counter() - t0
    queue.put(("image", result))


# ==============================================================================
# ORQUESTADOR PRINCIPAL
# ==============================================================================
def run_pipeline(
    image_path: str,
    output_path: str | None = None,
    mood: str = "happy",
    instrument: str = "synth",
    era: str = "actual",
    casa: str = "pop",
    language: str = "ca",
    skip_music: bool = True,
    gpu_music: int | None = 0,
    gpu_face: int | None = 1,
    image_delay: float = 0.0,
) -> dict:
    OUTPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_IMAGES_DIR.mkdir(exist_ok=True)
    OUTPUT_MUSIC_DIR.mkdir(exist_ok=True)
    OUTPUT_VIDEO_DIR.mkdir(exist_ok=True)
    OUTPUT_LANDMARKS_DIR.mkdir(exist_ok=True)

    stem = Path(image_path).stem
    if output_path is None:
        output_path = str(OUTPUT_IMAGES_DIR / f"{stem}_styled_{language}.png")

    timings = {}
    queue = mp.Queue()

    print("\n" + "=" * 60)
    print(" PROCESAMIENTO EN PARALELO (2 procesos / 2 GPUs)")
    print(f"   GPU música : {gpu_music}   |   GPU cara : {gpu_face}")
    print("=" * 60)

    # ── t = 0s: llegan mood/instrument/era/casa → arranca la GPU de música ──
    p_music = None
    t_pipeline_start = time.perf_counter()
    if not skip_music:
        p_music = mp.Process(
            target=_music_process,
            args=(mood, instrument, era, casa, gpu_music, queue),
        )
        p_music.start()
        print(f"[orquestador] Proceso de música lanzado (PID {p_music.pid}) en GPU {gpu_music}")

    # ── t = +image_delay: "llega" la imagen → arranca la GPU de cara ────────
    if image_delay > 0:
        print(f"[orquestador] Esperando {image_delay:.1f}s a que llegue la imagen "
              f"(simulación del delay real de captura)...")
        time.sleep(image_delay)

    p_face = mp.Process(
        target=_image_process,
        args=(image_path, output_path, language, gpu_face, queue),
    )
    p_face.start()
    print(f"[orquestador] Proceso de cara lanzado (PID {p_face.pid}) en GPU {gpu_face}")

    # ── Recoger resultados ───────────────────────────────────────────────────
    results = {}
    n_expected = 2 if p_music else 1
    for _ in range(n_expected):
        tag, res = queue.get()  # bloquea hasta que llegue un resultado
        results[tag] = res

    if p_music:
        p_music.join()
    p_face.join()

    timings["wall_clock_total_gpu_stage"] = time.perf_counter() - t_pipeline_start

    music_result = results.get("music")
    image_result = results.get("image")

    if music_result:
        timings["step_music_wall"] = music_result.get("wall_time_s", 0.0)
    if image_result:
        timings["step_image_wall"] = image_result.get("wall_time_s", 0.0)
        timings.update(image_result.get("timings", {}))

    if not image_result or not image_result.get("success"):
        return {
            "success": False,
            "error": (image_result or {}).get("error", "unknown image pipeline error"),
            "timings": timings,
        }

    artist_match   = image_result["artist_match"]
    tribe_poster   = image_result["tribe_poster"]
    landmarks_path = image_result["landmarks_path"]

    # ── Vídeo final: en local, se queda guardado en disco tal cual ──────────
    final_video = None
    audio_path = music_result.get("audio_path") if music_result and music_result.get("success") else None

    if tribe_poster and audio_path:
        print("\n[video] Generando vídeo final (local, CPU/ffmpeg)...")
        t0 = time.perf_counter()
        video_output = str(OUTPUT_VIDEO_DIR / f"{stem}_final_{language}.mp4")
        final_video = step_rich_video(
            polaroid_path=tribe_poster,
            landmarks_path=landmarks_path,
            audio_path=audio_path,
            artist_match=artist_match,
            casa=_normalise_tribe(casa),
            output_path=video_output,
            language=language,
        )
        timings["step_video"] = time.perf_counter() - t0
        if final_video:
            print(f"[video] Guardado en disco → {final_video}")
    else:
        timings["step_video"] = 0.0
        print("[video] No se generó vídeo final (falta audio o poster).")

    return {
        "success":      True,
        "artist_match": artist_match,
        "styled_image": image_result.get("styled_image"),
        "tribe_poster": tribe_poster,
        "music":        music_result,
        "final_video":  final_video,
        "timings":      timings,
    }


# ==============================================================================
# CLI
# ==============================================================================
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # obligatorio para aislar CUDA por proceso

    parser = argparse.ArgumentParser(description="Festival Cruilla — Tal Cara, Tal Beat (2 GPUs)")
    parser.add_argument("--image",       required=True,  help="Ruta a la foto del usuario")
    parser.add_argument("--output",      default=None,   help="Ruta de salida de imagen estilizada")
    parser.add_argument("--mood",        default="happy")
    parser.add_argument("--instrument",  default="synth")
    parser.add_argument("--era",         default="actual")
    parser.add_argument("--casa",        default="pop", choices=["indie", "pop", "rock", "tecno", "urban"])
    parser.add_argument("--language",    default="ca", choices=["en", "es", "ca"])
    parser.add_argument("--with-music",  action="store_true", help="Generar música (si no, se omite ese proceso)")
    parser.add_argument("--gpu-music",   type=int, default=0, help="ID de GPU para el proceso de música")
    parser.add_argument("--gpu-face",    type=int, default=1, help="ID de GPU para el proceso de cara/vestuario")
    parser.add_argument("--image-delay", type=float, default=0.0,
                         help="Segundos a esperar antes de lanzar el proceso de imagen "
                              "(simula la llegada real 5-10s después de los parámetros)")
    args = parser.parse_args()

    total_start = time.perf_counter()

    result = run_pipeline(
        image_path=args.image, output_path=args.output,
        mood=args.mood, instrument=args.instrument, era=args.era, casa=args.casa,
        language=args.language, skip_music=not args.with_music,
        gpu_music=args.gpu_music, gpu_face=args.gpu_face,
        image_delay=args.image_delay,
    )

    total_duration = time.perf_counter() - total_start

    if result["success"]:
        print("\n" + "=" * 50)
        print(" PIPELINE COMPLETADO CON ÉXITO")
        print("=" * 50)
        print(f"Artista detectado : {result['artist_match']['name']}")
        print(f"Casa seleccionada : {args.casa.upper()}")
        print(f"Polaroid generada : {result['tribe_poster']}")
        if result["music"] and result["music"].get("audio_path"):
            print(f"Audio guardado    : {result['music']['audio_path']}")
        if result["final_video"]:
            print(f"Video MP4 final   : {result['final_video']}")

        print("\n" + "-" * 50)
        print("TIEMPOS DE EJECUCIÓN")
        print("-" * 50)
        for step, dur in result["timings"].items():
            print(f"{step:<28} : {dur:.2f} segundos")
        print("-" * 50)
        print(f"TIEMPO TOTAL (Wall-Clock): {total_duration:.2f} segundos")
        print("=" * 50)
    else:
        print(f"\nPipeline fallido: {result['error']}")
        if "timings" in result:
            print("\n--- Tiempos hasta el fallo ---")
            for step, dur in result["timings"].items():
                print(f"{step}: {dur:.2f}s")
        sys.exit(1)