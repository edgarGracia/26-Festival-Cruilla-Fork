"""
================================================================================
 GENERADOR DE VÍDEO ULTRA-RÁPIDO (ffmpeg puro) - Festival Cruïlla
================================================================================
Genera un vídeo horizontal de 20s en un estilo "scrapbook" dinámico: fotos tipo
polaroid rotadas con borde amarillo grueso y sombra, tipografía condensada en
negrita, y acentos decorativos con la cruz "X" de la marca — todo generado por
código (PIL), sin depender de plantillas de Canva exportadas como PNG.

Arquitectura por frame (1920×1080):
  LEFT  half (0–959):   polaroid image (static across all 4 blocks)
  RIGHT half (960–1919): escena dinámica por bloque (ver abajo)

Bloques (derecha, con Ken Burns + transiciones variadas entre ellos). Textos en
catalán, salvo "Tal cara, tal beat" (nombre del proyecto, se deja tal cual):
  0-5s   landmarks   -> foto del mesh de landmarks + "Així et veu la IA"
  5-10s  artistas    -> collage de fotos rotadas + "...t'ha identificat com {nombre}"
  10-15s casa        -> sticker de la casa + "...t'identifica amb la casa dels {nombre}"
  15-20s música       -> tarjeta final amarilla + "la música que sona és la teva identitat"

Audio: pista de música durante los 20s.

Requisitos: ffmpeg + pillow
================================================================================
"""

import os
import subprocess
import time
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps

# ==============================================================================
# CONFIGURACIÓN - LINKS PRUEBA
# ==============================================================================
CONFIG = {
    "polaroid_path":     "/home/cvcadmin/cruilla/Festival-Cruilla/outputs/images/sonia1_tribe_poster_ca.png",
    "landmarks_path":    "/home/cvcadmin/cruilla/Festival-Cruilla/outputs/landmarks/sonia1_landmarks.png",
    "music_path":        "/home/cvcadmin/cruilla/Festival-Cruilla/outputs/music/af05bfeb-bd23-859d-133c-cb776de370ad.wav",
    "casa_sticker_path": "/home/cvcadmin/cruilla/Festival-Cruilla/final_video/casas/Casa_Techno.png",
    "casa_nombre":       "Tecno",
    "artistas": [
        {"name": "Lena Pulse", "confidence": 8,  "image": "/home/cvcadmin/cruilla/Festival-Cruilla/Fake_Artists/4/4_1.png"},
        {"name": "Artista 2",  "confidence": 6,  "image": "/home/cvcadmin/cruilla/Festival-Cruilla/Fake_Artists/2/2_1.png"},
        {"name": "Artista 3",  "confidence": 5,  "image": "/home/cvcadmin/cruilla/Festival-Cruilla/Fake_Artists/19/image (1).png"},
    ],
    "output_path": "/home/cvcadmin/cruilla/Festival-Cruilla/final_video/resultado_ffmpeg.mp4",

    # Technical parameters
    "resolucion":          (1920, 1080),
    "fps":                 30,
    "duracion_total":      20,
    "duracion_bloque":     5,
    "duracion_transicion": 0.55,  # con cuerpo suficiente para sentirse suave, no un corte brusco
    "usar_gpu":            True,
    "ffmpeg_preset":       "ultrafast",
    "crf":                 23,
    "threads":             0,
}


# ==============================================================================
# IMAGE UTILITIES
# ==============================================================================

def cover_resize(img, target_w, target_h):
    """Resize + center-crop to fill target_w × target_h (CSS object-fit: cover)."""
    img = img.convert("RGB")
    src_w, src_h = img.size
    if src_w / src_h > target_w / target_h:
        new_h, new_w = target_h, int(target_h * src_w / src_h)
    else:
        new_w, new_h = target_w, int(target_w * src_h / src_w)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top  = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def contain_resize(img, target_w, target_h):
    """Resize to fit entirely within target_w × target_h (no crop)."""
    img = img.convert("RGBA")
    img.thumbnail((target_w, target_h), Image.LANCZOS)
    return img


# ==============================================================================
# BRAND STYLING — scrapbook cards, cross-motif accents, condensed headlines
# ==============================================================================

BRAND_YELLOW = (255, 221, 35)
BRAND_INK    = (15, 15, 15)

# Anton (bundled here, OFL) is the festival's actual official typeface —
# tall, condensed, heavy-weight poster sans, as used on cruillabarcelona.cat
# itself ("ÚLTIMO CAMBIO DE PRECIO") — not a stylistic guess.
_FONT_CANDIDATES_DISPLAY = [
    os.path.join(os.path.dirname(__file__), "fonts", "Anton.ttf"),
    "/usr/share/fonts/abattis-cantarell/Cantarell-Bold.otf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf",
]
_FONT_CANDIDATES_BOLD = [
    "/usr/share/fonts/abattis-cantarell/Cantarell-Bold.otf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf",
]


def _cargar_fuente(candidates, size):
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def cargar_fuente_display(size):
    """The brand's hand-marker display face, for headlines/badges/titles."""
    return _cargar_fuente(_FONT_CANDIDATES_DISPLAY, size)


def cargar_fuente_bold(size):
    return _cargar_fuente(_FONT_CANDIDATES_BOLD, size)


def rounded_mask(size, radius):
    w, h = size
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
    return mask


def polaroid_card(img_or_path, w, h, border=16, radius=20, shadow_blur=20, shadow_offset=16):
    """A tilted-scrapbook-style photo card: thick yellow rounded border + soft
    drop shadow, ready to be rotated and dropped onto a scene."""
    src = img_or_path if isinstance(img_or_path, Image.Image) else Image.open(img_or_path)
    photo = cover_resize(src, w, h)

    card_w, card_h = w + border * 2, h + border * 2
    card = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
    ImageDraw.Draw(card).rounded_rectangle([0, 0, card_w - 1, card_h - 1], radius=radius, fill=(*BRAND_YELLOW, 255))
    card.paste(photo, (border, border), rounded_mask((w, h), max(radius - border, 6)))

    pad = shadow_blur * 3
    canvas = Image.new("RGBA", (card_w + pad * 2, card_h + pad * 2), (0, 0, 0, 0))
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        [pad + shadow_offset, pad + shadow_offset, pad + shadow_offset + card_w, pad + shadow_offset + card_h],
        radius=radius, fill=(0, 0, 0, 180),
    )
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(shadow_blur)))
    canvas.alpha_composite(card, (pad, pad))
    return canvas


def paste_rotated(base, card, cx, cy, angle):
    """Rotates `card` and drops it onto `base` centered at (cx, cy)."""
    rotated = card.rotate(angle, expand=True, resample=Image.BICUBIC)
    base.alpha_composite(rotated, (cx - rotated.width // 2, cy - rotated.height // 2))


def cross_burst(size, color=BRAND_YELLOW, thickness_ratio=0.20, angle=0):
    """Decorative accent shaped like the brand's own 'X' cross mark (instead
    of a generic star/sparkle) — used to scatter accents around a scene."""
    s = size
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    t = int(s * thickness_ratio)
    pad = int(s * 0.10)
    d.line([(pad, pad), (s - pad, s - pad)], fill=(*color, 255), width=t)
    d.line([(s - pad, pad), (pad, s - pad)], fill=(*color, 255), width=t)
    for x, y in [(pad, pad), (s - pad, s - pad), (s - pad, pad), (pad, s - pad)]:
        d.ellipse([x - t / 2, y - t / 2, x + t / 2, y + t / 2], fill=(*color, 255))
    if angle:
        img = img.rotate(angle, expand=True, resample=Image.BICUBIC)
    return img


def headline(canvas, x, y, kicker, title, max_w, kicker_color=BRAND_YELLOW, title_color=(255, 255, 255)):
    """Small kicker line + big wrapped title, both set in the brand's hand-marker
    face — the 'MIS FAVORITOS / MODA 2024' two-tier headline pattern."""
    d = ImageDraw.Draw(canvas)
    f_kicker = cargar_fuente_display(26)
    f_title = cargar_fuente_display(62)

    ty = y
    if kicker:
        d.text((x, y), kicker.upper(), font=f_kicker, fill=(*kicker_color, 255))
        ty = d.textbbox((x, y), kicker.upper(), font=f_kicker)[3] + 16

    words, lines, cur = title.upper().split(), [], ""
    for word in words:
        test = (cur + " " + word).strip()
        if cur and d.textbbox((0, 0), test, font=f_title)[2] > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = test
    if cur:
        lines.append(cur)
    for line in lines:
        d.text((x, ty), line, font=f_title, fill=(*title_color, 255))
        ty += 70
    return ty


# ==============================================================================
# SCENES — one per block, each builds a full w_half × H right-column frame
# ==============================================================================

def scene_landmarks(w, h, img_path, kicker, title):
    canvas = Image.new("RGBA", (w, h), (*BRAND_INK, 255))
    end_y = headline(canvas, 50, 46, kicker, title, w - 100)

    card = polaroid_card(img_path, 760, 640)
    paste_rotated(canvas, card, w // 2, end_y + 380, 3)

    canvas.alpha_composite(cross_burst(100, angle=-10), (40, h - 200))
    canvas.alpha_composite(cross_burst(70, angle=18), (w - 120, end_y + 60))
    return canvas


def _bw(img_path):
    """Grayscale version of a photo (still returned as RGB so it composites
    cleanly with the rest of the — colored — scene)."""
    return ImageOps.grayscale(Image.open(img_path)).convert("RGB")


def scene_artistas(w, h, artistas, kicker):
    """Podium layout: the best match large and centered on top, the two
    runner-ups the same size as each other side by side below — a ranking,
    not a scattered collage — all photos desaturated to black & white."""
    canvas = Image.new("RGBA", (w, h), (*BRAND_INK, 255))
    # Explicitly pick the highest-confidence match for the big photo instead
    # of trusting the caller to have pre-sorted the list — the "who does the
    # AI say you look like" claim has to point at the actual best match.
    ordenados = sorted(
        [a for a in artistas if a.get("image") and os.path.exists(a["image"])],
        key=lambda a: a.get("confidence", 0), reverse=True,
    )
    if not ordenados:
        return canvas
    end_y = headline(canvas, 50, 46, kicker, ordenados[0]["name"], w - 100)

    gutter = max(28, int(w * 0.05))
    content_top = end_y + 30
    content_h = (h - 40) - content_top
    top_h = int(content_h * 0.58)
    bottom_h = content_h - top_h - gutter

    top_w = int(w * 0.60)
    top_x = (w - top_w) // 2
    bottom_w = (top_w - gutter) // 2
    left_cx = top_x + bottom_w // 2
    right_cx = top_x + bottom_w + gutter + bottom_w // 2
    bottom_cy = content_top + top_h + gutter + bottom_h // 2

    canvas.alpha_composite(cross_burst(90, angle=12), (w - 120, content_top + 20))
    canvas.alpha_composite(cross_burst(70, angle=-8), (30, h - 150))

    card_top = polaroid_card(_bw(ordenados[0]["image"]), top_w, top_h)
    paste_rotated(canvas, card_top, w // 2, content_top + top_h // 2, 0)

    secundarios = ordenados[1:3]
    if len(secundarios) > 0:
        card_l = polaroid_card(_bw(secundarios[0]["image"]), bottom_w, bottom_h)
        paste_rotated(canvas, card_l, left_cx, bottom_cy, 0)
    if len(secundarios) > 1:
        card_r = polaroid_card(_bw(secundarios[1]["image"]), bottom_w, bottom_h)
        paste_rotated(canvas, card_r, right_cx, bottom_cy, 0)

    return canvas


def _casa_content(sticker_path, w, h):
    """House sticker over its own blurred backdrop (avoids exposing the
    sticker's flat white background inside the card)."""
    raw = Image.open(sticker_path).convert("RGBA")
    backdrop = cover_resize(raw.convert("RGB"), w, h)
    backdrop = ImageEnhance.Brightness(backdrop.filter(ImageFilter.GaussianBlur(24))).enhance(0.4)
    base = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    base.paste(backdrop, (0, 0))
    sticker = contain_resize(raw, int(w * 0.86), int(h * 0.86))
    base.alpha_composite(sticker, ((w - sticker.width) // 2, (h - sticker.height) // 2))
    return base


def scene_casa(w, h, sticker_path, casa_nombre, kicker):
    canvas = Image.new("RGBA", (w, h), (*BRAND_INK, 255))
    end_y = headline(canvas, 50, 46, kicker, f"la casa dels {casa_nombre}", w - 100)

    card = polaroid_card(_casa_content(sticker_path, 620, 520), 620, 520)
    paste_rotated(canvas, card, w // 2, end_y + 330, -3)

    canvas.alpha_composite(cross_burst(90, angle=15), (60, h - 220))
    canvas.alpha_composite(cross_burst(60, angle=-20), (w - 110, end_y + 40))
    return canvas


def scene_musica(w, h, kicker, title):
    """Final branded card — inverted to a full-bleed yellow background (like
    the finale of a reel), big centered title, ink-colored 'X' cross accents
    tucked into the corners so they don't compete with the text."""
    canvas = Image.new("RGBA", (w, h), (*BRAND_YELLOW, 255))
    d = ImageDraw.Draw(canvas)

    f_kicker = cargar_fuente_display(30)
    f_title = cargar_fuente_display(88)

    max_w = w - 140
    words, lines, cur = title.upper().split(), [], ""
    for word in words:
        test = (cur + " " + word).strip()
        if cur and d.textbbox((0, 0), test, font=f_title)[2] > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = test
    if cur:
        lines.append(cur)

    line_h, kicker_h = 100, 64
    y = (h - (kicker_h + len(lines) * line_h)) // 2

    kicker_up = kicker.upper()
    kw = d.textbbox((0, 0), kicker_up, font=f_kicker)[2]
    d.text(((w - kw) // 2, y), kicker_up, font=f_kicker, fill=(*BRAND_INK, 255))
    y += kicker_h

    for line in lines:
        lw = d.textbbox((0, 0), line, font=f_title)[2]
        d.text(((w - lw) // 2, y), line, font=f_title, fill=(*BRAND_INK, 255))
        y += line_h

    canvas.alpha_composite(cross_burst(150, color=BRAND_INK, thickness_ratio=0.18, angle=10), (-35, -35))
    canvas.alpha_composite(cross_burst(150, color=BRAND_INK, thickness_ratio=0.18, angle=-12), (w - 115, h - 115))
    return canvas


# ==============================================================================
# COMPOSERS
# left  = polaroid only, rendered once, reused static for the whole 20s
# right = one scene per block, kept as a separate w_half-wide image (instead
#         of a single 1920-wide frame) so ffmpeg can apply Ken-Burns zoom
#         motion to just this column without ever touching/animating the
#         polaroid.
# ==============================================================================

def _compose_left(cfg, W, H):
    w_half = W // 2
    return cover_resize(Image.open(cfg["polaroid_path"]), w_half, H)


def frame_landmarks_right(cfg, W, H):
    w_half = W // 2
    return scene_landmarks(w_half, H, cfg["landmarks_path"], "Tal cara, tal beat...", "Així et veu la IA")


def frame_artistas_right(cfg, W, H):
    w_half = W // 2
    artistas = cfg.get("artistas") or []
    if not artistas:
        return Image.new("RGBA", (w_half, H), (*BRAND_INK, 255))
    return scene_artistas(w_half, H, artistas, "Tal cara, tal beat t'ha identificat com...")


def frame_casa_right(cfg, W, H):
    w_half = W // 2
    sticker_path = cfg.get("casa_sticker_path")
    nombre = cfg.get("casa_nombre", "").strip()
    if not (sticker_path and os.path.exists(sticker_path) and nombre):
        return Image.new("RGBA", (w_half, H), (*BRAND_INK, 255))
    return scene_casa(w_half, H, sticker_path, nombre, "Tal cara, tal beat t'identifica amb...")


def frame_musica_right(cfg, W, H):
    w_half = W // 2
    return scene_musica(w_half, H, "Tal cara, tal beat", "La música que sona és la teva identitat")


def frame_preview(cfg, W, H, right_fn):
    """Convenience: full 1920x1080 still (left + right) for quick visual checks —
    not used by the real (animated) render pipeline."""
    w_half = W // 2
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    canvas.paste(_compose_left(cfg, W, H).convert("RGBA"), (0, 0))
    canvas.paste(right_fn(cfg, W, H).convert("RGB"), (w_half, 0))
    return canvas


# ==============================================================================
# FFMPEG ENCODER SELECTION
# ==============================================================================

_nvenc_available = None  # cached across calls within the same process — see below


def _detectar_encoder(cfg):
    global _nvenc_available

    if not cfg.get("usar_gpu", True):
        return "libx264", ["-preset", cfg["ffmpeg_preset"], "-crf", str(cfg["crf"])]

    # The nvenc probe is a real `subprocess.run` (fork+exec), which measured
    # ~0.5-0.65s here even though the equivalent raw shell command takes
    # ~0.02s — that gap is Python/OS process-spawn overhead, not ffmpeg
    # itself. Hardware capability can't change between calls in the same
    # process, so we pay that cost once and reuse the answer, instead of on
    # every single video generated.
    if _nvenc_available is None:
        try:
            test = subprocess.run(
                ["ffmpeg", "-hide_banner", "-f", "lavfi", "-i", "color=black:s=64x64:d=0.1",
                 "-c:v", "h264_nvenc", "-f", "null", "-"],
                capture_output=True, timeout=10,
            )
            _nvenc_available = (test.returncode == 0)
        except Exception:
            _nvenc_available = False
        if not _nvenc_available:
            print("[!] GPU NVENC not available, using CPU (libx264).")

    if _nvenc_available:
        return "h264_nvenc", ["-preset", "p1", "-tune", "ll"]
    return "libx264", ["-preset", cfg["ffmpeg_preset"], "-crf", str(cfg["crf"])]


# ==============================================================================
# MAIN VIDEO GENERATOR
# ==============================================================================

ZOOM_MAX = 0.08  # subtle Ken Burns: 8% zoom-in over each block's duration


def generar_video(cfg=CONFIG):
    t0 = time.time()
    W, H   = cfg["resolucion"]
    w_half = W // 2
    fps    = cfg["fps"]
    dur    = cfg["duracion_bloque"]

    workdir = "/tmp/festival_render"
    os.makedirs(workdir, exist_ok=True)

    # ---- 1) Render left (static polaroid, once) + 4 right-column blocks (PIL) ----
    # Saved as JPEG (quality=92) rather than PNG: these are throwaway frames
    # immediately consumed by ffmpeg below, and PNG's default compression
    # takes ~0.6-0.7s per frame vs ~0.01s for JPEG — that alone was ~2.5s of
    # the render budget for zero visible benefit.
    print("-> Rendering frames (PIL)...")
    _compose_left(cfg, W, H).save(f"{workdir}/left.jpg", quality=92)
    frame_landmarks_right(cfg, W, H).convert("RGB").save(f"{workdir}/b0.jpg", quality=92)
    frame_artistas_right( cfg, W, H).convert("RGB").save(f"{workdir}/b1.jpg", quality=92)
    frame_casa_right(     cfg, W, H).convert("RGB").save(f"{workdir}/b2.jpg", quality=92)
    frame_musica_right(   cfg, W, H).convert("RGB").save(f"{workdir}/b3.jpg", quality=92)
    t1 = time.time()
    print(f"   ({t1 - t0:.2f}s)")

    # ---- 2) Single ffmpeg call: Ken Burns zoom + xfade transitions + hstack + audio ----
    # Every cut slides left-to-right (same direction throughout, per brief)
    # at the same smooth duration — consistent motion reads as a deliberate
    # edit rhythm rather than three different templated wipes.
    cut1, cut2, cut3 = "slideright", "slideright", "slideright"
    trans1 = trans2 = trans3 = cfg["duracion_transicion"]

    # Block durations: d0=dur, d1/d2/d3=dur+(their own incoming transition)
    # so each xfade offset lines up and the total still lands on 20s exactly
    # regardless of the per-cut transition lengths (the padding each block
    # carries is exactly cancelled out by the corresponding xfade overlap).
    d0 = dur
    d1 = dur + trans1
    d2 = dur + trans2
    d3 = dur + trans3
    duracion_total = d0 + d1 + d2 + d3 - trans1 - trans2 - trans3  # = 4*dur = 20s

    off1 = d0 - trans1
    off2 = d0 + d1 - trans1 - trans2
    off3 = d0 + d1 + d2 - trans1 - trans2 - trans3

    t_enc0 = time.time()
    encoder, encoder_flags = _detectar_encoder(cfg)
    t_enc1 = time.time()
    print(f"-> Encoder: {encoder} (detectado en {t_enc1 - t_enc0:.2f}s)")

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-loop", "1", "-framerate", str(fps), "-t", str(duracion_total), "-i", f"{workdir}/left.jpg",
        "-loop", "1", "-framerate", str(fps), "-t", str(d0), "-i", f"{workdir}/b0.jpg",
        "-loop", "1", "-framerate", str(fps), "-t", str(d1), "-i", f"{workdir}/b1.jpg",
        "-loop", "1", "-framerate", str(fps), "-t", str(d2), "-i", f"{workdir}/b2.jpg",
        "-loop", "1", "-framerate", str(fps), "-t", str(d3), "-i", f"{workdir}/b3.jpg",
        "-stream_loop", "-1", "-i", cfg["music_path"],
    ]

    # zoompan turns each static (looped) right-column image into motion:
    # `d=1` because the input is already `-loop 1 -framerate fps` (i.e.
    # already the right number of duplicate frames at the right pace) —
    # zoompan just needs to nudge the zoom by one step per frame it
    # receives, not multiply the frame count itself.
    #
    # The zoom expression layers three things:
    #  - `scale=<2x>:-1` BEFORE zoompan: without this, zoompan's crop x/y get
    #    rounded to whole pixels *of the small source* every frame, and that
    #    rounding jumps inconsistently frame to frame — the well-known
    #    "shaky/jittery zoompan" bug (ffmpeg #4298). Upscaling first gives the
    #    crop math enough sub-pixel headroom that the rounding error becomes
    #    invisible after zoompan's own downscale to the final size. This is
    #    what actually eliminates jitter — an intentional shake would just
    #    bring back the exact thing this fixes. 1.3x was measured as the
    #    smallest factor that still keeps the motion visibly smooth (checked
    #    via a frame-by-frame contact strip) — 2x fixed it too but cost
    #    ~0.35s more per render for no visible difference.
    #  - an ease-out cubic base curve (1-(1-t)^3) instead of a linear crawl:
    #    fast at the start of the block, settling smoothly by the end —
    #    reads as intentional motion instead of a flat pan.
    #  - a decaying "bounce" at the start of each block (punches in past the
    #    target zoom, settles with shrinking oscillations, like a ball
    #    landing) plus a very small continuous "breathing" sine on top, so a
    #    held shot never goes fully static. Both are tiny relative to the
    #    ease-out curve and `max(1.001, ...)` guarantees zoom never dips to
    #    or past 1.0, which would reveal the source's uncropped edge.
    # More overshoot than before (0.12 -> 0.19) but a longer decay/slower
    # frequency so it settles as a couple of soft, springy wobbles instead
    # of a snap — the "gelatin" read, without reintroducing the jittery
    # feel that was explicitly fixed earlier.
    BOUNCE_AMP, BOUNCE_DECAY, BOUNCE_FREQ = 0.19, 11.0, 0.65
    BREATHE_AMP, BREATHE_FREQ = 0.012, 0.12

    def zoompan(block_idx, frames):
        t = f"(on/{frames - 1})"
        ease = f"(1-pow(1-{t},3))"
        base = f"1+{ZOOM_MAX}*{ease}"
        bounce = f"{BOUNCE_AMP}*exp(-on/{BOUNCE_DECAY})*abs(cos(on*{BOUNCE_FREQ}))"
        breathe = f"{BREATHE_AMP}*sin(on*{BREATHE_FREQ})"
        z = f"max(1.001,({base})+({bounce})+({breathe}))"
        return (
            f"[{block_idx}:v]scale={int(w_half * 1.3)}:-1:flags=lanczos,"
            f"zoompan=z='{z}':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d=1:s={w_half}x{H}:fps={fps},format=yuv420p[z{block_idx}]"
        )

    filter_complex = (
        f"[0:v]format=yuv420p[left];"
        f"{zoompan(1, d0 * fps)};"
        f"{zoompan(2, d1 * fps)};"
        f"{zoompan(3, d2 * fps)};"
        f"{zoompan(4, d3 * fps)};"
        f"[z1][z2]xfade=transition={cut1}:duration={trans1}:offset={off1}[x1];"
        f"[x1][z3]xfade=transition={cut2}:duration={trans2}:offset={off2}[x2];"
        f"[x2][z4]xfade=transition={cut3}:duration={trans3}:offset={off3}[right];"
        f"[left][right]hstack=inputs=2[video]"
    )

    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[video]",
        "-map", "5:a",
        "-t", str(duracion_total),
        "-c:v", encoder, *encoder_flags,
        "-c:a", "aac", "-b:a", "192k",
        "-threads", str(cfg.get("threads", 0)),
        "-movflags", "+faststart",
        cfg["output_path"],
    ]

    os.makedirs(os.path.dirname(cfg["output_path"]), exist_ok=True)
    print("-> Rendering video (ffmpeg)...")
    t_ff0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    t2 = time.time()

    if result.returncode != 0:
        print("ERROR in ffmpeg:")
        print(result.stderr[-3000:])
        raise RuntimeError("ffmpeg failed — see log above")

    print(f"   ({t2 - t_ff0:.2f}s)")
    print(
        f"Done in {t2 - t0:.2f}s total -> {cfg['output_path']} "
        f"[PIL {t1 - t0:.2f}s | encoder-detect {t_enc1 - t_enc0:.2f}s | ffmpeg {t2 - t_ff0:.2f}s]"
    )
    return cfg["output_path"]


if __name__ == "__main__":
    generar_video(CONFIG)
