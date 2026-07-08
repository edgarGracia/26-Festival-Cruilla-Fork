import math
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

# ==============================================================================
# CONFIGURACIÓN - LINKS PRUEBA
# ==============================================================================
CONFIG = {
    "polaroid_path":     "/home/cvcadmin/cruilla/Festival-Cruilla/final_video/img/image_tribe_poster_es.png",
    "fondo_derecha_path":"/home/cvcadmin/cruilla/Festival-Cruilla/final_video/img/fondo.png",
    "landmarks_path":    "/home/cvcadmin/cruilla/Festival-Cruilla/final_video/img/image_landmarks.png",
    "music_path":        "/home/cvcadmin/cruilla/Festival-Cruilla/final_video/img/8e02cfb3-9c6e-3654-d5fd-e5e6fc4b1298.wav",
    "casa_sticker_path": "/home/cvcadmin/cruilla/Festival-Cruilla/final_video/casas/Casa_Urban.png",
    "casa_nombre":       "Urban",
    "artistas": [
        {"name": "Lena Pulse", "confidence": 8,  "image": "/home/cvcadmin/cruilla/Festival-Cruilla/Fake_Artists/4/4_1.png"},
        {"name": "Artista 2",  "confidence": 6,  "image": "/home/cvcadmin/cruilla/Festival-Cruilla/Fake_Artists/2/2_1.png"},
        {"name": "Artista 3",  "confidence": 5,  "image": "/home/cvcadmin/cruilla/Festival-Cruilla/Fake_Artists/19/image (1).png"},
    ],
    "output_path": "/home/cvcadmin/cruilla/Festival-Cruilla/final_video/resultado_NV5.mp4",

    # Idioma de los textos en pantalla: "ca" | "es" | "en"
    "language": "ca",

    "resolucion":          (1920, 1080),
    "fps":                 30,
    "duracion_total":      20,
    "duracion_bloque":     5,
    "duracion_transicion": 0.6,
    "usar_gpu":            True,
    "ffmpeg_preset":       "p1",
    "crf":                 23,
    "threads":             0,

    "split_min_frac": 0.36,
    "split_max_frac": 0.52,

    "card_w_frac": 0.58,
    "card_aspect": 1.1875,

    # ETAPA 1 (texto): fade-in simple de todo el fondo (texto+acentos) al
    # arrancar el bloque.
    "texto_fade_dur": 0.30,
    # ETAPA 2 (foto): entra después, con SU PROPIO fade (de transparente a
    # opaca) — una transición distinta a la del texto, no la misma.
    "foto_delay":      0.45,
    "foto_fade_dur":   0.40,
}

"""
        "ffmpeg_preset":     "ultrafast",
        "crf":               23,
        "threads":           0,
"""

# ==============================================================================
# I18N — on-screen copy per language.
# "Tal cara, tal beat" is the project's own name, so it is NEVER translated
# and always appears verbatim, in every language.
# ==============================================================================

TEXTS = {
    "ca": {
        "landmarks_kicker": "Així és com...",
        "landmarks_title":  "Et veu la IA!",
        "artistas_kicker":  "Tal cara, tal beat t'ha identificat com...",
        "artistas_caption": "ALTRES ARTISTES IDENTIFICATS",
        "casa_kicker":      "Tal cara, tal beat t'identifica amb...",
        "casa_title":       "la casa dels {nombre}",
        "musica_kicker":    "Però...",
        "musica_title":     "La música que sona és la teva identitat",
    },
    "es": {
        "landmarks_kicker": "Así es como...",
        "landmarks_title":  "¡Te ve la IA!",
        "artistas_kicker":  "Tal cara, tal beat te ha identificado como...",
        "artistas_caption": "OTROS ARTISTAS IDENTIFICADOS",
        "casa_kicker":      "Tal cara, tal beat te identifica con...",
        "casa_title":       "la casa de los {nombre}",
        "musica_kicker":    "Pero...",
        "musica_title":     "La música que suena es tu identidad",
    },
    "en": {
        "landmarks_kicker": "This is how...",
        "landmarks_title":  "AI sees you!",
        "artistas_kicker":  "Tal cara, tal beat matched you with...",
        "artistas_caption": "OTHER ARTISTS IDENTIFIED",
        "casa_kicker":      "Tal cara, tal beat puts you in...",
        "casa_title":       "the house of {nombre}",
        "musica_kicker":    "But...",
        "musica_title":     "The music playing is your identity",
    },
}

DEFAULT_LANGUAGE = "ca"


def get_texts(language):
    """Returns the copy dict for `language`, falling back to the catalan
    (original/default) set if the code isn't recognized."""
    return TEXTS.get((language or "").strip().lower(), TEXTS[DEFAULT_LANGUAGE])


# ==============================================================================
# IMAGE UTILITIES
# ==============================================================================

def cover_resize(img, target_w, target_h):
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
    img = img.convert("RGBA")
    img.thumbnail((target_w, target_h), Image.LANCZOS)
    return img


def calcular_split(polaroid_path, W, H, min_frac=0.36, max_frac=0.52):
    with Image.open(polaroid_path) as img:
        aspect = img.width / img.height
    ideal_w_left = int(round(H * aspect))
    w_left = max(int(W * min_frac), min(ideal_w_left, int(W * max_frac)))
    return w_left, W - w_left


# ==============================================================================
# BRAND STYLING
# ==============================================================================

BRAND_YELLOW = (255, 221, 35)
BRAND_INK    = (15, 15, 15)
BRAND_WHITE  = (255, 255, 255)

_FONT_CANDIDATES_DISPLAY = [
    os.path.join(os.path.dirname(__file__), "fonts", "Anton.ttf"),
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
    return _cargar_fuente(_FONT_CANDIDATES_DISPLAY, size)


def rounded_mask(size, radius):
    w, h = size
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
    return mask


def circle_mask(diameter):
    mask = Image.new("L", (diameter, diameter), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, diameter - 1, diameter - 1], fill=255)
    return mask


def polaroid_card(img_or_path, w, h, border=16, radius=20, shadow_blur=20, shadow_offset=16):
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


def circle_card(img_path, diameter, ring=10, shadow_blur=14, shadow_offset=10):
    src = Image.open(img_path)
    photo = cover_resize(src, diameter, diameter)
    ring_d = diameter + ring * 2
    card = Image.new("RGBA", (ring_d, ring_d), (0, 0, 0, 0))
    ImageDraw.Draw(card).ellipse([0, 0, ring_d - 1, ring_d - 1], fill=(*BRAND_INK, 255))
    card.paste(photo, (ring, ring), circle_mask(diameter))
    pad = shadow_blur * 3
    canvas = Image.new("RGBA", (ring_d + pad * 2, ring_d + pad * 2), (0, 0, 0, 0))
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).ellipse(
        [pad + shadow_offset, pad + shadow_offset, pad + shadow_offset + ring_d, pad + shadow_offset + ring_d],
        fill=(0, 0, 0, 170),
    )
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(shadow_blur)))
    canvas.alpha_composite(card, (pad, pad))
    return canvas


def rotar(card, angle):
    if not angle:
        return card
    return card.rotate(angle, expand=True, resample=Image.BICUBIC)


def cross_burst(size, color=BRAND_YELLOW, thickness_ratio=0.30, angle=0):
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


TEXT_MARGIN_X = 90


def headline(canvas, y, kicker, title, w, kicker_color=BRAND_YELLOW, title_color=(255, 255, 255),
             kicker_size=46, title_size=86):
    d = ImageDraw.Draw(canvas)
    x = TEXT_MARGIN_X
    max_w = w - TEXT_MARGIN_X - 50
    f_kicker = cargar_fuente_display(kicker_size)
    f_title = cargar_fuente_display(title_size)
    line_h = int(title_size * 1.12)

    ty = y
    if kicker:
        d.text((x, y), kicker.upper(), font=f_kicker, fill=(*kicker_color, 255))
        ty = d.textbbox((x, y), kicker.upper(), font=f_kicker)[3] + 2

    if title:
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
            ty += line_h
    return ty


# ==============================================================================
# BLOQUES — cada build_* devuelve:
#   {
#     "full_text": <img w×h RGB>,   # fondo + kicker + título juntos (etapa 1)
#     "fotos": [ (img_rgba_pequeña, cx, cy), ... ],  # etapa 2, tamaño real
#   }
# Ya NO hay etapa intermedia "solo kicker": el texto completo aparece con un
# fade simple al arrancar el bloque, y la/s foto/s entran después con su
# propio fade — dos transiciones distintas, no una escalonada en 3 pasos.
#
# Los textos (kicker/título/caption) se resuelven vía get_texts(cfg["language"])
# — "Tal cara, tal beat" nunca se traduce.
# ==============================================================================

def _bg_base(w, h, cfg):
    fondo_path = cfg.get("fondo_derecha_path")
    if fondo_path and os.path.exists(fondo_path):
        with Image.open(fondo_path) as img_fondo:
            return cover_resize(img_fondo, w, h).convert("RGBA")
    return Image.new("RGBA", (w, h), (*BRAND_INK, 255))


def build_landmarks(cfg, w, h, card_w, card_h):
    t = get_texts(cfg.get("language"))
    kicker, title = t["landmarks_kicker"], t["landmarks_title"]

    full = _bg_base(w, h, cfg)
    end_y = headline(full, 50, kicker, title, w)
    full.alpha_composite(cross_burst(int(w * 0.10), angle=8), (40, h - int(h * 0.40)))
    full.alpha_composite(cross_burst(int(w * 0.07), angle=18), (w - int(w * 0.13), end_y + 60))

    card = rotar(polaroid_card(cfg["landmarks_path"], card_w, card_h), 3)
    cy = min(end_y + int(card_h * 0.68) + 70, h - card_h // 2 - 30)
    cx = w // 2

    return {"full_text": full.convert("RGB"), "fotos": [(card, cx, cy)]}


def _bw(img_path):
    return ImageOps.grayscale(Image.open(img_path)).convert("RGB")


def build_artistas(cfg, w, h):
    """Tarjeta principal y círculos secundarios más grandes y más separados
    del texto (antes quedaban chicos y pegados)."""
    t = get_texts(cfg.get("language"))
    artistas = cfg.get("artistas") or []
    ordenados = sorted(
        [a for a in artistas if a.get("image") and os.path.exists(a["image"])],
        key=lambda a: a.get("confidence", 0), reverse=True,
    )
    if not ordenados:
        blank = _bg_base(w, h, cfg).convert("RGB")
        return {"full_text": blank, "fotos": []}

    kicker = t["artistas_kicker"]
    title = ordenados[0]["name"]

    full = _bg_base(w, h, cfg)
    end_y = headline(full, 50, kicker, title, w)

    fotos = []
    avail = h - end_y - 40

    gap0 = int(avail * 0.10)          # aire extra entre el texto y la 1ª foto
    main_h = int(avail * 0.52)        # antes 0.34 -> tarjeta principal más grande
    main_w = int(main_h * 1.1875)
    gap1 = int(avail * 0.07)
    caption_h = 50
    gap2 = int(avail * 0.02)
    circle_d = int(avail * 0.22)      # antes 0.22 -> círculos más grandes

    main_cy = end_y + gap0 + main_h // 2
    fotos.append((polaroid_card(_bw(ordenados[0]["image"]), main_w, main_h), w // 2, main_cy))

    secundarios = ordenados[1:3]
    caption_y = main_cy + main_h // 2 + gap1
    if secundarios:
        d = ImageDraw.Draw(full)
        f_cap = cargar_fuente_display(30)
        caption = t["artistas_caption"]
        cw = d.textbbox((0, 0), caption, font=f_cap)[2]
        d.text(((w - cw) // 2, caption_y), caption, font=f_cap, fill=(*BRAND_YELLOW, 255))

        gap_circles = int(w * 0.07)
        circles_cy = caption_y + caption_h + gap2 + circle_d // 2
        left_cx = w // 2 - gap_circles // 2 - circle_d // 2
        right_cx = w // 2 + gap_circles // 2 + circle_d // 2
        c1 = circle_card(secundarios[0]["image"], circle_d) if len(secundarios) > 0 else None
        c2 = circle_card(secundarios[1]["image"], circle_d) if len(secundarios) > 1 else None
        if c1 and c2:
            # Pre-composited side by side (PIL, one-off cost) instead of two
            # separate zoompan+overlay chains in ffmpeg for two elements that
            # always move/fade in lockstep anyway — same pixels on screen,
            # one fewer filter chain per render.
            offset = right_cx - left_cx
            comb = Image.new("RGBA", (c1.width + offset, max(c1.height, c2.height)), (0, 0, 0, 0))
            comb.alpha_composite(c1, (0, (comb.height - c1.height) // 2))
            comb.alpha_composite(c2, (offset, (comb.height - c2.height) // 2))
            fotos.append((comb, (left_cx + right_cx) // 2, circles_cy))
        elif c1:
            fotos.append((c1, left_cx, circles_cy))

    full.alpha_composite(cross_burst(int(w * 0.08)), (w - int(w * 0.12), 300))
    full.alpha_composite(cross_burst(int(w * 0.06), angle=-8), (30, h - int(h * 0.12)))

    return {"full_text": full.convert("RGB"), "fotos": fotos}


def build_casa(cfg, w, h, card_w, card_h):
    t = get_texts(cfg.get("language"))
    sticker_path = cfg.get("casa_sticker_path")
    nombre = cfg.get("casa_nombre", "").strip()
    if not (sticker_path and os.path.exists(sticker_path) and nombre):
        blank = _bg_base(w, h, cfg).convert("RGB")
        return {"full_text": blank, "fotos": []}

    kicker = t["casa_kicker"]
    title = t["casa_title"].format(nombre=nombre)

    full = _bg_base(w, h, cfg)
    end_y = headline(full, 50, kicker, title, w)
    full.alpha_composite(cross_burst(int(w * 0.09), angle=10), (60, h - int(h * 0.30)))
    full.alpha_composite(cross_burst(int(w * 0.06), angle=-20), (w - int(w * 0.12), end_y + 40))

    sticker = contain_resize(Image.open(sticker_path), card_w, card_h)
    cy = min(end_y + int(sticker.height * 0.68) + 60, h - sticker.height // 2 - 30)
    cx = w // 2

    return {"full_text": full.convert("RGB"), "fotos": [(sticker, cx, cy)]}


def build_musica(cfg, w, h):
    t = get_texts(cfg.get("language"))
    kicker, title = t["musica_kicker"], t["musica_title"]
    canvas = Image.new("RGBA", (w, h), (*BRAND_YELLOW, 255))
    d = ImageDraw.Draw(canvas)
    f_kicker = cargar_fuente_display(70)
    f_title = cargar_fuente_display(100)

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

    line_h, kicker_h = 114, 96
    y = (h - (kicker_h + len(lines) * line_h)) // 2
    kicker_up = kicker.upper()
    kw = d.textbbox((0, 0), kicker_up, font=f_kicker)[2]
    d.text(((w - kw) // 2, y), kicker_up, font=f_kicker, fill=(*BRAND_WHITE, 255))
    y += kicker_h
    for line in lines:
        lw = d.textbbox((0, 0), line, font=f_title)[2]
        d.text(((w - lw) // 2, y), line, font=f_title, fill=(*BRAND_INK, 255))
        y += line_h

    canvas.alpha_composite(cross_burst(int(w * 0.16), color=BRAND_INK, thickness_ratio=0.30, angle=10), (-35, -35))
    canvas.alpha_composite(cross_burst(int(w * 0.16), color=BRAND_INK, thickness_ratio=0.30, angle=-12),
                            (w - int(w * 0.235), h - int(h * 0.2)))

    return {"full_text": canvas.convert("RGB"), "fotos": []}


# ==============================================================================
# FFMPEG ENCODER
# ==============================================================================

def _encoder_flags(cfg):
    if cfg.get("usar_gpu", True):
        return "h264_nvenc", ["-preset", "p1", "-tune", "ll", "-cq", "24", "-pix_fmt", "yuv420p"]
    return "libx264", ["-preset", "ultrafast", "-crf", str(cfg["crf"]), "-pix_fmt", "yuv420p"]


# ==============================================================================
# MAIN VIDEO GENERATOR
# ==============================================================================

ZOOM_MAX = 0.09
BOUNCE_AMP = 0.09
BOUNCE_DECAY = 9.0
BOUNCE_FREQ = 0.9
BREATHE_AMP = 0.010
BREATHE_FREQ = 0.17


def _zoom_expr(frames):
    return (f"max(1.001,(1+{ZOOM_MAX}*(1-pow(1-(on/{max(frames - 1, 1)}),3)))"
            f"+({BOUNCE_AMP}*exp(-on/{BOUNCE_DECAY})*abs(cos(on*{BOUNCE_FREQ})))"
            f"+({BREATHE_AMP}*sin(on*{BREATHE_FREQ})))")


# How long the per-photo fade/zoom/bounce is actually *visible* for: fade
# completes at foto_delay+foto_fade (~0.85s), bounce is decayed to ~0 by
# ~2s (exp(-frame/BOUNCE_DECAY) with BOUNCE_DECAY=9 frames ≈0.3s). Past that,
# all that's left is a ~1%-amplitude breathing wobble — imperceptible once
# frozen. Rendering only this window in PIL (instead of the whole ~5-5.6s
# block) is what makes pre-rendering cheaper than ffmpeg's per-frame
# zoompan+overlay for the full block duration.
ANIM_WINDOW_SEC = 1.3

# The breathing wobble (BREATHE_AMP/BREATHE_FREQ) never decays — unlike the
# ease-out zoom and the bounce, it's meant to keep going the whole time a
# photo is on screen. Rendering it for the *entire* remaining block would
# undo the whole point of ANIM_WINDOW_SEC, so instead we render exactly one
# period of the sine once and let ffmpeg loop that short clip for the rest
# of the block — a few extra frames, rendered once, instead of ~90-120 more.
BREATHE_LOOP_FRAMES = round(2 * math.pi / BREATHE_FREQ)


def _zoom_at(n, frames):
    """Python port of `_zoom_expr`, evaluated at a single frame `n` instead
    of compiled as an ffmpeg expression — same formula, same numbers."""
    t = n / max(frames - 1, 1)
    base = 1 + ZOOM_MAX * (1 - (1 - t) ** 3)
    bounce = BOUNCE_AMP * math.exp(-n / BOUNCE_DECAY) * abs(math.cos(n * BOUNCE_FREQ))
    breathe = BREATHE_AMP * math.sin(n * BREATHE_FREQ)
    return max(1.001, base + bounce + breathe)


def _ease_base_at(n, frames):
    t = n / max(frames - 1, 1)
    return 1 + ZOOM_MAX * (1 - (1 - t) ** 3)


def _zoom_at_frozen_base(base_frozen, n):
    """Same bounce+breathe as `_zoom_at`, but with the ease-out base pinned
    to a constant instead of still climbing. Used for the looped breathing
    cycle: the ease-out hasn't actually finished rising yet at n_anim (it's
    only ~2/3 there for a 5s block), so looping a slice of the *real*
    zoom_at would repeat a still-rising ramp and jump backwards on every
    wrap. Freezing the base at exactly its n_anim value keeps the loop
    perfectly periodic and matches the intro frame with zero discontinuity
    at the handoff — the (subtle, and already very slow at this point)
    remainder of the zoom-in creep is the one thing this trades away in
    exchange for looping cheaply; the breathing itself is unaffected."""
    bounce = BOUNCE_AMP * math.exp(-n / BOUNCE_DECAY) * abs(math.cos(n * BOUNCE_FREQ))
    breathe = BREATHE_AMP * math.sin(n * BREATHE_FREQ)
    return max(1.001, base_frozen + bounce + breathe)


def _zoomed_foto(foto, zoom):
    """PIL equivalent of the ffmpeg zoompan(d=1) trick: scale up by `zoom`
    and center-crop back to the original size, so the result drops in at
    the exact same (cx, cy) placement as the un-zoomed photo.
    BILINEAR, not LANCZOS: this runs once per animated frame (up to ~240
    times per render) and the zoom range here is tiny (~0-12%), so the two
    are visually indistinguishable (checked: mean per-pixel diff <1/255) —
    LANCZOS was ~1.7x slower for no visible gain at this zoom range."""
    fw, fh = foto.size
    zw, zh = max(1, round(fw * zoom)), max(1, round(fh * zoom))
    resized = foto.resize((zw, zh), Image.BILINEAR)
    left = (zw - fw) // 2
    top = (zh - fh) // 2
    return resized.crop((left, top, left + fw, top + fh))


def _render_block_frame(blk, w, h, n, frames_full, texto_fade_f, foto_delay_f, foto_fade_f, zoom_fn=None):
    """One fully-composited RGB frame (background fade-in + every photo's
    own zoom+fade) for animated frame index `n` — the PIL-side equivalent of
    what the ffmpeg fade/zoompan/overlay filter chain computed per-frame.
    `zoom_fn(n)` defaults to the real `_zoom_at`; the looped breathing cycle
    passes a frozen-base variant instead (see `_zoom_at_frozen_base`)."""
    if zoom_fn is None:
        zoom_fn = lambda n: _zoom_at(n, frames_full)

    bg_alpha = min(1.0, n / max(texto_fade_f, 1))
    if bg_alpha >= 1.0:
        frame = blk["full_text"].copy()
    else:
        black = Image.new("RGB", (w, h), (0, 0, 0))
        frame = Image.blend(black, blk["full_text"], bg_alpha)
    frame = frame.convert("RGBA")

    for foto, cx, cy in blk["fotos"]:
        if n < foto_delay_f:
            continue
        foto_alpha = min(1.0, (n - foto_delay_f) / max(foto_fade_f, 1))
        piece = _zoomed_foto(foto, zoom_fn(n))
        if foto_alpha < 1.0:
            r, g, b, a = piece.split()
            a = a.point(lambda v: int(v * foto_alpha))
            piece = Image.merge("RGBA", (r, g, b, a))
        frame.alpha_composite(piece, (cx - piece.width // 2, cy - piece.height // 2))

    return frame.convert("RGB")


def generar_video(cfg=CONFIG):
    t0 = time.time()
    W, H = cfg["resolucion"]
    fps  = cfg["fps"]
    dur  = cfg["duracion_bloque"]
    idioma = cfg.get("language", DEFAULT_LANGUAGE)

    w_left, w_right = calcular_split(
        cfg["polaroid_path"], W, H,
        cfg.get("split_min_frac", 0.36), cfg.get("split_max_frac", 0.52),
    )

    card_w = int(w_right * cfg.get("card_w_frac", 0.58))
    card_h = int(card_w / cfg.get("card_aspect", 1.1875))

    # A fresh directory per call, not a fixed shared path: a fixed path left
    # stale numbered frames (b{i}_anim_%04d / b{i}_loop_%04d) behind between
    # runs whenever a config change shrank the frame count — ffmpeg's image2
    # pattern reads *every* sequentially-numbered file it finds, so leftover
    # frames from a previous render (different photo/casa/etc.) got spliced
    # onto the end of the current one. This also makes concurrent renders
    # (e.g. two requests at once in a long-lived pipeline process) safe,
    # since they no longer share one directory.
    workdir = tempfile.mkdtemp(prefix="festival_render_")

    print(f"-> Rendering frames (PIL) [language={idioma}]...")
    cover_resize(Image.open(cfg["polaroid_path"]), w_left, H).save(f"{workdir}/left.jpg", quality=90)

    bloques = [
        build_landmarks(cfg, w_right, H, card_w, card_h),
        build_artistas(cfg, w_right, H),
        build_casa(cfg, w_right, H, card_w, card_h),
        build_musica(cfg, w_right, H),
    ]

    cut1, cut2, cut3 = "smoothright", "smoothleft", "smoothright"
    trans1 = trans2 = trans3 = cfg["duracion_transicion"]

    d0 = dur
    d1 = dur + trans1
    d2 = dur + trans2
    d3 = dur + trans3
    duracion_total = d0 + d1 + d2 + d3 - trans1 - trans2 - trans3
    durs = [d0, d1, d2, d3]

    off1 = d0 - trans1
    off2 = d0 + d1 - trans1 - trans2
    off3 = d0 + d1 + d2 - trans1 - trans2 - trans3

    texto_fade = cfg.get("texto_fade_dur", 0.30)
    foto_delay = cfg.get("foto_delay", 0.45)
    foto_fade = cfg.get("foto_fade_dur", 0.40)
    texto_fade_f = texto_fade * fps
    foto_delay_f = foto_delay * fps
    foto_fade_f = foto_fade * fps

    # Blocks with photos get their fade/zoom/bounce baked in PIL for the
    # short window where it's actually visible (ANIM_WINDOW_SEC), then a
    # short breathing cycle (one sine period, BREATHE_LOOP_FRAMES) that
    # ffmpeg loops for the rest of the block — same choreography (including
    # the continuous breathing wobble the whole time the photo is on
    # screen), but ffmpeg no longer has to run zoompan+overlay+fade per
    # frame for the full ~5-5.6s of every block. Blocks with no photos
    # (musica) are untouched: a plain fade on a single static frame was
    # already cheap.
    def _render_one_block(item):
        i, blk, d = item
        if not blk["fotos"]:
            blk["full_text"].save(f"{workdir}/b{i}_full.jpg", quality=90)
            return i, None
        frames_full = max(int(d * fps), 1)
        n_anim = max(1, min(round(ANIM_WINDOW_SEC * fps), frames_full - 1))
        for n in range(n_anim):
            _render_block_frame(blk, w_right, H, n, frames_full, texto_fade_f, foto_delay_f, foto_fade_f) \
                .save(f"{workdir}/b{i}_anim_{n:04d}.jpg", quality=90)

        # Freeze the ease-out base at its n_anim value for the loop (see
        # `_zoom_at_frozen_base`): the base is still rising at n_anim (only
        # ~2/3 of the way there for a 5s block), so looping a slice of the
        # true zoom_at would jump backwards on every wrap.
        base_frozen = _ease_base_at(n_anim, frames_full)
        loop_zoom_fn = lambda n: _zoom_at_frozen_base(base_frozen, n)
        remaining_frames = frames_full - n_anim
        n_loop = max(1, min(BREATHE_LOOP_FRAMES, remaining_frames))
        for k in range(n_loop):
            _render_block_frame(blk, w_right, H, n_anim + k, frames_full, texto_fade_f, foto_delay_f, foto_fade_f,
                                 zoom_fn=loop_zoom_fn) \
                .save(f"{workdir}/b{i}_loop_{k:04d}.jpg", quality=90)

        return i, {"n_anim": n_anim, "loop_dur": d - (n_anim / fps)}

    # Each block's animated frames are independent of the others, and
    # Pillow's resize/blend releases the GIL for the bulk of the work, so
    # rendering the (up to 3) foto-bearing blocks concurrently is a real
    # ~2x wall-clock win with zero change to the actual pixels produced.
    anim_meta = [None, None, None, None]
    with ThreadPoolExecutor(max_workers=len(bloques)) as ex:
        for i, meta in ex.map(_render_one_block, [(i, blk, d) for i, (blk, d) in enumerate(zip(bloques, durs))]):
            anim_meta[i] = meta

    t1 = time.time()
    print(f"   ({t1 - t0:.2f}s)")

    encoder, encoder_flags = _encoder_flags(cfg)
    print(f"-> Encoder: {encoder}")
    print(f"-> Split izquierda/derecha: {w_left}px / {w_right}px")

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-loop", "1", "-framerate", str(fps), "-t", str(duracion_total), "-i", f"{workdir}/left.jpg",
    ]
    input_idx = 1
    block_inputs = []
    for i, meta in enumerate(anim_meta):
        if meta is None:
            cmd += ["-loop", "1", "-framerate", str(fps), "-t", str(durs[i]), "-i", f"{workdir}/b{i}_full.jpg"]
            block_inputs.append({"full": input_idx})
            input_idx += 1
        else:
            cmd += ["-framerate", str(fps), "-i", f"{workdir}/b{i}_anim_%04d.jpg"]
            idx_anim = input_idx; input_idx += 1
            cmd += ["-stream_loop", "-1", "-framerate", str(fps), "-t", str(meta["loop_dur"]), "-i", f"{workdir}/b{i}_loop_%04d.jpg"]
            idx_loop = input_idx; input_idx += 1
            block_inputs.append({"anim": idx_anim, "loop": idx_loop})

    audio_idx = input_idx
    cmd += ["-stream_loop", "-1", "-i", cfg["music_path"]]

    filter_parts = ["[0:v]format=yuv420p[left]"]
    for i, info in enumerate(block_inputs):
        if "full" in info:
            filter_parts.append(f"[{info['full']}:v]format=yuv420p,fade=t=in:st=0:d={texto_fade}[blk{i}]")
        else:
            filter_parts.append(f"[{info['anim']}:v]format=yuv420p,fps={fps}[a{i}]")
            filter_parts.append(f"[{info['loop']}:v]format=yuv420p,fps={fps}[s{i}]")
            filter_parts.append(f"[a{i}][s{i}]concat=n=2:v=1:a=0,fps={fps}[blk{i}]")

    filter_parts += [
        f"[blk0][blk1]xfade=transition={cut1}:duration={trans1}:offset={off1}[x1]",
        f"[x1][blk2]xfade=transition={cut2}:duration={trans2}:offset={off2}[x2]",
        f"[x2][blk3]xfade=transition={cut3}:duration={trans3}:offset={off3}[right]",
        f"[left][right]hstack=inputs=2[video]",
    ]
    filter_complex = ";".join(filter_parts)

    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[video]",
        "-map", f"{audio_idx}:a",
        "-t", str(duracion_total),
        "-c:v", encoder, *encoder_flags,
        "-c:a", "aac", "-b:a", "128k",
        "-threads", "4",
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
        shutil.rmtree(workdir, ignore_errors=True)
        raise RuntimeError("ffmpeg failed")

    print(f"   ({t2 - t_ff0:.2f}s)")
    print(f"Done in {t2 - t0:.2f}s total -> {cfg['output_path']}")
    shutil.rmtree(workdir, ignore_errors=True)
    return cfg["output_path"]


if __name__ == "__main__":
    generar_video(CONFIG)