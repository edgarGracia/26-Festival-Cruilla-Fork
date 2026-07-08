import math
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
CONFIG = {
    "polaroid_path":     "/home/spG07/code/Festival-Cruilla/final_video/img/image_tribe_poster_es.png",
    "fondo_derecha_path":"/home/spG07/code/Festival-Cruilla/final_video/img/fondo.png",
    "landmarks_path":    "/home/spG07/code/Festival-Cruilla/final_video/img/image_landmarks.png",
    "music_path":        "/home/spG07/code/Festival-Cruilla/final_video/img/8e02cfb3-9c6e-3654-d5fd-e5e6fc4b1298.wav",
    "casa_sticker_path": "/home/spG07/code/Festival-Cruilla/final_video/casas/Casa_Urban.png",
    "casa_nombre":       "Urban",
    "artistas": [
        {"name": "Lena Pulse", "confidence": 8,  "image": "/home/cvcadmin/cruilla/Festival-Cruilla/Fake_Artists/4/4_1.png"},
        {"name": "Artista 2",  "confidence": 6,  "image": "/home/cvcadmin/cruilla/Festival-Cruilla/Fake_Artists/2/2_1.png"},
        {"name": "Artista 3",  "confidence": 5,  "image": "/home/cvcadmin/cruilla/Festival-Cruilla/Fake_Artists/19/image (1).png"},
    ],
    "output_path": "/home/spG07/code/Festival-Cruilla/final_video/resultado_FERNANDO.mp4",

    "resolucion":          (1920, 1080),
    "fps":                 30,
    "duracion_bloque":     5,
    "duracion_transicion": 0.6,
    "usar_gpu":            True,
    "ffmpeg_preset":       "p1",
    "crf":                 23,
    "threads":             4,

    "texto_fade_dur": 0.30,
    "foto_delay":      0.45,
    "foto_fade_dur":   0.40,
}

BRAND_YELLOW = (255, 221, 35)
BRAND_INK    = (15, 15, 15)
BRAND_WHITE  = (255, 255, 255)

# ==============================================================================
# IMAGE UTILITIES & BRAND STYLING
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

def _cargar_fuente(size):
    candidates = [
        os.path.join(os.path.dirname(__file__), "fonts", "Anton.ttf"),
        "/usr/share/fonts/abattis-cantarell/Cantarell-Bold.otf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try: return ImageFont.truetype(path, size)
            except: continue
    return ImageFont.load_default()

def rounded_mask(size, radius):
    w, h = size
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
    return mask

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

def headline(canvas, y, kicker, title, w, kicker_color=BRAND_YELLOW, title_color=(255, 255, 255), kicker_size=45, title_size=74):
    d = ImageDraw.Draw(canvas)
    x = 60
    max_w = w - 120
    f_kicker = _cargar_fuente(kicker_size)
    f_title = _cargar_fuente(title_size)
    line_h = int(title_size * 1.12)
    ty = y
    if kicker:
        d.text((x, ty), kicker.upper(), font=f_kicker, fill=(*kicker_color, 255))
        ty = d.textbbox((x, ty), kicker.upper(), font=f_kicker)[3] + 4
    if title:
        words, lines, cur = title.upper().split(), [], ""
        for word in words:
            test = (cur + " " + word).strip()
            if cur and d.textbbox((0, 0), test, font=f_title)[2] > max_w:
                lines.append(cur)
                cur = word
            else: cur = test
        if cur: lines.append(cur)
        for line in lines:
            d.text((x, ty), line, font=f_title, fill=(*title_color, 255))
            ty += line_h
    return ty

def draw_centered_huge_text(canvas, text, base_font_size, text_color, custom_second_line_size=None):
    """Dibuja el texto centrado respetando los saltos de línea explícitos sin mezclar palabras."""
    d = ImageDraw.Draw(canvas)
    w, h = canvas.size
    lines = text.upper().split('\n')
    
    # Calcular alturas y cargar fuentes por línea
    line_fonts = []
    line_heights = []
    total_h = 0
    
    for idx, line in enumerate(lines):
        size = custom_second_line_size if (idx == 1 and custom_second_line_size) else base_font_size
        font = _cargar_fuente(size)
        line_fonts.append(font)
        lh = int(size * 1.3)
        line_heights.append(lh)
        total_h += lh
        
    y = (h - total_h) // 2
    for idx, line in enumerate(lines):
        font = line_fonts[idx]
        lw = d.textbbox((0, 0), line, font=font)[2]
        d.text(((w - lw) // 2, y), line, font=font, fill=(*text_color, 255))
        y += line_heights[idx]

# ==============================================================================
# CONSTRUCTORES DE CONTENIDO
# ==============================================================================
def _bg_dark(w, h, cfg):
    fondo_path = cfg.get("fondo_derecha_path")
    if fondo_path and os.path.exists(fondo_path):
        return cover_resize(Image.open(fondo_path), w, h).convert("RGBA")
    return Image.new("RGBA", (w, h), (*BRAND_INK, 255))

def build_landmarks(cfg, w, h, card_w, card_h):
    full = _bg_dark(w, h, cfg)
    end_y = headline(full, 60, "Així és com...", "La IA et llegeix", w)
    card = rotar(polaroid_card(cfg["landmarks_path"], card_w, card_h), 4)
    return {"full_text": full.convert("RGB"), "fotos": [(card, w // 2, h // 2 + 80)]}

def build_artistas(cfg, w, h):
    full = _bg_dark(w, h, cfg)
    artistas = cfg.get("artistas", [])
    ordenados = sorted([a for a in artistas if os.path.exists(a.get("image", ""))], key=lambda a: a.get("confidence", 0), reverse=True)
    
    if not ordenados:
        return {"full_text": full.convert("RGB"), "fotos": []}
        
    end_y = headline(full, 60, "Tal cara, tal beat t'ha identificat com...", ordenados[0]["name"], w)
    
    avail = h - end_y - 40
    main_h = int(avail * 0.48)
    main_w = int(main_h * 1.1875)
    circle_d = int(avail * 0.22)
    
    main_cy = end_y + int(main_h // 2) + 80
    fotos = [(polaroid_card(ImageOps.grayscale(Image.open(ordenados[0]["image"])).convert("RGB"), main_w, main_h), w // 2, main_cy)]
    
    secundarios = ordenados[1:3]
    if secundarios:
        caption_y = main_cy + main_h // 2 + 70
        d = ImageDraw.Draw(full)
        f_cap = _cargar_fuente(35)
        caption = "ALTRES ARTISTES IDENTIFICATS"
        cw = d.textbbox((0, 0), caption, font=f_cap)[2]
        d.text(((w - cw) // 2, caption_y), caption, font=f_cap, fill=(*BRAND_YELLOW, 255))
        
        circles_cy = caption_y + 70 + circle_d // 2
        c1 = circle_card(secundarios[0]["image"], circle_d) if len(secundarios) > 0 else None
        c2 = circle_card(secundarios[1]["image"], circle_d) if len(secundarios) > 1 else None
        
        if c1 and c2:
            offset = int(w * 0.28)
            comb = Image.new("RGBA", (c1.width + offset, max(c1.height, c2.height)), (0, 0, 0, 0))
            comb.alpha_composite(c1, (0, (comb.height - c1.height) // 2))
            comb.alpha_composite(c2, (offset, (comb.height - c2.height) // 2))
            fotos.append((comb, w // 2, circles_cy))
        elif c1:
            fotos.append((c1, w // 2, circles_cy))
            
    return {"full_text": full.convert("RGB"), "fotos": fotos}

"""
def build_casa(cfg, w, h, card_w, card_h):
    full = Image.new("RGBA", (w, h), (*BRAND_WHITE, 255))
    end_y = headline(full, 60, "Ets a casa...", "LA TEVA IDENTITAT", w, kicker_color=BRAND_INK, title_color=BRAND_INK)
    sticker = contain_resize(Image.open(cfg["casa_sticker_path"]), card_w, card_h)
    return {"full_text": full.convert("RGB"), "fotos": [(sticker, w // 2, h // 2 + 80)]}
"""
def build_casa(cfg, w, h, card_w, card_h):
    full = Image.new("RGBA", (w, h), (*BRAND_WHITE, 255))
    end_y = headline(full, 60, "Ets a casa...", "LA TEVA IDENTITAT", w, kicker_color=BRAND_INK, title_color=BRAND_INK)
    
    # 1. Cargar y redimensionar sticker original
    sticker_raw = contain_resize(Image.open(cfg["casa_sticker_path"]), card_w, card_h)
    sw, sh = sticker_raw.size
    
    # 2. Configuración de sombra optimizada
    shadow_blur = 15
    shadow_offset = 12
    
    # 3. Margen mínimo estricto para absorber el ZOOM_MAX (9%) + BOUNCE_AMP (9%) del efecto breath
    # Un 25% de margen extra es más que suficiente para evitar el recorte visual
    pad_x = int(sw * 0.25) + shadow_blur
    pad_y = int(sh * 0.25) + shadow_blur
    
    canvas_w = sw + pad_x * 2
    canvas_h = sh + pad_y * 2
    
    # Lienzo final optimizado
    sticker_canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    
    # 4. Pre-renderizar la sombra en un lienzo pequeño ajustado al sticker
    # Esto se ejecuta UNA sola vez aquí, no en el bucle de frames
    shadow_zone = Image.new("RGBA", (sw + shadow_blur * 2, sh + shadow_blur * 2), (0, 0, 0, 0))
    shadow_mask = sticker_raw.getchannel('A')
    
    # Dibujar la silueta de la sombra (opacidad 140 para compensar el radio menor)
    ImageDraw.Draw(shadow_zone).bitmap((shadow_blur, shadow_blur), shadow_mask, fill=(0, 0, 0, 140))
    shadow_blurred = shadow_zone.filter(ImageFilter.GaussianBlur(shadow_blur))
    
    # 5. Composición estática única en el centro del lienzo con padding
    # Coordenadas de pegado idénticas y fijas
    sx = pad_x - shadow_blur + shadow_offset
    sy = pad_y - shadow_blur + shadow_offset
    stx = pad_x
    sty = pad_y
    
    sticker_canvas.alpha_composite(shadow_blurred, (sx, sy))
    sticker_canvas.alpha_composite(sticker_raw, (stx, sty))
    
    return {"full_text": full.convert("RGB"), "fotos": [(sticker_canvas, w // 2, h // 2 + 80)]}
    
# ==============================================================================
# CORE ANIMATION ENGINE (EVALUATORS)
# ==============================================================================
ZOOM_MAX = 0.09
BOUNCE_AMP = 0.09
BOUNCE_DECAY = 9.0
BOUNCE_FREQ = 0.9
BREATHE_AMP = 0.010
BREATHE_FREQ = 0.17
ANIM_WINDOW_SEC = 1.3
BREATHE_LOOP_FRAMES = round(2 * math.pi / BREATHE_FREQ)

def _ease_base_at(n, frames):
    t = n / max(frames - 1, 1)
    return 1 + ZOOM_MAX * (1 - (1 - t) ** 3)

def _zoom_at(n, frames):
    t = n / max(frames - 1, 1)
    base = 1 + ZOOM_MAX * (1 - (1 - t) ** 3)
    bounce = BOUNCE_AMP * math.exp(-n / BOUNCE_DECAY) * abs(math.cos(n * BOUNCE_FREQ))
    breathe = BREATHE_AMP * math.sin(n * BREATHE_FREQ)
    return max(1.001, base + bounce + breathe)

def _zoom_at_frozen_base(base_frozen, n):
    bounce = BOUNCE_AMP * math.exp(-n / BOUNCE_DECAY) * abs(math.cos(n * BOUNCE_FREQ))
    breathe = BREATHE_AMP * math.sin(n * BREATHE_FREQ)
    return max(1.001, base_frozen + bounce + breathe)

def _zoomed_foto(foto, zoom):
    fw, fh = foto.size
    zw, zh = max(1, round(fw * zoom)), max(1, round(fh * zoom))
    resized = foto.resize((zw, zh), Image.BILINEAR)
    return resized.crop(((zw - fw) // 2, (zh - fh) // 2, (zw - fw) // 2 + fw, (zh - fh) // 2 + fh))

def _render_pane_frame(blk, w, h, n, frames_full, texto_fade_f, foto_delay_f, foto_fade_f, zoom_fn=None):
    if zoom_fn is None: zoom_fn = lambda n: _zoom_at(n, frames_full)
    bg_alpha = min(1.0, n / max(texto_fade_f, 1))
    
    if bg_alpha >= 1.0: frame = blk["full_text"].copy()
    else: frame = Image.blend(Image.new("RGB", (w, h), (0,0,0)), blk["full_text"], bg_alpha)
    frame = frame.convert("RGBA")
    
    for foto, cx, cy in blk["fotos"]:
        if n < foto_delay_f: continue
        foto_alpha = min(1.0, (n - foto_delay_f) / max(foto_fade_f, 1))
        piece = _zoomed_foto(foto, zoom_fn(n))
        if foto_alpha < 1.0:
            r, g, b, a = piece.split()
            a = a.point(lambda v: int(v * foto_alpha))
            piece = Image.merge("RGBA", (r, g, b, a))
        frame.alpha_composite(piece, (cx - piece.width // 2, cy - piece.height // 2))
    return frame.convert("RGB")

# ==============================================================================
# MAIN GENERATOR
# ==============================================================================
def generar_video(cfg=CONFIG):
    t0 = time.time()
    W, H = cfg["resolucion"]
    fps  = cfg["fps"]
    dur  = cfg["duracion_bloque"]
    
    # --- CÁLCULO ASIMÉTRICO DE LA POLAROID (PUNTO 1) ---
    # Cargamos imagen original para ajustar contención vertical perfecta sin recortar
    img_polaroid_orig = Image.open(cfg["polaroid_path"])
    orig_w, orig_h = img_polaroid_orig.size
    w_polaroid = int(H * (orig_w / orig_h)) # Ancho exacto proporcional a la pantalla completa
    w_casa = W - w_polaroid                 # Ancho restante dinámico para la casa
    
    # Para los bloques simétricos (Bloque 1)
    w_half = W // 2
    card_w = int(w_half * 0.62)
    card_h = int(card_w / 1.1875)
    
    workdir = tempfile.mkdtemp(prefix="festival_rebuilt_")
    print(f"-> Rendering frames concurrently (PIL)...")
    
    # --------------------------------------------------------------------------
    # PANTALLAS COMPLETAS ESTÁTICAS / FADES (Bloques Full-Screen 0 y 2) - PUNTOS 2 Y 3
    # --------------------------------------------------------------------------
    # Capa 1 (Bloque 0) con división exacta y tamaño grande en la segunda línea
    b0_base = Image.new("RGBA", (W, H), (*BRAND_INK, 255))
    draw_centered_huge_text(b0_base, "Cruïlla Presenta...\ncom et veu la IA", 58, BRAND_WHITE, custom_second_line_size=110)

    # Dos cruces blancas (esquinas superior-izq e inferior-der) sobre la Capa 1
    cross_size_b0 = int((min(W, H) * 0.14)+20)
    cross_w1 = cross_burst(cross_size_b0, color=BRAND_WHITE, angle=-15)
    cross_w2 = cross_burst(cross_size_b0, color=BRAND_WHITE, angle=15)
    b0_base.alpha_composite(cross_w1, (int((W * 0.04)+10), int(H * 0.06)))
    b0_base.alpha_composite(cross_w2, (W - cross_w2.width - int((W * 0.04)+20), H - cross_w2.height - int(H * 0.06)))

    # Guardamos únicamente los frames del fade-in inicial (el resto del bloque es
    # visualmente idéntico, así que se reutiliza un único frame estático vía ffmpeg
    # -loop, evitando renderizar y codificar cientos de JPEGs redundantes en PIL).
    texto_fade_f = cfg["texto_fade_dur"] * fps
    frames_b0 = int(dur * fps)
    fade_frames_b0 = max(1, min(int(math.ceil(texto_fade_f)) + 1, frames_b0))
    for n in range(fade_frames_b0):
        alpha = min(1.0, n / max(texto_fade_f, 1))
        if alpha >= 1.0:
            frame_b0 = b0_base.convert("RGB")
        else:
            frame_b0 = Image.blend(Image.new("RGB", (W, H), (0, 0, 0)), b0_base.convert("RGB"), alpha)
        frame_b0.save(f"{workdir}/b0_frame_{n:04d}.jpg", quality=90)
    b0_base.convert("RGB").save(f"{workdir}/b0_static.jpg", quality=90)
    b0_static_dur = max(dur - (fade_frames_b0 / fps), 1.0 / fps)
    
    # Capa 3 (Bloque 2) con dos líneas
    b2_static = Image.new("RGBA", (W, H), (*BRAND_YELLOW, 255))
    draw_centered_huge_text(b2_static, "Però la melodia que sona\nrepresenta la teva identitat", 90, BRAND_INK)

    # Dos cruces negras (esquinas superior-der e inferior-izq) sobre la Capa 3
    cross_size_b2 = int((min(W, H) * 0.14)+30)
    cross_b1 = cross_burst(cross_size_b2, color=BRAND_INK, angle=-15)
    cross_b2 = cross_burst(cross_size_b2, color=BRAND_INK, angle=20)
    b2_static.alpha_composite(cross_b1, (W - cross_b1.width - int((W * 0.04)-30), int(H * 0.06)))
    b2_static.alpha_composite(cross_b2, (int((W * 0.04)+20), H - cross_b2.height - int(H * 0.06)))

    b2_static.convert("RGB").save(f"{workdir}/b2_full.jpg", quality=90)

    # --------------------------------------------------------------------------
    # PREPARACIÓN DE LAS TRANSICIONES ANIMADAS (Bloques Split 1 y 3)
    # --------------------------------------------------------------------------
    pane_landmarks = build_landmarks(cfg, w_half, H, card_w, card_h)
    pane_artistas  = build_artistas(cfg, w_half, H)
    
    # El Bloque 3 ahora usa el ancho asimétrico óptimo de la Polaroid sin recortes
    pane_casa      = build_casa(cfg, w_casa, H, int(w_casa * 0.65), int(w_casa * 0.65 / 1.1875))
    pane_polaroid_static = contain_resize(img_polaroid_orig, w_polaroid, H)
    if pane_polaroid_static.size != (w_polaroid, H):
        # thumbnail() only guarantees fitting *within* the box, rounding can leave
        # it 1px off in either dimension, which crashes Image.blend() downstream.
        pane_polaroid_static = pane_polaroid_static.resize((w_polaroid, H), Image.LANCZOS)
    pane_polaroid = {"full_text": pane_polaroid_static.convert("RGB"), "fotos": []}

    foto_delay_f = cfg["foto_delay"] * fps
    foto_fade_f  = cfg["foto_fade_dur"] * fps
    frames_full  = max(int((dur + cfg["duracion_transicion"]) * fps), 1)
    
    n_anim = max(1, min(round(ANIM_WINDOW_SEC * fps), frames_full - 1))
    base_frozen = _ease_base_at(n_anim, frames_full)
    n_loop = max(1, min(BREATHE_LOOP_FRAMES, frames_full - n_anim))
    
    loop_dur = (dur + cfg["duracion_transicion"]) - (n_anim / fps)

    # Renderizador dinámico adaptado a anchos independientes por panel
    def _render_pane_sequence(args):
        prefix, blk, current_w = args
        for n in range(n_anim):
            _render_pane_frame(blk, current_w, H, n, frames_full, texto_fade_f, foto_delay_f, foto_fade_f)\
                .save(f"{workdir}/{prefix}_anim_{n:04d}.jpg", quality=90)
                
        loop_zoom_fn = lambda n: _zoom_at_frozen_base(base_frozen, n)
        for k in range(n_loop):
            _render_pane_frame(blk, current_w, H, n_anim + k, frames_full, texto_fade_f, foto_delay_f, foto_fade_f, zoom_fn=loop_zoom_fn)\
                .save(f"{workdir}/{prefix}_loop_{k:04d}.jpg", quality=90)

    # Inyección de anchos calculados a la piscina paralela
    tasks = [
        ("landmarks", pane_landmarks, w_half),
        ("artists", pane_artistas, w_half),
        ("casa", pane_casa, w_casa),
        ("polaroid", pane_polaroid, w_polaroid)
    ]
    with ThreadPoolExecutor(max_workers=4) as executor:
        # list() forces iteration so exceptions raised inside worker threads
        # propagate here instead of being silently dropped by the unread map().
        list(executor.map(_render_pane_sequence, tasks))

    t1 = time.time()
    print(f"   PIL Render Completed in ({t1 - t0:.2f}s)")

    # --------------------------------------------------------------------------
    # CONFIGURACIÓN PIPELINE FFMPEG ENCODER
    # --------------------------------------------------------------------------
    encoder = "h264_nvenc" if cfg.get("usar_gpu", True) else "libx264"
    encoder_flags = ["-preset", "p1", "-tune", "ll", "-cq", "24", "-pix_fmt", "yuv420p"] if cfg.get("usar_gpu", True) else ["-preset", "ultrafast", "-crf", str(cfg["crf"]), "-pix_fmt", "yuv420p"]
    
    trans = cfg["duracion_transicion"]
    d_full = dur + trans
    
    off1 = dur - trans
    off2 = dur + d_full - trans * 2
    off3 = dur + d_full * 2 - trans * 3
    duracion_total = dur + d_full * 3 - trans * 3

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        # Secuencia de imágenes del Bloque 0: solo los frames del fade-in (con su fade-in
        # ya pre-renderizado en PIL) + un frame estático repetido por ffmpeg (-loop) para
        # el resto de la duración, en vez de cientos de JPEGs idénticos.
        "-framerate", str(fps), "-i", f"{workdir}/b0_frame_%04d.jpg",
        "-loop", "1", "-framerate", str(fps), "-t", str(b0_static_dur), "-i", f"{workdir}/b0_static.jpg",
        # Resto de entradas estáticas y bucles dinámicos
        "-loop", "1", "-framerate", str(fps), "-t", str(d_full), "-i", f"{workdir}/b2_full.jpg",
        "-framerate", str(fps), "-i", f"{workdir}/landmarks_anim_%04d.jpg",
        "-stream_loop", "-1", "-framerate", str(fps), "-t", str(loop_dur), "-i", f"{workdir}/landmarks_loop_%04d.jpg",
        "-framerate", str(fps), "-i", f"{workdir}/artists_anim_%04d.jpg",
        "-stream_loop", "-1", "-framerate", str(fps), "-t", str(loop_dur), "-i", f"{workdir}/artists_loop_%04d.jpg",
        "-framerate", str(fps), "-i", f"{workdir}/polaroid_anim_%04d.jpg",
        "-stream_loop", "-1", "-framerate", str(fps), "-t", str(loop_dur), "-i", f"{workdir}/polaroid_loop_%04d.jpg",
        "-framerate", str(fps), "-i", f"{workdir}/casa_anim_%04d.jpg",
        "-stream_loop", "-1", "-framerate", str(fps), "-t", str(loop_dur), "-i", f"{workdir}/casa_loop_%04d.jpg",
        "-stream_loop", "-1", "-i", cfg["music_path"]
    ]

    filter_parts = [
        # Ensamblaje Bloque 0 (fade-in pre-renderizado + frame estático repetido por ffmpeg)
        f"[0:v]format=yuv420p[b0_anim];[1:v]format=yuv420p[b0_loop];[b0_anim][b0_loop]concat=n=2:v=1:a=0,fps={fps}[blk0]",

        # Ensamblaje Bloque 1 (Split Simétrico 50/50)
        f"[3:v]format=yuv420p[l_anim];[4:v]format=yuv420p[l_loop];[l_anim][l_loop]concat=n=2:v=1:a=0[l_full]",
        f"[5:v]format=yuv420p[r_anim];[6:v]format=yuv420p[r_loop];[r_anim][r_loop]concat=n=2:v=1:a=0[r_full]",
        f"[l_full][r_full]hstack=inputs=2,fps={fps}[blk1]",
        
        # Ensamblaje Bloque 3 (Split Asimétrico Adaptativo)
        f"[7:v]format=yuv420p[p_anim];[8:v]format=yuv420p[p_loop];[p_anim][p_loop]concat=n=2:v=1:a=0[p_full]",
        f"[9:v]format=yuv420p[c_anim];[10:v]format=yuv420p[c_loop];[c_anim][c_loop]concat=n=2:v=1:a=0[c_full]",
        f"[p_full][c_full]hstack=inputs=2,fps={fps}[blk3]",
        
        # Formatear Bloque Estático/Pre-renderizado de fondo (Capa 3)
        f"[2:v]format=yuv420p,fps={fps}[blk2]",
        
        # Cadena de transiciones encadenadas xfade nativas en C
        f"[blk0][blk1]xfade=transition=fade:duration={trans}:offset={off1}[x1]",
        f"[x1][blk2]xfade=transition=fade:duration={trans}:offset={off2}[x2]",
        f"[x2][blk3]xfade=transition=fade:duration={trans}:offset={off3}[video]"
    ]
    
    cmd += [
        "-filter_complex", ";".join(filter_parts),
        "-map", "[video]",
        "-map", "11:a",
        "-t", str(duracion_total),
        "-c:v", encoder, *encoder_flags,
        "-c:a", "aac", "-b:a", "128k",
        "-threads", str(cfg["threads"]),
        "-movflags", "+faststart",
        cfg["output_path"]
    ]

    os.makedirs(os.path.dirname(cfg["output_path"]), exist_ok=True)
    print("-> Mixing & Encoding final video via FFmpeg...")
    
    t_ff0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    t2 = time.time()

    if result.returncode != 0:
        print("FFMPEG CRITICAL ERROR:")
        print(result.stderr[-3000:])
        shutil.rmtree(workdir, ignore_errors=True)
        raise RuntimeError("ffmpeg execution failed")

    print(f"   FFmpeg completed in ({t2 - t_ff0:.2f}s)")
    print(f"✨ Rebuilt Done successfully in {t2 - t0:.2f}s total -> {cfg['output_path']}")
    shutil.rmtree(workdir, ignore_errors=True)
    return cfg["output_path"]

if __name__ == "__main__":
    generar_video(CONFIG)