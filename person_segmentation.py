from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

YOLO_WEIGHTS = "yolov8n-seg.pt"
DEBUG_OUTPUT_PATH = Path(__file__).parent / "segmented_image.png"

_yolo_model = None


def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(YOLO_WEIGHTS)
    return _yolo_model


def _center_person_mask(np_image: np.ndarray) -> np.ndarray | None:
    """Mask (uint8, image size) of the person closest to the image centre.

    `np_image` is RGB (as produced by PIL). Ultralytics treats raw ndarray
    inputs as already BGR (OpenCV convention) and unconditionally flips them
    to RGB internally, so an RGB array must be pre-flipped to BGR here —
    otherwise the model receives color-inverted input, hurting detection.
    """
    results = _get_yolo()(np_image[:, :, ::-1], verbose=False)

    for r in results:
        if r.masks is None or len(r.masks.data) == 0:
            continue

        classes = r.boxes.cls.cpu().numpy().astype(int)
        person_idx = np.where(classes == 0)[0]  # COCO class 0 = person
        if len(person_idx) == 0:
            continue

        image_center = np.array([np_image.shape[1] / 2, np_image.shape[0] / 2])
        box_centers = r.boxes.xywh[:, :2].cpu().numpy()
        distances = np.linalg.norm(box_centers[person_idx] - image_center, axis=1)
        best_idx = person_idx[np.argmin(distances)]

        mask = (r.masks.data[best_idx].cpu().numpy() * 255).astype(np.uint8)
        mask = Image.fromarray(mask).resize(
            (np_image.shape[1], np_image.shape[0]), resample=Image.BILINEAR
        )
        return np.array(mask)

    return None


def remove_background_center_person(image: Image.Image) -> Image.Image:
    """rembg cutout (RGBA) restricted to the person closest to the centre.

    If no person is detected, returns the plain rembg cutout as before.
    """
    from rembg import remove

    subject = remove(image)
    if subject.mode != "RGBA":
        subject = subject.convert("RGBA")

    np_image = np.array(image.convert("RGB"))
    person_mask = _center_person_mask(np_image)
    if person_mask is None:
        print("[background] No person detected; keeping full rembg cutout")
        subject.save(DEBUG_OUTPUT_PATH)
        return subject

    # The YOLO mask is coarse (predicted at 640px), so grow it before using it
    # as a gate, letting rembg keep fine edges (hair, fingers) of the chosen
    # person while everything outside it becomes transparent.
    grow_px = max(5, int(0.02 * max(image.size)))
    grown = Image.fromarray(person_mask).filter(ImageFilter.GaussianBlur(grow_px))
    keep = np.array(grown) > 16

    r, g, b, a = subject.split()
    alpha = np.array(a, dtype=np.uint8)
    alpha[~keep] = 0
    result = Image.merge("RGBA", (r, g, b, Image.fromarray(alpha)))
    result.save(DEBUG_OUTPUT_PATH)
    return result
