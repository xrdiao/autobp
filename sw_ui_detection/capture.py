from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageGrab

from sw_ui_detection.detector import BoundingBox, _clip_box


@dataclass(frozen=True)
class CanvasLocation:
    bbox: BoundingBox
    confidence: float
    used_fallback: bool = False


def capture_screen_image() -> Image.Image:
    """Capture the visible desktop through the system screenshot API.

    This intentionally does not inspect processes, window handles, memory, or
    graphics hooks. The caller must trigger the capture explicitly.
    """
    try:
        return ImageGrab.grab(all_screens=True)
    except (TypeError, OSError):
        return ImageGrab.grab()


def save_screen_capture(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = capture_screen_image()
    if image.mode not in {"RGB", "RGBA"}:
        image = image.convert("RGB")
    image.save(output_path)
    return output_path


def locate_game_canvas(image: np.ndarray) -> CanvasLocation:
    """Find the large warm-toned game canvas inside a full-screen capture."""
    if image is None or image.size == 0:
        raise ValueError("image is empty")

    height, width = image.shape[:2]
    full_box = BoundingBox(x=0, y=0, w=width, h=height)
    if width < 320 or height < 240:
        return CanvasLocation(full_box, confidence=0.0, used_fallback=True)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    warm_mask = (
        (hue <= 42)
        & (saturation >= 32)
        & (value >= 18)
    ).astype(np.uint8) * 255
    warm_mask = cv2.morphologyEx(
        warm_mask,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), dtype=np.uint8),
        iterations=2,
    )
    warm_mask = cv2.morphologyEx(
        warm_mask,
        cv2.MORPH_OPEN,
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    )

    components, _, stats, _ = cv2.connectedComponentsWithStats(warm_mask, connectivity=8)
    image_area = float(width * height)
    candidates: list[tuple[float, BoundingBox, float]] = []
    for component_id in range(1, components):
        x, y, w, h, area = [int(value) for value in stats[component_id]]
        if w < width * 0.55 or h < height * 0.50:
            continue
        area_ratio = area / image_area
        if area_ratio < 0.25:
            continue
        aspect = w / max(1.0, float(h))
        if aspect < 1.30 or aspect > 2.35:
            continue
        aspect_score = max(0.0, 1.0 - abs(aspect - 1.72) / 0.95)
        area_score = min(1.0, area_ratio / 0.55)
        score = 0.65 * area_score + 0.35 * aspect_score
        candidates.append((score, BoundingBox(x=x, y=y, w=w, h=h), area_ratio))

    if not candidates:
        return CanvasLocation(full_box, confidence=0.0, used_fallback=True)

    candidates.sort(key=lambda item: item[0], reverse=True)
    score, candidate, area_ratio = candidates[0]
    candidate = _trim_title_bar(image, candidate, hsv)
    if candidate.x <= 2 and candidate.y <= 2 and candidate.w >= width * 0.98 and candidate.h >= height * 0.98:
        return CanvasLocation(full_box, confidence=float(score), used_fallback=True)

    pad_x = max(0, int(round(candidate.w * 0.006)))
    pad_y = 0
    expanded = _clip_box(
        candidate.x - pad_x,
        candidate.y - pad_y,
        candidate.w + pad_x * 2,
        candidate.h + pad_y * 2,
        image.shape,
    )
    confidence = float(min(1.0, score * (0.75 + 0.25 * min(1.0, area_ratio / 0.55))))
    return CanvasLocation(expanded, confidence=confidence, used_fallback=False)


def crop_game_canvas(image: np.ndarray) -> tuple[np.ndarray, CanvasLocation]:
    location = locate_game_canvas(image)
    box = location.bbox
    return image[box.y : box.y2, box.x : box.x2].copy(), location


def _trim_title_bar(image: np.ndarray, bbox: BoundingBox, hsv: np.ndarray) -> BoundingBox:
    """Remove a bright desktop/window title bar above the game canvas."""
    if bbox.y >= image.shape[0] - 1:
        return bbox

    x1 = max(0, bbox.x)
    x2 = min(image.shape[1], bbox.x2)
    if x2 <= x1:
        return bbox

    saturation = hsv[..., 1]
    value = hsv[..., 2]
    hue = hsv[..., 0]
    warm = (hue <= 42) & (saturation >= 32) & (value >= 18)
    max_scan = min(bbox.y + max(24, int(round(bbox.h * 0.12))), bbox.y2)
    for y in range(bbox.y, max_scan):
        end = min(max_scan, y + 4)
        row_saturation = float(saturation[y:end, x1:x2].mean())
        row_warm_ratio = float(warm[y:end, x1:x2].mean())
        if row_saturation >= 38.0 and row_warm_ratio >= 0.35:
            if y - bbox.y >= 6:
                return BoundingBox(bbox.x, y, bbox.w, bbox.y2 - y)
            break
    return bbox
