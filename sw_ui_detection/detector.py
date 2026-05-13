from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from sw_ui_detection.monster_bank import MonsterTemplateBank


@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SlotDetection:
    bbox: BoundingBox
    contour_score: float
    row_index: int = -1
    monster_name: str | None = None
    monster_score: float = 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["bbox"] = self.bbox.to_dict()
        return data


@dataclass
class UIDetectionResult:
    image_width: int
    image_height: int
    is_probable_sw_ui: bool
    confidence: float
    ui_kind: str
    screen_bbox: BoundingBox | None
    candidate_slots: list[SlotDetection]
    validated_slots: list[SlotDetection]
    row_groups: list[list[int]]

    def to_dict(self) -> dict:
        return {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "is_probable_sw_ui": self.is_probable_sw_ui,
            "confidence": self.confidence,
            "ui_kind": self.ui_kind,
            "screen_bbox": None if self.screen_bbox is None else self.screen_bbox.to_dict(),
            "candidate_slots": [slot.to_dict() for slot in self.candidate_slots],
            "validated_slots": [slot.to_dict() for slot in self.validated_slots],
            "row_groups": self.row_groups,
        }


def _clip_box(x: int, y: int, w: int, h: int, image_shape: tuple[int, int, int] | tuple[int, int]) -> BoundingBox:
    image_h, image_w = image_shape[:2]
    x = max(0, min(int(x), image_w - 1))
    y = max(0, min(int(y), image_h - 1))
    max_w = max(1, image_w - x)
    max_h = max(1, image_h - y)
    return BoundingBox(x=x, y=y, w=max(1, min(int(w), max_w)), h=max(1, min(int(h), max_h)))


def _iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    inter_x1 = max(box_a.x, box_b.x)
    inter_y1 = max(box_a.y, box_b.y)
    inter_x2 = min(box_a.x2, box_b.x2)
    inter_y2 = min(box_a.y2, box_b.y2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union = box_a.area + box_b.area - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _group_rows(slots: list[SlotDetection]) -> list[list[int]]:
    if not slots:
        return []

    side_lengths = np.array([slot.bbox.w for slot in slots], dtype=np.float32)
    tolerance = max(18.0, float(np.median(side_lengths)) * 0.75)

    row_centers: list[float] = []
    row_groups: list[list[int]] = []

    for idx, slot in enumerate(sorted(range(len(slots)), key=lambda i: (slots[i].bbox.y, slots[i].bbox.x))):
        y_center = slots[slot].bbox.y + slots[slot].bbox.h * 0.5
        best_row = -1
        best_delta = float("inf")
        for row_index, center in enumerate(row_centers):
            delta = abs(y_center - center)
            if delta < tolerance and delta < best_delta:
                best_row = row_index
                best_delta = delta

        if best_row < 0:
            row_centers.append(y_center)
            row_groups.append([slot])
        else:
            row_groups[best_row].append(slot)
            row_centers[best_row] = float(
                np.mean([slots[item].bbox.y + slots[item].bbox.h * 0.5 for item in row_groups[best_row]])
            )

    for members in row_groups:
        members.sort(key=lambda i: slots[i].bbox.x)

    row_groups.sort(key=lambda members: min(slots[item].bbox.y for item in members))
    for row_index, members in enumerate(row_groups):
        for member in members:
            slots[member].row_index = row_index
    return row_groups


class SummonersWarUIDetector:
    def __init__(
        self,
        template_bank: MonsterTemplateBank | None = None,
        min_slot_size_ratio: float = 0.03,
        max_slot_size_ratio: float = 0.18,
        validation_score_threshold: float = 0.48,
        max_candidates: int = 32,
        detect_max_side: int = 1600,
    ):
        self.template_bank = template_bank
        self.min_slot_size_ratio = float(min_slot_size_ratio)
        self.max_slot_size_ratio = float(max_slot_size_ratio)
        self.validation_score_threshold = float(validation_score_threshold)
        self.max_candidates = int(max_candidates)
        self.detect_max_side = int(detect_max_side)

    def detect(self, image_or_path: str | Path | np.ndarray, validate_monsters: bool = True) -> UIDetectionResult:
        original = self._load_image(image_or_path)
        detect_image, scale = self._resize_for_detection(original)
        candidates = self._detect_slot_candidates(detect_image)

        slots = [
            SlotDetection(
                bbox=BoundingBox(
                    x=int(round(candidate["bbox"].x / scale)),
                    y=int(round(candidate["bbox"].y / scale)),
                    w=int(round(candidate["bbox"].w / scale)),
                    h=int(round(candidate["bbox"].h / scale)),
                ),
                contour_score=float(candidate["score"]),
            )
            for candidate in candidates
        ]

        row_groups = _group_rows(slots)
        validated_slots = self._validate_slots(original, slots) if validate_monsters else []
        confidence = self._score_result(slots, row_groups, validated_slots)
        screen_bbox = self._compute_screen_bbox(slots, original.shape)

        is_probable = bool(
            confidence >= 0.45 and len(slots) >= 4 and any(len(row) >= 2 for row in row_groups)
        ) or len(validated_slots) >= 2

        ui_kind = "draft_like" if is_probable else "unknown"
        return UIDetectionResult(
            image_width=int(original.shape[1]),
            image_height=int(original.shape[0]),
            is_probable_sw_ui=is_probable,
            confidence=float(round(confidence, 4)),
            ui_kind=ui_kind,
            screen_bbox=screen_bbox,
            candidate_slots=slots,
            validated_slots=validated_slots,
            row_groups=row_groups,
        )

    def annotate(self, image_or_path: str | Path | np.ndarray, result: UIDetectionResult) -> np.ndarray:
        image = self._load_image(image_or_path).copy()
        validated_ids = {id(slot) for slot in result.validated_slots}

        for index, slot in enumerate(result.candidate_slots):
            color = (0, 200, 0) if id(slot) in validated_ids else (0, 180, 255)
            cv2.rectangle(
                image,
                (slot.bbox.x, slot.bbox.y),
                (slot.bbox.x2, slot.bbox.y2),
                color,
                2,
            )
            label = f"{index + 1} r{slot.row_index}"
            if slot.monster_name:
                label = f"{label} {slot.monster_name} {slot.monster_score:.2f}"
            cv2.putText(
                image,
                label,
                (slot.bbox.x, max(18, slot.bbox.y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        if result.screen_bbox is not None:
            box = result.screen_bbox
            cv2.rectangle(image, (box.x, box.y), (box.x2, box.y2), (255, 0, 0), 2)

        header = f"sw_ui={result.is_probable_sw_ui} conf={result.confidence:.3f} slots={len(result.candidate_slots)}"
        cv2.putText(image, header, (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return image

    def _load_image(self, image_or_path: str | Path | np.ndarray) -> np.ndarray:
        if isinstance(image_or_path, np.ndarray):
            image = image_or_path.copy()
        else:
            image = cv2.imread(str(image_or_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError("failed to load image")
        return image

    def _resize_for_detection(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = image.shape[:2]
        max_side = max(height, width)
        if max_side <= self.detect_max_side:
            return image, 1.0
        scale = self.detect_max_side / float(max_side)
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return resized, scale

    def _detect_slot_candidates(self, image: np.ndarray) -> list[dict]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 140)
        morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=2)

        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_dim = float(min(image.shape[:2]))
        min_side = max(28.0, min_dim * self.min_slot_size_ratio)
        max_side = min_dim * self.max_slot_size_ratio

        raw_candidates: list[dict] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < min_side or h < min_side or w > max_side or h > max_side:
                continue

            aspect = w / float(h)
            if aspect < 0.78 or aspect > 1.22:
                continue

            rect_area = float(w * h)
            contour_area = float(cv2.contourArea(contour))
            if rect_area <= 0:
                continue
            fill_ratio = contour_area / rect_area
            if fill_ratio < 0.35:
                continue

            roi_edges = morph[y : y + h, x : x + w]
            if roi_edges.size == 0:
                continue
            edge_density = float(np.count_nonzero(roi_edges)) / float(roi_edges.size)
            if edge_density < 0.05:
                continue

            side_consistency = 1.0 - min(1.0, abs(w - h) / max(w, h))
            fill_score = min(1.0, fill_ratio / 0.75)
            density_score = min(1.0, edge_density / 0.18)
            score = 0.45 * side_consistency + 0.25 * fill_score + 0.30 * density_score

            raw_candidates.append(
                {
                    "bbox": BoundingBox(x=x, y=y, w=w, h=h),
                    "score": float(score),
                }
            )

        raw_candidates.sort(key=lambda item: item["score"], reverse=True)
        selected: list[dict] = []
        for candidate in raw_candidates:
            if any(_iou(candidate["bbox"], existing["bbox"]) > 0.45 for existing in selected):
                continue
            selected.append(candidate)
            if len(selected) >= self.max_candidates:
                break
        return selected

    def _validate_slots(self, image: np.ndarray, slots: list[SlotDetection]) -> list[SlotDetection]:
        if not slots:
            return []
        if self.template_bank is None:
            self.template_bank = MonsterTemplateBank()

        validated = []
        for slot in slots:
            inset = max(2, int(round(min(slot.bbox.w, slot.bbox.h) * 0.06)))
            crop_box = _clip_box(
                slot.bbox.x + inset,
                slot.bbox.y + inset,
                slot.bbox.w - inset * 2,
                slot.bbox.h - inset * 2,
                image.shape,
            )
            crop = image[crop_box.y : crop_box.y2, crop_box.x : crop_box.x2]
            matches = self.template_bank.match_crop(crop, top_k=1)
            if not matches:
                continue
            best_match = matches[0]
            slot.monster_name = best_match.monster
            slot.monster_score = float(best_match.score)
            if best_match.score >= self.validation_score_threshold:
                validated.append(slot)
        return validated

    def _score_result(
        self,
        slots: list[SlotDetection],
        row_groups: list[list[int]],
        validated_slots: Iterable[SlotDetection],
    ) -> float:
        if not slots:
            return 0.0

        slot_count_score = min(1.0, len(slots) / 8.0)
        row_score = min(1.0, sum(1 for row in row_groups if len(row) >= 2) / 2.0)

        sides = np.array([slot.bbox.w for slot in slots], dtype=np.float32)
        mean_side = float(np.mean(sides))
        std_side = float(np.std(sides))
        consistency_score = 0.0 if mean_side <= 1e-6 else max(0.0, 1.0 - std_side / mean_side)

        validated_slots = list(validated_slots)
        validation_score = min(1.0, len(validated_slots) / max(3.0, len(slots) * 0.5))
        if validated_slots:
            avg_match = float(np.mean([slot.monster_score for slot in validated_slots]))
            validation_score = 0.6 * validation_score + 0.4 * max(0.0, min(1.0, avg_match))

        confidence = (
            0.30 * slot_count_score
            + 0.20 * row_score
            + 0.15 * consistency_score
            + 0.35 * validation_score
        )
        return max(0.0, min(1.0, confidence))

    def _compute_screen_bbox(self, slots: list[SlotDetection], image_shape: tuple[int, int, int]) -> BoundingBox | None:
        if not slots:
            return None

        min_x = min(slot.bbox.x for slot in slots)
        min_y = min(slot.bbox.y for slot in slots)
        max_x = max(slot.bbox.x2 for slot in slots)
        max_y = max(slot.bbox.y2 for slot in slots)
        margin = int(round(np.median([slot.bbox.w for slot in slots]) * 0.8))
        return _clip_box(min_x - margin, min_y - margin, (max_x - min_x) + margin * 2, (max_y - min_y) + margin * 2, image_shape)
