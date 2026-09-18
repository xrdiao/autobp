from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from bp_core.transformer_common import PROJECT_ROOT

DEFAULT_MONSTER_IMAGE_MAP = PROJECT_ROOT / "json" / "monster_images.json"
DEFAULT_MONSTER_CATALOG = PROJECT_ROOT / "json" / "lucksack_monster_catalog.json"


@dataclass
class MonsterMatch:
    monster: str
    score: float
    feature_score: float = 0.0
    template_score: float = 0.0
    color_score: float = 0.0
    edge_score: float = 0.0
    good_matches: int = 0
    inlier_matches: int = 0


def _center_crop_square(image: np.ndarray, margin_ratio: float = 0.08) -> np.ndarray:
    h, w = image.shape[:2]
    side = min(h, w)
    y = (h - side) // 2
    x = (w - side) // 2
    cropped = image[y : y + side, x : x + side]

    if margin_ratio <= 0:
        return cropped

    margin = int(side * margin_ratio)
    if margin * 2 >= side:
        return cropped
    return cropped[margin : side - margin, margin : side - margin]


def _prepare_gray(image: np.ndarray, size: int = 64, is_query: bool = False) -> np.ndarray:
    cropped = _crop_art_region(image, is_query=is_query)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) if cropped.ndim == 3 else cropped.copy()
    gray = _center_crop_square(gray, margin_ratio=0.04)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)
    return gray


def _crop_art_region(image: np.ndarray, is_query: bool) -> np.ndarray:
    h, w = image.shape[:2]
    if h <= 4 or w <= 4:
        return image

    # Keep the crop focused on the portrait art and consistently exclude
    # most star / level overlays from both query slots and template icons.
    y1 = int(round(h * 0.17))
    y2 = int(round(h * (0.79 if is_query else 0.81)))
    x1 = int(round(w * 0.10))
    x2 = int(round(w * 0.90))

    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    return image[y1:y2, x1:x2]


def _prepare_feature_gray(image: np.ndarray, is_query: bool, size: int = 96) -> np.ndarray:
    cropped = _crop_art_region(image, is_query=is_query)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) if cropped.ndim == 3 else cropped.copy()
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(gray)
    return gray


def _prepare_color_hist(image: np.ndarray, is_query: bool) -> np.ndarray:
    cropped = _crop_art_region(image, is_query=is_query)
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def _prepare_edge_map(image: np.ndarray, is_query: bool, size: int = 64) -> np.ndarray:
    gray = _prepare_feature_gray(image, is_query=is_query, size=size)
    edges = cv2.Canny(gray, 50, 140)
    edges = cv2.GaussianBlur(edges, (3, 3), 0)
    return edges.astype(np.float32) / 255.0


def _create_feature_extractor():
    return "orb", cv2.ORB_create(nfeatures=600, fastThreshold=7), cv2.NORM_HAMMING


class MonsterTemplateBank:
    def __init__(
        self,
        image_map_path: str | Path = DEFAULT_MONSTER_IMAGE_MAP,
        catalog_path: str | Path = DEFAULT_MONSTER_CATALOG,
        template_size: int = 64,
        feature_prefilter_limit: int = 64,
    ):
        self.image_map_path = Path(image_map_path)
        self.catalog_path = Path(catalog_path)
        self.template_size = int(template_size)
        self.feature_prefilter_limit = max(1, int(feature_prefilter_limit))
        self.feature_name, self.feature_extractor, matcher_norm = _create_feature_extractor()
        self.matcher = cv2.BFMatcher(matcher_norm)
        self.refine_feature_extractor = cv2.SIFT_create(nfeatures=240) if hasattr(cv2, "SIFT_create") else None
        self.refine_matcher = cv2.BFMatcher(cv2.NORM_L2) if self.refine_feature_extractor is not None else None
        self.monster_image_map: dict[str, Path] = {}
        self.templates = self._load_templates()

    def _load_templates(self) -> list[dict]:
        with open(self.image_map_path, "r", encoding="utf-8") as f:
            image_map = json.load(f)

        image_entries = list((str(monster), str(relative_path)) for monster, relative_path in image_map.items())
        if self.catalog_path.exists():
            with open(self.catalog_path, "r", encoding="utf-8") as f:
                catalog = json.load(f)
            known_names = {monster for monster, _path in image_entries}
            for row in catalog:
                if not isinstance(row, dict):
                    continue
                monster = str(row.get("slabel") or row.get("label") or row.get("slug") or "").strip()
                relative_path = str(row.get("image") or "").strip()
                if monster and relative_path and monster not in known_names:
                    image_entries.append((monster, relative_path))
                    known_names.add(monster)

        templates = []
        loaded_paths: set[Path] = set()
        for monster, relative_path in image_entries:
            image_path = PROJECT_ROOT / relative_path
            if not image_path.exists():
                image_path = PROJECT_ROOT / "images" / relative_path
            if not image_path.exists():
                continue
            image_path = image_path.resolve()
            if image_path in loaded_paths:
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            loaded_paths.add(image_path)
            self.monster_image_map[str(monster)] = image_path
            template_gray = _prepare_gray(image, size=self.template_size, is_query=False)
            feature_gray = _prepare_feature_gray(image, is_query=False, size=96)
            keypoints, descriptors = self.feature_extractor.detectAndCompute(feature_gray, None)
            templates.append(
                {
                    "monster": str(monster),
                    "gray": template_gray,
                    "gray_norm": cv2.normalize(template_gray.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX),
                    "edge": _prepare_edge_map(image, is_query=False, size=self.template_size),
                    "hist": _prepare_color_hist(image, is_query=False),
                    "keypoints": keypoints or [],
                    "descriptors": descriptors,
                    "keypoint_count": 0 if keypoints is None else len(keypoints),
                    "feature_gray": feature_gray,
                    "refine_keypoints": None,
                    "refine_descriptors": None,
                }
            )
        return templates

    def __len__(self) -> int:
        return len(self.templates)

    def get_monster_image_path(self, monster: str) -> Path | None:
        return self.monster_image_map.get(str(monster))

    def match_crop(self, crop_bgr: np.ndarray, top_k: int = 3) -> list[MonsterMatch]:
        if crop_bgr is None or crop_bgr.size == 0:
            return []

        top_k = max(1, int(top_k))
        query = _prepare_gray(crop_bgr, size=self.template_size, is_query=True).astype(np.float32)
        query = cv2.normalize(query, None, 0.0, 1.0, cv2.NORM_MINMAX)
        query_edge = _prepare_edge_map(crop_bgr, is_query=True, size=self.template_size)
        query_hist = _prepare_color_hist(crop_bgr, is_query=True)
        query_hist_f = query_hist.astype(np.float32)
        query_feature = _prepare_feature_gray(crop_bgr, is_query=True, size=96)
        query_keypoints, query_descriptors = self.feature_extractor.detectAndCompute(query_feature, None)

        quick_candidates = []
        for template in self.templates:
            template_score = float(cv2.matchTemplate(query, template["gray_norm"], cv2.TM_CCOEFF_NORMED)[0, 0])
            edge_score = float(cv2.matchTemplate(query_edge, template["edge"], cv2.TM_CCOEFF_NORMED)[0, 0])
            color_raw = float(
                cv2.compareHist(
                    query_hist_f,
                    template["hist"].astype(np.float32),
                    cv2.HISTCMP_CORREL,
                )
            )
            color_score = max(0.0, min(1.0, (color_raw + 1.0) * 0.5))
            quick_score = (
                0.58 * color_score
                + 0.24 * max(0.0, edge_score)
                + 0.18 * max(0.0, template_score)
            )
            quick_candidates.append(
                {
                    "template": template,
                    "template_score": float(template_score),
                    "edge_score": float(edge_score),
                    "color_score": float(color_score),
                    "quick_score": float(quick_score),
                }
            )

        quick_candidates.sort(key=lambda item: item["quick_score"], reverse=True)
        refine_needed = bool(
            self.refine_feature_extractor is not None
            and len(quick_candidates) >= 2
            and (
                quick_candidates[0]["quick_score"] < 0.73
                or quick_candidates[0]["quick_score"] - quick_candidates[1]["quick_score"] < 0.10
            )
        )
        refine_keypoints = None
        refine_descriptors = None
        if refine_needed:
            refine_keypoints, refine_descriptors = self.refine_feature_extractor.detectAndCompute(query_feature, None)
        if query_descriptors is None:
            feature_pool = quick_candidates[: max(top_k, min(len(quick_candidates), self.feature_prefilter_limit // 2))]
        else:
            prefilter_limit = max(top_k * 12, self.feature_prefilter_limit)
            feature_pool = quick_candidates[: min(len(quick_candidates), prefilter_limit)]

        matches = []
        refine_limit = max(top_k * 3, min(self.feature_prefilter_limit, 12))
        for candidate_index, candidate in enumerate(feature_pool):
            template = candidate["template"]
            monster = template["monster"]
            template_score = candidate["template_score"]
            edge_score = candidate["edge_score"]
            color_score = candidate["color_score"]
            good_matches = 0
            inlier_matches = 0
            feature_score = 0.0
            template_descriptors = template["descriptors"]
            matcher = self.matcher
            active_query_keypoints = query_keypoints
            active_query_descriptors = query_descriptors
            active_template_keypoints = template["keypoints"]
            if (
                refine_needed
                and candidate_index < refine_limit
                and refine_descriptors is not None
                and self.refine_matcher is not None
            ):
                if template["refine_descriptors"] is None:
                    (
                        template["refine_keypoints"],
                        template["refine_descriptors"],
                    ) = self.refine_feature_extractor.detectAndCompute(template["feature_gray"], None)
                matcher = self.refine_matcher
                active_query_keypoints = refine_keypoints
                active_query_descriptors = refine_descriptors
                active_template_keypoints = template["refine_keypoints"] or []
                template_descriptors = template["refine_descriptors"]
            if (
                active_query_descriptors is not None
                and template_descriptors is not None
                and len(template_descriptors) >= 3
            ):
                knn_matches = matcher.knnMatch(active_query_descriptors, template_descriptors, k=2)
                filtered = []
                for pair in knn_matches:
                    if len(pair) < 2:
                        continue
                    m, n = pair
                    if m.distance < 0.75 * n.distance:
                        filtered.append(m)
                good_matches = len(filtered)
                denom = max(8.0, min(24.0, float(template["keypoint_count"])))
                ratio_score = max(0.0, min(1.0, good_matches / denom))
                if active_query_keypoints is not None and len(filtered) >= 4 and active_template_keypoints:
                    source_points = np.float32(
                        [active_query_keypoints[match.queryIdx].pt for match in filtered]
                    ).reshape(-1, 1, 2)
                    target_points = np.float32(
                        [active_template_keypoints[match.trainIdx].pt for match in filtered]
                    ).reshape(-1, 1, 2)
                    _homography, inlier_mask = cv2.findHomography(
                        source_points,
                        target_points,
                        cv2.RANSAC,
                        5.0,
                    )
                    if inlier_mask is not None:
                        inlier_matches = int(np.count_nonzero(inlier_mask))
                inlier_score = max(0.0, min(1.0, inlier_matches / denom))
                feature_score = max(0.0, min(1.0, 0.55 * ratio_score + 0.45 * inlier_score))

            appearance_score = 0.52 * feature_score + 0.30 * color_score + 0.18 * min(feature_score, color_score)
            score = (
                0.82 * appearance_score
                + 0.08 * max(0.0, edge_score)
                + 0.06 * max(0.0, template_score)
            )
            if feature_score < 0.18:
                score *= 0.82
            if color_score < 0.56:
                score *= 0.88
            matches.append(
                MonsterMatch(
                    monster=monster,
                    score=float(score),
                    feature_score=float(feature_score),
                    template_score=float(template_score),
                    color_score=float(color_score),
                    edge_score=float(edge_score),
                    good_matches=int(good_matches),
                    inlier_matches=int(inlier_matches),
                )
            )

        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:top_k]
