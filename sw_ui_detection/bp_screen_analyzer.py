from __future__ import annotations

import os
import json
from dataclasses import asdict, dataclass, field
from html import escape
from pathlib import Path

import cv2
import numpy as np

from bp_transformer_common import PROJECT_ROOT
from sw_ui_detection.detector import BoundingBox, SlotDetection, SummonersWarUIDetector, _clip_box, _iou
from sw_ui_detection.monster_bank import MonsterTemplateBank


@dataclass
class BPSlot:
    slot_id: str
    side: str
    bbox: BoundingBox
    matched_layout: bool
    layout_score: float
    monster_name: str | None = None
    monster_score: float = 0.0
    monster_candidates: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["bbox"] = self.bbox.to_dict()
        return data


@dataclass
class BPScreenAnalysis:
    image_width: int
    image_height: int
    is_bp_screen: bool
    confidence: float
    first_pick_side: str
    current_pick_side: str
    my_slots: list[BPSlot]
    enemy_slots: list[BPSlot]
    my_box_region: BoundingBox
    draft_region: BoundingBox
    generic_slot_count: int

    def to_dict(self) -> dict:
        return {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "is_bp_screen": self.is_bp_screen,
            "confidence": self.confidence,
            "first_pick_side": self.first_pick_side,
            "current_pick_side": self.current_pick_side,
            "my_slots": [slot.to_dict() for slot in self.my_slots],
            "enemy_slots": [slot.to_dict() for slot in self.enemy_slots],
            "my_box_region": self.my_box_region.to_dict(),
            "draft_region": self.draft_region.to_dict(),
            "generic_slot_count": self.generic_slot_count,
        }


class BPDraftScreenAnalyzer:
    # Draft region is still estimated roughly from the whole image,
    # but all slot anchors below are normalized inside draft_region.
    BOX_REGION = (0.014, 0.680, 0.700, 0.260)
    DRAFT_REGION = (0.108, 0.180, 0.788, 0.390)
    MY_FIRST_LAYOUTS = {
        "my": [
            ("my_1", (0.029, 0.274, 0.098, 0.310)),
            ("my_2", (0.141, 0.108, 0.098, 0.297)),
            ("my_3", (0.252, 0.108, 0.098, 0.297)),
            ("my_4", (0.141, 0.469, 0.098, 0.297)),
            ("my_5", (0.252, 0.469, 0.098, 0.297)),
        ],
        "enemy": [
            ("enemy_1", (0.660, 0.108, 0.098, 0.297)),
            ("enemy_2", (0.772, 0.108, 0.098, 0.297)),
            ("enemy_3", (0.660, 0.469, 0.098, 0.297)),
            ("enemy_4", (0.772, 0.469, 0.098, 0.297)),
            ("enemy_5", (0.883, 0.274, 0.098, 0.310)),
        ],
    }
    my_offset_x = 0.01
    my_offset_y = 0.02
    ene_offset_x = -0.03
    ene_offset_y = 0.02
    ENEMY_FIRST_LAYOUTS = {
        "my": [
            ("my_1", (0.029+my_offset_x, 0.108+my_offset_y, 0.098, 0.297)),
            ("my_2", (0.141+my_offset_x, 0.108+my_offset_y, 0.098, 0.297)),
            ("my_3", (0.029+my_offset_x, 0.469+my_offset_y, 0.098, 0.297)),
            ("my_4", (0.141+my_offset_x, 0.469+my_offset_y, 0.098, 0.297)),
            ("my_5", (0.242+my_offset_x, 0.274+my_offset_y, 0.098, 0.310)),
        ],
        "enemy": [
            ("enemy_1", (0.666+ene_offset_x, 0.266+ene_offset_y, 0.104, 0.316)),
            ("enemy_2", (0.783+ene_offset_x, 0.108+ene_offset_y, 0.098, 0.297)),
            ("enemy_3", (0.879+ene_offset_x, 0.108+ene_offset_y, 0.098, 0.297)),
            ("enemy_4", (0.778+ene_offset_x, 0.469+ene_offset_y, 0.098, 0.297)),
            ("enemy_5", (0.884+ene_offset_x, 0.469+ene_offset_y, 0.098, 0.297)),
        ],
    }
    LAYOUTS_BY_FIRST_PICK = {
        "my": MY_FIRST_LAYOUTS,
        "enemy": ENEMY_FIRST_LAYOUTS,
    }
    SLOT_SEARCH_OFFSETS = (-1.0, -0.5, 0.0, 0.5, 1.0)
    SLOT_SEARCH_SCALES = (0.94, 1.0, 1.06)
    SLOT_SEARCH_ASPECTS = (0.96, 1.0, 1.04)
    WIDE_SLOT_SEARCH_OFFSETS = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)
    MONSTER_MARGIN_THRESHOLD = 0.05

    def __init__(
        self,
        ui_detector: SummonersWarUIDetector | None = None,
        template_bank: MonsterTemplateBank | None = None,
        monster_score_threshold: float = 0.42,
    ):
        self.template_bank = template_bank
        self.ui_detector = ui_detector or SummonersWarUIDetector(template_bank=template_bank)
        self.monster_score_threshold = float(monster_score_threshold)

    def analyze(
        self,
        image_or_path: str | Path | np.ndarray,
        validate_monsters: bool = True,
        candidate_top_k: int = 4,
        validate_sides: tuple[str, ...] | None = None,
    ) -> BPScreenAnalysis:
        image = self.ui_detector._load_image(image_or_path)
        generic_result = self.ui_detector.detect(image, validate_monsters=False)
        draft_region = self._estimate_draft_region(image, generic_result.candidate_slots)

        best_variant: dict | None = None
        for first_pick_side, layouts in self.LAYOUTS_BY_FIRST_PICK.items():
            my_slots = self._build_side_slots(
                image,
                generic_result.candidate_slots,
                layouts["my"],
                "my",
                first_pick_side,
                draft_region,
            )
            enemy_slots = self._build_side_slots(
                image,
                generic_result.candidate_slots,
                layouts["enemy"],
                "enemy",
                first_pick_side,
                draft_region,
            )
            variant_score = self._score_variant(my_slots, enemy_slots)
            if best_variant is None or variant_score > best_variant["score"]:
                best_variant = {
                    "first_pick_side": first_pick_side,
                    "my_slots": my_slots,
                    "enemy_slots": enemy_slots,
                    "score": variant_score,
                }

        if best_variant is None:
            best_variant = {
                "first_pick_side": "unknown",
                "my_slots": [],
                "enemy_slots": [],
                "score": 0.0,
            }

        my_slots = best_variant["my_slots"]
        enemy_slots = best_variant["enemy_slots"]

        if validate_monsters:
            normalized_sides = {str(side).strip().lower() for side in (validate_sides or ("my", "enemy"))}
            slots_to_validate: list[BPSlot] = []
            if "my" in normalized_sides:
                slots_to_validate.extend(my_slots)
            if "enemy" in normalized_sides:
                slots_to_validate.extend(enemy_slots)
            self._validate_filled_slots(
                image,
                slots_to_validate,
                candidate_top_k=max(1, int(candidate_top_k)),
            )

        my_box_region = self._estimate_box_region(image, draft_region)
        current_pick_side = self._detect_current_pick_side(image, my_slots, enemy_slots)
        confidence = self._score_layout(
            generic_result,
            my_slots,
            enemy_slots,
            my_box_region,
            best_variant["score"],
        )

        return BPScreenAnalysis(
            image_width=int(image.shape[1]),
            image_height=int(image.shape[0]),
            is_bp_screen=confidence >= 0.58,
            confidence=float(round(confidence, 4)),
            first_pick_side=str(best_variant["first_pick_side"]),
            current_pick_side=current_pick_side,
            my_slots=my_slots,
            enemy_slots=enemy_slots,
            my_box_region=my_box_region,
            draft_region=draft_region,
            generic_slot_count=len(generic_result.candidate_slots),
        )

    def annotate(self, image_or_path: str | Path | np.ndarray, analysis: BPScreenAnalysis) -> np.ndarray:
        image = self.ui_detector._load_image(image_or_path).copy()

        self._draw_region(image, analysis.draft_region, (255, 160, 0), "draft")
        self._draw_region(image, analysis.my_box_region, (0, 180, 255), "my_box")

        for slot in analysis.my_slots + analysis.enemy_slots:
            color = (0, 210, 0) if slot.matched_layout else (0, 180, 255)
            is_active_pick_slot = (
                (analysis.current_pick_side == "my" and slot.slot_id == "my_1")
                or (analysis.current_pick_side == "enemy" and slot.slot_id == "enemy_5")
            )
            if is_active_pick_slot:
                color = (255, 255, 255)
            cv2.rectangle(image, (slot.bbox.x, slot.bbox.y), (slot.bbox.x2, slot.bbox.y2), color, 2)
            label = f"{slot.slot_id} {slot.layout_score:.2f}"
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

        header = (
            f"bp={analysis.is_bp_screen} conf={analysis.confidence:.3f} "
            f"first_pick={analysis.first_pick_side} current={analysis.current_pick_side} "
            f"generic_slots={analysis.generic_slot_count}"
        )
        cv2.putText(image, header, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        return image

    def save_result(
        self,
        image_or_path: str | Path | np.ndarray,
        output_json_path: str | Path,
        output_preview_path: str | Path | None = None,
        output_report_path: str | Path | None = None,
        validate_monsters: bool = True,
        candidate_top_k: int = 4,
        validate_sides: tuple[str, ...] | None = None,
    ) -> BPScreenAnalysis:
        analysis = self.analyze(
            image_or_path,
            validate_monsters=validate_monsters,
            candidate_top_k=candidate_top_k,
            validate_sides=validate_sides,
        )
        output_json_path = Path(output_json_path)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(analysis.to_dict(), f, ensure_ascii=False, indent=2)

        if output_preview_path:
            preview = self.annotate(image_or_path, analysis)
            output_preview_path = Path(output_preview_path)
            output_preview_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_preview_path), preview)
        if output_report_path:
            self.save_html_report(
                image_or_path,
                analysis,
                output_html_path=output_report_path,
                preview_path=output_preview_path,
            )
        return analysis

    def save_html_report(
        self,
        image_or_path: str | Path | np.ndarray,
        analysis: BPScreenAnalysis,
        output_html_path: str | Path,
        preview_path: str | Path | None = None,
    ) -> Path:
        image = self.ui_detector._load_image(image_or_path)
        output_html_path = Path(output_html_path)
        output_html_path.parent.mkdir(parents=True, exist_ok=True)
        assets_dir = output_html_path.parent / f"{output_html_path.stem}_assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        slot_entries = self._save_slot_report_assets(
            image=image,
            analysis=analysis,
            assets_dir=assets_dir,
            html_parent=output_html_path.parent,
        )
        html = self._build_html_report(
            image_or_path=image_or_path,
            analysis=analysis,
            slot_entries=slot_entries,
            html_parent=output_html_path.parent,
            preview_path=Path(preview_path) if preview_path else None,
        )
        output_html_path.write_text(html, encoding="utf-8")
        return output_html_path

    def _save_slot_report_assets(
        self,
        image: np.ndarray,
        analysis: BPScreenAnalysis,
        assets_dir: Path,
        html_parent: Path,
    ) -> list[dict]:
        slot_entries: list[dict] = []
        if self.template_bank is None:
            self.template_bank = MonsterTemplateBank()

        for slot in analysis.my_slots + analysis.enemy_slots:
            crop = self._extract_slot_crop(image, slot.bbox, inset_ratio=0.0)
            crop_name = f"{slot.slot_id}.png"
            crop_path = assets_dir / crop_name
            cv2.imwrite(str(crop_path), crop)

            candidates = []
            for candidate in slot.monster_candidates:
                candidate_name = str(candidate.get("monster", ""))
                ref_path = None
                monster_image_path = self.template_bank.get_monster_image_path(candidate_name)
                if monster_image_path and monster_image_path.exists():
                    ref_path = self._relative_path_for_html(monster_image_path, html_parent)
                candidates.append(
                    {
                        "monster": candidate_name,
                        "score": float(candidate.get("score", 0.0)),
                        "feature_score": float(candidate.get("feature_score", 0.0)),
                        "template_score": float(candidate.get("template_score", 0.0)),
                        "color_score": float(candidate.get("color_score", 0.0)),
                        "edge_score": float(candidate.get("edge_score", 0.0)),
                        "good_matches": int(candidate.get("good_matches", 0)),
                        "reference_path": ref_path,
                    }
                )

            slot_entries.append(
                {
                    "slot_id": slot.slot_id,
                    "side": slot.side,
                    "crop_path": self._relative_path_for_html(crop_path, html_parent),
                    "matched_layout": bool(slot.matched_layout),
                    "layout_score": float(slot.layout_score),
                    "monster_name": slot.monster_name,
                    "monster_score": float(slot.monster_score),
                    "is_empty": bool(self._is_empty_slot(crop)),
                    "bbox": slot.bbox.to_dict(),
                    "candidates": candidates,
                }
            )
        return slot_entries

    def _build_html_report(
        self,
        image_or_path: str | Path | np.ndarray,
        analysis: BPScreenAnalysis,
        slot_entries: list[dict],
        html_parent: Path,
        preview_path: Path | None,
    ) -> str:
        original_path = None
        if isinstance(image_or_path, (str, Path)):
            image_path = Path(image_or_path)
            if image_path.exists():
                original_path = self._relative_path_for_html(image_path, html_parent)
        preview_rel = None
        if preview_path and preview_path.exists():
            preview_rel = self._relative_path_for_html(preview_path, html_parent)

        filled_slots = sum(1 for item in slot_entries if item["monster_name"])
        matched_slots = sum(1 for item in slot_entries if item["matched_layout"])

        parts = [
            "<!DOCTYPE html>",
            "<html lang=\"zh-CN\">",
            "<head>",
            "<meta charset=\"utf-8\">",
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
            f"<title>BP 识别报告 - {escape(str(getattr(image_or_path, 'name', image_or_path)) if image_or_path is not None else 'report')}</title>",
            "<style>",
            "body{font-family:Segoe UI,Arial,sans-serif;background:#111827;color:#e5e7eb;margin:0;padding:24px;}",
            "h1,h2,h3,p{margin:0;}",
            ".header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:18px;}",
            ".pill{padding:6px 10px;border-radius:999px;background:#1f2937;font-size:13px;}",
            ".pill.ok{background:#14532d;color:#dcfce7;}",
            ".pill.bad{background:#7f1d1d;color:#fee2e2;}",
            ".summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:18px;}",
            ".card{background:#1f2937;border:1px solid #374151;border-radius:14px;padding:14px;}",
            ".images{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px;margin-bottom:22px;}",
            ".images img{width:100%;height:auto;border-radius:10px;border:1px solid #374151;background:#0b1220;}",
            ".slots{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px;}",
            ".slot-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;gap:10px;}",
            ".slot-meta{font-size:12px;color:#9ca3af;}",
            ".slot-body{display:grid;grid-template-columns:120px 1fr;gap:12px;align-items:start;}",
            ".crop{width:120px;height:120px;object-fit:cover;border-radius:10px;border:1px solid #4b5563;background:#0b1220;}",
            ".status{font-size:12px;padding:4px 8px;border-radius:999px;background:#334155;display:inline-block;}",
            ".status.good{background:#14532d;color:#dcfce7;}",
            ".status.warn{background:#78350f;color:#fef3c7;}",
            ".status.bad{background:#7f1d1d;color:#fee2e2;}",
            ".candidate{display:grid;grid-template-columns:56px 1fr;gap:10px;padding:8px 0;border-top:1px solid #374151;}",
            ".candidate:first-child{border-top:none;padding-top:0;}",
            ".candidate img{width:56px;height:56px;object-fit:cover;border-radius:8px;border:1px solid #4b5563;background:#0b1220;}",
            ".candidate-name{font-weight:600;}",
            ".candidate-meta{font-size:12px;color:#9ca3af;line-height:1.45;}",
            ".small{font-size:12px;color:#9ca3af;}",
            "a{color:#93c5fd;text-decoration:none;}",
            "</style>",
            "</head>",
            "<body>",
            "<div class=\"header\">",
            f"<h1>BP 识别报告</h1>",
            f"<span class=\"pill {'ok' if analysis.is_bp_screen else 'bad'}\">BP界面: {escape(str(analysis.is_bp_screen))}</span>",
            f"<span class=\"pill\">置信度: {analysis.confidence:.3f}</span>",
            f"<span class=\"pill\">先手: {escape(analysis.first_pick_side)}</span>",
            f"<span class=\"pill\">当前手: {escape(analysis.current_pick_side)}</span>",
            "</div>",
            "<div class=\"summary\">",
            f"<div class=\"card\"><h3>已命名槽位</h3><p style=\"margin-top:8px;font-size:28px;\">{filled_slots}/10</p></div>",
            f"<div class=\"card\"><h3>布局匹配槽位</h3><p style=\"margin-top:8px;font-size:28px;\">{matched_slots}/10</p></div>",
            f"<div class=\"card\"><h3>通用槽位数</h3><p style=\"margin-top:8px;font-size:28px;\">{analysis.generic_slot_count}</p></div>",
            f"<div class=\"card\"><h3>图像尺寸</h3><p style=\"margin-top:8px;font-size:20px;\">{analysis.image_width} × {analysis.image_height}</p></div>",
            "</div>",
            "<div class=\"images\">",
        ]

        if original_path:
            parts.extend(
                [
                    "<div class=\"card\">",
                    "<h3 style=\"margin-bottom:10px;\">原图</h3>",
                    f"<img src=\"{escape(original_path)}\" alt=\"original\">",
                    "</div>",
                ]
            )
        if preview_rel:
            parts.extend(
                [
                    "<div class=\"card\">",
                    "<h3 style=\"margin-bottom:10px;\">标注预览</h3>",
                    f"<img src=\"{escape(preview_rel)}\" alt=\"preview\">",
                    "</div>",
                ]
            )
        parts.append("</div>")
        parts.append("<h2 style=\"margin-bottom:14px;\">槽位逐项核验</h2>")
        parts.append("<div class=\"slots\">")

        for item in slot_entries:
            predicted_name = item["monster_name"] or "未命名"
            status_class = "good" if item["monster_name"] else ("warn" if item["matched_layout"] else "bad")
            status_text = "已识别魔灵" if item["monster_name"] else ("只识别到槽位" if item["matched_layout"] else "槽位可疑")
            parts.extend(
                [
                    "<div class=\"card\">",
                    "<div class=\"slot-head\">",
                    f"<div><h3>{escape(item['slot_id'])}</h3><div class=\"slot-meta\">{escape(item['side'])} · bbox=({item['bbox']['x']}, {item['bbox']['y']}, {item['bbox']['w']}, {item['bbox']['h']})</div></div>",
                    f"<span class=\"status {status_class}\">{escape(status_text)}</span>",
                    "</div>",
                    "<div class=\"slot-body\">",
                    f"<img class=\"crop\" src=\"{escape(item['crop_path'])}\" alt=\"{escape(item['slot_id'])}\">",
                    "<div>",
                    f"<div><strong>预测:</strong> {escape(predicted_name)}</div>",
                    f"<div class=\"small\" style=\"margin-top:4px;\">monster_score={item['monster_score']:.3f} · layout_score={item['layout_score']:.3f} · empty={escape(str(item['is_empty']))}</div>",
                ]
            )

            if item["candidates"]:
                parts.append("<div style=\"margin-top:10px;\">")
                for candidate in item["candidates"]:
                    parts.append("<div class=\"candidate\">")
                    if candidate["reference_path"]:
                        parts.append(f"<img src=\"{escape(candidate['reference_path'])}\" alt=\"{escape(candidate['monster'])}\">")
                    else:
                        parts.append("<div style=\"width:56px;height:56px;border-radius:8px;background:#0b1220;border:1px solid #4b5563;\"></div>")
                    parts.append("<div>")
                    parts.append(f"<div class=\"candidate-name\">{escape(candidate['monster'])}</div>")
                    parts.append(
                        "<div class=\"candidate-meta\">"
                        f"总分 {candidate['score']:.3f} · 特征 {candidate['feature_score']:.3f} · "
                        f"模板 {candidate['template_score']:.3f} · 边缘 {candidate.get('edge_score', 0.0):.3f} · "
                        f"颜色 {candidate['color_score']:.3f} · "
                        f"good_matches {candidate['good_matches']}"
                        "</div>"
                    )
                    parts.append("</div></div>")
                parts.append("</div>")
            else:
                parts.append("<div style=\"margin-top:10px;\" class=\"small\">没有候选结果，通常表示该槽位被判为空，或模板匹配分数不足。</div>")

            parts.extend(["</div>", "</div>", "</div>"])

        parts.extend(["</div>", "</body>", "</html>"])
        return "\n".join(parts)

    def _relative_path_for_html(self, target_path: Path, html_parent: Path) -> str:
        return Path(os.path.relpath(target_path, html_parent)).as_posix()

    def _build_side_slots(
        self,
        image: np.ndarray,
        generic_candidates: list[SlotDetection],
        layout: list[tuple[str, tuple[float, float, float, float]]],
        side: str,
        first_pick_side: str,
        draft_region: BoundingBox,
    ) -> list[BPSlot]:
        slots: list[BPSlot] = []
        for slot_id, normalized_box in layout:
            anchor = self._tighten_slot_box(self._norm_box_in_region(normalized_box, draft_region), image.shape)
            best_candidate = None
            best_score = 0.0
            for candidate in generic_candidates:
                overlap = _iou(anchor, candidate.bbox)
                if overlap <= 0:
                    continue

                anchor_center = np.array([anchor.x + anchor.w * 0.5, anchor.y + anchor.h * 0.5], dtype=np.float32)
                candidate_center = np.array(
                    [candidate.bbox.x + candidate.bbox.w * 0.5, candidate.bbox.y + candidate.bbox.h * 0.5],
                    dtype=np.float32,
                )
                center_distance = float(np.linalg.norm(anchor_center - candidate_center))
                center_penalty = max(anchor.w, anchor.h) * 0.8
                proximity = max(0.0, 1.0 - center_distance / max(1.0, center_penalty))
                score = 0.70 * overlap + 0.30 * proximity
                if score > best_score:
                    best_candidate = candidate
                    best_score = score

            slot_box = anchor
            if best_candidate is not None and best_score >= 0.18:
                slot_box = self._blend_slot_box(anchor, best_candidate.bbox, image.shape)
            slot_box, visual_score = self._refine_slot_box(
                image,
                slot_box,
                anchor,
                slot_id=slot_id,
                side=side,
                first_pick_side=first_pick_side,
            )
            layout_score = max(best_score, visual_score * 0.70)
            slots.append(
                BPSlot(
                    slot_id=slot_id,
                    side=side,
                    bbox=slot_box,
                    matched_layout=(best_candidate is not None and best_score >= 0.18) or visual_score >= 0.28,
                    layout_score=float(round(layout_score, 4)),
                )
            )
        return slots

    def _tighten_slot_box(
        self,
        bbox: BoundingBox,
        image_shape: tuple[int, int, int] | tuple[int, int],
    ) -> BoundingBox:
        pad_x = max(2, int(round(bbox.w * 0.03)))
        pad_y = max(1, int(round(bbox.h * 0.015)))
        return _clip_box(
            bbox.x + pad_x,
            bbox.y + pad_y,
            bbox.w - pad_x * 2,
            bbox.h - pad_y * 2,
            image_shape,
        )

    def _blend_slot_box(
        self,
        anchor: BoundingBox,
        candidate: BoundingBox,
        image_shape: tuple[int, int, int] | tuple[int, int],
    ) -> BoundingBox:
        anchor_center_x = anchor.x + anchor.w * 0.5
        anchor_center_y = anchor.y + anchor.h * 0.5
        candidate_center_x = candidate.x + candidate.w * 0.5
        candidate_center_y = candidate.y + candidate.h * 0.5

        center_x = 0.40 * anchor_center_x + 0.60 * candidate_center_x
        center_y = 0.40 * anchor_center_y + 0.60 * candidate_center_y

        anchor_aspect = anchor.w / max(1.0, float(anchor.h))
        candidate_aspect = candidate.w / max(1.0, float(candidate.h))
        target_aspect = float(np.clip(anchor_aspect * 0.45 + candidate_aspect * 0.55, 0.92, 1.08))

        base_h = 0.68 * anchor.h + 0.32 * candidate.h
        height = int(round(np.clip(base_h, anchor.h * 0.90, anchor.h * 1.08)))
        width = int(round(height * target_aspect))
        width = int(round(np.clip(width, anchor.w * 0.88, anchor.w * 1.08)))

        return _clip_box(
            int(round(center_x - width * 0.5)),
            int(round(center_y - height * 0.5)),
            width,
            height,
            image_shape,
        )

    def _refine_slot_box(
        self,
        image: np.ndarray,
        seed_box: BoundingBox,
        reference_box: BoundingBox,
        slot_id: str,
        side: str,
        first_pick_side: str,
    ) -> tuple[BoundingBox, float]:
        best_box = seed_box
        best_visual = self._slot_visual_score(self._extract_slot_crop(image, seed_box, inset_ratio=0.0))
        best_score = self._slot_search_score(best_visual, seed_box, seed_box, reference_box)

        seed_center_x = seed_box.x + seed_box.w * 0.5
        seed_center_y = seed_box.y + seed_box.h * 0.5
        seed_aspect = seed_box.w / max(1.0, float(seed_box.h))
        search_offsets = self.SLOT_SEARCH_OFFSETS
        shift_scale = 1.0
        if side == "enemy" and slot_id == "enemy_1" and first_pick_side == "enemy":
            search_offsets = self.WIDE_SLOT_SEARCH_OFFSETS
            shift_scale = 1.45

        shift_x = max(3, int(round(reference_box.w * 0.06 * shift_scale)))
        shift_y = max(3, int(round(reference_box.h * 0.06 * shift_scale)))

        for scale in self.SLOT_SEARCH_SCALES:
            base_h = max(18, int(round(reference_box.h * scale)))
            for aspect_scale in self.SLOT_SEARCH_ASPECTS:
                aspect = float(np.clip(seed_aspect * aspect_scale, 0.90, 1.08))
                width = max(18, int(round(base_h * aspect)))
                for offset_x in search_offsets:
                    center_x = seed_center_x + offset_x * shift_x
                    for offset_y in search_offsets:
                        center_y = seed_center_y + offset_y * shift_y
                        candidate_box = _clip_box(
                            int(round(center_x - width * 0.5)),
                            int(round(center_y - base_h * 0.5)),
                            width,
                            base_h,
                            image.shape,
                        )
                        visual_score = self._slot_visual_score(
                            self._extract_slot_crop(image, candidate_box, inset_ratio=0.0)
                        )
                        score = self._slot_search_score(visual_score, candidate_box, seed_box, reference_box)
                        if score > best_score:
                            best_box = candidate_box
                            best_visual = visual_score
                            best_score = score
        return best_box, best_visual

    def _slot_search_score(
        self,
        visual_score: float,
        candidate_box: BoundingBox,
        seed_box: BoundingBox,
        reference_box: BoundingBox,
    ) -> float:
        candidate_center = np.array(
            [candidate_box.x + candidate_box.w * 0.5, candidate_box.y + candidate_box.h * 0.5],
            dtype=np.float32,
        )
        seed_center = np.array([seed_box.x + seed_box.w * 0.5, seed_box.y + seed_box.h * 0.5], dtype=np.float32)
        center_distance = float(np.linalg.norm(candidate_center - seed_center))
        max_distance = max(6.0, max(reference_box.w, reference_box.h) * 0.18)
        proximity_bonus = max(0.0, 1.0 - center_distance / max_distance)

        size_delta = (
            abs(candidate_box.w - seed_box.w) / max(1.0, float(seed_box.w))
            + abs(candidate_box.h - seed_box.h) / max(1.0, float(seed_box.h))
        )
        size_bonus = max(0.0, 1.0 - size_delta / 0.24)
        return visual_score + 0.08 * proximity_bonus + 0.04 * size_bonus

    def _score_variant(self, my_slots: list[BPSlot], enemy_slots: list[BPSlot]) -> float:
        slots = my_slots + enemy_slots
        if not slots:
            return 0.0
        matched_ratio = float(np.mean([1.0 if slot.matched_layout else 0.0 for slot in slots]))
        mean_layout = float(np.mean([slot.layout_score for slot in slots]))
        top_scores = sorted([slot.layout_score for slot in slots], reverse=True)[:4]
        top_mean = float(np.mean(top_scores)) if top_scores else 0.0
        return 0.30 * matched_ratio + 0.35 * mean_layout + 0.35 * top_mean

    def _validate_filled_slots(self, image: np.ndarray, slots: list[BPSlot], candidate_top_k: int = 4) -> None:
        if not slots:
            return
        if self.template_bank is None:
            self.template_bank = MonsterTemplateBank()

        candidate_top_k = max(1, int(candidate_top_k))
        for slot in slots:
            if slot.layout_score < 0.10 and not slot.matched_layout:
                continue
            crop = self._extract_slot_crop(image, slot.bbox, inset_ratio=0.03)
            if self._is_empty_slot(crop):
                continue
            matches = self.template_bank.match_crop(crop, top_k=candidate_top_k)
            if not matches:
                continue
            slot.monster_candidates = [
                {
                    "monster": match.monster,
                    "score": round(match.score, 4),
                    "feature_score": round(match.feature_score, 4),
                    "template_score": round(match.template_score, 4),
                    "color_score": round(match.color_score, 4),
                    "edge_score": round(match.edge_score, 4),
                    "good_matches": int(match.good_matches),
                }
                for match in matches
            ]
            best = matches[0]
            second = matches[1] if len(matches) > 1 else None
            if self._should_accept_monster_match(best, second, slot):
                slot.monster_name = best.monster
                slot.monster_score = float(best.score)

    def _should_accept_monster_match(self, best, second, slot: BPSlot) -> bool:
        if best.score < self.monster_score_threshold:
            return False
        if slot.layout_score < 0.16 and best.score < self.monster_score_threshold + 0.05:
            return False
        appearance_score = 0.60 * best.feature_score + 0.40 * best.color_score
        if appearance_score < 0.56:
            return False
        if second is None:
            return appearance_score >= 0.62 or best.good_matches >= 12

        margin = best.score - second.score
        appearance_margin = (
            0.60 * (best.feature_score - second.feature_score)
            + 0.40 * (best.color_score - second.color_score)
        )
        strong_signal = (
            appearance_score >= 0.66
            or (best.feature_score >= 0.52 and best.color_score >= 0.70)
            or (best.good_matches >= 12 and best.color_score >= 0.68)
        )
        return (
            margin >= self.MONSTER_MARGIN_THRESHOLD
            or (margin >= 0.03 and strong_signal)
            or (appearance_margin >= 0.045 and strong_signal)
        )

    def _detect_current_pick_side(self, image: np.ndarray, my_slots: list[BPSlot], enemy_slots: list[BPSlot]) -> str:
        if not my_slots or not enemy_slots:
            return "unknown"

        my_focus = my_slots[0].bbox
        enemy_focus = enemy_slots[-1].bbox
        my_score = self._highlight_score(self._extract_slot_crop(image, my_focus, inset_ratio=0.0))
        enemy_score = self._highlight_score(self._extract_slot_crop(image, enemy_focus, inset_ratio=0.0))
        if my_score - enemy_score >= 0.08:
            return "my"
        if enemy_score - my_score >= 0.08:
            return "enemy"
        return "unknown"

    def _score_layout(
        self,
        generic_result,
        my_slots: list[BPSlot],
        enemy_slots: list[BPSlot],
        my_box_region: BoundingBox,
        variant_score: float,
    ) -> float:
        slots = my_slots + enemy_slots
        matched_ratio = float(np.mean([1.0 if slot.matched_layout else 0.0 for slot in slots])) if slots else 0.0
        layout_score = float(np.mean([slot.layout_score for slot in slots])) if slots else 0.0

        box_hint = min(1.0, my_box_region.w / max(1, generic_result.image_width) / 0.70)
        generic_hint = min(1.0, generic_result.confidence + min(0.25, generic_result.image_height / max(1, generic_result.image_width) * 0.15))

        confidence = 0.42 * matched_ratio + 0.28 * layout_score + 0.18 * generic_hint + 0.12 * box_hint
        confidence = 0.65 * confidence + 0.35 * variant_score
        return max(0.0, min(1.0, confidence))

    def _extract_slot_crop(self, image: np.ndarray, bbox: BoundingBox, inset_ratio: float = 0.08) -> np.ndarray:
        inset = int(round(min(bbox.w, bbox.h) * max(0.0, inset_ratio)))
        crop_box = _clip_box(bbox.x + inset, bbox.y + inset, bbox.w - inset * 2, bbox.h - inset * 2, image.shape)
        return image[crop_box.y : crop_box.y2, crop_box.x : crop_box.x2]

    def _is_empty_slot(self, crop: np.ndarray) -> bool:
        if crop.size == 0:
            return True
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        edges = cv2.Canny(gray_blur, 40, 120)
        brightness = float(gray.mean()) / 255.0
        std = float(gray.std()) / 255.0
        saturation = float(hsv[..., 1].mean()) / 255.0
        edge_density = float(np.mean(edges > 0))
        return (
            brightness < 0.33
            and std < 0.11
            and saturation < 0.20
            and edge_density < 0.05
        ) or (
            brightness < 0.26
            and saturation < 0.28
            and edge_density < 0.06
        )

    def _highlight_score(self, crop: np.ndarray) -> float:
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean()) / 255.0
        bright_ratio = float(np.mean(gray >= 140))
        border = self._border_mask(gray.shape[0], gray.shape[1], border_ratio=0.13)
        border_brightness = float(gray[border].mean()) / 255.0 if np.any(border) else brightness
        return 0.40 * brightness + 0.35 * border_brightness + 0.25 * bright_ratio

    def _slot_visual_score(self, crop: np.ndarray) -> float:
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 40, 120)

        edge_density = float(np.mean(edges > 0))
        border = self._border_mask(gray.shape[0], gray.shape[1], border_ratio=0.15)
        border_edge_density = float(np.mean(edges[border] > 0)) if np.any(border) else edge_density
        inner_mask = ~border
        inner_edge_density = float(np.mean(edges[inner_mask] > 0)) if np.any(inner_mask) else edge_density

        border_mean = float(gray[border].mean()) / 255.0 if np.any(border) else 0.0
        inner_mean = float(gray[inner_mask].mean()) / 255.0 if np.any(inner_mask) else border_mean
        contrast = abs(border_mean - inner_mean)

        score = (
            0.35 * min(1.0, edge_density / 0.10)
            + 0.40 * min(1.0, border_edge_density / 0.18)
            + 0.10 * min(1.0, inner_edge_density / 0.12)
            + 0.15 * min(1.0, contrast / 0.16)
        )
        return max(0.0, min(1.0, score))

    def _border_mask(self, h: int, w: int, border_ratio: float) -> np.ndarray:
        mask = np.zeros((h, w), dtype=bool)
        if h <= 0 or w <= 0:
            return mask
        border = max(1, int(round(min(h, w) * border_ratio)))
        mask[:border, :] = True
        mask[-border:, :] = True
        mask[:, :border] = True
        mask[:, -border:] = True
        return mask

    def _draw_region(self, image: np.ndarray, bbox: BoundingBox, color: tuple[int, int, int], label: str) -> None:
        cv2.rectangle(image, (bbox.x, bbox.y), (bbox.x2, bbox.y2), color, 2)
        cv2.putText(
            image,
            label,
            (bbox.x + 4, max(18, bbox.y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    def _norm_box(self, box: tuple[float, float, float, float], image_shape: tuple[int, int, int]) -> BoundingBox:
        h, w = image_shape[:2]
        x, y, bw, bh = box
        return BoundingBox(
            x=int(round(x * w)),
            y=int(round(y * h)),
            w=int(round(bw * w)),
            h=int(round(bh * h)),
        )

    def _norm_box_in_region(self, box: tuple[float, float, float, float], region: BoundingBox) -> BoundingBox:
        x, y, bw, bh = box
        return BoundingBox(
            x=int(round(region.x + x * region.w)),
            y=int(round(region.y + y * region.h)),
            w=int(round(bw * region.w)),
            h=int(round(bh * region.h)),
        )

    def _estimate_draft_region(self, image: np.ndarray, generic_candidates: list[SlotDetection]) -> BoundingBox:
        fallback = self._norm_box(self.DRAFT_REGION, image.shape)
        image_h = image.shape[0]
        upper_candidates = [
            candidate
            for candidate in generic_candidates
            if (candidate.bbox.y + candidate.bbox.h * 0.5) <= image_h * 0.66
        ]
        if len(upper_candidates) < 4:
            return fallback

        boxes = [candidate.bbox for candidate in upper_candidates]
        slot_w = int(round(np.median([box.w for box in boxes])))
        slot_h = int(round(np.median([box.h for box in boxes])))
        x1 = min(box.x for box in boxes)
        y1 = min(box.y for box in boxes)
        x2 = max(box.x2 for box in boxes)
        y2 = max(box.y2 for box in boxes)

        estimated = _clip_box(
            x1 - int(round(slot_w * 0.55)),
            y1 - int(round(slot_h * 0.55)),
            (x2 - x1) + int(round(slot_w * 1.10)),
            (y2 - y1) + int(round(slot_h * 1.70)),
            image.shape,
        )

        fallback_center = np.array([fallback.x + fallback.w * 0.5, fallback.y + fallback.h * 0.5], dtype=np.float32)
        estimate_center = np.array([estimated.x + estimated.w * 0.5, estimated.y + estimated.h * 0.5], dtype=np.float32)
        center_shift = float(np.linalg.norm(fallback_center - estimate_center))
        if center_shift > max(fallback.w, fallback.h) * 0.35:
            return fallback
        return estimated

    def _estimate_box_region(self, image: np.ndarray, draft_region: BoundingBox) -> BoundingBox:
        fallback = self._norm_box(self.BOX_REGION, image.shape)
        x = max(0, draft_region.x - int(round(draft_region.w * 0.12)))
        y = min(image.shape[0] - 1, draft_region.y2 + int(round(image.shape[0] * 0.10)))
        w = min(image.shape[1] - x, int(round(draft_region.w * 0.86)))
        h = min(image.shape[0] - y, int(round(image.shape[0] * 0.25)))
        estimated = BoundingBox(x=x, y=y, w=max(1, w), h=max(1, h))

        size_ratio = estimated.w / max(1, fallback.w)
        if size_ratio < 0.65 or size_ratio > 1.35:
            return fallback
        return estimated


def analyze_bp_screenshot(
    image_or_path: str | Path | np.ndarray,
    validate_monsters: bool = True,
) -> BPScreenAnalysis:
    analyzer = BPDraftScreenAnalyzer()
    return analyzer.analyze(image_or_path, validate_monsters=validate_monsters)


def save_bp_screenshot_result(
    image_or_path: str | Path | np.ndarray,
    output_json_path: str | Path,
    output_preview_path: str | Path | None = None,
    validate_monsters: bool = True,
) -> BPScreenAnalysis:
    analyzer = BPDraftScreenAnalyzer()
    return analyzer.save_result(
        image_or_path,
        output_json_path=output_json_path,
        output_preview_path=output_preview_path,
        validate_monsters=validate_monsters,
    )
