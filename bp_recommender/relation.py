import argparse
import json
import math
from itertools import combinations

import torch

from bp_core.lucksack_stats import BPLucksackStats, DEFAULT_STATS_PATH
from bp_core.relation_common import (
    BPDraftRelationNet,
    BPDraftRelationNetV2,
    MonsterVocab,
    encode_pick_list,
    infer_pick_count,
    infer_pick_count_for_side,
    valid_vocab_slugs,
)
from bp_core.transformer_common import PAD_TOKEN, PROJECT_ROOT, UNK_TOKEN, normalize_name, slugify_fallback
from bp_core.personal_history import BPPersonalHistoryModel

LEGACY_MODEL_PATH = PROJECT_ROOT / "models" / "bp_relation_net.pt"
LEGACY_META_PATH = PROJECT_ROOT / "models" / "bp_relation_meta.json"
LEGACY_VOCAB_PATH = PROJECT_ROOT / "models" / "bp_relation_vocab.json"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "bp_relation_net_v2.pt"
DEFAULT_META_PATH = PROJECT_ROOT / "models" / "bp_relation_meta_v2.json"
DEFAULT_VOCAB_PATH = PROJECT_ROOT / "models" / "bp_relation_vocab_v2.json"


class BPRelationRecommender:
    def __init__(
        self,
        model_path=str(DEFAULT_MODEL_PATH),
        meta_path=str(DEFAULT_META_PATH),
        vocab_path=str(DEFAULT_VOCAB_PATH),
        device=None,
        personal_history_model=None,
        max_personal_weight=0.30,
        exploration_bonus_enabled=False,
        low_exposure_threshold=3.0,
        high_relation_threshold=0.75,
        exploration_bonus=0.03,
        external_stats_path=str(DEFAULT_STATS_PATH),
        stats_rank_group=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.personal_history_model = personal_history_model
        self.lucksack_stats = BPLucksackStats(
            stats_path=external_stats_path,
            rank_group=stats_rank_group,
        )
        self.max_personal_weight = min(0.30, max(0.0, float(max_personal_weight)))
        self.exploration_bonus_enabled = bool(exploration_bonus_enabled)
        self.low_exposure_threshold = float(low_exposure_threshold)
        self.high_relation_threshold = float(high_relation_threshold)
        self.exploration_bonus = float(exploration_bonus)
        if (
            model_path == str(DEFAULT_MODEL_PATH)
            and not all(path.exists() for path in (DEFAULT_MODEL_PATH, DEFAULT_META_PATH, DEFAULT_VOCAB_PATH))
            and all(path.exists() for path in (LEGACY_MODEL_PATH, LEGACY_META_PATH, LEGACY_VOCAB_PATH))
        ):
            model_path = str(LEGACY_MODEL_PATH)
            meta_path = str(LEGACY_META_PATH)
            vocab_path = str(LEGACY_VOCAB_PATH)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_dict = json.load(f)
        self.vocab = MonsterVocab(vocab_dict)
        self.display_name_map = {
            normalize_name(key): value
            for key, value in (self.meta.get("display_name_map", {}) or {}).items()
            if key and value
        }
        self.image_name_map = {
            normalize_name(key): value
            for key, value in (self.meta.get("image_name_map", {}) or {}).items()
            if key and value
        }
        self.name_slug_aliases = {
            normalize_name(key): normalize_name(value)
            for key, value in (self.meta.get("name_slug_aliases", {}) or {}).items()
            if key and value
        }
        for slug, display_name in self.display_name_map.items():
            self.name_slug_aliases.setdefault(slug, slug)
            self.name_slug_aliases.setdefault(normalize_name(display_name), slug)
            self.name_slug_aliases.setdefault(slugify_fallback(display_name), slug)

        self.pad_id = self.vocab.get_id(PAD_TOKEN)
        self.model_version = int(self.meta.get("architecture_version", 1))
        model_cls = BPDraftRelationNetV2 if self.model_version >= 2 else BPDraftRelationNet
        model_kwargs = {
            "vocab_size": len(self.vocab),
            "embed_dim": int(self.meta.get("embed_dim", 128)),
            "state_dim": int(self.meta.get("state_dim", 256)),
            "action_dim": int(self.meta.get("action_dim", 256)),
            "hidden_dim": int(self.meta.get("hidden_dim", 256)),
            "dropout": float(self.meta.get("dropout", 0.15)),
            "pad_idx": self.pad_id,
        }
        if self.model_version >= 2:
            model_kwargs.update(
                {
                    "num_heads": int(self.meta.get("num_heads", 4)),
                    "num_layers": int(self.meta.get("num_layers", 2)),
                    "ffn_dim": int(self.meta.get("ffn_dim", 512)),
                }
            )
        self.model = model_cls(**model_kwargs)
        try:
            state = torch.load(model_path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self._known_model_slugs = set(valid_vocab_slugs(self.vocab))
        self._external_candidate_slugs = frozenset(self.lucksack_stats.candidate_slugs())

    def set_personal_history_model(self, personal_model):
        self.personal_history_model = personal_model

    def update_personal_records(self, records, deduplicate=True):
        if self.personal_history_model is None:
            self.personal_history_model = BPPersonalHistoryModel(deduplicate=deduplicate)
        self.personal_history_model.update_records(records, deduplicate=deduplicate)

    def clear_personal_history(self):
        if self.personal_history_model is not None:
            self.personal_history_model.clear()

    def _personal_weight(self, personal):
        if not personal:
            return 0.0
        return max(0.0, min(1.0, self.max_personal_weight * float(personal.get("confidence", 0.0))))

    def _apply_personal_blend(self, relation_score, personal):
        relation_score = float(relation_score)
        personal_weight = self._personal_weight(personal)
        personal_score = 0.5 if not personal else float(personal.get("personal_score", 0.5))
        final_score = (1.0 - personal_weight) * relation_score + personal_weight * personal_score
        bonus = 0.0
        reasons = [] if not personal else list(personal.get("reasons", []))
        raw_sample_count = 0.0 if not personal else float(personal.get("raw_sample_count", personal.get("sample_count", 0.0)) or 0.0)
        confidence = 0.0 if not personal else float(personal.get("confidence", 0.0) or 0.0)
        if relation_score >= self.high_relation_threshold and confidence <= 0.15:
            reasons.append("high relation score preserved despite low personal exposure")
        if (
            self.exploration_bonus_enabled
            and raw_sample_count < self.low_exposure_threshold
            and relation_score > self.high_relation_threshold
        ):
            bonus = self.exploration_bonus
            final_score += bonus
            reasons.append("exploration bonus: low exposure with high relation score")
        return final_score, personal_weight, bonus, reasons

    def _to_slug(self, value):
        normalized = normalize_name(value)
        if not normalized:
            return ""
        fallback = slugify_fallback(value)
        sanitized_fallback = fallback.replace("[", "").replace("]", "")
        external_slug = self.lucksack_stats.resolve_slug(value)
        for key in (normalized, fallback, sanitized_fallback):
            resolved = self.name_slug_aliases.get(key)
            if resolved:
                return normalize_name(resolved)
        return normalize_name(external_slug or sanitized_fallback or fallback or normalized)

    def _to_display_name(self, slug):
        slug = normalize_name(slug)
        return self.display_name_map.get(slug) or self.lucksack_stats.to_display_name(slug) or slug

    def _extract_owned_set(self, my_box):
        owned = set()
        if not my_box:
            return owned
        if isinstance(my_box, dict):
            for key, value in my_box.items():
                if value:
                    owned.add(self._to_slug(key))
            return owned
        if isinstance(my_box, (list, tuple, set)):
            for item in my_box:
                if isinstance(item, str):
                    owned.add(self._to_slug(item))
                elif isinstance(item, dict):
                    name = item.get("name") or item.get("monster") or item.get("label") or item.get("value")
                    if item.get("owned", True) and name:
                        owned.add(self._to_slug(name))
        return owned

    def _swap_first_pick_side(self, first_pick_side):
        first_pick_side = normalize_name(first_pick_side)
        if first_pick_side not in {"me", "enemy"}:
            return None
        return "enemy" if first_pick_side == "me" else "me"

    def _allowed_candidates(self, my_picks, enemy_picks, candidate_pool=None, my_box=None, use_my_box_for_reco=True):
        occupied = {self._to_slug(x) for x in (my_picks or []) + (enemy_picks or []) if x}
        external_candidates = getattr(self, "_external_candidate_slugs", frozenset())
        if candidate_pool is not None:
            allowed = {self._to_slug(x) for x in candidate_pool if x}
        else:
            allowed = set(valid_vocab_slugs(self.vocab)) | set(external_candidates)
        blocked = {normalize_name(PAD_TOKEN), normalize_name(UNK_TOKEN), "", *occupied}
        allowed = {
            slug
            for slug in allowed
            if slug not in blocked
            and (
                slug in self._known_model_slugs
                or candidate_pool is not None
                or slug in external_candidates
            )
        }
        if use_my_box_for_reco:
            owned = self._extract_owned_set(my_box)
            if not owned:
                return []
            allowed &= owned
        return sorted(allowed, key=lambda slug: self._to_display_name(slug))

    def _normalize_scores(self, raw_scores):
        if not raw_scores:
            return []
        if self.model_version >= 2 and "relation_score_center" in self.meta:
            center = float(self.meta.get("relation_score_center", 0.0))
            scale = max(0.5, float(self.meta.get("relation_score_scale", 1.0)))
            return [1.0 / (1.0 + math.exp(-(float(score) - center) / scale)) for score in raw_scores]
        lo = min(raw_scores)
        hi = max(raw_scores)
        if hi - lo < 1e-8:
            return [0.5 for _ in raw_scores]
        return [(score - lo) / (hi - lo) for score in raw_scores]

    def _policy_probabilities(self, raw_scores):
        """Calibrate a candidate set into a comparable policy distribution."""
        if not raw_scores:
            return []
        temperature = max(0.05, float(self.meta.get("policy_temperature", 1.0)))
        logits = [float(score) / temperature for score in raw_scores]
        peak = max(logits)
        weights = [math.exp(logit - peak) for logit in logits]
        total = sum(weights)
        if total <= 0.0:
            return [1.0 / len(weights)] * len(weights)
        return [weight / total for weight in weights]

    def _blend_relation_stats(self, relation_score, stats_detail, relation_model_known=True):
        relation_model_known = stats_detail.get(
            "_relation_model_known",
            relation_model_known,
        )
        stats_score = float(stats_detail.get("score", 0.5))
        mechanism_score = float(stats_detail.get("mechanism_score", 0.5))
        mechanism_available = bool(stats_detail.get("mechanism_available"))
        matchup_score = float(stats_detail.get("matchup_score", 0.5))
        matchup_coverage = float(stats_detail.get("matchup_coverage", 0.0))
        coverage_values = [
            float(stats_detail[key] or 0.0)
            for key in ("coverage", "data_coverage")
            if key in stats_detail
        ]
        stats_coverage = min(coverage_values) if coverage_values else 0.0
        stats_available = bool(getattr(self.lucksack_stats, "ready", False)) and stats_coverage > 0.0
        mechanism_coverage = float(stats_detail.get("mechanism_coverage", 0.0) or 0.0)
        mechanism_confidence = mechanism_coverage if mechanism_available else 0.0
        matchup_available = matchup_coverage > 0.0
        relation_confidence = 1.0 if relation_model_known else 0.15
        relation_value = 0.5 + (float(relation_score) - 0.5) * relation_confidence

        if stats_available and mechanism_available and matchup_available:
            base_weights = (0.65, 0.10, 0.15, 0.10)
        elif stats_available and mechanism_available:
            base_weights = (0.70, 0.10, 0.20, 0.0)
        elif stats_available and matchup_available:
            base_weights = (0.75, 0.15, 0.0, 0.10)
        elif stats_available:
            base_weights = (0.85, 0.15, 0.0, 0.0)
        elif mechanism_available:
            base_weights = (0.80, 0.0, 0.20, 0.0)
        else:
            base_weights = (1.0, 0.0, 0.0, 0.0)

        weights = (
            base_weights[0] * relation_confidence,
            base_weights[1] * stats_coverage,
            base_weights[2] * mechanism_confidence,
            base_weights[3] * matchup_coverage,
        )
        values = (
            relation_value,
            stats_score,
            mechanism_score,
            matchup_score,
        )
        weight_total = sum(weights)
        combined_score = (
            sum(value * weight for value, weight in zip(values, weights)) / weight_total
            if weight_total > 0.0
            else float(relation_score)
        )
        return (
            combined_score,
            stats_score,
            mechanism_score,
            mechanism_available,
            matchup_score,
            matchup_available,
        )

    @staticmethod
    def _select_evidence_diverse_plans(plans, top_k):
        """Keep high-value plans from independent evidence sources before reranking."""
        plans = list(plans or [])
        limit = max(0, int(top_k))
        if limit <= 0 or len(plans) <= limit:
            return plans

        selected = []
        selected_ids = set()
        anchor_count = max(1, limit // 4)
        component_keys = (
            ("plan_score", lambda row: float(row.get("plan_score", 0.0))),
            (
                "mechanism_score",
                lambda row: float((row.get("detail") or {}).get("mechanism_score", 0.5)),
            ),
            (
                "draft_stats_score",
                lambda row: float((row.get("detail") or {}).get("draft_stats_score", 0.5)),
            ),
            (
                "matchup_score",
                lambda row: float((row.get("detail") or {}).get("matchup_score", 0.5)),
            ),
        )

        for key, score_fn in component_keys:
            candidates = plans
            if key == "mechanism_score":
                candidates = [
                    row
                    for row in plans
                    if bool((row.get("detail") or {}).get("mechanism_available"))
                ]
            elif key == "draft_stats_score":
                candidates = [
                    row
                    for row in plans
                    if float((row.get("detail") or {}).get("draft_stats_coverage", 0.0)) > 0.0
                ]
            elif key == "matchup_score":
                candidates = [
                    row
                    for row in plans
                    if float((row.get("detail") or {}).get("matchup_coverage", 0.0)) > 0.0
                ]
            for row in sorted(candidates, key=score_fn, reverse=True)[:anchor_count]:
                row_id = id(row)
                if row_id in selected_ids:
                    continue
                selected.append(row)
                selected_ids.add(row_id)
                if len(selected) >= limit:
                    return selected

        for row in sorted(plans, key=lambda item: float(item.get("plan_score", 0.0)), reverse=True):
            if id(row) in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(id(row))
            if len(selected) >= limit:
                break
        return selected

    @staticmethod
    def _select_candidate_diverse(
        allowed,
        quick_scores,
        quick_details,
        limit,
        synergy_pairs=None,
    ):
        """Keep statistical, matchup, and mechanism anchors before pair expansion."""
        limit = min(len(allowed), max(1, int(limit)))
        if len(allowed) <= limit:
            return list(allowed)
        rows = [
            {
                "slug": slug,
                "quick_score": float(score),
                "detail": detail or {},
            }
            for slug, score, detail in zip(allowed, quick_scores, quick_details)
        ]
        selected = []
        selected_slugs = set()
        available_slugs = {row["slug"] for row in rows}
        pair_anchor_limit = min(limit, max(2, limit // 4))
        pair_anchor_count = 0
        for pair in synergy_pairs or []:
            pair_slugs = [
                slug
                for slug in pair.get("slugs", [])
                if slug in available_slugs
            ]
            if len(pair_slugs) != 2 or pair_anchor_count + 2 > pair_anchor_limit:
                continue
            for slug in pair_slugs:
                if slug in selected_slugs:
                    continue
                selected.append(slug)
                selected_slugs.add(slug)
            pair_anchor_count += len(pair_slugs)
            if pair_anchor_count >= pair_anchor_limit:
                break
        anchors = (
            ("quick_score", lambda row: row["quick_score"]),
            ("mechanism", lambda row: float(row["detail"].get("mechanism_score", 0.5))),
            ("matchup", lambda row: float(row["detail"].get("matchup_score", 0.5))),
            ("draft_stats", lambda row: float(row["detail"].get("score", 0.5))),
        )
        anchor_count = max(1, limit // 8)
        for key, score_fn in anchors:
            candidates = rows
            if key == "mechanism":
                candidates = [row for row in rows if row["detail"].get("mechanism_available")]
            elif key == "matchup":
                candidates = [
                    row for row in rows
                    if float(row["detail"].get("matchup_coverage", 0.0) or 0.0) > 0.0
                ]
            elif key == "draft_stats":
                candidates = [
                    row for row in rows
                    if float(row["detail"].get("coverage", 0.0) or 0.0) > 0.0
                ]
            for row in sorted(candidates, key=score_fn, reverse=True)[:anchor_count]:
                if len(selected) >= limit:
                    return selected[:limit]
                if row["slug"] in selected_slugs:
                    continue
                selected.append(row["slug"])
                selected_slugs.add(row["slug"])
                if len(selected) >= limit:
                    return selected
        for row in sorted(rows, key=lambda item: item["quick_score"], reverse=True):
            if len(selected) >= limit:
                break
            if row["slug"] in selected_slugs:
                continue
            selected.append(row["slug"])
            selected_slugs.add(row["slug"])
            if len(selected) >= limit:
                break
        return selected

    def _score_actions_with_unknowns(
        self,
        my_picks,
        enemy_picks,
        actions,
        turn,
        first_pick_side,
        batch_size=1024,
        action_type=None,
        current_side="me",
    ):
        actions = list(actions or [])
        if not actions:
            return []

        known_indices = [
            index
            for index, action in enumerate(actions)
            if all(slug in self._known_model_slugs for slug in action)
        ]
        center = float(self.meta.get("relation_score_center", 0.0))
        scores = [center for _action in actions]
        if known_indices:
            known_actions = [actions[index] for index in known_indices]
            known_scores = self._score_actions(
                my_picks=my_picks,
                enemy_picks=enemy_picks,
                actions=known_actions,
                turn=turn,
                first_pick_side=first_pick_side,
                batch_size=batch_size,
                action_type=action_type,
                current_side=current_side,
            )
            for index, score in zip(known_indices, known_scores):
                scores[index] = score
        return scores

    @torch.inference_mode()
    def _score_actions(
        self,
        my_picks,
        enemy_picks,
        actions,
        turn,
        first_pick_side,
        batch_size=1024,
        action_type=None,
        current_side="me",
    ):
        my_ids, my_mask = encode_pick_list(my_picks, self.vocab, self._to_slug)
        enemy_ids, enemy_mask = encode_pick_list(enemy_picks, self.vocab, self._to_slug)
        first_pick_value = 1.0 if normalize_name(first_pick_side) == "me" else 0.0
        current_side_value = 1.0 if normalize_name(current_side) == "me" else 0.0
        my_ids_tensor = torch.tensor([my_ids], dtype=torch.long, device=self.device)
        my_mask_tensor = torch.tensor([my_mask], dtype=torch.float32, device=self.device)
        enemy_ids_tensor = torch.tensor([enemy_ids], dtype=torch.long, device=self.device)
        enemy_mask_tensor = torch.tensor([enemy_mask], dtype=torch.float32, device=self.device)
        turn_tensor = torch.tensor([float(turn)], dtype=torch.float32, device=self.device)
        first_pick_tensor = torch.tensor([first_pick_value], dtype=torch.float32, device=self.device)
        if self.model_version >= 2:
            state_cache = self.model.encode_state(
                my_picks=my_ids_tensor,
                my_mask=my_mask_tensor,
                enemy_picks=enemy_ids_tensor,
                enemy_mask=enemy_mask_tensor,
                turn=turn_tensor,
                first_pick_side=first_pick_tensor,
                current_side=torch.tensor([current_side_value], dtype=torch.float32, device=self.device),
            )
        else:
            state_cache = self.model.encode_state(
                my_ids_tensor,
                my_mask_tensor,
                enemy_ids_tensor,
                enemy_mask_tensor,
                turn_tensor,
                first_pick_tensor,
            )
        scores = []
        for start in range(0, len(actions), batch_size):
            chunk = actions[start : start + batch_size]
            size = len(chunk)
            if action_type is None:
                action_types = [1 if len(action) == 2 else 0 for action in chunk]
            else:
                action_types = [int(action_type)] * size
            action_mask = [[1.0, 1.0 if action_types[idx] == 1 else 0.0] for idx, _action in enumerate(chunk)]
            c1 = [self.vocab.get_id(action[0]) for action in chunk]
            c2 = [self.vocab.get_id(action[1]) if len(action) == 2 else self.pad_id for action in chunk]
            score = self.model.score_from_state(
                state_cache,
                candidate_1=torch.tensor(c1, dtype=torch.long, device=self.device),
                candidate_2=torch.tensor(c2, dtype=torch.long, device=self.device),
                action_mask=torch.tensor(action_mask, dtype=torch.float32, device=self.device),
                action_type=torch.tensor(action_types, dtype=torch.long, device=self.device),
            )
            scores.extend(float(x) for x in score.detach().cpu().tolist())
        return scores

    def recommend_pick_plans(
        self,
        my_picks,
        enemy_picks,
        pick_count=None,
        turn=None,
        first_pick_side="me",
        candidate_pool=None,
        my_box=None,
        use_my_box_for_reco=True,
        top_k=6,
        max_pair_candidates=400,
        preserve_evidence=False,
    ):
        if turn is None:
            turn = len(my_picks or []) + len(enemy_picks or []) + 1
        if pick_count is None:
            pick_count = infer_pick_count(
                [self._to_slug(x) for x in (my_picks or [])],
                [self._to_slug(x) for x in (enemy_picks or [])],
                first_pick_side,
            )
        pick_count = int(pick_count)
        allowed = self._allowed_candidates(my_picks, enemy_picks, candidate_pool, my_box, use_my_box_for_reco)
        if pick_count <= 0 or not allowed:
            return []

        quick_score_by_slug = {}
        synergy_pairs = []
        if pick_count == 1:
            actions = [(slug,) for slug in allowed]
        else:
            pair_candidate_limit = int(max_pair_candidates or 0)
            if pair_candidate_limit > 0:
                synergy_pairs = self.lucksack_stats.candidate_synergy_pairs(
                    allowed,
                    limit=max(8, int(top_k) * 4),
                )
                quick_raw_scores = self._score_actions_with_unknowns(
                    my_picks,
                    enemy_picks,
                    [(slug,) for slug in allowed],
                    turn,
                    first_pick_side,
                )
                quick_relation_scores = self._normalize_scores(quick_raw_scores)
                quick_scores = []
                quick_details = []
                for slug, quick_relation in zip(allowed, quick_relation_scores):
                    stats_detail = self.lucksack_stats.score_action(
                        my_picks=my_picks,
                        enemy_picks=enemy_picks,
                        action=(slug,),
                        turn=turn,
                    )
                    quick_score, _stats_score, _mechanism_score, _mechanism_available, _matchup_score, _matchup_available = (
                        self._blend_relation_stats(
                            quick_relation,
                            {
                                **stats_detail,
                                "_relation_model_known": slug in self._known_model_slugs,
                            },
                        )
                    )
                    quick_scores.append(quick_score)
                    quick_details.append(stats_detail)
                quick_score_by_slug = dict(zip(allowed, quick_scores))
            if pair_candidate_limit > 0 and len(allowed) > pair_candidate_limit:
                allowed = self._select_candidate_diverse(
                    allowed,
                    quick_scores,
                    quick_details,
                    pair_candidate_limit,
                    synergy_pairs=synergy_pairs,
                )
            actions = list(combinations(allowed, 2))
        if not actions:
            return []

        relation_scores = self._score_actions_with_unknowns(
            my_picks,
            enemy_picks,
            actions,
            turn,
            first_pick_side,
        )
        normalized_relation = self._normalize_scores(relation_scores)
        if pick_count == 2 and len(actions) > max(64, int(top_k) * 8):
            score_limit = min(len(actions), max(96, int(top_k) * 16))
            pruning_scores = []
            for action, norm_score in zip(actions, normalized_relation):
                singleton_score = sum(
                    quick_score_by_slug.get(slug, 0.5) for slug in action
                ) / len(action)
                pruning_scores.append(0.75 * norm_score + 0.25 * singleton_score)
            selected_indices = sorted(
                range(len(actions)),
                key=lambda index: pruning_scores[index],
                reverse=True,
            )[:score_limit]
            actions = [actions[index] for index in selected_indices]
            relation_scores = [relation_scores[index] for index in selected_indices]
            normalized_relation = [normalized_relation[index] for index in selected_indices]

        policy_probabilities = self._policy_probabilities(relation_scores)

        plans = []
        action_type = "pair" if pick_count == 2 else "single"
        synergy_pair_keys = {
            frozenset(pair.get("slugs", []))
            for pair in synergy_pairs
        }
        for action, raw_score, norm_score, policy_prob in zip(
            actions, relation_scores, normalized_relation, policy_probabilities
        ):
            stats_detail = self.lucksack_stats.score_action(
                my_picks=my_picks,
                enemy_picks=enemy_picks,
                action=action,
                turn=turn,
            )
            relation_model_known = all(slug in self._known_model_slugs for slug in action)
            (
                combined_score,
                stats_score,
                mechanism_score,
                mechanism_available,
                matchup_score,
                matchup_available,
            ) = self._blend_relation_stats(
                norm_score,
                {
                    **stats_detail,
                    "_relation_model_known": relation_model_known,
                },
            )
            personal = None
            personal_weight = 0.0
            exploration_bonus = 0.0
            personal_reasons = []
            final_score = float(combined_score)
            if self.personal_history_model is not None:
                action_display = [self._to_display_name(slug) for slug in action]
                personal = self.personal_history_model.score_pick_action(
                    my_picks=my_picks,
                    enemy_picks=enemy_picks,
                    action=action_display,
                )
                final_score, personal_weight, exploration_bonus, personal_reasons = self._apply_personal_blend(combined_score, personal)
            plans.append(
                {
                    "plan": [self._to_display_name(slug) for slug in action],
                    "plan_score": float(final_score),
                    "estimated_win_rate": 0.0,
                    "source": "relation",
                    "reasons": list(stats_detail.get("mechanism_reasons") or []),
                    "detail": {
                        "relation_score": float(raw_score),
                        "policy_prob": float(policy_prob),
                        "pair_prob": float(policy_prob) if pick_count == 2 else None,
                        "normalized_relation_score": float(norm_score),
                        "draft_stats_score": stats_score,
                        "draft_stats_solo_score": float(stats_detail.get("solo_score", 0.5)),
                        "draft_stats_synergy_score": float(stats_detail.get("synergy_score", 0.5)),
                        "draft_position_score": float(stats_detail.get("draft_position_score", 0.5)),
                        "draft_position_coverage": float(stats_detail.get("draft_position_coverage", 0.0)),
                        "rta_tier_score": float(stats_detail.get("rta_tier_score", 0.5)),
                        "rta_tier_coverage": float(stats_detail.get("rta_tier_coverage", 0.0)),
                        "swrt_solo_score": float(stats_detail.get("swrt_solo_score", 0.5)),
                        "swrt_solo_coverage": float(
                            stats_detail.get("swrt_solo_coverage", 0.0)
                        ),
                        "matchup_score": matchup_score,
                        "matchup_coverage": float(
                            stats_detail.get("matchup_coverage", 0.0)
                        ),
                        "matchup_weight": float(stats_detail.get("matchup_weight", 0.0)),
                        "matchup_available": matchup_available,
                        "draft_stats_coverage": float(stats_detail.get("coverage", 0.0)),
                        "draft_stats_data_coverage": float(
                            stats_detail.get("data_coverage", 0.0)
                        ),
                        "summary_first_pick_win_rate": float(
                            stats_detail.get("summary_first_pick_win_rate", 0.5)
                        ),
                        "summary_second_pick_win_rate": float(
                            stats_detail.get("summary_second_pick_win_rate", 0.5)
                        ),
                        "summary_total_battles": int(
                            stats_detail.get("summary_total_battles", 0)
                        ),
                        "stats_rank_group": stats_detail.get("rank_group"),
                        "stats_season": stats_detail.get("season", {}),
                        "stats_patch": stats_detail.get("patch", {}),
                        "draft_stats_source": stats_detail.get("source", "unavailable"),
                        "draft_stats_evidence": stats_detail.get("evidence", []),
                        "mechanism_score": mechanism_score,
                        "mechanism_available": mechanism_available,
                        "mechanism_coverage": float(stats_detail.get("mechanism_coverage", 0.0)),
                        "mechanism_coverage_before": float(
                            stats_detail.get("mechanism_coverage_before", 0.0)
                        ),
                        "mechanism_coverage_gain": float(
                            stats_detail.get("mechanism_coverage_gain", 0.0)
                        ),
                        "mechanism_threats": stats_detail.get("mechanism_threats", []),
                        "mechanism_action_tags": stats_detail.get("mechanism_action_tags", []),
                        "mechanism_reasons": stats_detail.get("mechanism_reasons", []),
                        "combined_relation_stats_score": float(combined_score),
                        "relation_model_known": relation_model_known,
                        "relation_model_confidence": 1.0 if relation_model_known else 0.15,
                        "evidence_confidence": float(
                            stats_detail.get("evidence_confidence", stats_detail.get("data_coverage", 0.0))
                        ),
                        "evidence_uncertainty": float(
                            stats_detail.get("evidence_uncertainty", 1.0)
                        ),
                        "sample_count": int(stats_detail.get("sample_count", 0)),
                        "personal_score": None if personal is None else float(personal.get("personal_score", 0.5)),
                        "personal_confidence": None if personal is None else float(personal.get("confidence", 0.0)),
                        "personal_sample_count": None if personal is None else float(personal.get("sample_count", 0.0)),
                        "personal_raw_sample_count": None if personal is None else float(personal.get("raw_sample_count", personal.get("sample_count", 0.0))),
                        "personal_weight": float(personal_weight),
                        "personal_exploration_bonus": float(exploration_bonus),
                        "personal_reasons": personal_reasons,
                        "final_score": float(final_score),
                        "action_type": action_type,
                        "pair_candidate_anchor": (
                            frozenset(action) in synergy_pair_keys
                            if pick_count == 2
                            else False
                        ),
                        "pair_candidate_anchor_count": len(synergy_pairs),
                    },
                }
            )
        if preserve_evidence:
            plans = self._select_evidence_diverse_plans(plans, top_k)
        plans.sort(key=lambda row: row["plan_score"], reverse=True)
        return plans[: int(top_k)]

    def recommend_enemy_next_picks(
        self,
        my_picks,
        enemy_picks,
        first_pick_side,
        top_k=5,
        candidate_pool=None,
        max_pair_candidates=400,
        preserve_evidence=False,
        pick_count=None,
        **_unused,
    ):
        turn = len(my_picks or []) + len(enemy_picks or []) + 1
        swapped_first_pick_side = self._swap_first_pick_side(first_pick_side)
        if pick_count is None:
            pick_count = infer_pick_count_for_side(
                my_picks,
                enemy_picks,
                first_pick_side,
                "enemy",
            )
        plans = self.recommend_pick_plans(
            my_picks=enemy_picks,
            enemy_picks=my_picks,
            pick_count=pick_count,
            turn=turn,
            first_pick_side=swapped_first_pick_side,
            candidate_pool=candidate_pool,
            my_box=None,
            use_my_box_for_reco=False,
            top_k=top_k,
            max_pair_candidates=max_pair_candidates,
            preserve_evidence=preserve_evidence,
        )
        predictions = []
        for plan in plans:
            monsters = [monster for monster in plan.get("plan", []) if monster]
            monster = monsters[0] if monsters else ""
            detail = plan.get("detail", {}) or {}
            predictions.append(
                {
                    "monster": monster,
                    "display_name": " + ".join(monsters) if monsters else monster,
                    "plan": monsters,
                    "monsters": monsters,
                    "monster_slug": self._to_slug(monster),
                    "image_name": monster,
                    "policy_prob": None,
                    "completion_prob": None,
                    "final_score": round(float(plan.get("plan_score", 0.0)), 6),
                    "estimated_win_rate": 0.0,
                    "plan_archetype": None,
                    "adaptation_score": None,
                    "reasons": [
                        "Relation model score from the enemy-side draft perspective.",
                        *list(detail.get("mechanism_reasons") or []),
                    ],
                    "detail": {
                        "relation_score": detail.get("relation_score"),
                        "normalized_relation_score": detail.get("normalized_relation_score"),
                        "policy_prob": detail.get("policy_prob"),
                        "pair_prob": detail.get("pair_prob"),
                        "draft_stats_score": detail.get("draft_stats_score"),
                        "draft_stats_solo_score": detail.get("draft_stats_solo_score"),
                        "draft_stats_synergy_score": detail.get("draft_stats_synergy_score"),
                        "draft_position_score": detail.get("draft_position_score"),
                        "draft_position_coverage": detail.get("draft_position_coverage"),
                        "rta_tier_score": detail.get("rta_tier_score"),
                        "rta_tier_coverage": detail.get("rta_tier_coverage"),
                        "swrt_solo_score": detail.get("swrt_solo_score"),
                        "swrt_solo_coverage": detail.get("swrt_solo_coverage"),
                        "matchup_score": detail.get("matchup_score"),
                        "matchup_coverage": detail.get("matchup_coverage"),
                        "matchup_weight": detail.get("matchup_weight"),
                        "matchup_available": detail.get("matchup_available"),
                        "draft_stats_coverage": detail.get("draft_stats_coverage"),
                        "draft_stats_data_coverage": detail.get("draft_stats_data_coverage"),
                        "summary_first_pick_win_rate": detail.get("summary_first_pick_win_rate"),
                        "summary_second_pick_win_rate": detail.get("summary_second_pick_win_rate"),
                        "summary_total_battles": detail.get("summary_total_battles"),
                        "stats_rank_group": detail.get("stats_rank_group"),
                        "stats_season": detail.get("stats_season", {}),
                        "stats_patch": detail.get("stats_patch", {}),
                        "draft_stats_source": detail.get("draft_stats_source"),
                        "draft_stats_evidence": detail.get("draft_stats_evidence", []),
                        "mechanism_score": detail.get("mechanism_score"),
                        "mechanism_available": detail.get("mechanism_available"),
                        "mechanism_coverage": detail.get("mechanism_coverage"),
                        "mechanism_coverage_before": detail.get("mechanism_coverage_before"),
                        "mechanism_coverage_gain": detail.get("mechanism_coverage_gain"),
                        "mechanism_threats": detail.get("mechanism_threats", []),
                        "mechanism_action_tags": detail.get("mechanism_action_tags", []),
                        "mechanism_reasons": detail.get("mechanism_reasons", []),
                        "relation_model_known": detail.get("relation_model_known"),
                        "relation_model_confidence": detail.get("relation_model_confidence"),
                        "evidence_confidence": detail.get("evidence_confidence"),
                        "evidence_uncertainty": detail.get("evidence_uncertainty"),
                        "sample_count": detail.get("sample_count"),
                        "action_type": detail.get("action_type", "single"),
                        "pick_count": int(pick_count),
                        "source": "relation",
                    },
                }
            )
        return predictions

    def recommend_ban_plans(
        self,
        my_picks,
        enemy_picks,
        top_k=5,
        first_pick_side="me",
    ):
        my_slugs = [self._to_slug(x) for x in (my_picks or []) if x]
        enemy_slugs = [self._to_slug(x) for x in (enemy_picks or []) if x]
        enemy_slugs = list(dict.fromkeys(slug for slug in enemy_slugs if self.vocab.get_id(slug) > 1))
        if len(my_slugs) != 5 or len(enemy_slugs) != 5:
            return []

        actions = [(slug,) for slug in enemy_slugs]
        relation_scores = self._score_actions(
            my_picks=my_slugs,
            enemy_picks=enemy_slugs,
            actions=actions,
            turn=10,
            first_pick_side=first_pick_side,
            action_type=2,
        )
        normalized_relation = self._normalize_scores(relation_scores)

        plans = []
        for slug, raw_score, norm_score in zip(enemy_slugs, relation_scores, normalized_relation):
            display = self._to_display_name(slug)
            personal = None
            personal_weight = 0.0
            exploration_bonus = 0.0
            personal_reasons = []
            final_score = float(norm_score)
            if self.personal_history_model is not None:
                personal = self.personal_history_model.score_ban_action(
                    my_picks=my_picks,
                    enemy_picks=enemy_picks,
                    ban_candidate=display,
                )
                final_score, personal_weight, exploration_bonus, personal_reasons = self._apply_personal_blend(norm_score, personal)
            plans.append(
                {
                    "ban": display,
                    "monster": display,
                    "plan": [display],
                    "plan_score": float(final_score),
                    "estimated_win_rate": 0.0,
                    "source": "relation_ban",
                    "detail": {
                        "ban_score": float(raw_score),
                        "relation_score": float(raw_score),
                        "normalized_relation_score": float(norm_score),
                        "personal_score": None if personal is None else float(personal.get("personal_score", 0.5)),
                        "personal_confidence": None if personal is None else float(personal.get("confidence", 0.0)),
                        "personal_sample_count": None if personal is None else float(personal.get("sample_count", 0.0)),
                        "personal_raw_sample_count": None if personal is None else float(personal.get("raw_sample_count", personal.get("sample_count", 0.0))),
                        "personal_weight": float(personal_weight),
                        "personal_exploration_bonus": float(exploration_bonus),
                        "personal_reasons": personal_reasons,
                        "final_score": float(final_score),
                        "reason": "relation ban priority",
                    },
                }
            )
        plans.sort(key=lambda row: row["plan_score"], reverse=True)
        return plans[: int(top_k)]

    def apply_personal_ban_adjustment(self, plans, my_picks, enemy_picks, relation_score_key="plan_score"):
        if self.personal_history_model is None or not plans:
            return plans
        relation_scores = [float(plan.get(relation_score_key, plan.get("score", 0.0)) or 0.0) for plan in plans]
        normalized_relation = self._normalize_scores(relation_scores)
        adjusted = []
        for plan, norm_score in zip(plans, normalized_relation):
            item = dict(plan)
            detail = dict(item.get("detail", {}) or {})
            ban_candidate = item.get("ban") or item.get("monster") or (item.get("plan") or [""])[0]
            personal = self.personal_history_model.score_ban_action(
                my_picks=my_picks,
                enemy_picks=enemy_picks,
                ban_candidate=ban_candidate,
            )
            final_score, personal_weight, exploration_bonus, personal_reasons = self._apply_personal_blend(norm_score, personal)
            item["plan_score"] = float(final_score)
            detail.update(
                {
                    "normalized_relation_score": float(norm_score),
                    "personal_score": float(personal.get("personal_score", 0.5)),
                    "personal_confidence": float(personal.get("confidence", 0.0)),
                    "personal_sample_count": float(personal.get("sample_count", 0.0)),
                    "personal_raw_sample_count": float(personal.get("raw_sample_count", personal.get("sample_count", 0.0))),
                    "personal_weight": float(personal_weight),
                    "personal_exploration_bonus": float(exploration_bonus),
                    "personal_reasons": personal_reasons,
                    "final_score": float(final_score),
                }
            )
            item["detail"] = detail
            adjusted.append(item)
        adjusted.sort(key=lambda row: row.get("plan_score", 0.0), reverse=True)
        return adjusted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test-ban", action="store_true")
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()
    if not args.test and not args.test_ban:
        parser.print_help()
        return
    v2_ready = all(path.exists() for path in (DEFAULT_MODEL_PATH, DEFAULT_META_PATH, DEFAULT_VOCAB_PATH))
    legacy_ready = all(path.exists() for path in (LEGACY_MODEL_PATH, LEGACY_META_PATH, LEGACY_VOCAB_PATH))
    if not v2_ready and not legacy_ready:
        print("relation model bundle is missing; run python -m training.relation first")
        for path in (DEFAULT_MODEL_PATH, DEFAULT_META_PATH, DEFAULT_VOCAB_PATH):
            if not path.exists():
                print(f"missing: {path}")
        return
    recommender = BPRelationRecommender()
    if args.test_ban:
        slugs = valid_vocab_slugs(recommender.vocab)[:10]
        my_picks = [recommender._to_display_name(slug) for slug in slugs[:5]]
        enemy_picks = [recommender._to_display_name(slug) for slug in slugs[5:10]]
        plans = recommender.recommend_ban_plans(my_picks=my_picks, enemy_picks=enemy_picks, top_k=args.top_k)
        print("my_picks:", my_picks)
        print("enemy_picks:", enemy_picks)
        print("top ban recommendations:")
        for plan in plans:
            detail = plan["detail"]
            print(
                f"ban={plan['ban']} ban_score={detail['ban_score']:.4f} "
            )
        if len(enemy_picks) >= 2:
            positive = enemy_picks[0]
            negative = enemy_picks[1]
            scores = recommender._score_actions(
                my_picks=my_picks,
                enemy_picks=enemy_picks,
                actions=[(recommender._to_slug(positive),), (recommender._to_slug(negative),)],
                turn=10,
                first_pick_side="me",
                action_type=2,
            )
            print("sanity check:")
            print("positive_ban_candidate:", positive)
            print("negative_ban_candidate:", negative)
            print("positive_score:", scores[0])
            print("negative_score:", scores[1])
            print("positive > negative ?", scores[0] > scores[1])
        return

    my_picks = []
    enemy_picks = []
    pick_count = 1
    plans = recommender.recommend_pick_plans(
        my_picks=my_picks,
        enemy_picks=enemy_picks,
        pick_count=pick_count,
        turn=1,
        first_pick_side="me",
        use_my_box_for_reco=False,
        top_k=args.top_k,
    )
    print("my_picks:", my_picks)
    print("enemy_picks:", enemy_picks)
    print("pick_count:", pick_count)
    print("top recommendations:")
    for plan in plans:
        detail = plan["detail"]
        print(
            f"{detail['action_type']} action={plan['plan']} "
            f"relation_score={detail['relation_score']:.4f} "
            f"policy_prob={detail['policy_prob']} pair_prob={detail['pair_prob']} "
            f"final_score={detail['final_score']:.4f}"
        )


if __name__ == "__main__":
    main()
