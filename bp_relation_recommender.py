import argparse
import json
from itertools import combinations

import torch

from bp_relation_common import BPDraftRelationNet, MonsterVocab, encode_pick_list, infer_pick_count, valid_vocab_slugs
from bp_transformer_common import PAD_TOKEN, PROJECT_ROOT, UNK_TOKEN, normalize_name, slugify_fallback
from bp_personal_history_model import BPPersonalHistoryModel

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "bp_relation_net.pt"
DEFAULT_META_PATH = PROJECT_ROOT / "models" / "bp_relation_meta.json"
DEFAULT_VOCAB_PATH = PROJECT_ROOT / "models" / "bp_relation_vocab.json"


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
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.personal_history_model = personal_history_model
        self.max_personal_weight = min(0.30, max(0.0, float(max_personal_weight)))
        self.exploration_bonus_enabled = bool(exploration_bonus_enabled)
        self.low_exposure_threshold = float(low_exposure_threshold)
        self.high_relation_threshold = float(high_relation_threshold)
        self.exploration_bonus = float(exploration_bonus)
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
        self.model = BPDraftRelationNet(
            vocab_size=len(self.vocab),
            embed_dim=int(self.meta.get("embed_dim", 128)),
            state_dim=int(self.meta.get("state_dim", 256)),
            action_dim=int(self.meta.get("action_dim", 256)),
            hidden_dim=int(self.meta.get("hidden_dim", 256)),
            dropout=float(self.meta.get("dropout", 0.15)),
            pad_idx=self.pad_id,
        )
        try:
            state = torch.load(model_path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()

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
        for key in (normalized, fallback, sanitized_fallback):
            resolved = self.name_slug_aliases.get(key)
            if resolved:
                return normalize_name(resolved)
        return sanitized_fallback or fallback or normalized

    def _to_display_name(self, slug):
        slug = normalize_name(slug)
        return self.display_name_map.get(slug, slug)

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
        return "enemy" if normalize_name(first_pick_side) == "me" else "me"

    def _allowed_candidates(self, my_picks, enemy_picks, candidate_pool=None, my_box=None, use_my_box_for_reco=True):
        occupied = {self._to_slug(x) for x in (my_picks or []) + (enemy_picks or []) if x}
        if candidate_pool:
            allowed = {self._to_slug(x) for x in candidate_pool if x}
        else:
            allowed = set(valid_vocab_slugs(self.vocab))
        blocked = {normalize_name(PAD_TOKEN), normalize_name(UNK_TOKEN), "", *occupied}
        allowed = {slug for slug in allowed if slug not in blocked and self.vocab.get_id(slug) > 1}
        if use_my_box_for_reco:
            owned = self._extract_owned_set(my_box)
            if not owned:
                return []
            allowed &= owned
        return sorted(allowed, key=lambda slug: self._to_display_name(slug))

    def _normalize_scores(self, raw_scores):
        if not raw_scores:
            return []
        lo = min(raw_scores)
        hi = max(raw_scores)
        if hi - lo < 1e-8:
            return [0.5 for _ in raw_scores]
        return [(score - lo) / (hi - lo) for score in raw_scores]

    @torch.no_grad()
    def _score_actions(self, my_picks, enemy_picks, actions, turn, first_pick_side, batch_size=1024, action_type=None):
        my_ids, my_mask = encode_pick_list(my_picks, self.vocab, self._to_slug)
        enemy_ids, enemy_mask = encode_pick_list(enemy_picks, self.vocab, self._to_slug)
        first_pick_value = 1.0 if normalize_name(first_pick_side) == "me" else 0.0
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
            score = self.model(
                torch.tensor([my_ids] * size, dtype=torch.long, device=self.device),
                torch.tensor([my_mask] * size, dtype=torch.float32, device=self.device),
                torch.tensor([enemy_ids] * size, dtype=torch.long, device=self.device),
                torch.tensor([enemy_mask] * size, dtype=torch.float32, device=self.device),
                torch.tensor(c1, dtype=torch.long, device=self.device),
                torch.tensor(c2, dtype=torch.long, device=self.device),
                torch.tensor(action_mask, dtype=torch.float32, device=self.device),
                torch.full((size,), float(turn), dtype=torch.float32, device=self.device),
                torch.full((size,), first_pick_value, dtype=torch.float32, device=self.device),
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
        max_pair_candidates=220,
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

        if pick_count == 1:
            actions = [(slug,) for slug in allowed]
        else:
            if len(allowed) > max_pair_candidates:
                quick_scores = self._score_actions(my_picks, enemy_picks, [(slug,) for slug in allowed], turn, first_pick_side)
                ordered = [slug for _, slug in sorted(zip(quick_scores, allowed), reverse=True)]
                allowed = ordered[:max_pair_candidates]
            actions = list(combinations(allowed, 2))
        if not actions:
            return []

        relation_scores = self._score_actions(my_picks, enemy_picks, actions, turn, first_pick_side)
        normalized_relation = self._normalize_scores(relation_scores)

        plans = []
        action_type = "pair" if pick_count == 2 else "single"
        for action, raw_score, norm_score in zip(actions, relation_scores, normalized_relation):
            personal = None
            personal_weight = 0.0
            exploration_bonus = 0.0
            personal_reasons = []
            final_score = float(norm_score)
            if self.personal_history_model is not None:
                action_display = [self._to_display_name(slug) for slug in action]
                personal = self.personal_history_model.score_pick_action(
                    my_picks=my_picks,
                    enemy_picks=enemy_picks,
                    action=action_display,
                )
                final_score, personal_weight, exploration_bonus, personal_reasons = self._apply_personal_blend(norm_score, personal)
            plans.append(
                {
                    "plan": [self._to_display_name(slug) for slug in action],
                    "plan_score": float(final_score),
                    "estimated_win_rate": 0.0,
                    "source": "relation",
                    "detail": {
                        "relation_score": float(raw_score),
                        "policy_prob": None,
                        "pair_prob": None,
                        "normalized_relation_score": float(norm_score),
                        "personal_score": None if personal is None else float(personal.get("personal_score", 0.5)),
                        "personal_confidence": None if personal is None else float(personal.get("confidence", 0.0)),
                        "personal_sample_count": None if personal is None else float(personal.get("sample_count", 0.0)),
                        "personal_raw_sample_count": None if personal is None else float(personal.get("raw_sample_count", personal.get("sample_count", 0.0))),
                        "personal_weight": float(personal_weight),
                        "personal_exploration_bonus": float(exploration_bonus),
                        "personal_reasons": personal_reasons,
                        "final_score": float(final_score),
                        "action_type": action_type,
                    },
                }
            )
        plans.sort(key=lambda row: row["plan_score"], reverse=True)
        return plans[: int(top_k)]

    def recommend_enemy_next_picks(
        self,
        my_picks,
        enemy_picks,
        first_pick_side,
        top_k=5,
        candidate_pool=None,
        **_unused,
    ):
        turn = len(my_picks or []) + len(enemy_picks or []) + 1
        plans = self.recommend_pick_plans(
            my_picks=enemy_picks,
            enemy_picks=my_picks,
            pick_count=1,
            turn=turn,
            first_pick_side=self._swap_first_pick_side(first_pick_side),
            candidate_pool=candidate_pool,
            my_box=None,
            use_my_box_for_reco=False,
            top_k=top_k,
        )
        predictions = []
        for plan in plans:
            monster = plan.get("plan", [""])[0]
            detail = plan.get("detail", {}) or {}
            predictions.append(
                {
                    "monster": monster,
                    "display_name": monster,
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
                    ],
                    "detail": {
                        "relation_score": detail.get("relation_score"),
                        "action_type": "single",
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
        old_ban_recommender=None,
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
            first_pick_side="me",
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
    missing = [path for path in (DEFAULT_MODEL_PATH, DEFAULT_META_PATH, DEFAULT_VOCAB_PATH) if not path.exists()]
    if missing:
        print("relation model bundle is missing; run python train_bp_relation_net.py first")
        for path in missing:
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
