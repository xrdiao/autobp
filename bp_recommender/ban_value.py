import json

import torch

from bp_core.lucksack_stats import BPLucksackStats, DEFAULT_STATS_PATH
from bp_core.probability_calibration import apply_probability_calibration, resolve_probability_calibration
from bp_core.transformer_common import PAD_TOKEN, PROJECT_ROOT, UNK_TOKEN, normalize_name, slugify_fallback
from training.draft_value import BPDraftValueNet as TransformerDraftValueNet

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "bp_draft_value_net.pt"
DEFAULT_META_PATH = PROJECT_ROOT / "models" / "bp_draft_value_meta.json"
DEFAULT_VOCAB_PATH = PROJECT_ROOT / "models" / "bp_draft_value_vocab.json"
BAN_VALUE_WEIGHT = 0.75
BAN_MECHANISM_WEIGHT = 0.25
BAN_STATS_WEIGHT = 0.10
BAN_WORST_RESPONSE_WEIGHT = 0.70


class MonsterVocab:
    def __init__(self, slug_to_id):
        self.slug_to_id = dict(slug_to_id)
        self.id_to_slug = {v: k for k, v in self.slug_to_id.items()}

    def __len__(self):
        return len(self.slug_to_id)

    def get_id(self, slug):
        slug = normalize_name(slug)
        return self.slug_to_id.get(slug, self.slug_to_id.get(UNK_TOKEN, 1))


class BPBanValueRecommender:
    def __init__(
        self,
        model_path=str(DEFAULT_MODEL_PATH),
        meta_path=str(DEFAULT_META_PATH),
        vocab_path=str(DEFAULT_VOCAB_PATH),
        device=None,
        external_stats_path=str(DEFAULT_STATS_PATH),
        stats_rank_group=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lucksack_stats = BPLucksackStats(
            stats_path=external_stats_path,
            rank_group=stats_rank_group,
        )

        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_dict = json.load(f)

        self.vocab = MonsterVocab(vocab_dict)
        blocked_tokens = {normalize_name(PAD_TOKEN), normalize_name(UNK_TOKEN)}
        self._known_model_slugs = {
            normalize_name(slug)
            for slug in self.vocab.slug_to_id
            if normalize_name(slug) not in blocked_tokens
        }
        self.display_name_map = self.meta.get("display_name_map", {})
        self.image_name_map = self.meta.get("image_name_map", {})
        self.prob_calibration = resolve_probability_calibration(self.meta, role="draft_value")
        self.prob_temperature = float(self.prob_calibration["temperature"])
        self.prob_prior_weight = float(self.prob_calibration["prior_weight"])
        self.prob_clip_min = float(self.prob_calibration["clip_min"])
        self.prob_clip_max = float(self.prob_calibration["clip_max"])

        model_type = str(self.meta.get("model_type", "")).lower()
        if "transformer" in model_type or self.meta.get("architecture") == "transformer_encoder":
            model_cls = TransformerDraftValueNet
            model_kwargs = {
                "vocab_size": len(self.vocab),
                "embed_dim": self.meta["embed_dim"],
                "hidden_dim": self.meta["hidden_dim"],
                "dropout": self.meta["dropout"],
                "pad_idx": self.vocab.get_id(PAD_TOKEN),
                "num_heads": int(self.meta.get("num_heads", 4)),
                "num_layers": int(self.meta.get("num_layers", 3)),
                "ffn_dim": int(self.meta.get("ffn_dim", max(self.meta["hidden_dim"], self.meta["embed_dim"] * 3))),
            }
        else:
            raise RuntimeError(
                f"Unsupported draft value model type: {self.meta.get('model_type')}. "
                "Retrain with python -m training.draft_value to produce the current transformer model."
            )

        self.model = model_cls(**model_kwargs)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def _to_slug(self, value):
        normalized = normalize_name(value)
        if not normalized:
            return ""
        external_slug = self.lucksack_stats.resolve_slug(value)
        return external_slug or slugify_fallback(value)

    def _to_display_name(self, slug):
        slug = normalize_name(slug)
        return (
            self.display_name_map.get(slug)
            or self.lucksack_stats.to_display_name(slug)
            or slug
        )

    def _to_image_name(self, slug):
        slug = normalize_name(slug)
        return self.image_name_map.get(slug, self._to_display_name(slug))

    def _encode_list(self, arr, max_len=5):
        ids = [self.vocab.get_id(self._to_slug(x)) for x in (arr or [])[:max_len]]
        mask = [1] * len(ids)
        while len(ids) < max_len:
            ids.append(self.vocab.get_id(PAD_TOKEN))
            mask.append(0)
        return ids, mask

    @torch.inference_mode()
    def evaluate_states(self, states, return_detail=False):
        states = list(states or [])
        if not states:
            return []

        encoded_rows = []
        for state in states:
            my_ids, my_mask = self._encode_list(state.get("my_picks"), 5)
            enemy_ids, enemy_mask = self._encode_list(state.get("enemy_picks"), 5)
            pad_id = self.vocab.get_id(PAD_TOKEN)
            my_ban_id = (
                self.vocab.get_id(self._to_slug(state["my_ban"]))
                if state.get("my_ban")
                else pad_id
            )
            enemy_ban_id = (
                self.vocab.get_id(self._to_slug(state["enemy_ban"]))
                if state.get("enemy_ban")
                else pad_id
            )
            my_leader_id = (
                self.vocab.get_id(self._to_slug(state["my_leader"]))
                if state.get("my_leader")
                else pad_id
            )
            enemy_leader_id = (
                self.vocab.get_id(self._to_slug(state["enemy_leader"]))
                if state.get("enemy_leader")
                else pad_id
            )
            encoded_rows.append(
                {
                    "my_ids": my_ids,
                    "my_mask": my_mask,
                    "enemy_ids": enemy_ids,
                    "enemy_mask": enemy_mask,
                    "my_ban_id": my_ban_id,
                    "enemy_ban_id": enemy_ban_id,
                    "my_leader_id": my_leader_id,
                    "enemy_leader_id": enemy_leader_id,
                }
            )

        batch = {
            "my_picks": torch.tensor(
                [row["my_ids"] for row in encoded_rows],
                dtype=torch.long,
                device=self.device,
            ),
            "my_mask": torch.tensor(
                [row["my_mask"] for row in encoded_rows],
                dtype=torch.float32,
                device=self.device,
            ),
            "enemy_picks": torch.tensor(
                [row["enemy_ids"] for row in encoded_rows],
                dtype=torch.long,
                device=self.device,
            ),
            "enemy_mask": torch.tensor(
                [row["enemy_mask"] for row in encoded_rows],
                dtype=torch.float32,
                device=self.device,
            ),
            "my_ban": torch.tensor(
                [row["my_ban_id"] for row in encoded_rows],
                dtype=torch.long,
                device=self.device,
            ),
            "enemy_ban": torch.tensor(
                [row["enemy_ban_id"] for row in encoded_rows],
                dtype=torch.long,
                device=self.device,
            ),
            "my_leader": torch.tensor(
                [row["my_leader_id"] for row in encoded_rows],
                dtype=torch.long,
                device=self.device,
            ),
            "enemy_leader": torch.tensor(
                [row["enemy_leader_id"] for row in encoded_rows],
                dtype=torch.long,
                device=self.device,
            ),
        }
        raw_probs = torch.sigmoid(self.model(**batch)).detach().cpu().tolist()
        calibrated_probs = [
            apply_probability_calibration(raw_prob, self.prob_calibration)
            for raw_prob in raw_probs
        ]
        if not return_detail:
            return calibrated_probs
        return [
            {
                "raw_win_rate": float(raw_prob),
                "calibrated_win_rate": float(calibrated_prob),
            }
            for raw_prob, calibrated_prob in zip(raw_probs, calibrated_probs)
        ]

    @torch.inference_mode()
    def evaluate_state(
        self,
        my_picks,
        enemy_picks,
        my_ban=None,
        enemy_ban=None,
        my_leader=None,
        enemy_leader=None,
        return_detail=False,
    ):
        return self.evaluate_states(
            [
                {
                    "my_picks": my_picks,
                    "enemy_picks": enemy_picks,
                    "my_ban": my_ban,
                    "enemy_ban": enemy_ban,
                    "my_leader": my_leader,
                    "enemy_leader": enemy_leader,
                }
            ],
            return_detail=return_detail,
        )[0]

    @torch.inference_mode()
    def recommend_bans(
        self,
        my_picks,
        enemy_picks,
        my_leader=None,
        enemy_leader=None,
        enemy_ban=None,
        top_k=5,
    ):
        candidates = [self._to_slug(x) for x in (enemy_picks or []) if x]
        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            return []

        # Do not invent an opponent Ban from our picks. The legacy model was
        # trained mostly with one recorded Ban side, so that synthetic input is
        # out of distribution and makes the before/after value misleading.
        response_bans = [self._to_slug(enemy_ban)] if enemy_ban else [None]
        response_bans = [ban for ban in response_bans if ban] or [None]
        base_states = [
            {
                "my_picks": my_picks,
                "enemy_picks": enemy_picks,
                "my_ban": None,
                "enemy_ban": response_ban,
                "my_leader": my_leader,
                "enemy_leader": enemy_leader,
            }
            for response_ban in response_bans
        ]
        target_states = [
            {
                "my_picks": my_picks,
                "enemy_picks": enemy_picks,
                "my_ban": target,
                "enemy_ban": response_ban,
                "my_leader": my_leader,
                "enemy_leader": enemy_leader,
            }
            for target in candidates
            for response_ban in response_bans
        ]
        evaluations = self.evaluate_states(
            base_states + target_states,
            return_detail=True,
        )
        base_evaluations = evaluations[: len(base_states)]
        target_evaluations = evaluations[len(base_states) :]
        base_wr_by_response = [
            float(item["calibrated_win_rate"]) for item in base_evaluations
        ]
        base_raw_wr_by_response = [
            float(item["raw_win_rate"]) for item in base_evaluations
        ]

        plans = []
        response_count = len(response_bans)
        for candidate_index, target in enumerate(candidates):
            start = candidate_index * response_count
            target_batch = target_evaluations[start : start + response_count]
            target_wrs = [
                float(item["calibrated_win_rate"]) for item in target_batch
            ]
            target_raw_wrs = [
                float(item["raw_win_rate"]) for item in target_batch
            ]
            worst_index = min(range(response_count), key=target_wrs.__getitem__)
            mean_target_wr = sum(target_wrs) / response_count
            mean_base_wr = sum(base_wr_by_response) / response_count
            wr = (
                BAN_WORST_RESPONSE_WEIGHT * target_wrs[worst_index]
                + (1.0 - BAN_WORST_RESPONSE_WEIGHT) * mean_target_wr
            )
            raw_wr = target_raw_wrs[worst_index]
            base_wr = (
                BAN_WORST_RESPONSE_WEIGHT * base_wr_by_response[worst_index]
                + (1.0 - BAN_WORST_RESPONSE_WEIGHT) * mean_base_wr
            )
            base_raw_wr = base_raw_wr_by_response[worst_index]
            gain = wr - base_wr
            raw_gain = raw_wr - base_raw_wr
            ban_stats = self.lucksack_stats.score_ban_action(
                my_picks=my_picks,
                enemy_picks=enemy_picks,
                target=target,
            )
            mechanism_score = float(ban_stats.get("mechanism_score", 0.5))
            mechanism_available = bool(ban_stats.get("mechanism_available"))
            stats_score = float(ban_stats.get("ban_priority_score", 0.5))
            stats_available = (
                float(ban_stats.get("target_solo_coverage", 0.0)) > 0.0
                or float(ban_stats.get("target_contest_coverage", 0.0)) > 0.0
            )
            weighted_score = BAN_VALUE_WEIGHT * wr
            weight_total = BAN_VALUE_WEIGHT
            source_parts = ["value_0.75"]
            if mechanism_available:
                weighted_score += BAN_MECHANISM_WEIGHT * mechanism_score
                weight_total += BAN_MECHANISM_WEIGHT
                source_parts.append("mechanism_0.25")
            if stats_available:
                weighted_score += BAN_STATS_WEIGHT * stats_score
                weight_total += BAN_STATS_WEIGHT
                source_parts.append("stats_0.10")
            decision_score = weighted_score / max(weight_total, 1e-6)
            ranking_source = "ban_robust_response_" + "_".join(source_parts)

            display = self._to_display_name(target)
            image_name = self._to_image_name(target)
            response_rows = [
                {
                    "enemy_ban": self._to_display_name(response_ban)
                    if response_ban
                    else None,
                    "enemy_ban_slug": response_ban,
                    "pre_ban_win_rate": round(base_wr_by_response[index], 6),
                    "post_ban_win_rate": round(target_wrs[index], 6),
                    "win_rate_gain": round(
                        target_wrs[index] - base_wr_by_response[index],
                        6,
                    ),
                }
                for index, response_ban in enumerate(response_bans)
            ]

            plans.append(
                {
                    "ban": display,
                    "ban_slug": target,
                    "plan": [display],
                    "plan_slugs": [target],
                    "image_names": [image_name],
                    "score": round(decision_score, 6),
                    "plan_score": round(decision_score, 6),
                    "estimated_win_rate": round(wr, 6),
                    "source": str(self.meta.get("model_type", "bp_draft_value_model")),
                    "summary": "Response-aware ban recommendation from full-draft value model",
                    "reasons": [
                        "Compares full draft win rate before and after banning the target",
                        "Includes the strongest opposing ban response",
                        "Ranks bans by learned value plus mechanism threat removal",
                        *list(ban_stats.get("mechanism_reasons") or []),
                    ],
                    "detail": {
                        "ban_target": display,
                        "decision_score": round(decision_score, 6),
                        "pre_ban_win_rate": round(base_wr, 6),
                        "post_ban_win_rate": round(wr, 6),
                        "win_rate_gain": round(gain, 6),
                        "raw_pre_ban_win_rate": round(base_raw_wr, 6),
                        "raw_post_ban_win_rate": round(raw_wr, 6),
                        "raw_win_rate_gain": round(raw_gain, 6),
                        "mean_pre_ban_win_rate": round(
                            sum(base_wr_by_response) / response_count,
                            6,
                        ),
                        "mean_post_ban_win_rate": round(
                            sum(target_wrs) / response_count,
                            6,
                        ),
                        "worst_pre_ban_win_rate": round(
                            base_wr_by_response[worst_index],
                            6,
                        ),
                        "worst_post_ban_win_rate": round(
                            target_wrs[worst_index],
                            6,
                        ),
                        "response_worst_weight": BAN_WORST_RESPONSE_WEIGHT,
                        "worst_enemy_ban": response_rows[worst_index]["enemy_ban"],
                        "worst_enemy_ban_slug": response_bans[worst_index],
                        "opponent_ban_responses": response_rows,
                        "opponent_response_count": response_count,
                        "response_aware": response_count > 1,
                        "mechanism_score": round(mechanism_score, 6),
                        "mechanism_available": mechanism_available,
                        "mechanism_threat_reduction": round(
                            float(ban_stats.get("mechanism_threat_reduction", 0.0)),
                            6,
                        ),
                        "mechanism_threats": ban_stats.get("mechanism_threats", []),
                        "mechanism_reasons": ban_stats.get("mechanism_reasons", []),
                        "target_solo_score": round(stats_score, 6),
                        "target_solo_coverage": round(
                            float(ban_stats.get("target_solo_coverage", 0.0)),
                            6,
                        ),
                        "target_contest_score": round(
                            float(ban_stats.get("target_contest_score", 0.5)),
                            6,
                        ),
                        "target_matchup_score": round(
                            float(ban_stats.get("target_matchup_score", 0.5)),
                            6,
                        ),
                        "target_matchup_coverage": round(
                            float(ban_stats.get("target_matchup_coverage", 0.0)),
                            6,
                        ),
                        "ban_priority_score": round(stats_score, 6),
                        "draft_stats_source": ban_stats.get("source", "unavailable"),
                        "draft_stats_evidence": ban_stats.get("evidence", []),
                        "relation_model_known": target in self._known_model_slugs,
                        "ranking_source": ranking_source,
                        "prob_temperature": round(self.prob_temperature, 6),
                        "prob_prior_weight": round(self.prob_prior_weight, 6),
                    },
                    "monster": display,
                }
            )

        plans.sort(
            key=lambda row: (
                float((row.get("detail") or {}).get("post_ban_win_rate", row.get("estimated_win_rate", 0.0))),
                float(row.get("plan_score", 0.0)),
            ),
            reverse=True,
        )
        return plans[:top_k]
