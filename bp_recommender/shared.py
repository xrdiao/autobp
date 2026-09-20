"""Inference and ranking adapter for the single Shared BP Core model."""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import torch

from bp_core.action_space import (
    DraftActionEvent,
    DraftTimeline,
    timeline_from_picks,
    transition_state,
    validate_timeline,
)
from bp_core.bp_network import BPSharedNetwork
from bp_core.lucksack_stats import BPLucksackStats, DEFAULT_STATS_PATH
from bp_core.bp_common import (
    MonsterVocab,
    infer_pick_count_for_side,
    side_for_order,
    valid_vocab_slugs,
)
from bp_core.names import (
    PAD_TOKEN,
    PROJECT_ROOT,
    UNK_TOKEN,
    normalize_name,
    slugify_name,
)

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "bp_shared.pt"
DEFAULT_META_PATH = PROJECT_ROOT / "models" / "bp_shared_meta.json"
DEFAULT_VOCAB_PATH = PROJECT_ROOT / "models" / "bp_shared_vocab.json"


class BPSharedRecommender:
    """Expose pick, opponent, pair, Ban and value heads through one API."""

    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        meta_path=DEFAULT_META_PATH,
        vocab_path=DEFAULT_VOCAB_PATH,
        device=None,
        external_stats_path=DEFAULT_STATS_PATH,
        stats_rank_group=None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        bundle = tuple(Path(path) for path in (model_path, meta_path, vocab_path))
        missing = [str(path) for path in bundle if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Shared BP Core 模型不完整："
                + ", ".join(missing)
                + "。请先运行 python -m training.shared_bp"
            )

        with open(meta_path, "r", encoding="utf-8") as file:
            self.meta = json.load(file)
        required_meta = {
            "architecture",
            "architecture_version",
            "embed_dim",
            "state_dim",
            "hidden_dim",
            "max_events",
            "slots_per_event",
            "num_heads",
            "num_layers",
            "ffn_dim",
            "dropout",
        }
        missing_meta = sorted(required_meta - self.meta.keys())
        if missing_meta:
            raise ValueError(f"Shared BP Core 元数据缺少字段: {', '.join(missing_meta)}")
        if self.meta["architecture"] != "shared_event_v1" or int(self.meta["architecture_version"]) != 1:
            raise ValueError("模型架构不是 shared_event_v1，请重新训练 Shared BP Core")
        with open(vocab_path, "r", encoding="utf-8") as file:
            self.vocab = MonsterVocab(json.load(file))

        self.display_name_map = {
            normalize_name(key): str(value).strip()
            for key, value in (self.meta.get("display_name_map") or {}).items()
            if key and value
        }
        self.name_slug_aliases = {
            normalize_name(key): normalize_name(value)
            for key, value in (self.meta.get("name_slug_aliases") or {}).items()
            if key and value
        }
        for slug, display_name in self.display_name_map.items():
            self.name_slug_aliases.setdefault(slug, slug)
            self.name_slug_aliases.setdefault(normalize_name(display_name), slug)
            self.name_slug_aliases.setdefault(slugify_name(display_name), slug)

        self.lucksack_stats = BPLucksackStats(
            stats_path=external_stats_path,
            rank_group=stats_rank_group,
        )
        self.pad_id = self.vocab.get_id(PAD_TOKEN)
        self.unk_id = self.vocab.get_id(UNK_TOKEN)
        self.external_slugs = frozenset(
            normalize_name(value)
            for value in self.lucksack_stats.candidate_slugs()
            if normalize_name(value)
        )
        self.model = BPSharedNetwork(
            vocab_size=len(self.vocab),
            embed_dim=int(self.meta["embed_dim"]),
            state_dim=int(self.meta["state_dim"]),
            hidden_dim=int(self.meta["hidden_dim"]),
            dropout=float(self.meta["dropout"]),
            pad_idx=self.pad_id,
            max_events=int(self.meta["max_events"]),
            slots_per_event=int(self.meta["slots_per_event"]),
            num_heads=int(self.meta["num_heads"]),
            num_layers=int(self.meta["num_layers"]),
            ffn_dim=int(self.meta["ffn_dim"]),
        )
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        self.known_slugs = frozenset(valid_vocab_slugs(self.vocab))

    def _to_slug(self, value):
        normalized = normalize_name(value)
        if not normalized:
            return ""
        generated = slugify_name(value).replace("[", "").replace("]", "")
        for key in (normalized, slugify_name(value), generated):
            resolved = self.name_slug_aliases.get(key)
            if resolved and normalize_name(resolved) in self.known_slugs:
                return normalize_name(resolved)
        resolved = normalize_name(self.lucksack_stats.resolve_slug(value))
        return resolved if resolved in self.known_slugs or resolved in self.external_slugs else ""

    def _vocab_id(self, value):
        """Map an external catalog monster to the model's unknown token."""
        try:
            return self.vocab.get_id(value)
        except KeyError:
            return self.unk_id

    def _to_display_name(self, slug):
        slug = normalize_name(slug)
        return self.display_name_map.get(slug) or self.lucksack_stats.to_display_name(slug) or slug

    def _owned_slugs(self, my_box):
        if isinstance(my_box, dict):
            values = [key for key, owned in my_box.items() if owned]
        else:
            values = []
            for item in my_box or ():
                if isinstance(item, str):
                    values.append(item)
                elif isinstance(item, dict) and item.get("owned", True):
                    value = item.get("name") or item.get("monster") or item.get("label")
                    if value:
                        values.append(value)
        return {self._to_slug(value) for value in values if value}

    def _allowed_candidates(
        self,
        my_picks,
        enemy_picks,
        candidate_pool=None,
        my_box=None,
        use_my_box_for_reco=False,
    ):
        occupied = {self._to_slug(value) for value in [*(my_picks or ()), *(enemy_picks or ())]}
        external_slugs = set(getattr(self, "external_slugs", ()))
        if candidate_pool is None:
            allowed = set(self.known_slugs) | external_slugs
        else:
            allowed = {self._to_slug(value) for value in candidate_pool if value}
            allowed &= set(self.known_slugs) | external_slugs
        allowed -= {"", normalize_name(PAD_TOKEN), normalize_name(UNK_TOKEN), *occupied}
        if use_my_box_for_reco:
            allowed &= self._owned_slugs(my_box)
        return sorted(allowed)

    @staticmethod
    def _canonical_action(action):
        if isinstance(action, str):
            action = (action,)
        values = tuple(dict.fromkeys(normalize_name(value) for value in action if value))
        return tuple(sorted(values)) if len(values) == 2 else values

    def _state(self, my_picks, enemy_picks, first_pick_side, turn, events, current_side):
        input_my = [self._to_slug(value) for value in my_picks or ()]
        input_enemy = [self._to_slug(value) for value in enemy_picks or ()]
        timeline = DraftTimeline(events)
        slug_events = DraftTimeline(
            tuple(
                DraftActionEvent(
                    side=event.side,
                    monsters=tuple(self._to_slug(value) for value in event.monsters),
                    action_type=event.action_type,
                    order=event.order,
                )
                for event in timeline.events
            )
        )
        validate_timeline(slug_events, first_pick_side)
        raw_my = list(slug_events.picks_for("me"))
        raw_enemy = list(slug_events.picks_for("enemy"))
        if raw_my != input_my or raw_enemy != input_enemy:
            raise ValueError("side pick lists do not match canonical events")
        expected_turn = slug_events.slots_used + 1
        actual_turn = int(turn)
        if actual_turn != expected_turn:
            raise ValueError("BP turn does not match the canonical event timeline")
        expected_side = side_for_order(actual_turn, normalize_name(first_pick_side))
        if actual_turn == 11:
            expected_side = "me"
        if expected_side is None:
            raise ValueError("first_pick_side and turn do not describe a legal BP state")
        if normalize_name(current_side) != expected_side:
            raise ValueError("current_side does not match the canonical BP turn")
        state = {
            "my_picks": raw_my,
            "enemy_picks": raw_enemy,
            "first_pick_side": normalize_name(first_pick_side),
            "turn": actual_turn,
            "events": slug_events.to_dict(),
        }
        state["current_side"] = expected_side
        return state

    def _slug_timeline(self, state):
        timeline = DraftTimeline(state["events"])
        return DraftTimeline(
            tuple(
                DraftActionEvent(
                    side=event.side,
                    monsters=tuple(self._to_slug(value) for value in event.monsters),
                    action_type=event.action_type,
                    order=event.order,
                )
                for event in timeline.events
            )
        )

    def _forward_state(self, state, pair_candidates=None):
        first_pick_side = 1.0 if normalize_name(state["first_pick_side"]) == "me" else 0.0
        current_side = 1.0 if normalize_name(state["current_side"]) == "me" else 0.0
        return self.model.forward_timeline(
            self._slug_timeline(state),
            self.vocab,
            turn=float(state["turn"]),
            first_pick_side=first_pick_side,
            current_side=current_side,
            pair_candidates=pair_candidates,
        )

    @torch.inference_mode()
    def _score_actions(self, state, actions, head="pick", batch_size=1024):
        actions = [self._canonical_action(action) for action in actions]
        if not actions:
            return []
        if any(len(action) not in {1, 2} for action in actions):
            raise ValueError("每个候选动作必须包含一个或两个魔灵")
        pair_mask = [len(action) == 2 for action in actions]
        if any(pair_mask) and not all(pair_mask):
            raise ValueError("单选和双选候选不能放在同一批次评分")

        if not any(pair_mask):
            output = self._forward_state(state)
            logits = {
                "pick": output["pick_logits"],
                "opponent": output["opponent_logits"],
                "ban": output["ban_logits"],
            }[head][0]
            return [float(logits[self._vocab_id(action[0])].detach().cpu()) for action in actions]

        scores = []
        for start in range(0, len(actions), int(batch_size)):
            chunk = actions[start : start + int(batch_size)]
            pair_ids = torch.tensor(
                [[self._vocab_id(first), self._vocab_id(second)] for first, second in chunk],
                dtype=torch.long,
                device=self.device,
            )
            output = self._forward_state(state, pair_candidates=pair_ids)
            scores.extend(float(value) for value in output["pair_logits"][0].detach().cpu())
        return scores

    @staticmethod
    def _sigmoid(value):
        value = max(-60.0, min(60.0, float(value)))
        return 1.0 / (1.0 + math.exp(-value))

    @staticmethod
    def _blend_policy_and_evidence(policy_score, detail):
        components = [(float(policy_score), 0.75)]
        coverage = float(detail.get("data_coverage", detail.get("coverage", 0.0)) or 0.0)
        if coverage > 0.0:
            components.append((float(detail.get("score", 0.5)), 0.10 * min(1.0, coverage)))
        mechanism_coverage = float(detail.get("mechanism_coverage", 0.0) or 0.0)
        if detail.get("mechanism_available") and mechanism_coverage > 0.0:
            components.append(
                (float(detail.get("mechanism_score", 0.5)), 0.10 * min(1.0, mechanism_coverage))
            )
        matchup_coverage = float(detail.get("matchup_coverage", 0.0) or 0.0)
        if matchup_coverage > 0.0:
            components.append(
                (float(detail.get("matchup_score", 0.5)), 0.05 * min(1.0, matchup_coverage))
            )
        weight = sum(item_weight for _value, item_weight in components)
        return sum(value * item_weight for value, item_weight in components) / weight

    @staticmethod
    def _detail(policy_logit, policy_score, stats_detail):
        detail = {
            "policy_logit": float(policy_logit),
            "policy_score": float(policy_score),
            "normalized_policy_score": float(policy_score),
            "draft_stats_score": float(stats_detail.get("score", 0.5)),
            "draft_stats_solo_score": float(stats_detail.get("solo_score", 0.5)),
            "draft_stats_synergy_score": float(stats_detail.get("synergy_score", 0.5)),
            "draft_stats_coverage": float(stats_detail.get("coverage", 0.0)),
            "draft_stats_data_coverage": float(stats_detail.get("data_coverage", 0.0)),
            "draft_stats_source": stats_detail.get("source", "unavailable"),
            "draft_stats_evidence": list(stats_detail.get("evidence") or []),
            "matchup_score": float(stats_detail.get("matchup_score", 0.5)),
            "matchup_coverage": float(stats_detail.get("matchup_coverage", 0.0)),
            "matchup_available": float(stats_detail.get("matchup_coverage", 0.0)) > 0.0,
            "mechanism_score": float(stats_detail.get("mechanism_score", 0.5)),
            "mechanism_available": bool(stats_detail.get("mechanism_available")),
            "mechanism_coverage": float(stats_detail.get("mechanism_coverage", 0.0)),
            "mechanism_reasons": list(stats_detail.get("mechanism_reasons") or []),
            "sample_count": int(stats_detail.get("sample_count", 0)),
            "evidence_confidence": float(stats_detail.get("evidence_confidence", 0.0)),
            "evidence_uncertainty": float(stats_detail.get("evidence_uncertainty", 1.0)),
        }
        return detail

    def _candidate_actions(self, state, allowed, pick_count, head, max_pair_candidates, top_k):
        if int(pick_count) == 1:
            return [(slug,) for slug in allowed]
        if int(pick_count) != 2 or len(allowed) < 2:
            return []

        # A pair can be strong because of its interaction even when neither
        # singleton is near the top of the policy head.  Keep a broad seed
        # pool, then let the pair head score the combinations.  The previous
        # sqrt(limit) pool made a 400-pair budget use only 29 singletons and
        # discarded most real double-pick actions before pair scoring.
        limit = max(int(top_k), int(max_pair_candidates))
        generation_limit = max(2048, limit * 12)
        singleton_actions = [(slug,) for slug in allowed]
        singleton_scores = self._score_actions(state, singleton_actions, head=head)
        ranked = [slug for _score, slug in sorted(zip(singleton_scores, allowed), reverse=True)]
        pool_size = int((1 + math.sqrt(1 + 8 * generation_limit)) // 2)
        pool_size = max(64, min(128, pool_size, len(ranked)))
        pool = ranked[: min(len(ranked), pool_size)]
        synergy_rows = self.lucksack_stats.candidate_synergy_pairs(
            allowed,
            limit=max(32, int(top_k) * 16),
        )
        actions = {tuple(sorted(pair)) for pair in combinations(pool, 2)}
        synergy_actions = set()
        for row in synergy_rows:
            pair = self._canonical_action(row.get("slugs") or ())
            if len(pair) == 2:
                actions.add(pair)
                synergy_actions.add(pair)
        if len(actions) > limit:
            rank = {slug: index for index, slug in enumerate(ranked)}
            # Keep the stats anchors in the scoring set.  The final cap is
            # applied after pair logits are available, so this stage only
            # bounds inference work and does not decide the recommendation.
            ranked_actions = sorted(
                actions - synergy_actions,
                key=lambda pair: (
                    rank.get(pair[0], len(rank)) + rank.get(pair[1], len(rank)),
                    pair,
                ),
            )
            actions = set(synergy_actions)
            actions.update(ranked_actions[: max(0, generation_limit - len(actions))])
        return sorted(actions)

    def recommend_pick_plans(
        self,
        my_picks,
        enemy_picks,
        *,
        pick_count,
        turn,
        events,
        first_pick_side="me",
        candidate_pool=None,
        my_box=None,
        use_my_box_for_reco=False,
        top_k=6,
        max_pair_candidates=400,
    ):
        expected = infer_pick_count_for_side(my_picks, enemy_picks, first_pick_side, "me")
        if pick_count is not None and int(pick_count) != expected:
            raise ValueError(f"当前我方动作应选 {expected} 个魔灵，而不是 {pick_count} 个")
        if expected not in {1, 2}:
            raise ValueError("当前状态没有合法的我方选人动作")
        state = self._state(my_picks, enemy_picks, first_pick_side, turn, events, "me")
        allowed = self._allowed_candidates(
            my_picks,
            enemy_picks,
            candidate_pool,
            my_box,
            use_my_box_for_reco,
        )
        actions = self._candidate_actions(
            state, allowed, expected, "pick", max_pair_candidates, top_k
        )
        if not actions:
            raise RuntimeError("当前候选池没有满足本轮动作大小的合法魔灵")
        logits = self._score_actions(state, actions, head="pick")
        if expected == 2 and len(actions) > int(max_pair_candidates):
            selected = sorted(
                range(len(actions)),
                key=lambda index: logits[index],
                reverse=True,
            )[: int(max_pair_candidates)]
            actions = [actions[index] for index in selected]
            logits = [logits[index] for index in selected]
        plans = []
        for action, logit in zip(actions, logits):
            policy_score = self._sigmoid(logit)
            stats = self.lucksack_stats.score_action(my_picks, enemy_picks, action, turn=state["turn"])
            final_score = self._blend_policy_and_evidence(policy_score, stats)
            display = [self._to_display_name(slug) for slug in action]
            detail = self._detail(logit, policy_score, stats)
            detail.update(
                {
                    "action_type": "pair" if len(action) == 2 else "single",
                    "pick_count": len(action),
                    "plan_score": float(final_score),
                }
            )
            plans.append(
                {
                    "plan": display,
                    "action_key": "|".join(self._canonical_action(display)),
                    "action_type": detail["action_type"],
                    "pick_count": len(action),
                    "plan_score": float(final_score),
                    "estimated_win_rate": None,
                    "source": "shared_policy",
                    "reasons": list(detail["mechanism_reasons"]),
                    "detail": detail,
                }
            )
        plans.sort(key=lambda row: row["plan_score"], reverse=True)
        return plans[: int(top_k)]

    def predict_opponent_distribution(
        self,
        *,
        my_picks,
        enemy_picks,
        turn,
        events,
        first_pick_side="me",
        top_k=5,
        candidate_pool=None,
        max_pair_candidates=400,
    ):
        my_picks = list(my_picks)
        enemy_picks = list(enemy_picks)
        turn = int(turn)
        pick_count = infer_pick_count_for_side(my_picks, enemy_picks, first_pick_side, "enemy")
        if pick_count not in {1, 2}:
            raise ValueError("当前状态没有合法的对方选人动作")
        model_state = self._state(
            my_picks, enemy_picks, first_pick_side, turn, events, "enemy"
        )
        allowed = self._allowed_candidates(my_picks, enemy_picks, candidate_pool)
        actions = self._candidate_actions(
            model_state, allowed, pick_count, "opponent", max_pair_candidates, top_k
        )
        if not actions:
            raise RuntimeError("当前候选池没有满足本轮对方动作大小的合法魔灵")
        logits = self._score_actions(model_state, actions, head="opponent")
        if pick_count == 2 and len(actions) > int(max_pair_candidates):
            selected = sorted(
                range(len(actions)),
                key=lambda index: logits[index],
                reverse=True,
            )[: int(max_pair_candidates)]
            actions = [actions[index] for index in selected]
            logits = [logits[index] for index in selected]
        if not logits:
            raise RuntimeError("Shared opponent head 没有返回候选分数")
        probabilities = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=0).tolist()
        rows = []
        for action, logit, probability in zip(actions, logits, probabilities):
            display = [self._to_display_name(slug) for slug in action]
            rows.append(
                {
                    "plan": display,
                    "action_key": "|".join(self._canonical_action(display)),
                    "probability": float(probability),
                    "plan_score": float(probability),
                    "source": "shared_opponent_policy",
                    "action_type": "pair" if len(action) == 2 else "single",
                    "pick_count": len(action),
                    "detail": {
                        "policy_logit": float(logit),
                        "policy_score": self._sigmoid(logit),
                        "normalized_policy_score": self._sigmoid(logit),
                        "opponent_probability": float(probability),
                    },
                }
            )
        rows.sort(key=lambda row: row["probability"], reverse=True)
        rows = rows[: int(top_k)]
        total = sum(row["probability"] for row in rows)
        for row in rows:
            row["probability"] /= total
            row["plan_score"] = row["probability"]
            row["detail"]["opponent_probability"] = row["probability"]
        return rows

    def recommend_enemy_next_picks(
        self,
        my_picks,
        enemy_picks,
        *,
        first_pick_side,
        turn,
        events,
        top_k=5,
        candidate_pool=None,
        max_pair_candidates=400,
        pick_count=None,
    ):
        expected = infer_pick_count_for_side(my_picks, enemy_picks, first_pick_side, "enemy")
        if pick_count is not None and int(pick_count) != expected:
            raise ValueError(f"当前对方动作应选 {expected} 个魔灵，而不是 {pick_count} 个")
        rows = self.predict_opponent_distribution(
            my_picks=my_picks,
            enemy_picks=enemy_picks,
            first_pick_side=first_pick_side,
            turn=turn,
            events=events,
            top_k=max(int(top_k) * 4, int(top_k)),
            candidate_pool=candidate_pool,
            max_pair_candidates=max_pair_candidates,
        )
        for row in rows:
            action = tuple(self._to_slug(value) for value in row["plan"])
            stats = self.lucksack_stats.score_action(
                enemy_picks, my_picks, action, turn=turn
            )
            policy_score = float(row["detail"]["policy_score"])
            score = self._blend_policy_and_evidence(policy_score, stats)
            detail = self._detail(row["detail"]["policy_logit"], policy_score, stats)
            detail.update(
                {
                    "opponent_probability": row["probability"],
                    "action_type": row["action_type"],
                    "pick_count": row["pick_count"],
                }
            )
            row.update(
                {
                    "plan_score": float(score),
                    "estimated_win_rate": None,
                    "reasons": list(detail["mechanism_reasons"]),
                    "detail": detail,
                }
            )
        rows.sort(key=lambda row: row["plan_score"], reverse=True)
        return rows[: int(top_k)]

    @staticmethod
    def _swap_timeline(timeline):
        return DraftTimeline(
            tuple(
                DraftActionEvent(
                    side="enemy" if event.side == "me" else "me",
                    monsters=event.monsters,
                    action_type=event.action_type,
                    order=event.order,
                )
                for event in timeline.events
            )
        )

    @torch.inference_mode()
    def evaluate_states(self, states, return_detail=False):
        states = list(states or ())
        if not states:
            return []
        event_tensors = []
        turns = []
        first_sides = []
        invert = []
        for raw_state in states:
            if not isinstance(raw_state, dict):
                raise TypeError("Shared value evaluation requires canonical state mappings")
            state = dict(raw_state)
            if "turn" not in state or "events" not in state or "first_pick_side" not in state:
                raise ValueError("Shared value evaluation requires canonical events, turn, and first_pick_side")
            turn = int(state["turn"])
            first_side = normalize_name(state["first_pick_side"])
            if first_side not in {"me", "enemy"}:
                raise ValueError("Shared value evaluation requires a valid first_pick_side")
            current_side = normalize_name(state["current_side"])
            if current_side not in {"me", "enemy"}:
                raise ValueError("Shared value evaluation requires current_side")
            timeline = self._slug_timeline(state)
            validate_timeline(timeline, first_side)
            if timeline.slots_used != turn - 1:
                raise ValueError("Shared value evaluation turn does not match events")
            expected_side = side_for_order(turn, first_side)
            if turn <= 10 and expected_side != current_side:
                raise ValueError("Shared value evaluation current_side does not match turn")
            should_invert = current_side == "enemy"
            if should_invert:
                timeline = self._swap_timeline(timeline)
                first_side = "enemy" if first_side == "me" else "me"
            event_tensors.append(
                self.model.timeline_to_tensors(
                    timeline,
                    self.vocab,
                    max_events=self.model.max_events,
                    device=self.device,
                )
            )
            turns.append(float(turn))
            first_sides.append(1.0 if first_side == "me" else 0.0)
            invert.append(should_invert)

        event_ids = torch.cat([row[0] for row in event_tensors], dim=0)
        event_mask = torch.cat([row[1] for row in event_tensors], dim=0)
        event_side = torch.cat([row[2] for row in event_tensors], dim=0)
        event_type = torch.cat([row[3] for row in event_tensors], dim=0)
        output = self.model(
            event_ids,
            event_mask,
            event_side,
            event_type,
            turn=torch.tensor(turns, dtype=torch.float32, device=self.device),
            first_pick_side=torch.tensor(first_sides, dtype=torch.float32, device=self.device),
            current_side=torch.ones(len(states), dtype=torch.float32, device=self.device),
        )
        values = torch.sigmoid(output["value"]).detach().cpu().tolist()
        values = [1.0 - value if flip else value for value, flip in zip(values, invert)]
        if not return_detail:
            return values
        return [
            {
                "raw_win_rate": float(value),
                "calibrated_win_rate": float(value),
                "model_type": "shared_value_head",
            }
            for value in values
        ]

    def recommend_ban_plans(
        self,
        my_picks,
        enemy_picks,
        *,
        events,
        top_k=5,
        first_pick_side="me",
    ):
        if len(my_picks or ()) != 5 or len(enemy_picks or ()) != 5:
            raise ValueError("Ban 推荐要求双方各完成五个选人")
        enemy_slugs = list(dict.fromkeys(self._to_slug(value) for value in enemy_picks))
        supported_slugs = set(self.known_slugs) | set(getattr(self, "external_slugs", ()))
        if len(enemy_slugs) != 5 or any(slug not in supported_slugs for slug in enemy_slugs):
            raise ValueError("敌方最终阵容包含未知或重复魔灵")
        state = self._state(my_picks, enemy_picks, first_pick_side, 11, events, "me")
        actions = [(slug,) for slug in enemy_slugs]
        logits = self._score_actions(state, actions, head="ban")
        baseline = self.evaluate_states([state], return_detail=True)[0]
        post_ban_states = [
            transition_state(
                state,
                {"side": "me", "action_type": "ban", "monsters": [slug]},
            )
            for slug in enemy_slugs
        ]
        post_ban_values = self.evaluate_states(
            post_ban_states,
            return_detail=True,
        )
        pre_ban_win_rate = float(baseline["calibrated_win_rate"])
        plans = []
        for action, logit, post_ban in zip(actions, logits, post_ban_values):
            score = self._sigmoid(logit)
            display = self._to_display_name(action[0])
            post_ban_win_rate = float(post_ban["calibrated_win_rate"])
            plans.append(
                {
                    "plan": [display],
                    "action_key": self._canonical_action((display,))[0],
                    "action_type": "ban",
                    "pick_count": 1,
                    "plan_score": score,
                    "estimated_win_rate": post_ban_win_rate,
                    "source": "shared_ban",
                    "detail": {
                        "ban_logit": float(logit),
                        "policy_score": score,
                        "plan_score": score,
                        "pre_ban_win_rate": pre_ban_win_rate,
                        "post_ban_win_rate": post_ban_win_rate,
                        "win_rate_gain": post_ban_win_rate - pre_ban_win_rate,
                    },
                }
            )
        plans.sort(key=lambda row: row["plan_score"], reverse=True)
        return plans[: int(top_k)]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Test the Shared BP Core recommender.")
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args(argv)
    recommender = BPSharedRecommender()
    slugs = list(recommender.known_slugs)[:10]
    if len(slugs) < 10:
        raise RuntimeError("Shared BP vocabulary contains fewer than ten monsters")
    timeline = timeline_from_picks([], [], "me")
    plans = recommender.recommend_pick_plans(
        my_picks=[],
        enemy_picks=[],
        pick_count=1,
        first_pick_side="me",
        turn=1,
        events=timeline.to_dict(),
        candidate_pool=slugs,
        top_k=args.top_k,
    )
    print(json.dumps(plans, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
