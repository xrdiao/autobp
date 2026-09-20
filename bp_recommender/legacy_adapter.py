"""Adapter for the original multi-model BP recommender.

The old relation/value models remain responsible for ranking.  This adapter
keeps the current application's canonical event state at the boundary: event
lists are validated by the caller and the legacy models consume the derived
side pick lists and turn metadata.
"""

from __future__ import annotations

import math

from bp_core.names import normalize_name
from bp_recommender.ban_value import BPBanValueRecommender
from bp_recommender.prefix_value import BPPrefixValueRecommender
from bp_recommender.relation import BPRelationRecommender


class LegacyBPRecommender:
    """Use the old specialized models behind the unified BP state API."""

    model_family = "legacy_relation_prefix_ban"

    def __init__(self, device=None, stats_rank_group=None):
        self.relation = BPRelationRecommender(
            device=device,
            stats_rank_group=stats_rank_group,
        )
        self.value = BPPrefixValueRecommender(device=device)
        self.ban_value = BPBanValueRecommender(
            device=device,
            stats_rank_group=stats_rank_group,
        )
        self.known_slugs = frozenset(self.relation._known_model_slugs)
        self.external_slugs = frozenset(
            normalize_name(value)
            for value in self.relation.lucksack_stats.candidate_slugs()
            if normalize_name(value)
        )
        self.display_name_map = dict(self.relation.display_name_map)

    def _to_slug(self, value):
        return self.relation._to_slug(value)

    def _to_display_name(self, value):
        return self.relation._to_display_name(value)

    @staticmethod
    def _action_key(plan):
        values = [str(value) for value in plan if value]
        values = sorted(values, key=lambda value: (normalize_name(value), value.casefold(), value))
        return "|".join(normalize_name(value) for value in values)

    @classmethod
    def _normalize_plan(cls, row, action_type=None):
        row = dict(row or {})
        plan = list(row.get("plan") or row.get("monsters") or [])
        if not plan and row.get("monster"):
            plan = [row["monster"]]
        if len(plan) > 1:
            plan.sort(key=lambda value: (normalize_name(value), str(value).casefold(), str(value)))
        kind = action_type or row.get("action_type")
        if kind is None:
            kind = "pair" if len(plan) == 2 else "single"
        if kind == "ban":
            pick_count = 1
        else:
            pick_count = len(plan)
            kind = "pair" if pick_count == 2 else "single"
        detail = dict(row.get("detail") or {})
        if "normalized_policy_score" not in detail:
            detail["normalized_policy_score"] = float(
                detail.get("normalized_relation_score", row.get("plan_score", 0.5)) or 0.5
            )
        if "policy_score" not in detail:
            detail["policy_score"] = float(
                detail.get("relation_score", detail["normalized_policy_score"]) or 0.5
            )
        detail.setdefault("draft_stats_coverage", detail.get("draft_stats_coverage", 0.0))
        detail.setdefault("draft_stats_data_coverage", detail.get("draft_stats_data_coverage", 0.0))
        detail.setdefault("evidence_confidence", detail.get("evidence_confidence", 0.0))
        detail.setdefault("action_type", kind)
        detail.setdefault("pick_count", pick_count)
        row.update(
            {
                "plan": plan,
                "action_key": cls._action_key(plan),
                "action_type": kind,
                "pick_count": pick_count,
                "detail": detail,
                "plan_score": float(
                    row.get("plan_score", row.get("final_score", row.get("score", 0.0))) or 0.0
                ),
            }
        )
        return row

    def recommend_pick_plans(
        self,
        my_picks,
        enemy_picks,
        *,
        pick_count=None,
        turn=None,
        events=None,
        first_pick_side="me",
        candidate_pool=None,
        my_box=None,
        use_my_box_for_reco=False,
        top_k=6,
        max_pair_candidates=400,
        **_unused,
    ):
        plans = self.relation.recommend_pick_plans(
            my_picks=list(my_picks or []),
            enemy_picks=list(enemy_picks or []),
            pick_count=pick_count,
            turn=turn,
            first_pick_side=first_pick_side,
            candidate_pool=candidate_pool,
            my_box=my_box,
            use_my_box_for_reco=bool(use_my_box_for_reco),
            top_k=int(top_k),
            max_pair_candidates=int(max_pair_candidates),
            preserve_evidence=True,
        )
        return [self._normalize_plan(row) for row in plans]

    def recommend_enemy_next_picks(
        self,
        my_picks,
        enemy_picks,
        *,
        first_pick_side="me",
        turn=None,
        events=None,
        pick_count=None,
        top_k=5,
        candidate_pool=None,
        max_pair_candidates=400,
        **_unused,
    ):
        predictions = self.relation.recommend_enemy_next_picks(
            my_picks=list(my_picks or []),
            enemy_picks=list(enemy_picks or []),
            first_pick_side=first_pick_side,
            pick_count=pick_count,
            top_k=int(top_k),
            candidate_pool=candidate_pool,
            max_pair_candidates=int(max_pair_candidates),
            preserve_evidence=True,
        )
        return [self._normalize_plan(row) for row in predictions]

    def predict_opponent_distribution(self, **kwargs):
        rows = self.recommend_enemy_next_picks(**kwargs)
        if not rows:
            return []
        raw = [max(0.0, float(row.get("plan_score", 0.0))) for row in rows]
        total = sum(raw)
        probabilities = raw if total > 0.0 else [1.0] * len(rows)
        total = sum(probabilities)
        result = []
        for row, probability in zip(rows, probabilities):
            item = dict(row)
            item["probability"] = probability / total
            item["opponent_probability"] = item["probability"]
            item.setdefault("detail", {})["opponent_probability"] = item["probability"]
            result.append(item)
        return result

    def evaluate_states(self, states, return_detail=False):
        return self.value.evaluate_states(list(states or []), return_detail=return_detail)

    def recommend_ban_plans(
        self,
        my_picks,
        enemy_picks,
        *,
        events=None,
        top_k=5,
        first_pick_side="me",
        my_leader=None,
        enemy_leader=None,
        enemy_ban=None,
        **_unused,
    ):
        plans = self.ban_value.recommend_bans(
            my_picks=list(my_picks or []),
            enemy_picks=list(enemy_picks or []),
            my_leader=my_leader,
            enemy_leader=enemy_leader,
            enemy_ban=enemy_ban,
            top_k=int(top_k),
        )
        return [self._normalize_plan(row, action_type="ban") for row in plans]
