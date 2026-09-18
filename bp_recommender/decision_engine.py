"""Unified, evidence-aware BP decision engine.

The relation model proposes actions, the value model evaluates resulting
states, and a small robust search accounts for the opponent's strongest
likely response. Public statistics and mechanism data are used for candidate
coverage and evidence, not as a replacement for the learned value model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from bp_core.relation_common import side_for_order
from bp_core.transformer_common import normalize_name


@dataclass(frozen=True)
class SearchConfig:
    opponent_branch: int = 3
    continuation_branch: int = 2
    worst_response_weight: float = 0.65
    robust_weight: float = 0.60
    immediate_weight: float = 0.15
    relation_weight: float = 0.15
    evidence_weight: float = 0.10


class BPDecisionEngine:
    def __init__(self, relation_model, value_model=None, assets=None, config=None):
        self.relation_model = relation_model
        self.value_model = value_model
        self.assets = assets or {}
        self.config = config or SearchConfig()
        self._candidate_cache = {}
        self._value_cache = {}

    @staticmethod
    def _plan_monsters(item):
        values = item.get("plan") or item.get("monsters") or []
        if not values:
            value = item.get("monster") or item.get("display_name")
            values = [value] if value else []
        return [value for value in values if value]

    @staticmethod
    def _state_key(state):
        return (
            tuple(state.get("my_picks") or []),
            tuple(state.get("enemy_picks") or []),
            int(state.get("turn") or 1),
            normalize_name(state.get("first_pick_side")),
            normalize_name(state.get("current_side")),
            state.get("my_leader") or "",
            state.get("enemy_leader") or "",
        )

    @staticmethod
    def _side(state):
        return side_for_order(
            int(state.get("turn") or 1),
            state.get("first_pick_side"),
        )

    @staticmethod
    def _transition(state, side, action):
        next_state = dict(state)
        values = list(action)
        if side == "me":
            next_state["my_picks"] = list(state.get("my_picks") or []) + values
        else:
            next_state["enemy_picks"] = list(state.get("enemy_picks") or []) + values
        next_turn = int(state.get("turn") or 1) + len(values)
        next_state["turn"] = next_turn
        next_state["current_side"] = side_for_order(
            next_turn,
            next_state.get("first_pick_side"),
        )
        return next_state

    def _candidate_actions(self, state, side, limit):
        key = (side, self._state_key(state), int(limit))
        cached = self._candidate_cache.get(key)
        if cached is not None:
            return cached
        if int(state.get("turn") or 1) > 10 or not self.relation_model:
            return []
        common = {
            "first_pick_side": state["first_pick_side"],
            "candidate_pool": self.assets.get("options"),
            "max_pair_candidates": min(
                int(state.get("max_pair_candidates", 80)),
                80,
            ),
            "preserve_evidence": True,
        }
        if side == "me":
            plans = self.relation_model.recommend_pick_plans(
                my_picks=list(state.get("my_picks") or []),
                enemy_picks=list(state.get("enemy_picks") or []),
                turn=int(state.get("turn") or 1),
                my_box=state.get("my_box"),
                use_my_box_for_reco=bool(state.get("use_my_box", True)),
                top_k=int(limit),
                **common,
            )
        else:
            plans = self.relation_model.recommend_enemy_next_picks(
                my_picks=list(state.get("my_picks") or []),
                enemy_picks=list(state.get("enemy_picks") or []),
                top_k=int(limit),
                **common,
            )
        self._candidate_cache[key] = list(plans or [])
        return self._candidate_cache[key]

    def _value_key(self, state):
        return self._state_key(state)

    def _evaluate_states(self, states):
        states = list(states or [])
        if not states:
            return []
        values = [None] * len(states)
        pending = {}
        for index, state in enumerate(states):
            key = self._value_key(state)
            if key in self._value_cache:
                values[index] = self._value_cache[key]
            else:
                pending.setdefault(key, (index, state))
        if pending and self.value_model is not None:
            evaluations = self.value_model.evaluate_states(
                [state for _index, state in pending.values()],
                return_detail=True,
            )
            for key, evaluation in zip(pending, evaluations):
                value = float(evaluation.get("calibrated_win_rate", 0.5))
                self._value_cache[key] = value
                values[pending[key][0]] = value
        for index, value in enumerate(values):
            if value is None:
                values[index] = 0.5
        return values

    @staticmethod
    def _evidence_score(detail):
        components = []
        if detail.get("draft_stats_source") not in {None, "unavailable"}:
            coverage = min(
                [
                    float(detail.get(key, 0.0) or 0.0)
                    for key in ("draft_stats_coverage", "draft_stats_data_coverage")
                    if key in detail
                ]
                or [0.0]
            )
            components.append(
                (float(detail.get("draft_stats_score", 0.5) or 0.5), max(coverage, 1.0 if coverage <= 0 else coverage))
            )
        if detail.get("mechanism_available"):
            coverage = float(detail.get("mechanism_coverage", 0.0) or 0.0)
            components.append(
                (float(detail.get("mechanism_score", 0.5) or 0.5), max(coverage, 1.0 if coverage <= 0 else coverage))
            )
        if detail.get("matchup_available") or float(detail.get("matchup_coverage", 0.0) or 0.0) > 0:
            coverage = float(detail.get("matchup_coverage", 0.0) or 0.0)
            components.append(
                (float(detail.get("matchup_score", 0.5) or 0.5), max(coverage, 1.0 if coverage <= 0 else coverage))
            )
        if not components:
            return 0.5
        total = sum(weight for _value, weight in components)
        return sum(value * weight for value, weight in components) / max(total, 1e-6)

    @staticmethod
    def _decision_confidence(detail):
        """Combine model coverage and external evidence coverage for risk control."""
        relation_confidence = detail.get("relation_model_confidence")
        evidence_confidence = detail.get("evidence_confidence")
        if relation_confidence is None and evidence_confidence is None:
            return 1.0
        relation_confidence = 1.0 if relation_confidence is None else float(relation_confidence)
        evidence_confidence = 0.0 if evidence_confidence is None else float(evidence_confidence)
        return max(0.0, min(1.0, 0.65 * relation_confidence + 0.35 * evidence_confidence))

    @staticmethod
    def _weighted_response_value(rows):
        if not rows:
            return None, None, []
        priorities = [float(row.get("priority", 0.5)) for row in rows]
        values = [float(row["value"]) for row in rows]
        highest = max(priorities)
        weights = [math.exp((priority - highest) / 0.05) for priority in priorities]
        total = sum(weights) or 1.0
        expected = sum(value * weight for value, weight in zip(values, weights)) / total
        return min(values), expected, rows

    def _my_tree(self, root_plan, state):
        picked = self._plan_monsters(root_plan)
        post_state = self._transition(state, "me", picked)
        rows = [post_state]
        response_rows = []
        response_meta = []
        if self._side(post_state) == "enemy":
            responses = self._candidate_actions(
                post_state,
                "enemy",
                self.config.opponent_branch,
            )
            for response in responses:
                counter = self._plan_monsters(response)
                if not counter:
                    continue
                enemy_state = self._transition(post_state, "enemy", counter)
                leaves = [enemy_state]
                continuation_meta = []
                if self._side(enemy_state) == "me":
                    continuations = self._candidate_actions(
                        enemy_state,
                        "me",
                        self.config.continuation_branch,
                    )
                    leaves = []
                    for continuation in continuations:
                        followup = self._plan_monsters(continuation)
                        if not followup:
                            continue
                        leaves.append(self._transition(enemy_state, "me", followup))
                        continuation_meta.append(followup)
                    if not leaves:
                        leaves = [enemy_state]
                rows.extend(leaves)
                response_rows.append({"leaves": leaves, "continuations": continuation_meta})
                response_meta.append(
                    {
                        "plan": counter,
                        "priority": float(response.get("final_score", response.get("plan_score", 0.5))),
                    }
                )
        return rows, response_rows, response_meta

    def rank_my_plans(self, plans, state, top_k):
        plans = list(plans or [])
        if not plans:
            return []
        base_value = self._evaluate_states([state])[0]
        tree_data = []
        all_rows = []
        for plan in plans:
            rows, response_rows, response_meta = self._my_tree(plan, state)
            tree_data.append((rows, response_rows, response_meta))
            all_rows.extend(rows)
        all_values = self._evaluate_states(all_rows)
        value_index = 0
        enriched = []
        for plan, (rows, response_rows, response_meta) in zip(plans, tree_data):
            immediate = all_values[value_index]
            value_index += 1
            response_values = []
            for response, meta in zip(response_rows, response_meta):
                leaves = response["leaves"]
                leaf_values = all_values[value_index : value_index + len(leaves)]
                value_index += len(leaves)
                own_followup = max(leaf_values) if leaf_values else immediate
                response_values.append({**meta, "value": own_followup})
            worst, expected, response_values = self._weighted_response_value(response_values)
            robust = immediate if worst is None else (
                self.config.worst_response_weight * worst
                + (1.0 - self.config.worst_response_weight) * expected
            )
            item = dict(plan)
            detail = dict(item.get("detail", {}) or {})
            relation_score = float(
                detail.get("normalized_relation_score", item.get("plan_score", 0.0)) or 0.0
            )
            evidence_score = self._evidence_score(detail)
            decision_confidence = self._decision_confidence(detail)
            relation_confidence = float(detail.get("relation_model_confidence", 1.0))
            relation_component = 0.5 + (relation_score - 0.5) * relation_confidence
            uncertainty_penalty = 0.025 * (1.0 - decision_confidence)
            fused = (
                self.config.robust_weight * robust
                + self.config.immediate_weight * immediate
                + self.config.relation_weight * relation_component
                + self.config.evidence_weight * evidence_score
                - uncertainty_penalty
            )
            item["plan_score"] = round(fused, 6)
            item["estimated_win_rate"] = round(immediate, 6)
            detail.update(
                {
                    "pre_pick_win_rate": round(base_value, 6),
                    "post_pick_win_rate": round(immediate, 6),
                    "adversarial_win_rate": round(robust, 6),
                    "win_rate_gain": round(immediate - base_value, 6),
                    "value_component": round(immediate, 6),
                    "adversarial_value_component": round(robust, 6),
                    "relation_component": round(relation_component, 6),
                    "evidence_component": round(evidence_score, 6),
                    "decision_confidence": round(decision_confidence, 6),
                    "uncertainty_penalty": round(uncertainty_penalty, 6),
                    "worst_counter_win_rate": None if worst is None else round(worst, 6),
                    "mean_counter_win_rate": None if expected is None else round(expected, 6),
                    "counter_plans": [
                        {
                            "plan": row["plan"],
                            "win_rate": round(float(row["value"]), 6),
                            "priority": round(float(row["priority"]), 6),
                        }
                        for row in response_values
                    ],
                    "search_depth": 3,
                    "ranking_source": "robust_search_value_3ply_relation_stats_mechanism",
                }
            )
            item["detail"] = detail
            enriched.append(item)
        enriched.sort(key=lambda row: float(row.get("plan_score", 0.0)), reverse=True)
        return enriched[: int(top_k)]

    def rank_enemy_plans(self, plans, state, top_k):
        plans = list(plans or [])
        if not plans:
            return []
        tree_data = []
        all_rows = []
        for plan in plans:
            monsters = self._plan_monsters(plan)
            post_state = self._transition(state, "enemy", monsters)
            response_plans = self._candidate_actions(
                post_state,
                "me",
                self.config.opponent_branch,
            ) if self._side(post_state) == "me" else []
            rows = [post_state]
            for response_plan in response_plans:
                response = self._plan_monsters(response_plan)
                if response:
                    rows.append(self._transition(post_state, "me", response))
            tree_data.append((rows, response_plans))
            all_rows.extend(rows)
        values = self._evaluate_states(all_rows)
        offset = 0
        enriched = []
        for plan, (rows, response_plans) in zip(plans, tree_data):
            immediate = values[offset]
            response_values = values[offset + 1 : offset + len(rows)]
            offset += len(rows)
            best_response = max(response_values, default=immediate)
            item = dict(plan)
            detail = dict(item.get("detail", {}) or {})
            relation_score = float(
                detail.get("normalized_relation_score", item.get("final_score", 0.0)) or 0.0
            )
            evidence_score = self._evidence_score(detail)
            decision_confidence = self._decision_confidence(detail)
            relation_confidence = float(detail.get("relation_model_confidence", 1.0))
            relation_component = 0.5 + (relation_score - 0.5) * relation_confidence
            uncertainty_penalty = 0.025 * (1.0 - decision_confidence)
            final_score = (
                0.60 * (1.0 - best_response)
                + 0.15 * (1.0 - immediate)
                + 0.15 * relation_component
                + 0.10 * evidence_score
                - uncertainty_penalty
            )
            response_rows = []
            for response_plan, response_value in zip(response_plans, response_values):
                response_rows.append(
                    {
                        "plan": self._plan_monsters(response_plan),
                        "win_rate": round(float(response_value), 6),
                        "priority": round(float(response_plan.get("plan_score", 0.5)), 6),
                    }
                )
            item["estimated_win_rate"] = round(immediate, 6)
            item["final_score"] = round(final_score, 6)
            detail.update(
                {
                    "post_enemy_pick_win_rate": round(immediate, 6),
                    "best_response_win_rate": round(best_response, 6),
                    "enemy_adversarial_score": round(final_score, 6),
                    "relation_component": round(relation_component, 6),
                    "evidence_component": round(evidence_score, 6),
                    "decision_confidence": round(decision_confidence, 6),
                    "uncertainty_penalty": round(uncertainty_penalty, 6),
                    "response_plans": response_rows,
                    "search_depth": 2,
                    "ranking_source": "robust_enemy_search_value_relation_stats_mechanism",
                }
            )
            item["detail"] = detail
            enriched.append(item)
        enriched.sort(key=lambda row: float(row.get("final_score", 0.0)), reverse=True)
        return enriched[: int(top_k)]
