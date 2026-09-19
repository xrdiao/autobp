"""Search and ranking on top of the single Shared BP Core."""

from __future__ import annotations

from dataclasses import dataclass

from bp_core.action_space import events_from_state, transition_state, validate_timeline
from bp_core.bp_common import infer_pick_count_for_side, side_for_order
from bp_core.names import normalize_name


@dataclass(frozen=True)
class SearchConfig:
    opponent_branch: int = 3
    continuation_branch: int = 2
    worst_response_weight: float = 0.65
    robust_weight: float = 0.60
    immediate_weight: float = 0.15
    policy_weight: float = 0.15
    evidence_weight: float = 0.10


class BPDecisionEngine:
    def __init__(self, model, assets=None, config=None):
        self.model = model
        self.assets = assets or {}
        self.config = config or SearchConfig()
        self._candidate_cache = {}
        self._value_cache = {}

    @staticmethod
    def _canonical_action(action):
        """Return a stable action representation.

        A BP event may contain one pick or an unordered pair.  The event
            itself remains in chronological order, while the members of a pair
            are sorted by their normalized name.
        """
        if isinstance(action, str):
            action = [action]
        values = []
        seen = set()
        for value in action or ():
            if value is None:
                continue
            value = str(value).strip()
            if not value:
                continue
            key = normalize_name(value)
            if key not in seen:
                values.append(value)
                seen.add(key)
        if len(values) > 1:
            values.sort(key=lambda value: (normalize_name(value), value.casefold(), value))
        return tuple(values)

    @classmethod
    def canonicalize_action(cls, action):
        """Public alias used by adapters and evaluation code."""
        return cls._canonical_action(action)

    @classmethod
    def _action_key(cls, action):
        return "|".join(normalize_name(value) for value in cls._canonical_action(action))

    @classmethod
    def canonical_action_key(cls, action):
        return cls._action_key(action)

    @classmethod
    def _plan_monsters(cls, item):
        if not isinstance(item, dict):
            raise TypeError("Shared plan rows must be mappings")
        if "plan" not in item:
            raise KeyError("Shared plan row is missing canonical 'plan'")
        return list(cls._canonical_action(item["plan"]))

    @classmethod
    def _normalize_plan(cls, item, action_type=None):
        """Normalize Shared BP recommendation rows to one action schema."""
        if not isinstance(item, dict):
            raise TypeError("Shared plan rows must be mappings")
        normalized = dict(item)
        action = list(cls._plan_monsters(normalized))
        detail = dict(normalized.get("detail", {}) or {})
        kind = action_type or normalized.get("action_type")
        if kind is None:
            raise KeyError("Shared plan row is missing canonical 'action_type'")
        if kind not in {"single", "pair", "ban"}:
            raise ValueError(f"未知的 Shared 动作类型: {kind!r}")
        expected_count = 1 if kind in {"single", "ban"} else 2
        if len(action) != expected_count:
            raise ValueError(f"Shared {kind} 动作必须包含 {expected_count} 个魔灵")
        if "pick_count" not in normalized or int(normalized["pick_count"]) != expected_count:
            raise ValueError(f"Shared {kind} 动作的 pick_count 不正确")
        normalized["plan"] = action
        action_key = cls._action_key(action)
        supplied_key = normalized.get("action_key")
        if supplied_key is not None and str(supplied_key) != action_key:
            raise ValueError("Shared plan row 的 action_key 与 plan 不一致")
        normalized["action_key"] = action_key
        normalized["action_type"] = kind
        normalized["pick_count"] = expected_count
        detail.update(
            {
                "action_key": normalized["action_key"],
                "action_type": kind,
                "pick_count": len(action),
            }
        )
        normalized["detail"] = detail
        return normalized

    @classmethod
    def _normalize_plans(cls, plans, action_type=None):
        """Canonicalize and de-duplicate Shared rows."""
        selected = {}
        for raw in plans or ():
            item = cls._normalize_plan(raw, action_type=action_type)
            if not item["action_key"]:
                continue
            key = item["action_key"]
            if key not in selected or float(item["plan_score"]) > float(selected[key]["plan_score"]):
                selected[key] = item
        return list(selected.values())

    @classmethod
    def _canonicalize_state(cls, state):
        if not isinstance(state, dict):
            raise TypeError("BP state must be a canonical mapping")
        normalized = dict(state)
        first_pick_side = normalize_name(normalized.get("first_pick_side"))
        if first_pick_side not in {"me", "enemy"}:
            raise ValueError("BP state requires first_pick_side='me' or 'enemy'")
        required = ("events", "turn", "current_side")
        missing = [key for key in required if key not in normalized]
        if missing:
            raise KeyError(f"BP state is missing canonical fields: {', '.join(missing)}")
        timeline = events_from_state(normalized)
        validate_timeline(timeline, first_pick_side)
        normalized["events"] = timeline.to_dict()
        normalized["my_picks"] = list(timeline.picks_for("me"))
        normalized["enemy_picks"] = list(timeline.picks_for("enemy"))
        normalized["turn"] = int(normalized["turn"])
        if normalized["turn"] != timeline.slots_used + 1:
            raise ValueError("BP state turn does not match canonical events")
        expected_side = "me" if normalized["turn"] == 11 else side_for_order(normalized["turn"], first_pick_side)
        if expected_side is None or normalize_name(normalized["current_side"]) != expected_side:
            raise ValueError("BP state current_side does not match turn and first_pick_side")
        normalized["current_side"] = expected_side
        return normalized

    @staticmethod
    def _state_key(state):
        state = BPDecisionEngine._canonicalize_state(state)
        timeline = events_from_state(state)
        events_key = tuple(
            (
                event.side,
                event.action_type,
                tuple(normalize_name(value) for value in event.monsters),
            )
            for event in timeline.events
        )
        return (
            events_key,
            int(state["turn"]),
            normalize_name(state["first_pick_side"]),
            normalize_name(state["current_side"]),
            normalize_name(state.get("my_leader")),
            normalize_name(state.get("enemy_leader")),
        )

    @staticmethod
    def _side(state):
        return side_for_order(
            int(state["turn"]),
            state["first_pick_side"],
        )

    @staticmethod
    def _transition(state, side, action):
        next_state = BPDecisionEngine._canonicalize_state(state)
        values = list(BPDecisionEngine._canonical_action(action))
        if not values:
            raise ValueError("a search transition requires a non-empty action")
        return transition_state(
            next_state,
            {"side": side, "action_type": "pick", "monsters": values},
        )

    @classmethod
    def _distribution_rows(cls, raw):
        """Normalize the Shared opponent head's canonical rows."""
        rows = []
        for entry in raw or ():
            if not isinstance(entry, dict):
                raise TypeError("Shared opponent rows must be mappings")
            item = cls._normalize_plan(entry, action_type=entry.get("action_type"))
            probability = float(entry["probability"])
            if probability < 0.0:
                raise ValueError("opponent probabilities must be non-negative")
            item["probability"] = probability
            item["opponent_probability"] = probability
            item.setdefault("source", "shared_opponent_policy")
            item.setdefault("detail", {})["opponent_probability"] = probability
            rows.append(item)

        # A policy map can contain both ``[b, a]`` and ``[a, b]``.  Collapse
        # these before normalizing probabilities so the search sees one event.
        rows = cls._normalize_plans(rows)
        total = sum(row["probability"] for row in rows)
        if total > 0.0:
            for row in rows:
                normalized = row["probability"] / total
                row["probability"] = normalized
                row["opponent_probability"] = normalized
                row["plan_score"] = normalized
                row.setdefault("detail", {})["opponent_probability"] = normalized
        return rows

    def _opponent_action_distribution(self, state, limit):
        """Read the opponent head's legal action distribution."""
        model = self.model
        payload = {
            "my_picks": list(state["my_picks"]),
            "enemy_picks": list(state["enemy_picks"]),
            "turn": int(state["turn"]),
            "first_pick_side": state["first_pick_side"],
            "top_k": int(limit),
            "candidate_pool": self.assets.get("options"),
            "max_pair_candidates": int(state.get("max_pair_candidates", 400)),
            "events": state["events"],
        }
        rows = self._distribution_rows(model.predict_opponent_distribution(**payload))
        expected = infer_pick_count_for_side(
            state["my_picks"],
            state["enemy_picks"],
            state["first_pick_side"],
            "enemy",
        )
        matching = [row for row in rows if int(row.get("pick_count", 0)) == expected]
        if not matching:
            raise RuntimeError("Shared opponent policy returned no legal action for the current BP turn")
        return matching[: int(limit)]

    def _candidate_actions(self, state, side, limit):
        state = self._canonicalize_state(state)
        if side not in {"me", "enemy"}:
            raise ValueError("BP search side must be me or enemy")
        key = (side, self._state_key(state), int(limit))
        cached = self._candidate_cache.get(key)
        if cached is not None:
            return cached
        if int(state["turn"]) > 10:
            return []
        common = {
            "first_pick_side": state["first_pick_side"],
            "candidate_pool": self.assets.get("options"),
            "max_pair_candidates": int(state.get("max_pair_candidates", 80)),
        }
        if side == "enemy":
            distribution = self._opponent_action_distribution(state, limit)
            self._candidate_cache[key] = distribution
            return distribution
        if side == "me":
            plans = self.model.recommend_pick_plans(
                my_picks=list(state["my_picks"]),
                enemy_picks=list(state["enemy_picks"]),
                turn=int(state["turn"]),
                my_box=state.get("my_box"),
                # A missing Box means "no account constraint".  The UI sets
                # ``use_my_box`` explicitly when it wants the legality mask.
                use_my_box_for_reco=bool(
                    state.get("use_my_box", False)
                ),
                events=state["events"],
                top_k=int(limit),
                **common,
            )
        else:
            plans = self.model.recommend_enemy_next_picks(
                my_picks=list(state["my_picks"]),
                enemy_picks=list(state["enemy_picks"]),
                turn=int(state["turn"]),
                top_k=int(limit),
                events=state["events"],
                **common,
            )
        self._candidate_cache[key] = self._normalize_plans(plans)
        return self._candidate_cache[key]

    def _value_key(self, state):
        return self._state_key(state)

    def _evaluate_states(self, states):
        states = [self._canonicalize_state(state) for state in (states or [])]
        if not states:
            return []
        values = [None] * len(states)
        pending = {}
        pending_indices = {}
        for index, state in enumerate(states):
            key = self._value_key(state)
            if key in self._value_cache:
                values[index] = self._value_cache[key]
            else:
                pending.setdefault(key, state)
                pending_indices.setdefault(key, []).append(index)
        if pending:
            evaluations = self.model.evaluate_states(list(pending.values()), return_detail=True)
            for key, evaluation in zip(pending, evaluations):
                value = float(evaluation["calibrated_win_rate"])
                self._value_cache[key] = value
                for index in pending_indices[key]:
                    values[index] = value
        if any(value is None for value in values):
            raise RuntimeError("Shared BP value head returned an incomplete evaluation")
        return [float(value) for value in values]

    @staticmethod
    def _evidence_score(detail):
        components = []
        if detail["draft_stats_source"] not in {None, "unavailable"}:
            coverage = min(
                float(detail["draft_stats_coverage"]),
                float(detail["draft_stats_data_coverage"]),
            )
            if coverage > 0.0:
                components.append((float(detail["draft_stats_score"]), coverage))
        if detail["mechanism_available"]:
            coverage = float(detail["mechanism_coverage"])
            if coverage > 0.0:
                components.append((float(detail["mechanism_score"]), coverage))
        if detail["matchup_available"]:
            coverage = float(detail["matchup_coverage"])
            if coverage > 0.0:
                components.append((float(detail["matchup_score"]), coverage))
        if not components:
            return 0.0
        total = sum(weight for _value, weight in components)
        return sum(value * weight for value, weight in components) / total

    @staticmethod
    def _decision_confidence(detail):
        """Combine model coverage and external evidence coverage for risk control."""
        evidence_confidence = float(detail["evidence_confidence"])
        return max(0.0, min(1.0, evidence_confidence))

    @staticmethod
    def _weighted_response_value(rows):
        if not rows:
            return None, None, []
        values = [float(row["value"]) for row in rows]
        probabilities = [float(row["probability"]) for row in rows]
        if any(value < 0.0 for value in probabilities):
            raise ValueError("response probabilities must be non-negative")
        weights = probabilities
        total = sum(weights)
        if total <= 0.0:
            raise ValueError("response probabilities must contain positive mass")
        expected = sum(value * weight for value, weight in zip(values, weights)) / total
        return min(values), expected, rows

    def _my_tree(self, root_plan, state):
        root_plan = self._normalize_plan(root_plan)
        state = self._canonicalize_state(state)
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
            if not responses:
                raise RuntimeError("Shared opponent policy returned no response for a non-terminal BP state")
            for response in responses:
                counter = self._plan_monsters(response)
                if not counter:
                    raise ValueError("Shared opponent response contains an empty action")
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
                            raise ValueError("Shared continuation contains an empty action")
                        leaves.append(self._transition(enemy_state, "me", followup))
                        continuation_meta.append(followup)
                    if not leaves:
                        raise RuntimeError("Shared policy returned no continuation for a non-terminal BP state")
                rows.extend(leaves)
                response_rows.append({"leaves": leaves, "continuations": continuation_meta})
                response_meta.append(
                    {
                        "plan": counter,
                        "priority": float(response["plan_score"]),
                        "probability": float(response["probability"]),
                        "action_key": response["action_key"],
                    }
                )
        return rows, response_rows, response_meta

    def rank_my_plans(self, plans, state, top_k):
        plans = self._normalize_plans(plans)
        if not plans:
            return []
        state = self._canonicalize_state(state)
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
            policy_score = float(detail["normalized_policy_score"])
            evidence_score = self._evidence_score(detail)
            decision_confidence = self._decision_confidence(detail)
            policy_component = policy_score
            uncertainty_penalty = 0.025 * (1.0 - decision_confidence)
            fused = (
                self.config.robust_weight * robust
                + self.config.immediate_weight * immediate
                + self.config.policy_weight * policy_component
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
                    "policy_component": round(policy_component, 6),
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
                            "probability": (
                                None
                                if row.get("probability") is None
                                else round(float(row["probability"]), 6)
                            ),
                            "action_key": row.get("action_key", self._action_key(row["plan"])),
                        }
                        for row in response_values
                    ],
                    "search_depth": 3,
                    "ranking_source": "robust_search_value_3ply_shared_stats_mechanism",
                }
            )
            item["detail"] = detail
            enriched.append(item)
        enriched.sort(key=lambda row: float(row.get("plan_score", 0.0)), reverse=True)
        return enriched[: int(top_k)]

    def rank_enemy_plans(self, plans, state, top_k):
        plans = self._normalize_plans(plans)
        if not plans:
            return []
        state = self._canonicalize_state(state)
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
            if self._side(post_state) == "me" and not response_plans:
                raise RuntimeError("Shared policy returned no response for a non-terminal BP state")
            rows = [post_state]
            for response_plan in response_plans:
                response = self._plan_monsters(response_plan)
                if not response:
                    raise ValueError("Shared response contains an empty action")
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
            best_response = max(response_values) if response_values else immediate
            item = dict(plan)
            detail = dict(item.get("detail", {}) or {})
            policy_score = float(detail["normalized_policy_score"])
            evidence_score = self._evidence_score(detail)
            decision_confidence = self._decision_confidence(detail)
            policy_component = policy_score
            uncertainty_penalty = 0.025 * (1.0 - decision_confidence)
            final_score = (
                0.60 * (1.0 - best_response)
                + 0.15 * (1.0 - immediate)
                + 0.15 * policy_component
                + 0.10 * evidence_score
                - uncertainty_penalty
            )
            response_rows = []
            for response_plan, response_value in zip(response_plans, response_values):
                response_rows.append(
                    {
                        "plan": self._plan_monsters(response_plan),
                        "win_rate": round(float(response_value), 6),
                        "priority": round(float(response_plan["plan_score"]), 6),
                    }
                )
            item["estimated_win_rate"] = round(immediate, 6)
            item["plan_score"] = round(final_score, 6)
            detail.update(
                {
                    "post_enemy_pick_win_rate": round(immediate, 6),
                    "best_response_win_rate": round(best_response, 6),
                    "enemy_adversarial_score": round(final_score, 6),
                    "policy_component": round(policy_component, 6),
                    "evidence_component": round(evidence_score, 6),
                    "decision_confidence": round(decision_confidence, 6),
                    "uncertainty_penalty": round(uncertainty_penalty, 6),
                    "response_plans": response_rows,
                    "search_depth": 2,
                    "ranking_source": "robust_enemy_search_value_shared_stats_mechanism",
                }
            )
            item["detail"] = detail
            enriched.append(item)
        enriched.sort(key=lambda row: float(row["plan_score"]), reverse=True)
        return enriched[: int(top_k)]
