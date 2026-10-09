import unittest

from bp_core.bp_common import side_for_order
from bp_core.action_space import timeline_from_picks
from bp_recommender.ban_value import BPBanValueRecommender
from bp_recommender.decision_engine import BPDecisionEngine
from bp_recommender.relation import BPRelationRecommender
from bp_recommender.value_influence import (
    influence_weight,
    median_absolute_residual,
    value_adjustments,
)
from evaluation.replay import _percentile


def _state(my_picks, enemy_picks, first_pick_side):
    timeline = timeline_from_picks(my_picks, enemy_picks, first_pick_side)
    turn = timeline.slots_used + 1
    return {
        "events": timeline.to_dict(),
        "my_picks": list(timeline.picks_for("me")),
        "enemy_picks": list(timeline.picks_for("enemy")),
        "turn": turn,
        "first_pick_side": first_pick_side,
        "current_side": "me" if turn == 11 else side_for_order(turn, first_pick_side),
    }


class _ValueModel:
    def __init__(self, table, noise):
        self.table = table
        self.value = type("ValueHead", (), {"meta": {"value_residual_noise": noise}})()

    def evaluate_states(self, states, return_detail=True):
        rows = []
        for state in states:
            key = (tuple(state.get("my_picks") or ()), tuple(state.get("enemy_picks") or ()))
            rows.append({"calibrated_win_rate": self.table.get(key, 0.5)})
        return rows


def _plan(name, score):
    return {
        "plan": [name],
        "action_type": "single",
        "pick_count": 1,
        "plan_score": score,
        "detail": {},
    }


class ValueInfluenceTests(unittest.TestCase):
    def test_weight_stays_zero_inside_residual_and_rises_smoothly(self):
        values = [0.45, 0.55]
        self.assertEqual(influence_weight(values, None), 0.0)
        self.assertEqual(influence_weight(values, 1.0), 0.0)
        self.assertAlmostEqual(influence_weight(values, 0.04), 1.0)
        self.assertGreater(influence_weight(values, 0.05), 0.0)
        self.assertLess(influence_weight(values, 0.05), 1.0)

    def test_adjustment_is_centered_on_the_shortlist(self):
        weight, center, adjustments = value_adjustments([0.2, 0.8], 0.01)
        self.assertEqual(weight, 1.0)
        self.assertAlmostEqual(center, 0.5)
        self.assertAlmostEqual(adjustments[0], -0.3)
        self.assertAlmostEqual(adjustments[1], 0.3)

    def test_residual_is_the_weighted_median_absolute_error(self):
        noise = median_absolute_residual([0.2, 0.9, 0.4], [0.0, 1.0, 1.0], [1.0, 1.0, 5.0])
        self.assertAlmostEqual(noise, 0.6)


class DecisionRankingTests(unittest.TestCase):
    def _last_pick_state(self):
        return _state(
            ("m1", "m2", "m3", "m4"),
            ("e1", "e2", "e3", "e4", "e5"),
            "enemy",
        )

    def test_flat_value_keeps_policy_order(self):
        state = self._last_pick_state()
        base = (tuple(state["my_picks"]), tuple(state["enemy_picks"]))
        alpha = (tuple(state["my_picks"]) + ("alpha",), tuple(state["enemy_picks"]))
        beta = (tuple(state["my_picks"]) + ("beta",), tuple(state["enemy_picks"]))
        model = _ValueModel({base: 0.5, alpha: 0.2, beta: 0.9}, noise=1.0)
        ranked = BPDecisionEngine(model=model).rank_my_plans(
            [_plan("alpha", 0.8), _plan("beta", 0.7)],
            state,
            2,
        )
        self.assertEqual([row["plan"] for row in ranked], [["alpha"], ["beta"]])
        self.assertEqual(ranked[0]["detail"]["value_influence"], 0.0)

    def test_separated_value_moves_the_policy_score_continuously(self):
        state = self._last_pick_state()
        base = (tuple(state["my_picks"]), tuple(state["enemy_picks"]))
        alpha = (tuple(state["my_picks"]) + ("alpha",), tuple(state["enemy_picks"]))
        beta = (tuple(state["my_picks"]) + ("beta",), tuple(state["enemy_picks"]))
        model = _ValueModel({base: 0.5, alpha: 0.2, beta: 0.9}, noise=0.01)
        ranked = BPDecisionEngine(model=model).rank_my_plans(
            [_plan("alpha", 0.8), _plan("beta", 0.7)],
            state,
            2,
        )
        self.assertEqual(ranked[0]["plan"], ["beta"])
        self.assertEqual(ranked[0]["detail"]["value_influence"], 1.0)
        self.assertGreater(ranked[0]["detail"]["value_adjustment"], 0.0)
        self.assertLess(ranked[1]["detail"]["value_adjustment"], 0.0)

    def test_opponent_order_follows_policy_not_threat(self):
        state = _state(
            ("m1", "m2", "m3", "m4", "m5"),
            ("e1", "e2", "e3", "e4"),
            "me",
        )
        self.assertEqual(state["current_side"], "enemy")
        base = (tuple(state["my_picks"]), tuple(state["enemy_picks"]))
        likely = (tuple(state["my_picks"]), tuple(state["enemy_picks"]) + ("likely",))
        threat = (tuple(state["my_picks"]), tuple(state["enemy_picks"]) + ("threat",))
        model = _ValueModel({base: 0.5, likely: 0.8, threat: 0.1}, noise=0.01)
        ranked = BPDecisionEngine(model=model).rank_enemy_plans(
            [_plan("threat", 0.2), _plan("likely", 0.9)],
            state,
            2,
        )
        self.assertEqual([row["plan"] for row in ranked], [["likely"], ["threat"]])
        self.assertEqual(ranked[0]["detail"]["ranking_source"], "opponent_policy")


class SynergyShortlistTests(unittest.TestCase):
    def test_low_scoring_synergy_pair_survives_the_scoring_cap(self):
        actions = [("hot", "pick"), ("core", "pair")]
        selected = BPRelationRecommender._select_pair_indices_with_synergy(
            actions,
            [0.9, 0.1],
            {frozenset(("core", "pair"))},
            score_limit=1,
            reserve_count=1,
        )
        self.assertEqual(selected, [1])

    def test_returned_shortlist_keeps_a_synergy_seat(self):
        plans = [
            {"plan": ["a", "b"], "plan_score": 0.9, "action_key": "a|b", "detail": {}},
            {"plan": ["c", "d"], "plan_score": 0.2, "action_key": "c|d", "detail": {"pair_candidate_anchor": True}},
            {"plan": ["e", "f"], "plan_score": 0.8, "action_key": "e|f", "detail": {}},
        ]
        reserved = BPRelationRecommender._reserve_synergy_plans(plans, 2)
        self.assertIn("c|d", {row["action_key"] for row in reserved})
        self.assertEqual(len(reserved), 2)


class BanRankingTests(unittest.TestCase):
    def _recommender(self, noise):
        recommender = object.__new__(BPBanValueRecommender)
        recommender.meta = {"model_type": "test", "value_residual_noise": noise}
        recommender.prob_temperature = 1.0
        recommender.prob_prior_weight = 0.0
        recommender._known_model_slugs = set()
        recommender._to_slug = lambda value: str(value)
        recommender._to_display_name = lambda value: str(value)
        recommender._to_image_name = lambda value: str(value)

        def score_ban_action(my_picks, enemy_picks, target):
            high = target == "core"
            return {
                "mechanism_score": 0.8 if high else 0.2,
                "mechanism_available": True,
                "mechanism_coverage": 1.0,
                "mechanism_threat_reduction": 0.4 if high else 0.0,
                "mechanism_threats": [],
                "mechanism_reasons": [],
                "ban_priority_score": 0.7 if high else 0.2,
                "target_solo_coverage": 1.0,
                "target_contest_coverage": 0.0,
                "target_contest_score": 0.5,
                "target_matchup_score": 0.5,
                "target_matchup_coverage": 0.0,
                "source": "test",
                "evidence": [],
            }

        recommender.lucksack_stats = type("Stats", (), {"score_ban_action": staticmethod(score_ban_action)})()

        def evaluate_states(states, return_detail=True):
            rows = []
            for state in states:
                ban = state.get("my_ban")
                win_rate = 0.2 if ban == "core" else 0.95 if ban == "bait" else 0.5
                rows.append({"calibrated_win_rate": win_rate, "raw_win_rate": win_rate})
            return rows

        recommender.evaluate_states = evaluate_states
        return recommender

    def test_flat_value_ranks_the_mechanism_ban_first(self):
        plans = self._recommender(1.0).recommend_bans(["m1"], ["core", "bait"], top_k=2)
        self.assertEqual(plans[0]["plan"], ["core"])
        self.assertEqual(plans[0]["detail"]["value_influence"], 0.0)

    def test_separated_value_can_lift_the_other_ban(self):
        plans = self._recommender(0.01).recommend_bans(["m1"], ["core", "bait"], top_k=2)
        self.assertEqual(plans[0]["plan"], ["bait"])
        self.assertEqual(plans[0]["detail"]["value_influence"], 1.0)


class ReplayPercentileTests(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertAlmostEqual(_percentile([10, 20, 30, 40], 0.95), 38.5)


if __name__ == "__main__":
    unittest.main()
