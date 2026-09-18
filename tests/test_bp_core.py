import unittest
from types import SimpleNamespace

import torch

from bp_core.relation_common import side_for_order
from bp_core.account_profile import AccountProfile
from bp_recommender.ban_value import BPBanValueRecommender
from bp_core.draft_state import DraftState
from bp_recommender.decision_engine import BPDecisionEngine
from bp_core.feasibility import score_plan_feasibility
from bp_recommender.prefix_value_common import BPPrefixValueNet
from bp_recommender.prefix_value import BPPrefixValueRecommender
from bp_core.relation_common import BPDraftRelationNetV2
from bp_core.relation_common import MonsterVocab
from bp_recommender.relation import BPRelationRecommender
from bp_core.swex_profile import parse_swex_payload
from bp_core.transformer_common import PAD_TOKEN, UNK_TOKEN
from bp_core.rank_tiers import get_rank_context_from_record


class BPCoreTests(unittest.TestCase):
    def test_swex_profile_maps_units_and_keeps_unknown_ids_unparsed(self):
        profile = parse_swex_payload(
            {
                "unit_list": [
                    {
                        "unit_master_id": 11014,
                        "spd": 250,
                        "con": 12000,
                        "runes": [
                            {"rune_id": 10, "slot_no": 2, "set_id": 3}
                        ],
                    },
                    {"unit_master_id": 999999, "spd": 300},
                ],
                "runes": [
                    {
                        "rune_id": 20,
                        "slot_no": 4,
                        "set_id": 13,
                        "pri_eff": ["9", "63"],
                        "sec_eff": [["8", "18", "0", "0"]],
                    }
                ],
            },
            catalog=[{"com2us_id": "11014", "slug": "belladeon", "slabel": "Belladeon"}],
        )

        self.assertEqual(profile.owned_monsters, {"belladeon"})
        self.assertEqual(profile.runes_by_monster["belladeon"].speed, 250.0)
        self.assertEqual(profile.runes_by_monster["belladeon"].rune_set_ids, ("3",))
        self.assertEqual(profile.rune_inventory_count, 2)
        self.assertEqual(profile.rune_inventory[0].secondary_stats, ((8.0, 18.0, 0.0, 0.0),))
        self.assertIn("unit_master_id:999999", profile.unparsed_fields)
        self.assertIn("rta_equipment_pool", profile.unparsed_fields)
    def test_decision_engine_uses_own_followup_after_opponent_response(self):
        class FakeRelation:
            def recommend_enemy_next_picks(self, **_kwargs):
                return [{"plan": ["Enemy Counter"], "plan_score": 0.9}]

            def recommend_pick_plans(self, **kwargs):
                if kwargs["my_picks"] == ["Alpha"] and kwargs["enemy_picks"] == ["Enemy Counter"]:
                    return [{"plan": ["Own Followup"], "plan_score": 0.8}]
                return []

        class FakeValue:
            def evaluate_states(self, states, return_detail=False):
                values = []
                for state in states:
                    self.last_states = getattr(self, "last_states", []) + [dict(state)]
                    picks = state["my_picks"]
                    value = 0.9 if "Own Followup" in picks else 0.4 if "Enemy Counter" in state["enemy_picks"] else 0.5
                    values.append({"calibrated_win_rate": value})
                return values

        value_model = FakeValue()
        engine = BPDecisionEngine(
            FakeRelation(),
            value_model,
            {"options": ["Alpha", "Enemy Counter", "Own Followup"]},
        )
        plans = engine.rank_my_plans(
            [{"plan": ["Alpha"], "plan_score": 0.8, "detail": {"normalized_relation_score": 0.8}}],
            {
                "my_picks": [],
                "enemy_picks": [],
                "turn": 1,
                "first_pick_side": "me",
                "my_box": [],
                "use_my_box": False,
            },
            1,
        )

        self.assertEqual(plans[0]["detail"]["search_depth"], 3)
        self.assertEqual(plans[0]["detail"]["counter_plans"][0]["plan"], ["Enemy Counter"])
        self.assertIn("enemy", {state.get("current_side") for state in value_model.last_states})
        next_state = BPDecisionEngine._transition(
            {
                "my_picks": ["Alpha"],
                "enemy_picks": [],
                "turn": 2,
                "first_pick_side": "me",
            },
            "enemy",
            ["Enemy Counter", "Enemy Counter 2"],
        )
        self.assertEqual(next_state["current_side"], "me")

    def test_candidate_shortlist_keeps_mechanism_anchor(self):
        allowed = ["relation", "mechanism", "ordinary"]
        selected = BPRelationRecommender._select_candidate_diverse(
            allowed,
            [0.99, 0.40, 0.30],
            [
                {"score": 0.9},
                {"score": 0.4, "mechanism_available": True, "mechanism_score": 0.99},
                {"score": 0.3},
            ],
            2,
        )
        self.assertIn("mechanism", selected)
    def test_account_profile_keeps_only_explicit_rune_fields(self):
        profile = AccountProfile.from_payload(
            {
                "source": "manual",
                "parse_confidence": 0.8,
                "owned_monsters": ["Monster Name"],
                "runes_by_monster": {
                    "Monster Name": {
                        "speed": 312,
                        "verified_fields": ["speed"],
                        "unknown_value": 99,
                    }
                },
                "raw_swex": {"opaque": True},
            }
        )

        self.assertEqual(profile.owned_monsters, {"monster name"})
        self.assertEqual(profile.runes_by_monster["monster name"].speed, 312)
        self.assertTrue(profile.has_verified_runes)
        self.assertEqual(profile.unparsed_fields, ("raw_swex",))

    def test_feasibility_does_not_assume_opponent_runes(self):
        profile = AccountProfile.from_payload(
            {
                "parse_confidence": 1.0,
                "owned_monsters": ["Alpha"],
                "runes_by_monster": {
                    "Alpha": {"speed": 300, "verified_fields": ["speed"]}
                },
            }
        )

        own = score_plan_feasibility({"plan": ["Alpha"]}, profile)
        unknown = score_plan_feasibility({"plan": ["Enemy"]}, None)

        self.assertEqual(own.owned, True)
        self.assertTrue(own.verified_rune_data)
        self.assertIsNone(unknown.owned)
        self.assertEqual(unknown.score, 0.5)

    def test_draft_state_reports_turn_side_without_opponent_build(self):
        state = DraftState(turn=2, first_pick_side="enemy")

        self.assertEqual(state.current_side, "me")
        self.assertFalse(state.opponent_runes_known)
        self.assertIsNone(state.to_dict()["account_profile"])

    def test_unknown_first_pick_side_is_not_assigned(self):
        self.assertEqual(side_for_order(1, "me"), "me")
        self.assertEqual(side_for_order(1, "enemy"), "enemy")
        self.assertIsNone(side_for_order(1, None))
        self.assertIsNone(side_for_order(1, "unexpected"))

    def test_sample_weight_has_priority_over_rank_weight(self):
        context = get_rank_context_from_record(
            {
                "rank_tier": "three_red",
                "rank_weight": 1.25,
                "sample_weight": 0.75,
            }
        )
        self.assertEqual(context["sample_weight"], 0.75)

    def test_relation_state_cache_matches_forward(self):
        torch.manual_seed(7)
        model = BPDraftRelationNetV2(
            vocab_size=20,
            embed_dim=8,
            state_dim=8,
            action_dim=8,
            hidden_dim=8,
            dropout=0.0,
            pad_idx=0,
            num_heads=2,
            num_layers=1,
            ffn_dim=16,
        ).eval()
        my_picks = torch.randint(0, 20, (3, 5))
        enemy_picks = torch.randint(0, 20, (3, 5))
        my_mask = torch.ones(3, 5)
        enemy_mask = torch.ones(3, 5)
        candidate_1 = torch.randint(0, 20, (3,))
        candidate_2 = torch.randint(0, 20, (3,))
        action_mask = torch.ones(3, 2)
        turn = torch.ones(3)
        first_pick_side = torch.ones(3)
        current_side = torch.ones(3)

        direct = model(
            my_picks,
            my_mask,
            enemy_picks,
            enemy_mask,
            candidate_1,
            candidate_2,
            action_mask,
            turn,
            first_pick_side,
            current_side=current_side,
        )
        state_cache = model.encode_state(
            my_picks,
            my_mask,
            enemy_picks,
            enemy_mask,
            turn,
            first_pick_side,
            current_side=current_side,
        )
        cached = model.score_from_state(
            state_cache,
            candidate_1,
            candidate_2,
            action_mask,
        )
        torch.testing.assert_close(direct, cached)

    def test_enemy_double_turn_predicts_pair_action(self):
        class FakeRecommender:
            def _swap_first_pick_side(self, first_pick_side):
                return "enemy" if first_pick_side == "me" else "me"

            def _to_slug(self, value):
                return str(value or "").lower()

            def recommend_pick_plans(self, **kwargs):
                self.kwargs = kwargs
                return [
                    {
                        "plan": ["Moore", "Cheongpung"],
                        "plan_score": 0.91,
                        "detail": {
                            "relation_score": 2.0,
                            "normalized_relation_score": 0.8,
                            "action_type": "pair",
                        },
                    }
                ]

        fake = FakeRecommender()
        predictions = BPRelationRecommender.recommend_enemy_next_picks(
            fake,
            my_picks=["Oliver"],
            enemy_picks=[],
            first_pick_side="me",
            top_k=3,
        )

        self.assertEqual(fake.kwargs["pick_count"], 2)
        self.assertEqual(predictions[0]["plan"], ["Moore", "Cheongpung"])
        self.assertEqual(predictions[0]["detail"]["action_type"], "pair")

    def test_empty_candidate_pool_stays_empty(self):
        class FakeVocab:
            def get_id(self, _slug):
                return 2

        class FakeRecommender:
            vocab = FakeVocab()

            def _to_slug(self, value):
                return str(value or "").lower()

            def _to_display_name(self, value):
                return value

        recommender = FakeRecommender()
        allowed = BPRelationRecommender._allowed_candidates(
            recommender,
            my_picks=[],
            enemy_picks=[],
            candidate_pool=[],
            use_my_box_for_reco=False,
        )
        self.assertEqual(allowed, [])

    def test_prefix_batch_matches_single_state_evaluation(self):
        torch.manual_seed(11)
        recommender = object.__new__(BPPrefixValueRecommender)
        recommender.device = "cpu"
        recommender.meta = {"model_type": "transformer_encoder"}
        recommender.vocab = MonsterVocab(
            {
                PAD_TOKEN: 0,
                UNK_TOKEN: 1,
                "alpha": 2,
                "beta": 3,
                "gamma": 4,
            }
        )
        recommender.display_name_map = {}
        recommender.name_slug_aliases = {}
        recommender.prob_calibration = {
            "temperature": 1.0,
            "prior_weight": 0.0,
            "clip_min": 1e-4,
            "clip_max": 1.0 - 1e-4,
        }
        recommender.model = BPPrefixValueNet(
            vocab_size=5,
            embed_dim=8,
            hidden_dim=8,
            dropout=0.0,
            pad_idx=0,
            num_heads=2,
            num_layers=1,
            ffn_dim=16,
        ).eval()
        states = [
            {
                "my_picks": ["alpha"],
                "enemy_picks": ["beta"],
                "turn": 3,
                "first_pick_side": "me",
                "current_side": "enemy",
            },
            {
                "my_picks": ["gamma"],
                "enemy_picks": ["alpha", "beta"],
                "turn": 4,
                "first_pick_side": "enemy",
                "current_side": "me",
            },
        ]

        batched = recommender.evaluate_states(states, return_detail=True)
        singles = [
            recommender.evaluate_state(**state, return_detail=True)
            for state in states
        ]

        self.assertEqual(len(batched), len(singles))
        for batch_item, single_item in zip(batched, singles):
            self.assertAlmostEqual(
                batch_item["raw_win_rate"],
                single_item["raw_win_rate"],
                places=7,
            )
            self.assertAlmostEqual(
                batch_item["calibrated_win_rate"],
                single_item["calibrated_win_rate"],
                places=7,
            )

    def test_ban_recommendation_considers_opponent_response(self):
        class FakeStats:
            def score_ban_action(self, **_kwargs):
                return {
                    "mechanism_score": 0.5,
                    "mechanism_available": False,
                    "target_solo_coverage": 0.0,
                    "target_contest_coverage": 0.0,
                    "source": "fake",
                    "evidence": [],
                }

        recommender = object.__new__(BPBanValueRecommender)
        recommender.lucksack_stats = FakeStats()
        recommender.meta = {"model_type": "fake"}
        recommender._known_model_slugs = set()
        recommender.display_name_map = {}
        recommender.image_name_map = {}
        recommender.prob_temperature = 1.0
        recommender.prob_prior_weight = 0.0

        recommender._to_slug = lambda value: str(value or "").lower()
        recommender._to_display_name = lambda value: str(value or "")
        recommender._to_image_name = lambda value: str(value or "")

        def evaluate_states(states, return_detail=False):
            recommender.evaluated_states = states
            values = []
            for state in states:
                target = state.get("my_ban")
                response = state.get("enemy_ban")
                if not target:
                    value = 0.50
                elif target == "threat_a" and response == "carry_a":
                    value = 0.40
                elif target == "threat_a":
                    value = 0.80
                elif response == "carry_a":
                    value = 0.55
                else:
                    value = 0.56
                values.append(
                    {
                        "raw_win_rate": value,
                        "calibrated_win_rate": value,
                    }
                )
            return values

        recommender.evaluate_states = evaluate_states
        plans = BPBanValueRecommender.recommend_bans(
            recommender,
            my_picks=["carry_a", "carry_b"],
            enemy_picks=["threat_a", "threat_b"],
            top_k=2,
        )

        self.assertEqual(len(recommender.evaluated_states), 6)
        self.assertEqual(plans[0]["ban_slug"], "threat_b")
        self.assertTrue(plans[0]["detail"]["response_aware"])
        self.assertEqual(plans[0]["detail"]["worst_enemy_ban_slug"], "carry_a")
        self.assertAlmostEqual(
            plans[0]["detail"]["post_ban_win_rate"],
            0.5515,
            places=6,
        )

    def test_pair_shortlist_keeps_singleton_evidence_candidate(self):
        class FakeStats:
            def candidate_synergy_pairs(self, *_args, **_kwargs):
                return []

            def score_action(self, **kwargs):
                action = tuple(kwargs["action"])
                return {
                    "mechanism_score": 1.0
                    if set(action) == {"m0", "m1"}
                    else 0.5
                }

        recommender = object.__new__(BPRelationRecommender)
        recommender.lucksack_stats = FakeStats()
        recommender.personal_history_model = None
        recommender._known_model_slugs = set()
        recommender._allowed_candidates = lambda *_args, **_kwargs: [
            f"m{index}" for index in range(16)
        ]
        recommender._score_actions_with_unknowns = (
            lambda *_args, **kwargs: [
                1.0
                if len(action) == 1 and action[0] in {"m0", "m1"}
                else 0.5
                if len(action) == 1
                else 0.45
                if set(action) == {"m0", "m1"}
                else 0.5
                for action in kwargs.get("actions", _args[2])
            ]
        )
        recommender._normalize_scores = lambda scores: list(scores)

        def blend_relation_stats(relation_score, stats_detail):
            return (
                max(
                    float(relation_score),
                    float(stats_detail.get("mechanism_score", 0.5)),
                ),
                0.5,
                1.0,
                True,
                0.5,
                False,
            )

        recommender._blend_relation_stats = blend_relation_stats
        recommender._to_display_name = lambda slug: slug

        plans = BPRelationRecommender.recommend_pick_plans(
            recommender,
            my_picks=[],
            enemy_picks=[],
            pick_count=2,
            turn=1,
            first_pick_side="me",
            top_k=6,
            max_pair_candidates=400,
        )

        self.assertTrue(
            any(plan["plan"] == ["m0", "m1"] for plan in plans)
        )

    def test_evidence_diverse_plan_pool_keeps_mechanism_anchor(self):
        plans = [
            {
                "plan": ["relation"],
                "plan_score": 0.95,
                "detail": {
                    "mechanism_available": False,
                    "draft_stats_coverage": 0.0,
                    "matchup_coverage": 0.0,
                },
            },
            {
                "plan": ["mechanism"],
                "plan_score": 0.70,
                "detail": {
                    "mechanism_available": True,
                    "mechanism_score": 0.99,
                    "draft_stats_coverage": 0.0,
                    "matchup_coverage": 0.0,
                },
            },
            {
                "plan": ["ordinary"],
                "plan_score": 0.60,
                "detail": {
                    "mechanism_available": False,
                    "draft_stats_coverage": 0.0,
                    "matchup_coverage": 0.0,
                },
            },
        ]

        selected = BPRelationRecommender._select_evidence_diverse_plans(plans, 2)

        self.assertEqual({row["plan"][0] for row in selected}, {"relation", "mechanism"})

    def test_relation_stats_weight_shrinks_with_low_coverage(self):
        recommender = object.__new__(BPRelationRecommender)
        recommender.lucksack_stats = SimpleNamespace(ready=True)
        full_detail = {
            "score": 0.9,
            "coverage": 1.0,
            "data_coverage": 1.0,
            "mechanism_score": 0.9,
            "mechanism_available": True,
            "mechanism_coverage": 1.0,
            "matchup_score": 0.9,
            "matchup_coverage": 1.0,
        }
        low_detail = dict(full_detail)
        low_detail["coverage"] = 0.05
        low_detail["data_coverage"] = 0.05

        full_score = recommender._blend_relation_stats(0.2, full_detail)[0]
        low_score = recommender._blend_relation_stats(0.2, low_detail)[0]

        self.assertLess(low_score, full_score)

    def test_unknown_relation_score_is_shrunk_toward_neutral(self):
        recommender = object.__new__(BPRelationRecommender)
        recommender.lucksack_stats = SimpleNamespace(ready=False)

        score = recommender._blend_relation_stats(
            0.95,
            {"_relation_model_known": False},
        )[0]

        self.assertGreater(score, 0.5)
        self.assertLess(score, 0.6)


if __name__ == "__main__":
    unittest.main()
