import unittest

import torch

from audit_bp_prefix_data import side_for_order
from bp_relation_common import BPDraftRelationNetV2
from bp_relation_recommender import BPRelationRecommender
from rank_tiers import get_rank_context_from_record


class BPCoreTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
