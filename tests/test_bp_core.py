import unittest

import torch

from bp_core.account_profile import AccountProfile
from bp_core.action_space import DraftActionEvent, DraftTimeline, events_from_state, transition_state
from bp_core.bp_common import side_for_order
from bp_core.bp_network import BPSharedNetwork
from bp_core.draft_state import DraftState
from bp_core.rank_tiers import get_rank_context_from_record
from bp_recommender.decision_engine import BPDecisionEngine
from bp_recommender.shared import BPSharedRecommender


class BPCoreTests(unittest.TestCase):
    def test_account_profile_contains_only_owned_monsters(self):
        profile = AccountProfile.from_payload(
            {
                "owned_monsters": ["Monster Name"],
                "source": "manual_normalized",
            }
        )
        self.assertEqual(profile.owned_monsters, {"monster name"})
        self.assertEqual(profile.source, "manual_normalized")

    def test_turn_pattern_and_canonical_events(self):
        self.assertEqual(side_for_order(1, "me"), "me")
        self.assertEqual(side_for_order(2, "me"), "enemy")
        state = {
            "events": DraftTimeline(
                (
                    DraftActionEvent("me", ("a",)),
                    DraftActionEvent("enemy", ("x", "y")),
                    DraftActionEvent("me", ("b", "c")),
                )
            ).to_dict(),
            "turn": 6,
            "first_pick_side": "me",
        }
        events = events_from_state(state)
        self.assertEqual([event.side for event in events.events], ["me", "enemy", "me"])
        self.assertEqual(events.events[1].monsters, ("x", "y"))

    def test_pair_members_are_unordered_and_event_order_is_preserved(self):
        timeline = DraftTimeline((
            DraftActionEvent("me", ("a",)),
            DraftActionEvent("enemy", ("z", "b")),
            DraftActionEvent("me", ("c",)),
        ))
        self.assertEqual(timeline.events[1].monsters, ("b", "z"))
        self.assertEqual([event.side for event in timeline.events], ["me", "enemy", "me"])

    def test_transition_keeps_pair_atomic(self):
        state = DraftState(
            turn=2,
            first_pick_side="me",
            my_picks=("first",),
            events=(DraftActionEvent("me", ("first",)),),
        ).to_dict()
        updated = transition_state(
            state,
            {"side": "enemy", "action_type": "pick", "monsters": ["z", "a"]},
        )
        self.assertEqual(updated["enemy_picks"], ["a", "z"])
        self.assertEqual(updated["turn"], 4)
        self.assertEqual(updated["events"][1]["monsters"], ["a", "z"])

    def test_shared_network_pair_score_is_symmetric(self):
        torch.manual_seed(7)
        model = BPSharedNetwork(24, embed_dim=8, state_dim=8, hidden_dim=8, dropout=0.0, num_heads=2, num_layers=1, ffn_dim=16).eval()
        state = model(
            torch.zeros((1, 0, 2), dtype=torch.long),
            torch.zeros((1, 0, 2), dtype=torch.bool),
            torch.zeros((1, 0), dtype=torch.long),
            torch.zeros((1, 0), dtype=torch.long),
            turn=1,
            first_pick_side="me",
            current_side="me",
        )
        first = torch.tensor([[[7, 11]]])
        second = torch.tensor([[[11, 7]]])
        self.assertEqual(model.score_pair_candidates(state["state"], first).item(), model.score_pair_candidates(state["state"], second).item())

    def test_decision_engine_caches_shared_value_evaluation(self):
        class FakeShared:
            def __init__(self):
                self.calls = 0

            def evaluate_states(self, states, return_detail=True):
                self.calls += len(states)
                return [{"calibrated_win_rate": 0.7} for _ in states]

        model = FakeShared()
        engine = BPDecisionEngine(model=model)
        state = {
            "events": [],
            "my_picks": [],
            "enemy_picks": [],
            "turn": 1,
            "current_side": "me",
            "first_pick_side": "me",
        }
        self.assertEqual(engine._evaluate_states([state, state]), [0.7, 0.7])
        self.assertEqual(model.calls, 1)

    def test_shared_candidate_mask_excludes_unknown_and_occupied(self):
        recommender = object.__new__(BPSharedRecommender)
        recommender.known_slugs = frozenset({"a", "b", "c"})
        recommender._to_slug = lambda value: str(value or "").lower()
        allowed = recommender._allowed_candidates(["a"], [], ["a", "b", "unknown"], use_my_box_for_reco=False)
        self.assertEqual(allowed, ["b"])

    def test_rank_context_prefers_explicit_sample_weight(self):
        context = get_rank_context_from_record({"rank_tier": "three_red", "rank_weight": 1.25, "sample_weight": 0.75})
        self.assertEqual(context["sample_weight"], 0.75)


if __name__ == "__main__":
    unittest.main()
