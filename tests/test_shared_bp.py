import unittest

from bp_core.action_space import (
    DraftActionEvent,
    DraftTimeline,
    canonicalize_pair,
    events_from_state,
    is_legal_action,
    legal_action_sizes,
    next_event_size,
    timeline_from_picks,
    transition_state,
)
from bp_core.bp_common import side_for_order
from bp_core.draft_state import DraftState
from bp_core.lucksack_stats import is_usable_catalog_item
from bp_recommender.decision_engine import BPDecisionEngine


def canonical_state(my_picks=(), enemy_picks=(), first_pick_side="me"):
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


class ActionSpaceTests(unittest.TestCase):
    def test_catalog_keeps_awakened_and_secondary_awakened_only(self):
        self.assertTrue(is_usable_catalog_item({"natural_stars": 5, "searchable": True}))
        self.assertTrue(
            is_usable_catalog_item(
                {"natural_stars": 3, "searchable": True, "label": "Foo [2A]"}
            )
        )
        self.assertFalse(is_usable_catalog_item({"natural_stars": 3, "searchable": True}))
        self.assertFalse(is_usable_catalog_item({"natural_stars": 5, "searchable": False}))

    def test_pair_is_unordered_but_events_are_ordered(self):
        timeline = DraftTimeline(
            (
                DraftActionEvent("me", ("first",)),
                DraftActionEvent("enemy", ("z", "a")),
                DraftActionEvent("me", ("last",)),
            )
        )
        self.assertEqual(timeline.events[1].monsters, ("a", "z"))
        self.assertEqual([event.side for event in timeline.events], ["me", "enemy", "me"])
        self.assertEqual(canonicalize_pair("z", "a"), ("a", "z"))

    def test_turn_pattern_exposes_single_and_double_event_sizes(self):
        self.assertEqual(next_event_size(canonical_state()), 1)
        self.assertEqual(next_event_size(canonical_state(("a",))), 2)
        self.assertEqual(next_event_size(canonical_state(("a",), ("b", "c"))), 2)

    def test_transition_keeps_pair_atomic_and_serializable(self):
        state = canonical_state(("first",))
        updated = transition_state(
            state,
            {"side": "enemy", "action_type": "pick", "monsters": ["z", "a"]},
        )
        self.assertEqual(updated["enemy_picks"], ["a", "z"])
        self.assertEqual(updated["turn"], 4)
        self.assertEqual(updated["events"][1]["monsters"], ["a", "z"])

    def test_canonical_events_are_required(self):
        state = {
            "my_picks": ["old"],
            "enemy_picks": [],
            "turn": 2,
            "first_pick_side": "me",
        }
        with self.assertRaises(KeyError):
            events_from_state(state)

    def test_first_pick_side_is_required_for_event_state(self):
        with self.assertRaises(KeyError):
            events_from_state({"events": [], "turn": 1})

    def test_event_only_state_drives_legality_and_transition(self):
        state = {
            "events": [{"side": "me", "monsters": ["a"]}],
            "turn": 2,
            "first_pick_side": "me",
        }
        self.assertFalse(is_legal_action(state, ["a"], side="enemy"))
        state["current_side"] = "enemy"
        updated = transition_state(
            state,
            {"side": "enemy", "action_type": "pick", "monsters": ["b", "c"]},
        )
        self.assertEqual(updated["my_picks"], ["a"])
        self.assertEqual(updated["enemy_picks"], ["b", "c"])
        self.assertEqual(updated["turn"], 4)

    def test_event_only_dataclass_keeps_both_sides(self):
        state = DraftState(
            first_pick_side="me",
            turn=2,
            my_picks=("a",),
            events=(DraftActionEvent("me", ("a",)),),
        )
        updated = transition_state(
            state.to_dict(),
            {"side": "enemy", "action_type": "pick", "monsters": ["b", "c"]},
        )
        self.assertEqual(updated["my_picks"], ["a"])
        self.assertEqual(updated["enemy_picks"], ["b", "c"])
        self.assertEqual(updated["turn"], 4)

    def test_event_state_requires_explicit_current_side(self):
        state = {
            "events": [
                {"side": "me", "monsters": ["a"]},
                {"side": "enemy", "monsters": ["b", "c"]},
            ],
            "turn": 4,
            "first_pick_side": "me",
        }
        with self.assertRaises(KeyError):
            BPDecisionEngine._canonicalize_state(state)

    def test_timeline_accepts_one_mapping_event(self):
        timeline = DraftTimeline({"side": "me", "monsters": ["a"]})
        self.assertEqual(len(timeline.events), 1)
        self.assertEqual(timeline.events[0].monsters, ("a",))

    def test_timeline_accepts_an_existing_timeline(self):
        source = DraftTimeline(({"side": "me", "monsters": ["a"]},))
        copied = DraftTimeline(source)
        self.assertEqual(copied.to_dict(), source.to_dict())

    def test_draft_state_serializes_timeline_events(self):
        state = DraftState(
            turn=4,
            my_picks=("a",),
            enemy_picks=("b", "c"),
            events=DraftTimeline((
                DraftActionEvent("me", ("a",)),
                DraftActionEvent("enemy", ("b", "c")),
            )),
        )
        serialized = state.to_dict()
        self.assertEqual(events_from_state(serialized).to_dict()[0]["monsters"], ["a"])

    def test_illegal_duplicate_and_wrong_side_actions_are_rejected(self):
        state = canonical_state(("a",))
        self.assertFalse(is_legal_action(state, ["a", "b"], side="enemy"))
        self.assertFalse(is_legal_action(state, ["b"], side="me"))

    def test_event_bans_are_checked_for_duplicate_actions(self):
        timeline = timeline_from_picks(
            ("m1", "m2", "m3", "m4", "m5"),
            ("e1", "e2", "e3", "e4", "e5"),
            "me",
        )
        state = {
            "events": timeline.to_dict()
            + [{"side": "me", "action_type": "ban", "monsters": ["e1"]}],
            "first_pick_side": "me",
            "turn": 11,
        }
        self.assertFalse(
            is_legal_action(
                state,
                {"side": "enemy", "action_type": "ban", "monsters": ["e1"]},
            )
        )

    def test_bans_are_available_only_after_all_pick_slots(self):
        state = canonical_state()
        action = {"side": "me", "action_type": "ban", "monsters": ["e1"]}
        self.assertEqual(legal_action_sizes(state, side="me", action_type="ban"), ())
        self.assertFalse(is_legal_action(state, action))
        with self.assertRaises(ValueError):
            transition_state(state, action)

        complete = canonical_state(
            ("m1", "m2", "m3", "m4", "m5"),
            ("e1", "e2", "e3", "e4", "e5"),
        )
        self.assertEqual(complete["turn"], 11)
        self.assertEqual(legal_action_sizes(complete, side="me", action_type="ban"), (1,))
        self.assertTrue(is_legal_action(complete, action))


class DecisionCanonicalizationTests(unittest.TestCase):
    def test_string_action_is_one_monster(self):
        self.assertEqual(BPDecisionEngine.canonicalize_action("foo"), ("foo",))

    def test_duplicate_state_values_are_evaluated_once_and_reused(self):
        class ValueModel:
            def __init__(self):
                self.calls = 0

            def evaluate_states(self, states, return_detail=True):
                self.calls += len(states)
                return [{"calibrated_win_rate": 0.7} for _ in states]

        model = ValueModel()
        engine = BPDecisionEngine(model=model)
        state = canonical_state()
        self.assertEqual(engine._evaluate_states([state, state]), [0.7, 0.7])
        self.assertEqual(model.calls, 1)

    def test_state_cache_keeps_global_event_order(self):
        first = {
            **canonical_state(("m",), ("e1", "e2")),
            "events": [
                {"side": "me", "monsters": ["m"]},
                {"side": "enemy", "monsters": ["e1", "e2"]},
            ],
        }
        second = {
            **first,
            "events": [
                {"side": "enemy", "monsters": ["e1", "e2"]},
                {"side": "me", "monsters": ["m"]},
            ],
        }
        first["current_side"] = "me"
        second["current_side"] = "me"
        with self.assertRaises(ValueError):
            BPDecisionEngine._state_key(second)

    def test_missing_canonical_state_fields_are_rejected(self):
        with self.assertRaises(KeyError):
            BPDecisionEngine._canonicalize_state(
                {"my_picks": ["a"], "enemy_picks": ["b"], "first_pick_side": "me"}
            )


try:
    import torch
    from bp_core.bp_network import BPSharedNetwork
    from bp_core.bp_common import MonsterVocab
    from bp_core.names import PAD_TOKEN
    from training.shared_bp import SharedActionDataset, encode_state

    class SharedNetworkTests(unittest.TestCase):
        def test_pair_score_is_permutation_invariant_and_empty_state_works(self):
            torch.manual_seed(3)
            model = BPSharedNetwork(24, dropout=0.0).eval()
            empty_ids = torch.zeros((1, 0, 2), dtype=torch.long)
            empty_mask = torch.zeros((1, 0, 2), dtype=torch.bool)
            empty_side = torch.zeros((1, 0), dtype=torch.long)
            empty_type = torch.zeros((1, 0), dtype=torch.long)
            encoded = model(
                empty_ids,
                empty_mask,
                empty_side,
                empty_type,
                turn=1,
                first_pick_side="me",
                current_side="me",
            )
            first = torch.tensor([[[7, 11]]])
            second = torch.tensor([[[11, 7]]])
            self.assertEqual(
                model.score_pair_candidates(encoded["state"], first).item(),
                model.score_pair_candidates(encoded["state"], second).item(),
            )
            self.assertEqual(
                model.score_single_candidates(encoded["state"], [7, 11]).shape,
                (1, 2),
            )
            vocab = MonsterVocab()
            vocab.add_token("a")
            timeline_output = model.forward_timeline(
                DraftTimeline((DraftActionEvent("me", ("a",)),)),
                vocab,
                turn=1,
                first_pick_side="me",
                current_side="me",
            )
            self.assertEqual(tuple(timeline_output["state"].shape), (1, model.state_dim))
            list_output = model(
                [[[7, 0]]],
                [[[1, 0]]],
                [[0]],
                [[1]],
                turn=2,
                first_pick_side="me",
                current_side="me",
            )
            self.assertEqual(tuple(list_output["state"].shape), (1, model.state_dim))

        def test_empty_pair_candidate_pool_returns_empty_scores(self):
            model = BPSharedNetwork(24, dropout=0.0).eval()
            empty_ids = torch.zeros((2, 0, 2), dtype=torch.long)
            empty_mask = torch.zeros((2, 0, 2), dtype=torch.bool)
            empty_side = torch.zeros((2, 0), dtype=torch.long)
            empty_type = torch.zeros((2, 0), dtype=torch.long)
            encoded = model(
                empty_ids,
                empty_mask,
                empty_side,
                empty_type,
                turn=[1, 1],
                first_pick_side=[1, 1],
                current_side=[1, 1],
            )
            scores = model.score_pair_candidates(encoded["state"], empty_ids)
            self.assertEqual(tuple(scores.shape), (2, 0))

        def test_event_type_matches_mask_and_slot_count(self):
            model = BPSharedNetwork(
                24,
                embed_dim=8,
                state_dim=8,
                hidden_dim=8,
                dropout=0.0,
                num_heads=2,
                num_layers=1,
                ffn_dim=16,
            ).eval()
            event_ids = [[[7, 0]]]
            event_mask = [[[1, 0]]]
            event_side = [[0]]
            for event_type in ([[0]], [[2]]):
                with self.subTest(event_type=event_type):
                    with self.assertRaises(ValueError):
                        model(
                            event_ids,
                            event_mask,
                            event_side,
                            event_type,
                            turn=1,
                            first_pick_side="me",
                            current_side="me",
                        )
            with self.assertRaises(ValueError):
                model(
                    [[[7, 8]]],
                    [[[1, 0]]],
                    [[0]],
                    [[2]],
                    turn=1,
                    first_pick_side="me",
                    current_side="me",
                )
            output = model(
                [[[7, 8]]],
                [[[1, 1]]],
                [[0]],
                [[2]],
                turn=1,
                first_pick_side="me",
                current_side="me",
            )
            self.assertEqual(tuple(output["state"].shape), (1, model.state_dim))

        def test_explicit_event_encoding_keeps_pair_as_one_token(self):
            vocab = MonsterVocab()
            vocab.add_token("a")
            vocab.add_token("b")
            encoded = encode_state(
                {
                    "events": [
                        {"side": "me", "monsters": ["b", "a"]},
                    ]
                },
                vocab,
                10,
            )
            self.assertEqual(encoded[0][0], [2, 3])
            self.assertEqual(encoded[1][0], [1, 1])
            self.assertEqual(encoded[3][0], 2)

        def test_single_without_legal_negative_is_rejected(self):
            vocab = MonsterVocab()
            vocab.add_token("only")
            with self.assertRaises(ValueError):
                SharedActionDataset(
                    [
                        {
                            "my_picks": [],
                            "enemy_picks": [],
                            "candidate": (vocab.get_id("only"), vocab.get_id(PAD_TOKEN)),
                            "is_pair": False,
                            "turn": 1,
                            "first_pick_side": 1.0,
                            "events": [],
                            "outcome": None,
                        }
                    ],
                    [],
                    vocab,
                    negative_count=2,
                )

except ModuleNotFoundError:  # pragma: no cover - torch is optional for utilities.
    pass


if __name__ == "__main__":
    unittest.main()
