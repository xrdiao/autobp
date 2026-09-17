import json
import tempfile
import unittest
from pathlib import Path

from bp_mechanism import BPMechanismLayer
from build_dataset.sync_swarfarm_mechanisms import build_profile


class BPMechanismTests(unittest.TestCase):
    def test_candidate_adds_counter_coverage(self):
        snapshot = {
            "source": "test",
            "aliases_by_slug": {
                "giana": "giana",
                "haegang": "haegang",
                "plain": "plain",
            },
            "profiles_by_slug": {
                "giana": {
                    "name": "Giana",
                    "tags": ["control", "bomb_dot", "tempo"],
                    "threat_tags": ["control", "bomb_dot", "tempo"],
                },
                "haegang": {
                    "name": "Haegang",
                    "tags": ["cleanse", "strip", "atb_control", "tempo"],
                    "threat_tags": ["tempo"],
                },
                "plain": {
                    "name": "Plain",
                    "tags": [],
                    "threat_tags": [],
                },
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mechanisms.json"
            path.write_text(json.dumps(snapshot), encoding="utf-8")
            layer = BPMechanismLayer(path)

            counter = layer.score_action([], ["Giana"], ["Haegang"])
            neutral = layer.score_action([], ["Giana"], ["Plain"])

        self.assertGreater(counter["coverage_gain"], 0.0)
        self.assertGreater(counter["score"], neutral["score"])
        self.assertTrue(any("硬控" in reason for reason in counter["reasons"]))

    def test_without_enemy_draft_mechanism_signal_is_inactive(self):
        layer = BPMechanismLayer(Path("/does/not/exist.json"))
        result = layer.score_action([], [], ["Haegang"])
        self.assertFalse(result["available"])
        self.assertEqual(result["score"], 0.5)

    def test_profile_keeps_combat_metadata(self):
        profile = build_profile(
            {
                "name": "Alpha",
                "com2us_id": "1001",
                "speed": 105,
                "base_hp": 10000,
                "base_attack": 800,
                "base_defense": 600,
                "skills": [1],
            },
            {
                1: {
                    "name": "Skill",
                    "slot": 2,
                    "cooltime": 4,
                    "max_level": 5,
                    "hits": 2,
                    "scales_with": ["ATK"],
                    "multiplier_formula": "3.0*{ATK}",
                    "effects": [],
                }
            },
        )

        self.assertEqual(profile["base_stats"]["speed"], 105)
        self.assertEqual(profile["skills"][0]["cooltime"], 4)
        self.assertEqual(profile["skills"][0]["scales_with"], ["ATK"])
        self.assertEqual(profile["skills"][0]["hits"], 2)


if __name__ == "__main__":
    unittest.main()
