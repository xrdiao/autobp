import json
import tempfile
import unittest
from pathlib import Path

from bp_core.lucksack_stats import BPLucksackStats


class BPLucksackStatsTests(unittest.TestCase):
    def test_scores_solo_and_composition_evidence(self):
        snapshot = {
            "schema_version": 1,
            "source": "test",
            "default_rank_group": "102",
            "season": {"season_number": 1, "season_name": "Test"},
            "patch": {"patch_id": 1, "patch_name": "Test"},
            "meta_dashboard": {
                "draft_position": {
                    "102": {
                        "phases": [
                            {
                                "pick_phase": 1,
                                "points": [
                                    {
                                        "monster_slug": "alpha",
                                        "pick_count": 1000,
                                        "pick_rate": 0.2,
                                        "win_rate": 0.65,
                                    },
                                    {
                                        "monster_slug": "beta",
                                        "pick_count": 1000,
                                        "pick_rate": 0.2,
                                        "win_rate": 0.50,
                                    },
                                ],
                            }
                        ]
                    }
                }
            },
            "statistics": {
                "solo": {
                    "102": [
                        {
                            "monster_slug": "alpha",
                            "played_count": 1000,
                            "win_rate": 0.60,
                            "played_rate": 0.10,
                        },
                        {
                            "monster_slug": "beta",
                            "played_count": 1000,
                            "win_rate": 0.50,
                            "played_rate": 0.10,
                        },
                    ]
                },
                "duo": {
                    "102": [
                        {
                            "monster_slug": ["alpha", "beta"],
                            "played_count": 1000,
                            "win_rate": 0.70,
                        }
                    ]
                },
                "trio": {"102": []},
            },
        }
        catalog = [
            {
                "slug": "alpha",
                "slabel": "Alpha",
                "label": "Alpha (Fire)",
                "natural_stars": 5,
                "searchable": True,
                "image": "alpha.png",
                "com2us_id": "1001",
            },
            {
                "slug": "beta",
                "slabel": "Beta",
                "label": "Beta (Wind)",
                "natural_stars": 5,
                "searchable": True,
                "image": "beta.png",
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            stats_path = temp_dir / "stats.json"
            catalog_path = temp_dir / "catalog.json"
            tier_path = temp_dir / "tiers.json"
            stats_path.write_text(json.dumps(snapshot), encoding="utf-8")
            catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
            tier_path.write_text(
                json.dumps(
                    {
                        "tier_date": "2026-09-07",
                        "entries_by_id": {
                            "1001": {
                                "id": "1001",
                                "tier": "SSS",
                                "pick": 25.0,
                                "win": 55.0,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            stats = BPLucksackStats(
                stats_path=stats_path,
                catalog_path=catalog_path,
                swkit_tier_path=tier_path,
                swrt_stats_path=temp_dir / "missing-swrt.json",
            )
            result = stats.score_action([], [], ["Alpha", "Beta"], turn=1)
            string_result = stats.score_action([], [], "Alpha", turn=1)

        self.assertTrue(stats.ready)
        self.assertEqual(stats.resolve_slug("Alpha"), "alpha")
        self.assertEqual(stats.to_display_name("beta"), "Beta")
        self.assertEqual(stats.to_image_name("alpha"), "alpha.png")
        self.assertEqual(stats.candidate_slugs(), ["alpha", "beta"])
        synergy_pairs = stats.candidate_synergy_pairs(["alpha", "beta"])
        self.assertEqual(synergy_pairs[0]["slugs"], ("alpha", "beta"))
        self.assertGreater(result["solo_score"], 0.5)
        self.assertGreater(result["synergy_score"], 0.5)
        self.assertGreater(result["draft_position_score"], 0.5)
        self.assertGreater(result["rta_tier_score"], 0.5)
        self.assertGreater(result["score"], 0.5)
        self.assertEqual(string_result, stats.score_action([], [], ["Alpha"], turn=1))
        self.assertEqual(result["sample_count"], 1000)
        self.assertGreater(result["evidence_confidence"], 0.0)
        self.assertLess(result["evidence_uncertainty"], 1.0)
        self.assertEqual(
            len([item for item in result["evidence"] if item["kind"] == "duo"]),
            1,
        )
        self.assertEqual(
            {item["kind"] for item in result["evidence"]},
            {"solo", "duo", "draft_position", "swkit_rta_tier"},
        )

    def test_missing_snapshot_is_unavailable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stats = BPLucksackStats(
                stats_path=Path(temp_dir) / "missing-stats.json",
                catalog_path=Path(temp_dir) / "missing-catalog.json",
                swrt_stats_path=Path(temp_dir) / "missing-swrt.json",
            )

        self.assertFalse(stats.ready)
        self.assertEqual(stats.candidate_slugs(), [])
        result = stats.score_action([], [], ["new-monster"])
        self.assertEqual(result["score"], 0.5)
        self.assertEqual(result["evidence_confidence"], 0.0)
        self.assertEqual(result["evidence_uncertainty"], 1.0)

    def test_uses_swrt_matchup_evidence(self):
        snapshot = {
            "schema_version": 1,
            "source": "test",
            "statistics": [
                {
                    "monsterId": 1001,
                    "monsterName": "Alpha",
                    "monsterNameCn": "阿尔法",
                    "winRate": 0.55,
                    "pickRate": 0.10,
                    "banRate": 0.05,
                    "pickTotal": 5000,
                },
                {
                    "monsterId": 1099,
                    "monsterName": "Unawakened",
                    "winRate": 0.60,
                    "pickTotal": 5000,
                },
            ],
            "highdata_by_id": {
                "1001": {
                    "lowOneVsOneList": [
                        {
                            "opponentMonsterId": 1002,
                            "pickTotal": 3000,
                            "winRate": "0.35",
                        }
                    ],
                    "lowOneVsTwoList": [
                        {
                            "opponentMonsterOneId": 1002,
                            "opponentMonsterTwoId": 1003,
                            "pickTotal": 2000,
                            "winRate": "0.30",
                        }
                    ],
                }
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            stats_path = temp_dir / "stats.json"
            catalog_path = temp_dir / "catalog.json"
            swrt_path = temp_dir / "swrt.json"
            stats_path.write_text(
                json.dumps(
                    {
                        "default_rank_group": "102",
                        "statistics": {"solo": {"102": []}, "duo": {"102": []}, "trio": {"102": []}},
                    }
                ),
                encoding="utf-8",
            )
            catalog_path.write_text(
                json.dumps(
                    [
                        {"slug": "alpha", "slabel": "Alpha", "com2us_id": "1001", "natural_stars": 5},
                        {"slug": "beta", "slabel": "Beta", "com2us_id": "1002", "natural_stars": 5},
                        {"slug": "gamma", "slabel": "Gamma", "com2us_id": "1003", "natural_stars": 5},
                        {"slug": "unawakened", "slabel": "Unawakened", "com2us_id": "1099", "natural_stars": 3},
                    ]
                ),
                encoding="utf-8",
            )
            swrt_path.write_text(json.dumps(snapshot), encoding="utf-8")

            stats = BPLucksackStats(
                stats_path=stats_path,
                catalog_path=catalog_path,
                swrt_stats_path=swrt_path,
            )
            result = stats.score_action(["Alpha"], ["Beta"], ["Alpha"])

        self.assertEqual(stats.resolve_slug("阿尔法"), "alpha")
        self.assertNotIn("unawakened", stats.candidate_slugs())
        self.assertLess(result["matchup_score"], 0.5)
        self.assertGreater(result["matchup_coverage"], 0.0)
        self.assertIn("swrt_matchup", {item["kind"] for item in result["evidence"]})
        team_result = stats.score_action(["Alpha"], ["Beta", "Gamma"], ["Alpha"])
        self.assertIn(
            "swrt_team_matchup",
            {item["kind"] for item in team_result["evidence"]},
        )


if __name__ == "__main__":
    unittest.main()
