import unittest

from bp_core.data_common import NameResolver, build_name_resolution_resources, normalize_bp_sample
from build_dataset.expand_from_matches_raw import expand_one_match
from evaluation.audit_bp_data import audit_samples


class CanonicalDatasetTests(unittest.TestCase):
    def test_expansion_emits_one_row_per_bp_event(self):
        pattern = ("me", "enemy", "enemy", "me", "me", "enemy", "enemy", "me", "me", "enemy")
        records = []
        for order, side in enumerate(pattern, start=1):
            records.append(
                {
                    "order": order,
                    "slug": f"monster-{order}",
                    "name": f"Monster {order}",
                    "is_banned": side == "enemy" and order == 3,
                    "is_leader": order == 1,
                }
            )
        rows = expand_one_match(
            {
                "page_player_id": 7,
                "match_index": 1,
                "my_picks": [item for item, side in zip(records, pattern) if side == "me"],
                "enemy_picks": [item for item, side in zip(records, pattern) if side == "enemy"],
                "result": "win",
            }
        )
        self.assertEqual([row["turn"] for row in rows], [1, 2, 4, 6, 8, 10])
        self.assertEqual(audit_samples(rows)["issue_counts"], {})
        resources = build_name_resolution_resources(rows)
        resolver = NameResolver(resources["name_slug_aliases"], resources["display_name_map"])
        normalized = [normalize_bp_sample(row, resolver) for row in rows]
        self.assertEqual([len(row["events"]) for row in normalized], [0, 1, 2, 3, 4, 5])
        self.assertEqual(normalized[1]["next_action_slugs"], ["monster-2", "monster-3"])


if __name__ == "__main__":
    unittest.main()
