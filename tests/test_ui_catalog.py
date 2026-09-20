import unittest

from app_web import build_monster_options, matches_monster_search


class UICatalogTests(unittest.TestCase):
    def test_gallery_options_are_sorted_by_usage_frequency(self):
        options = build_monster_options(
            {"low": 2, "high": 3, "tie": 4},
            {},
            {"low": 12, "high": 98, "tie": 98},
        )
        self.assertEqual(options, ["high", "tie", "low"])

    def test_search_uses_the_same_name_variants_as_alias_index(self):
        self.assertTrue(
            matches_monster_search(
                "wind_aya",
                "Wind Aya - Nobara Kugisaki 风钉子",
            )
        )
        self.assertTrue(
            matches_monster_search(
                "暗大锤",
                "Zinc [2A] 暗大锤",
            )
        )


if __name__ == "__main__":
    unittest.main()
