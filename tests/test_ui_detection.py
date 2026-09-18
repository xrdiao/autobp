import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from sw_ui_detection.bp_screen_analyzer import BPDraftScreenAnalyzer
from sw_ui_detection.capture import locate_game_canvas
from sw_ui_detection.monster_bank import MonsterTemplateBank


class UIDetectionTests(unittest.TestCase):
    def test_canvas_locator_trims_window_title_bar(self):
        image = np.full((300, 500, 3), 245, dtype=np.uint8)
        image[42:, :] = (28, 48, 66)
        image[55:80, 20:480] = (80, 140, 190)

        location = locate_game_canvas(image)

        self.assertFalse(location.used_fallback)
        self.assertGreaterEqual(location.bbox.y, 40)
        self.assertGreater(location.confidence, 0.5)

    def test_empty_bp_slot_is_layout_evidence(self):
        crop = np.full((128, 128, 3), (16, 31, 41), dtype=np.uint8)
        cv2.rectangle(crop, (8, 8), (120, 120), (22, 40, 52), 4)

        analyzer = BPDraftScreenAnalyzer()

        self.assertTrue(analyzer._is_empty_slot(crop))
        self.assertGreater(analyzer._empty_slot_layout_score(crop), 0.3)

    def test_template_bank_uses_catalog_images_that_exist_locally(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            image_path = temp_dir / "unit_icon_test.png"
            cv2.imwrite(str(image_path), np.full((96, 96, 3), 120, dtype=np.uint8))
            map_path = temp_dir / "monster_images.json"
            catalog_path = temp_dir / "lucksack_catalog.json"
            map_path.write_text(json.dumps({}), encoding="utf-8")
            catalog_path.write_text(
                json.dumps([{"slabel": "Catalog Monster", "image": str(image_path)}]),
                encoding="utf-8",
            )

            bank = MonsterTemplateBank(image_map_path=map_path, catalog_path=catalog_path)

        self.assertEqual(len(bank), 1)
        self.assertIsNotNone(bank.get_monster_image_path("Catalog Monster"))


if __name__ == "__main__":
    unittest.main()
