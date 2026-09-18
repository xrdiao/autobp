import unittest

from evaluation.replay import _binary_metrics, _select_match_representatives


class ReplayEvaluationTests(unittest.TestCase):
    def test_match_representatives_are_reproducible_and_unique(self):
        samples = [
            {"match_id": "m1", "sample_id": "m1_1"},
            {"match_id": "m1", "sample_id": "m1_2"},
            {"match_id": "m2", "sample_id": "m2_1"},
        ]

        first = _select_match_representatives(samples, seed=7)
        second = _select_match_representatives(samples, seed=7)

        self.assertEqual(
            [row["sample_id"] for row in first],
            [row["sample_id"] for row in second],
        )
        self.assertEqual({row["match_id"] for row in first}, {"m1", "m2"})

    def test_binary_metrics_report_calibration(self):
        metrics = _binary_metrics(
            [
                {"probability": 0.9, "label": 1.0},
                {"probability": 0.1, "label": 0.0},
            ]
        )

        self.assertEqual(metrics["evaluated"], 2)
        self.assertAlmostEqual(metrics["brier_score"], 0.01, places=6)
        self.assertAlmostEqual(metrics["ece"], 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
