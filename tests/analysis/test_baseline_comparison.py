import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PIPELINE = os.path.join(ROOT, "synthenexa_pipeline")
if PIPELINE not in sys.path:
    sys.path.insert(0, PIPELINE)

from Analysis.compare_baseline import build_test_comparison  # noqa: E402


class BaselineComparisonTests(unittest.TestCase):
    def _rows(self):
        return [
            {
                "prompt_id": "prompt_0",
                "is_base": True,
                "dataset_split": "test",
                "evaluator_model": "model_a",
                "passed": 4,
                "total": 10,
                "task_ids": ["t1", "t2"],
            },
            {
                "prompt_id": "prompt_0",
                "is_base": True,
                "dataset_split": "test",
                "evaluator_model": "model_b",
                "passed": 5,
                "total": 10,
                "task_ids": ["t1", "t2"],
            },
            {
                "prompt_id": "prompt_best",
                "is_base": False,
                "dataset_split": "test",
                "evaluator_model": "model_a",
                "passed": 7,
                "total": 10,
                "task_ids": ["t1", "t2"],
            },
            {
                "prompt_id": "prompt_best",
                "is_base": False,
                "dataset_split": "test",
                "evaluator_model": "model_b",
                "passed": 8,
                "total": 10,
                "task_ids": ["t1", "t2"],
            },
        ]

    def test_returns_per_model_and_aggregate_improvement(self):
        comparison = build_test_comparison(self._rows(), "prompt_best")

        self.assertTrue(comparison["complete"])
        self.assertEqual(comparison["base_mean"], 0.45)
        self.assertEqual(comparison["refined_mean"], 0.75)
        self.assertAlmostEqual(comparison["absolute_delta"], 0.30)
        for row in comparison["models"]:
            self.assertAlmostEqual(row["absolute_delta"], 0.3)

    def test_rejects_missing_baseline_or_refined_rows(self):
        rows = self._rows()[:-1]
        with self.assertRaises(ValueError):
            build_test_comparison(rows, "prompt_best")

    def test_rejects_mismatched_test_task_ids(self):
        rows = self._rows()
        rows[-1]["task_ids"] = ["different_task"]
        with self.assertRaises(ValueError):
            build_test_comparison(rows, "prompt_best")

    def test_base_and_refined_rows_are_required_even_when_base_is_global_best(self):
        rows = self._rows()
        comparison = build_test_comparison(rows, "prompt_best")

        self.assertEqual(comparison["base_mean"], 0.45)
        self.assertEqual(comparison["refined_mean"], 0.75)


if __name__ == "__main__":
    unittest.main()
