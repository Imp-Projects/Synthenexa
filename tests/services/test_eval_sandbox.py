import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PIPELINE = os.path.join(ROOT, "synthenexa_pipeline")
if PIPELINE not in sys.path:
    sys.path.insert(0, PIPELINE)

from services.eval_sandbox import SandboxEvaluator  # noqa: E402


class SandboxEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = SandboxEvaluator(timeout_seconds=1.0)

    def test_runs_valid_code(self):
        task = {
            "entry_point": "add",
            "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
        }
        result = self.evaluator.run_code_on_task(
            "def add(a, b):\n    return a + b\n",
            task,
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["error_category"], "ok")

    def test_blocks_forbidden_imports(self):
        task = {
            "entry_point": "solve",
            "test": "def check(candidate):\n    assert candidate() == 1\n",
        }
        result = self.evaluator.run_code_on_task(
            "import os\n\ndef solve():\n    return 1\n",
            task,
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["error_category"], "runtime_error")
        self.assertIn("blocked by the sandbox", result["error_message"])

    def test_times_out_long_running_code(self):
        task = {
            "entry_point": "solve",
            "test": "def check(candidate):\n    candidate()\n",
        }
        result = self.evaluator.run_code_on_task(
            "def solve():\n    while True:\n        pass\n",
            task,
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["error_category"], "timeout")


if __name__ == "__main__":
    unittest.main()
