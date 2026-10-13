import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PIPELINE = os.path.join(ROOT, "synthenexa_pipeline")
if PIPELINE not in sys.path:
    sys.path.insert(0, PIPELINE)

from core.schema import EvalResult, PromptCandidate  # noqa: E402
from services.storage import StorageService  # noqa: E402


class FileStoragePhase3Tests(unittest.TestCase):
    def test_manifest_and_baseline_files_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = StorageService(base_dir=tmpdir, backend="file")
            storage.create_experiment_record(
                "run-1",
                "model-a",
                "experiment-a",
                {"seed": 4, "base_prompt": {"id": "prompt_0", "text": "Base"}},
            )
            storage.save_prompts([PromptCandidate(id="prompt_0", text="Base", is_base=True)], 0)
            storage.save_evaluations(
                [EvalResult(prompt_id="prompt_0", passed=2, failed=0, total=2, is_base=True)],
                0,
            )

            manifest = json.loads((Path(tmpdir) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["run_id"], "run-1")
            self.assertEqual(manifest["config"]["seed"], 4)
            self.assertTrue((Path(tmpdir) / "baseline.json").exists())
            self.assertTrue((Path(tmpdir) / "baseline_evaluations.jsonl").exists())

    def test_existing_artifact_directory_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = StorageService(base_dir=tmpdir, backend="file")
            first.create_experiment_record("run-1", "model-a", "experiment-a", {})
            second = StorageService(base_dir=tmpdir, backend="file")
            with self.assertRaises(FileExistsError):
                second.create_experiment_record("run-2", "model-a", "experiment-a", {})


if __name__ == "__main__":
    unittest.main()
