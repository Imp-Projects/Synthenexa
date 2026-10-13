import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PIPELINE = os.path.join(ROOT, "synthenexa_pipeline")
if PIPELINE not in sys.path:
    sys.path.insert(0, PIPELINE)

from services.results_repository import ResultsRepository  # noqa: E402


class ResultsRepositoryTests(unittest.TestCase):
    def test_loads_experiment_from_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_dir = Path(tmpdir) / "artifacts"
            experiment_dir = artifacts_dir / "demo_run"
            experiment_dir.mkdir(parents=True)

            (experiment_dir / "best_prompt.json").write_text(
                json.dumps(
                    {
                        "run_id": "run-123",
                        "model": "ensemble",
                        "prompt_id": "prompt_best",
                        "text": "Best prompt",
                        "score": 0.82,
                        "reward_history": {"1": 0.75, "2": 0.82},
                    }
                ),
                encoding="utf-8",
            )
            (experiment_dir / "snapshot_gen_1.json").write_text(
                json.dumps(
                    {
                        "generation": 1,
                        "best_score": 0.75,
                        "scores": {"prompt_a": 0.75},
                        "ensemble_scores": {"prompt_a": {"model_a": 0.75}},
                    }
                ),
                encoding="utf-8",
            )
            (experiment_dir / "prompts_gen_1.jsonl").write_text(
                json.dumps({"id": "prompt_a", "text": "Prompt A", "generation": 1}) + "\n",
                encoding="utf-8",
            )
            (experiment_dir / "evaluations_gen_1.jsonl").write_text(
                json.dumps(
                    {
                        "prompt_id": "prompt_a",
                        "evaluator_model": "model_a",
                        "passed": 3,
                        "failed": 1,
                        "total": 4,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (experiment_dir / "final_test_evaluations.jsonl").write_text(
                json.dumps(
                    {
                        "prompt_id": "prompt_0",
                        "is_base": True,
                        "dataset_split": "test",
                        "evaluator_model": "model_a",
                        "passed": 2,
                        "failed": 0,
                        "total": 2,
                        "task_ids": ["t1", "t2"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            repo = ResultsRepository(artifacts_dir=artifacts_dir, db_path=Path(tmpdir) / "missing.db")
            result = repo.load_experiment(experiment_name="demo_run", source="artifacts")

            self.assertEqual(result.run_id, "run-123")
            self.assertEqual(result.experiment_name, "demo_run")
            self.assertEqual(result.best_prompt["score"], 0.82)
            self.assertEqual(len(result.snapshots), 1)
            self.assertEqual(len(result.evaluations), 1)
            self.assertEqual(len(result.final_test_evaluations), 1)
            self.assertTrue(result.final_test_evaluations[0]["is_base"])

    def test_prefers_sqlite_when_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "experiments.db"
            con = sqlite3.connect(db_path)
            cur = con.cursor()
            cur.execute(
                "CREATE TABLE experiments (id INTEGER PRIMARY KEY, run_id TEXT, experiment_name TEXT, model_name TEXT, config TEXT, status TEXT)"
            )
            cur.execute(
                "CREATE TABLE prompts (id INTEGER PRIMARY KEY, prompt_id TEXT, experiment_id INTEGER, generation INTEGER, text TEXT, is_base INTEGER, is_best INTEGER, created_at TEXT)"
            )
            cur.execute(
                "CREATE TABLE evaluations (id INTEGER PRIMARY KEY, prompt_id TEXT, experiment_id INTEGER, generation INTEGER, evaluator_model TEXT, is_base INTEGER, dataset_split TEXT, task_ids TEXT, passed INTEGER, failed INTEGER, total INTEGER, syntax_errors INTEGER, errors TEXT, total_time_ms FLOAT, min_time_ms FLOAT, max_time_ms FLOAT, avg_time_ms FLOAT, created_at TEXT)"
            )
            cur.execute(
                "CREATE TABLE snapshots (id INTEGER PRIMARY KEY, experiment_id INTEGER, generation INTEGER, best_score FLOAT, best_prompt_id TEXT, best_prompt_text TEXT, scores TEXT, reward_history TEXT, best_ensemble_scores TEXT, model_leaderboard TEXT, ensemble_scores TEXT, created_at TEXT)"
            )
            cur.execute(
                "INSERT INTO experiments (id, run_id, experiment_name, model_name, config, status) VALUES (1, 'sqlite-run', 'sqlite_exp', 'ensemble:model_a,model_b', ?, 'completed')",
                (json.dumps({"ensemble_models": ["model_a", "model_b"]}),),
            )
            cur.execute(
                "INSERT INTO prompts (prompt_id, experiment_id, generation, text, is_base, is_best, created_at) VALUES ('prompt_win', 1, 1, 'Prompt Winner', 0, 1, '2026-07-27')"
            )
            cur.execute(
                "INSERT INTO evaluations (prompt_id, experiment_id, generation, evaluator_model, is_base, dataset_split, task_ids, passed, failed, total, syntax_errors, errors, total_time_ms, min_time_ms, max_time_ms, avg_time_ms, created_at) VALUES ('prompt_win', 1, 1, 'model_a', 0, 'train_validation', '[\"t1\"]', 4, 0, 4, 0, '[]', 10, 2, 4, 2.5, '2026-07-27')"
            )
            cur.executemany(
                "INSERT INTO evaluations (prompt_id, experiment_id, generation, evaluator_model, is_base, dataset_split, task_ids, passed, failed, total, syntax_errors, errors, total_time_ms, min_time_ms, max_time_ms, avg_time_ms, created_at) VALUES (?, 1, -1, ?, ?, 'test', '[\"t1\"]', ?, ?, 4, 0, '[]', 10, 2, 4, 2.5, '2026-07-27')",
                [
                    ("prompt_0", "model_a", 1, 2, 2),
                    ("prompt_win", "model_a", 0, 3, 1),
                ],
            )
            cur.execute(
                "INSERT INTO snapshots (experiment_id, generation, best_score, best_prompt_id, best_prompt_text, scores, reward_history, best_ensemble_scores, model_leaderboard, ensemble_scores, created_at) VALUES (1, 1, 1.0, 'prompt_win', 'Prompt Winner', ?, ?, ?, ?, ?, '2026-07-27')",
                (
                    json.dumps({"prompt_win": 1.0}),
                    json.dumps({"1": 1.0}),
                    json.dumps({"model_a": 1.0}),
                    json.dumps({"model_a": 1.0}),
                    json.dumps({"prompt_win": {"model_a": 1.0}}),
                ),
            )
            con.commit()
            con.close()

            repo = ResultsRepository(artifacts_dir=Path(tmpdir) / "artifacts", db_path=db_path)
            result = repo.load_experiment(run_id="sqlite-run", source="auto")

            self.assertEqual(result.source, "sqlite")
            self.assertEqual(result.experiment_name, "sqlite_exp")
            self.assertEqual(result.best_prompt["prompt_id"], "prompt_win")
            self.assertEqual(len(result.final_test_evaluations), 2)


if __name__ == "__main__":
    unittest.main()
