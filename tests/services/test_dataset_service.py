import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PIPELINE = os.path.join(ROOT, "synthenexa_pipeline")
if PIPELINE not in sys.path:
    sys.path.insert(0, PIPELINE)

from core.config import ExperimentConfig  # noqa: E402
from services.dataset_service import DatasetService  # noqa: E402


class DatasetPartitionTests(unittest.TestCase):
    def _tasks(self):
        return [{"task_id": f"task_{index}", "text": f"Task {index}"} for index in range(10)]

    def test_partitioning_is_deterministic(self):
        first = DatasetService.partition_dataset(self._tasks(), seed=42)
        second = DatasetService.partition_dataset(self._tasks(), seed=42)

        self.assertEqual(first["task_ids"], second["task_ids"])
        self.assertEqual(first["dataset_hash"], second["dataset_hash"])

    def test_partitions_are_disjoint_and_cover_source(self):
        tasks = self._tasks()
        result = DatasetService.partition_dataset(tasks, seed=7)
        split_ids = [set(result["task_ids"][name]) for name in ("train", "validation", "test")]

        self.assertEqual(set.union(*split_ids), {task["task_id"] for task in tasks})
        self.assertEqual(set.intersection(*split_ids), set())
        self.assertEqual(sum(len(ids) for ids in split_ids), len(tasks))

    def test_task_ids_and_source_hash_are_stable(self):
        tasks = self._tasks()
        result = DatasetService.partition_dataset(tasks, seed=1)

        self.assertEqual(result["dataset_hash"], DatasetService.dataset_hash(self._tasks()))
        self.assertEqual(
            sorted(task["task_id"] for task in result["train"] + result["validation"] + result["test"]),
            sorted(task["task_id"] for task in tasks),
        )
        partitioned_tasks = result["train"] + result["validation"] + result["test"]
        self.assertTrue(all(task["dataset_split"] in {"train", "validation", "test"} for task in partitioned_tasks))

    def test_partition_configuration_round_trips(self):
        config = ExperimentConfig()
        config.dataset_partition.seed = 17
        config.dataset_partition.test_ratio = 0.3
        restored = ExperimentConfig.from_json(config.to_json())

        self.assertEqual(restored.dataset_partition.seed, 17)
        self.assertEqual(restored.dataset_partition.test_ratio, 0.3)
        self.assertTrue(restored.dataset_partition.evaluate_test_after_selection)


if __name__ == "__main__":
    unittest.main()
