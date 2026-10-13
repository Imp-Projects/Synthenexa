"""
services/dataset_service.py — Load MBPP / HumanEval benchmark datasets.
"""

import hashlib
import json
import os
import random
from copy import deepcopy
from typing import Any, Dict, List, Optional


class DatasetService:
    @staticmethod
    def dataset_hash(tasks: List[Dict[str, Any]]) -> str:
        """Return a stable hash for the ordered task content and IDs."""
        payload = json.dumps(tasks, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def partition_dataset(
        tasks: List[Dict[str, Any]],
        seed: int = 0,
        train_ratio: float = 0.6,
        validation_ratio: float = 0.2,
        test_ratio: float = 0.2,
        train_size: Optional[int] = None,
        validation_size: Optional[int] = None,
        test_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create deterministic, disjoint train/validation/test task partitions."""
        if min(train_ratio, validation_ratio, test_ratio) < 0:
            raise ValueError("Partition ratios cannot be negative.")
        if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-9:
            raise ValueError("Partition ratios must sum to 1.0.")
        if any(size is not None and size < 0 for size in (train_size, validation_size, test_size)):
            raise ValueError("Partition sizes cannot be negative.")

        shuffled = deepcopy(tasks)
        source_hash = DatasetService.dataset_hash(tasks)
        random.Random(seed).shuffle(shuffled)
        total = len(shuffled)
        if train_size is None:
            train_size = int(total * train_ratio)
        if validation_size is None:
            validation_size = int(total * validation_ratio)
        if test_size is None:
            test_size = total - train_size - validation_size
        if train_size + validation_size + test_size != total:
            raise ValueError("Partition sizes must cover the dataset exactly.")

        train = shuffled[:train_size]
        validation = shuffled[train_size : train_size + validation_size]
        test = shuffled[train_size + validation_size :]
        for split_name, split_tasks in (("train", train), ("validation", validation), ("test", test)):
            for task in split_tasks:
                task["dataset_split"] = split_name

        return {
            "train": train,
            "validation": validation,
            "test": test,
            "seed": seed,
            "dataset_hash": source_hash,
            "task_ids": {
                "train": [task["task_id"] for task in train],
                "validation": [task["task_id"] for task in validation],
                "test": [task["task_id"] for task in test],
            },
        }

    @staticmethod
    def load_partitions(
        path: str,
        seed: int = 0,
        train_ratio: float = 0.6,
        validation_ratio: float = 0.2,
        test_ratio: float = 0.2,
    ) -> Dict[str, Any]:
        """Load a JSONL dataset and return deterministic partitions plus metadata."""
        tasks = DatasetService.load_dataset(path)
        return DatasetService.partition_dataset(
            tasks,
            seed=seed,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        )

    @staticmethod
    def load_dataset(path: str) -> List[Dict[str, Any]]:
        """
        Load an MBPP / HumanEval dataset from a JSONL file.

        Each line becomes a normalised task dict with keys:
            task_id, prompt, entry_point, test, text

        Returns an empty list (not an exception) if the file is missing.
        """
        if not os.path.exists(path):
            print(f"[DatasetService] WARNING: dataset file not found at '{path}'")
            return []

        dataset: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"[DatasetService] Skipping malformed line {lineno}: {exc}")
                    continue

                task: Dict[str, Any] = {
                    "task_id": raw.get("task_id", f"task_{lineno}"),
                    "prompt": raw.get("prompt", raw.get("text", "")),
                    "entry_point": raw.get("entry_point", ""),
                    "test": raw.get("test", ""),
                    "text": raw.get("text", ""),
                    # Keep the canonical solution for reference / offline eval
                    "canonical_solution": raw.get("canonical_solution", ""),
                }
                dataset.append(task)

        return dataset

    @staticmethod
    def build_task_prompt(task: Dict[str, Any]) -> str:
        """
        Build the user-facing task instruction that will be sent to the LLM
        for code generation.

        Includes:
          - The task description / docstring
          - The function entry point name (so the LLM knows the target signature)
          - A clear instruction to return only code (no prose)

        Example output:
            Task: Write a function to find the minimum cost path ...
            Function name: min_cost
            Return ONLY a complete, runnable Python function. No explanation.
        """
        description = (task.get("text") or task.get("prompt") or "").strip()
        entry_point = task.get("entry_point", "").strip()

        lines = []
        if description:
            lines.append(f"Task:\n{description}")
        if entry_point:
            lines.append(f"Function name: {entry_point}")
        lines.append("Return ONLY a complete, runnable Python function. No explanation, no markdown.")

        return "\n\n".join(lines)
