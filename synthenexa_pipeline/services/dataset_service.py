"""
services/dataset_service.py — Load MBPP / HumanEval benchmark datasets.
"""
import json
import os
from typing import Any, Dict, List


class DatasetService:

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
                    "task_id":     raw.get("task_id", f"task_{lineno}"),
                    "prompt":      raw.get("prompt", raw.get("text", "")),
                    "entry_point": raw.get("entry_point", ""),
                    "test":        raw.get("test", ""),
                    "text":        raw.get("text", ""),
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
