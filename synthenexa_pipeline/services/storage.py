"""
services/storage.py — Persist pipeline artifacts to disk.

Naming conventions:
  - prompts_gen_{N}.jsonl         — candidate prompts for generation N
  - evaluations_gen_{N}.jsonl     — eval results for generation N
  - snapshot_gen_{N}.json         — full generation state snapshot
  - run_logs.jsonl                — append-only run log
"""
import json
import os
from typing import Any, Dict, List

from core.schema import EvalResult, PromptCandidate


class StorageService:

    def __init__(self, base_dir: str = "experiments/artifacts"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    # ---------------------------------------------------------------- #
    #  Public JSONL API                                                 #
    # ---------------------------------------------------------------- #

    def save_jsonl(self, filename: str, data: List[Dict[str, Any]]) -> None:
        """
        Append rows to a JSONL file under base_dir.

        Args:
            filename: Relative filename inside base_dir (e.g. "run_logs.jsonl").
            data:     List of JSON-serialisable dicts to append.
        """
        path = os.path.join(self.base_dir, filename)
        with open(path, "a", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def load_jsonl(self, filename: str) -> List[Dict[str, Any]]:
        """
        Load all rows from a JSONL file under base_dir.

        Returns an empty list if the file does not exist.
        """
        path = os.path.join(self.base_dir, filename)
        if not os.path.exists(path):
            return []

        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    # ---------------------------------------------------------------- #
    #  Domain-specific save methods                                     #
    # ---------------------------------------------------------------- #

    def save_prompts(self, prompts: List[PromptCandidate], generation: int) -> None:
        """Save candidate prompts for a given generation."""
        data = [p.model_dump() for p in prompts]
        self.save_jsonl(f"prompts_gen_{generation}.jsonl", data)

    def save_evaluations(self, results: List[EvalResult], generation: int) -> None:
        """Save evaluation results for a given generation."""
        data = [r.model_dump() for r in results]
        self.save_jsonl(f"evaluations_gen_{generation}.jsonl", data)

    def save_generation_snapshot(self, context: Any, generation: int) -> None:
        """
        Persist a full generation-state snapshot to a JSON file.

        Captures: generation number, best prompt, best score, all candidate
        scores, and the reward history so far.
        """
        snapshot: Dict[str, Any] = {
            "generation":      generation,
            "best_score":      context.best_score,
            "best_prompt_id":  context.best_prompt.id if context.best_prompt else None,
            "best_prompt_text": context.best_prompt.text if context.best_prompt else None,
            "scores":          context.scores,
            "reward_history":  context.reward_history,
        }
        path = os.path.join(self.base_dir, f"snapshot_gen_{generation}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False, default=str)

    def save_run_log(self, log_entry: Dict[str, Any]) -> None:
        """Append a single log entry to the run log."""
        self.save_jsonl("run_logs.jsonl", [log_entry])
