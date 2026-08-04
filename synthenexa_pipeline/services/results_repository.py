"""
services/results_repository.py — Canonical experiment-results extraction layer.

Normalizes experiment data from either:
  - SQLite (`experiments/synthenexa.db`)
  - File artifacts (`experiments/artifacts/<experiment>/`)

This keeps analysis and visualization code focused on metrics instead of
backend-specific storage details.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from core.config import SQLITE_DB_PATH


@dataclass
class ExperimentResult:
    run_id: str
    experiment_name: str
    source: str
    model_name: str
    config: dict[str, Any]
    best_prompt: dict[str, Any]
    snapshots: list[dict[str, Any]]
    prompts: list[dict[str, Any]]
    evaluations: list[dict[str, Any]]


def _safe_json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if value in ("", "null"):
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


class ResultsRepository:
    """Read experiment results from SQLite or artifact directories."""

    def __init__(
        self,
        artifacts_dir: str | Path = "experiments/artifacts",
        db_path: str | Path = SQLITE_DB_PATH,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.db_path = Path(db_path)

    def load_experiment(
        self,
        experiment_name: Optional[str] = None,
        run_id: Optional[str] = None,
        source: str = "auto",
    ) -> ExperimentResult:
        resolved_source = source.lower()
        if resolved_source in ("auto", "sqlite"):
            sqlite_result = self._load_from_sqlite(experiment_name=experiment_name, run_id=run_id)
            if sqlite_result is not None:
                return sqlite_result
            if resolved_source == "sqlite":
                raise ValueError("No matching experiment found in SQLite storage.")

        file_result = self._load_from_artifacts(experiment_name=experiment_name, run_id=run_id)
        if file_result is not None:
            return file_result

        raise ValueError("No matching experiment found in SQLite or artifact storage.")

    def list_experiments(self, source: str = "auto") -> list[dict[str, Any]]:
        resolved_source = source.lower()
        results: list[dict[str, Any]] = []
        if resolved_source in ("auto", "sqlite") and self.db_path.exists():
            results.extend(self._list_sqlite_experiments())
            if resolved_source == "sqlite":
                return results
        if resolved_source in ("auto", "file", "artifacts"):
            results.extend(self._list_artifact_experiments())
        seen: set[tuple[str, str]] = set()
        deduped = []
        for item in results:
            key = (item.get("run_id", ""), item.get("experiment_name", ""))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _load_from_sqlite(
        self,
        experiment_name: Optional[str],
        run_id: Optional[str],
    ) -> Optional[ExperimentResult]:
        if not self.db_path.exists():
            return None

        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            query = "SELECT * FROM experiments"
            params: list[Any] = []
            clauses = []
            if run_id:
                clauses.append("run_id = ?")
                params.append(run_id)
            if experiment_name:
                clauses.append("experiment_name = ?")
                params.append(experiment_name)
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            query += " ORDER BY id DESC LIMIT 1"
            exp = con.execute(query, params).fetchone()
            if exp is None:
                return None

            exp_id = exp["id"]
            prompts = [
                dict(row)
                for row in con.execute(
                    "SELECT prompt_id, generation, text, is_base, is_best, created_at "
                    "FROM prompts WHERE experiment_id = ? ORDER BY generation, id",
                    (exp_id,),
                ).fetchall()
            ]
            evaluations = [
                dict(row)
                for row in con.execute(
                    "SELECT prompt_id, generation, evaluator_model, passed, failed, total, "
                    "syntax_errors, errors, total_time_ms, min_time_ms, max_time_ms, avg_time_ms, created_at "
                    "FROM evaluations WHERE experiment_id = ? ORDER BY generation, id",
                    (exp_id,),
                ).fetchall()
            ]
            snapshots = []
            for row in con.execute(
                "SELECT * FROM snapshots WHERE experiment_id = ? ORDER BY generation, id",
                (exp_id,),
            ).fetchall():
                snap = dict(row)
                for key in (
                    "scores",
                    "reward_history",
                    "best_ensemble_scores",
                    "model_leaderboard",
                    "ensemble_scores",
                ):
                    snap[key] = _safe_json_loads(snap.get(key), {})
                snapshots.append(snap)

            best_prompt_row = next((row for row in prompts if row.get("is_best")), None)
            last_snapshot = snapshots[-1] if snapshots else {}
            best_prompt = {
                "run_id": exp["run_id"],
                "model": exp["model_name"],
                "prompt_id": best_prompt_row.get("prompt_id") if best_prompt_row else last_snapshot.get("best_prompt_id"),
                "text": best_prompt_row.get("text") if best_prompt_row else last_snapshot.get("best_prompt_text", ""),
                "score": last_snapshot.get("best_score", 0.0),
                "reward_history": last_snapshot.get("reward_history", {}),
                "ensemble_models": _safe_json_loads(exp["config"], {}).get("ensemble_models", []),
                "ensemble_strategy": _safe_json_loads(exp["config"], {}).get("ensemble_strategy"),
            }

            return ExperimentResult(
                run_id=exp["run_id"],
                experiment_name=exp["experiment_name"] or exp["run_id"],
                source="sqlite",
                model_name=exp["model_name"],
                config=_safe_json_loads(exp["config"], {}),
                best_prompt=best_prompt,
                snapshots=snapshots,
                prompts=prompts,
                evaluations=[
                    {
                        **row,
                        "errors": _safe_json_loads(row.get("errors"), []),
                    }
                    for row in evaluations
                ],
            )
        finally:
            con.close()

    def _load_from_artifacts(
        self,
        experiment_name: Optional[str],
        run_id: Optional[str],
    ) -> Optional[ExperimentResult]:
        if not self.artifacts_dir.exists():
            return None

        target_dir: Optional[Path] = None
        if experiment_name:
            candidate = self.artifacts_dir / experiment_name
            if candidate.is_dir():
                target_dir = candidate
        else:
            dirs = sorted(
                (path for path in self.artifacts_dir.iterdir() if path.is_dir() and not path.name.startswith(".")),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            for candidate in dirs:
                best = self._read_json(candidate / "best_prompt.json")
                if run_id and best.get("run_id") != run_id:
                    continue
                target_dir = candidate
                break

        if target_dir is None or not target_dir.exists():
            return None

        best_prompt = self._read_json(target_dir / "best_prompt.json")
        if run_id and best_prompt and best_prompt.get("run_id") != run_id:
            return None

        snapshots = [
            self._read_json(path)
            for path in sorted(target_dir.glob("snapshot_gen_*.json"))
        ]
        prompts = []
        evaluations = []
        for path in sorted(target_dir.glob("prompts_gen_*.jsonl")):
            prompts.extend(self._read_jsonl(path))
        for path in sorted(target_dir.glob("evaluations_gen_*.jsonl")):
            evaluations.extend(self._read_jsonl(path))

        model_name = best_prompt.get("model", "ensemble")
        if isinstance(best_prompt.get("ensemble_models"), list):
            model_name = ",".join(best_prompt["ensemble_models"])

        return ExperimentResult(
            run_id=best_prompt.get("run_id", target_dir.name),
            experiment_name=target_dir.name,
            source="artifacts",
            model_name=model_name,
            config={
                "ensemble_models": best_prompt.get("ensemble_models", []),
                "ensemble_strategy": best_prompt.get("ensemble_strategy"),
            },
            best_prompt=best_prompt,
            snapshots=snapshots,
            prompts=prompts,
            evaluations=evaluations,
        )

    def _list_sqlite_experiments(self) -> list[dict[str, Any]]:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute(
                "SELECT id, run_id, experiment_name, model_name, status, created_at, config "
                "FROM experiments ORDER BY id DESC"
            ).fetchall()
            results = []
            for row in rows:
                config = _safe_json_loads(row["config"], {})
                last_snapshot = con.execute(
                    "SELECT best_score FROM snapshots WHERE experiment_id = ? ORDER BY generation DESC, id DESC LIMIT 1",
                    (row["id"],),
                ).fetchone()
                results.append(
                    {
                        "run_id": row["run_id"],
                        "experiment_name": row["experiment_name"] or row["run_id"],
                        "model_name": row["model_name"],
                        "status": row["status"],
                        "best_score": last_snapshot["best_score"] if last_snapshot else 0.0,
                        "created_at": row["created_at"],
                        "source": "sqlite",
                        "ensemble_models": config.get("ensemble_models", []),
                    }
                )
            return results
        finally:
            con.close()

    def _list_artifact_experiments(self) -> list[dict[str, Any]]:
        if not self.artifacts_dir.exists():
            return []
        results = []
        for exp_dir in sorted(self.artifacts_dir.iterdir()):
            if not exp_dir.is_dir() or exp_dir.name.startswith("."):
                continue
            best_prompt = self._read_json(exp_dir / "best_prompt.json")
            if not best_prompt:
                continue
            results.append(
                {
                    "run_id": best_prompt.get("run_id", exp_dir.name),
                    "experiment_name": exp_dir.name,
                    "model_name": best_prompt.get("model", "ensemble"),
                    "status": "completed",
                    "best_score": best_prompt.get("score", 0.0),
                    "created_at": None,
                    "source": "artifacts",
                    "ensemble_models": best_prompt.get("ensemble_models", []),
                }
            )
        return results

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not path.exists():
            return rows
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows


def default_repository() -> ResultsRepository:
    root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return ResultsRepository(
        artifacts_dir=root / "experiments" / "artifacts",
        db_path=SQLITE_DB_PATH,
    )
