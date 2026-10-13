"""
Analysis/import_artifacts.py — Import artifact folder into SQLite
=================================================================

Reads an experiment from a file-based artifact directory
(e.g. experiments/artifacts/testnew-2) and writes all records into
the shared SQLite database so that compare_baseline.py and
visualize_experiments.py can use them directly.

Usage (from synthenexa_pipeline/):
    python Analysis/import_artifacts.py --experiment testnew-2
    python Analysis/import_artifacts.py --experiment testnew-2 --force
    python Analysis/import_artifacts.py --list

Options:
    --experiment   Name of the artifact directory to import (required unless --list).
    --force        Overwrite an existing DB record with the same run_id.
    --list         List all importable artifact directories and exit.
    --db           Path to the SQLite DB (default: experiments/synthenexa.db).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE     = Path(__file__).resolve().parent
_PIPELINE = _HERE.parent
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from services.results_repository import default_repository
from services.database import DatabaseManager
from core.config import DatabaseConfig, SQLITE_DB_PATH
from core.db_models import Experiment, Prompt, Evaluation, Snapshot


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _infer_generation(prompt_id: str) -> int:
    marker = "_gen"
    if marker not in prompt_id:
        return 1
    try:
        remainder = prompt_id.split(marker, 1)[1]
        return int(remainder.split("_", 1)[0])
    except (IndexError, ValueError):
        return 1


def _infer_origin_model(prompt_id: str, ensemble_models: list) -> str:
    pid_lower = prompt_id.lower()
    for model in ensemble_models:
        slug = model.replace(":", "_").replace("-", "_").replace(".", "_").lower()
        if slug in pid_lower:
            return model
    return ensemble_models[0] if ensemble_models else "unknown"


# ─────────────────────────────────────────────────────────────────────────────
#  Core import
# ─────────────────────────────────────────────────────────────────────────────

def import_experiment(experiment_name: str, db_path: Path, force: bool = False) -> None:
    repo = default_repository()

    print(f"[import] Reading artifact: {experiment_name}")
    result = repo.load_experiment(experiment_name=experiment_name, source="artifacts")
    run_id = result.run_id
    print(f"[import] run_id={run_id}  models={result.config.get('ensemble_models', [])}")

    config = DatabaseConfig(db_path=str(db_path))
    db = DatabaseManager(config)
    db.create_all_tables()

    with db.get_session() as session:
        existing = session.query(Experiment).filter_by(run_id=run_id).first()
        if existing and not force:
            print(
                f"[import] Experiment with run_id='{run_id}' already exists in the DB "
                f"(id={existing.id}, name='{existing.experiment_name}').\n"
                "         Use --force to overwrite."
            )
            return
        if existing and force:
            print(f"[import] --force: deleting existing experiment id={existing.id}")
            session.delete(existing)
            session.flush()

        ensemble_models = result.config.get("ensemble_models", [])
        best_pid = result.best_prompt.get("prompt_id") or result.best_prompt.get("id") or ""

        # Experiment row
        exp = Experiment(
            run_id=run_id,
            experiment_name=result.experiment_name,
            model_name=result.model_name,
            status="completed",
            config=result.config,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        session.add(exp)
        session.flush()
        eid = exp.id
        print(f"[import] Created experiment id={eid}")

        # Prompts
        seen_prompt_keys = set()
        prompt_count = 0
        for row in result.prompts:
            pid = row.get("prompt_id") or row.get("id", "")
            generation = row.get("generation") if row.get("generation") is not None else _infer_generation(pid)
            key = (pid, eid, generation)
            if key in seen_prompt_keys:
                continue
            seen_prompt_keys.add(key)
            session.add(Prompt(
                prompt_id=pid,
                experiment_id=eid,
                generation=generation,
                text=row.get("text", ""),
                is_base=bool(row.get("is_base", False)),
                is_best=(pid == best_pid),
                created_at=datetime.now(timezone.utc),
            ))
            prompt_count += 1

        if best_pid and (best_pid, eid, _infer_generation(best_pid)) not in seen_prompt_keys:
            generation = _infer_generation(best_pid)
            session.add(Prompt(
                prompt_id=best_pid,
                experiment_id=eid,
                generation=generation,
                text=result.best_prompt.get("text", ""),
                is_base=False,
                is_best=True,
                created_at=datetime.now(timezone.utc),
            ))
            prompt_count += 1

        session.flush()
        print(f"[import] Inserted {prompt_count} prompt rows")

        # Evaluations
        eval_count = 0
        all_evaluations = list(result.evaluations) + [
            {**row, "generation": -1, "dataset_split": "test"}
            for row in result.final_test_evaluations
        ]
        for row in all_evaluations:
            pid = row.get("prompt_id", "")
            generation = row.get("generation") if row.get("generation") is not None else _infer_generation(pid)
            if row.get("is_base") and pid == "prompt_0" and row.get("dataset_split") != "test":
                generation = 0
            evaluator_model = row.get("evaluator_model") or _infer_origin_model(pid, ensemble_models)
            errors_val = row.get("errors", [])
            session.add(Evaluation(
                prompt_id=pid,
                experiment_id=eid,
                generation=generation,
                evaluator_model=evaluator_model,
                is_base=bool(row.get("is_base", False)),
                dataset_split=row.get("dataset_split", "all"),
                task_ids=row.get("task_ids", []),
                passed=int(row.get("passed", 0)),
                failed=int(row.get("failed", 0)),
                total=int(row.get("total", 0)),
                syntax_errors=int(row.get("syntax_errors", 0)),
                errors=errors_val if isinstance(errors_val, list) else [],
                total_time_ms=float(row.get("total_time_ms", 0.0)),
                min_time_ms=float(row.get("min_time_ms", 0.0)),
                max_time_ms=float(row.get("max_time_ms", 0.0)),
                avg_time_ms=float(row.get("avg_time_ms", 0.0)),
                created_at=datetime.now(timezone.utc),
            ))
            eval_count += 1

        session.flush()
        print(f"[import] Inserted {eval_count} evaluation rows")

        # Snapshots
        snap_count = 0
        for snap in result.snapshots:
            session.add(Snapshot(
                experiment_id=eid,
                generation=snap.get("generation", 0),
                best_score=snap.get("best_score"),
                best_prompt_id=snap.get("best_prompt_id", ""),
                best_prompt_text=snap.get("best_prompt_text", ""),
                scores=snap.get("scores"),
                reward_history=snap.get("reward_history"),
                best_ensemble_scores=snap.get("best_ensemble_scores"),
                model_leaderboard=snap.get("model_leaderboard"),
                ensemble_scores=snap.get("ensemble_scores"),
                created_at=datetime.now(timezone.utc),
            ))
            snap_count += 1

        session.flush()
        print(f"[import] Inserted {snap_count} snapshot rows")

    print(f"\n[✓] Import complete — '{result.experiment_name}' is now in the DB")
    print(f"    run_id: {run_id}  |  DB: {db_path}")
    print(f"\n    Next steps:")
    print(f"    python Analysis/compare_baseline.py --list-experiments")
    print(f"    python Analysis/visualize_experiments.py --list-experiments")


# ─────────────────────────────────────────────────────────────────────────────
#  Listing
# ─────────────────────────────────────────────────────────────────────────────

def list_importable(artifacts_dir: Path) -> None:
    if not artifacts_dir.exists():
        print(f"[import] Artifacts directory not found: {artifacts_dir}")
        return
    dirs = [d for d in sorted(artifacts_dir.iterdir()) if d.is_dir() and not d.name.startswith(".")]
    if not dirs:
        print("[import] No artifact directories found.")
        return
    print(f"\n{'NAME':<30}  {'RUN ID':<12}  {'BEST SCORE':<12}  MODELS")
    print("─" * 80)
    for d in dirs:
        bp_path = d / "best_prompt.json"
        if not bp_path.exists():
            continue
        bp = json.loads(bp_path.read_text())
        models = ", ".join(bp.get("ensemble_models", [bp.get("model", "?")]))
        print(f"{d.name:<30}  {bp.get('run_id', '?'):<12}  {bp.get('score', 0):<12.2%}  {models}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import a Synthenexa artifact folder into the SQLite database."
    )
    parser.add_argument("--experiment", help="Name of the artifact directory to import.")
    parser.add_argument("--db", type=Path, default=Path(SQLITE_DB_PATH),
                        help=f"Path to the SQLite database (default: {SQLITE_DB_PATH})")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing experiment with the same run_id.")
    parser.add_argument("--list", action="store_true",
                        help="List importable artifact directories and exit.")
    args = parser.parse_args()

    artifacts_dir = _PIPELINE / "experiments" / "artifacts"

    if args.list:
        list_importable(artifacts_dir)
        return

    if not args.experiment:
        parser.error("--experiment is required (or use --list to see available artifacts).")

    import_experiment(args.experiment, args.db, force=args.force)


if __name__ == "__main__":
    main()
