"""
scripts/inspect_experiments.py -- CLI tool to inspect experiment results.

Works with the file backend and SQLite backend. Use --storage to override STORAGE_BACKEND.

Sub-commands
------------
  list                  Table of all experiments
  show  <run_id>        Full details for one run
  best  <run_id>        Best prompt text + score
  compare               Cross-experiment score comparison

Usage (run from synthenexa_pipeline/):
    python scripts/inspect_experiments.py list
    python scripts/inspect_experiments.py --storage sqlite list
    python scripts/inspect_experiments.py show a1915724
    python scripts/inspect_experiments.py best a1915724
    python scripts/inspect_experiments.py compare
    python scripts/inspect_experiments.py --json list
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

# ── Ensure synthenexa_pipeline/ is on sys.path ──────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPELINE = os.path.dirname(_HERE)
if _PIPELINE not in sys.path:
    sys.path.insert(0, _PIPELINE)
# ─────────────────────────────────────────────────────────────────────────────

from core.config import STORAGE_BACKEND


# ======================================================================= #
#  Formatting helpers                                                      #
# ======================================================================= #

def _sep(widths: list[int]) -> str:
    return "  ".join("-" * w for w in widths)

def _row(widths: list[int], values: list[str]) -> str:
    parts = []
    for w, v in zip(widths, values):
        parts.append(str(v)[:w].ljust(w))
    return "  ".join(parts)

def _bar(score: float, width: int = 20) -> str:
    """ASCII progress bar safe for all console encodings."""
    filled = int(score * width)
    return "#" * filled + "." * (width - filled)


# ======================================================================= #
#  File backend readers                                                    #
# ======================================================================= #

class FileInspector:
    def __init__(self, artifacts_dir: str):
        self.root = Path(artifacts_dir)

    def _exp_dirs(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        return sorted(p for p in self.root.iterdir() if p.is_dir() and not p.name.startswith("."))

    def _read_best(self, exp_dir: Path) -> Optional[dict]:
        f = exp_dir / "best_prompt.json"
        if not f.exists():
            return None
        with open(f, encoding="utf-8") as fh:
            return json.load(fh)

    def _read_snapshots(self, exp_dir: Path) -> list[dict]:
        snaps = []
        for snap_file in sorted(exp_dir.glob("snapshot_gen_*.json")):
            with open(snap_file, encoding="utf-8") as fh:
                snaps.append(json.load(fh))
        return snaps

    def list_experiments(self) -> list[dict]:
        results = []
        for exp_dir in self._exp_dirs():
            best = self._read_best(exp_dir)
            if best is None:
                continue
            results.append({
                "run_id":          best.get("run_id", exp_dir.name[:8]),
                "experiment_name": exp_dir.name,
                "model_name":      best.get("model", "—"),
                "status":          "completed",
                "best_score":      best.get("score", 0.0),
                "created_at":      "—",
            })
        return results

    def show_experiment(self, run_id: str) -> Optional[dict]:
        for exp_dir in self._exp_dirs():
            best = self._read_best(exp_dir)
            if best and best.get("run_id", exp_dir.name[:8]) == run_id:
                snaps = self._read_snapshots(exp_dir)
                return {
                    "run_id":          run_id,
                    "experiment_name": exp_dir.name,
                    "model_name":      best.get("model", "—"),
                    "status":          "completed",
                    "best_score":      best.get("score", 0.0),
                    "best_prompt":     best.get("text", ""),
                    "snapshots":       snaps,
                    "reward_history":  best.get("reward_history", {}),
                }
        return None

    def get_best_prompt(self, run_id: str) -> Optional[dict]:
        detail = self.show_experiment(run_id)
        if detail is None:
            return None
        return {"run_id": run_id, "best_prompt": detail["best_prompt"], "best_score": detail["best_score"]}

    def compare(self) -> list[dict]:
        return self.list_experiments()


# ======================================================================= #
#  SQLite backend readers                                                   #
# ======================================================================= #

class SQLiteInspector:
    def __init__(self):
        from services.database import get_database_manager
        from core.config import DatabaseConfig
        self._db = get_database_manager(DatabaseConfig())

    def list_experiments(self) -> list[dict]:
        from core.db_models import Experiment, Prompt
        results = []
        with self._db.get_session() as session:
            experiments = session.query(Experiment).order_by(Experiment.created_at.desc()).all()
            for exp in experiments:
                best_prompt = (
                    session.query(Prompt)
                    .filter_by(experiment_id=exp.id, is_best=True)
                    .first()
                )
                # Derive best_score from snapshots if available
                from core.db_models import Snapshot
                last_snap = (
                    session.query(Snapshot)
                    .filter_by(experiment_id=exp.id)
                    .order_by(Snapshot.generation.desc())
                    .first()
                )
                results.append({
                    "run_id":          exp.run_id,
                    "experiment_name": exp.experiment_name or exp.run_id,
                    "model_name":      exp.model_name,
                    "status":          exp.status,
                    "best_score":      last_snap.best_score if last_snap else 0.0,
                    "created_at":      str(exp.created_at)[:16] if exp.created_at else "—",
                })
        return results

    def show_experiment(self, run_id: str) -> Optional[dict]:
        from core.db_models import Experiment, Prompt, Snapshot
        with self._db.get_session() as session:
            exp = session.query(Experiment).filter_by(run_id=run_id).first()
            if exp is None:
                return None
            snaps = (
                session.query(Snapshot)
                .filter_by(experiment_id=exp.id)
                .order_by(Snapshot.generation)
                .all()
            )
            best_prompt_obj = (
                session.query(Prompt)
                .filter_by(experiment_id=exp.id, is_best=True)
                .first()
            )
            reward_history = {}
            snapshot_list = []
            for s in snaps:
                reward_history[str(s.generation)] = s.best_score or 0.0
                snapshot_list.append({
                    "generation":      s.generation,
                    "best_score":      s.best_score,
                    "best_prompt_id":  s.best_prompt_id,
                    "best_prompt_text": s.best_prompt_text,
                    "scores":          s.scores,
                })
            last_snap = snaps[-1] if snaps else None
            return {
                "run_id":          exp.run_id,
                "experiment_name": exp.experiment_name or exp.run_id,
                "model_name":      exp.model_name,
                "status":          exp.status,
                "best_score":      last_snap.best_score if last_snap else 0.0,
                "best_prompt":     best_prompt_obj.text if best_prompt_obj else "",
                "snapshots":       snapshot_list,
                "reward_history":  reward_history,
            }

    def get_best_prompt(self, run_id: str) -> Optional[dict]:
        detail = self.show_experiment(run_id)
        if detail is None:
            return None
        return {"run_id": run_id, "best_prompt": detail["best_prompt"], "best_score": detail["best_score"]}

    def compare(self) -> list[dict]:
        return self.list_experiments()


# ======================================================================= #
#  Command handlers                                                        #
# ======================================================================= #

def cmd_list(inspector: Any, as_json: bool) -> None:
    experiments = inspector.list_experiments()
    if not experiments:
        print("No experiments found.")
        return
    if as_json:
        print(json.dumps(experiments, indent=2, default=str))
        return

    widths = [10, 30, 25, 10, 10, 16]
    headers = ["Run ID", "Experiment", "Model", "Status", "Best Score", "Created At"]
    print("\n" + "=" * 110)
    print("  Experiments")
    print("=" * 110)
    print(_row(widths, headers))
    print(_sep(widths))
    for exp in experiments:
        score = exp.get("best_score") or 0.0
        print(_row(widths, [
            exp["run_id"],
            exp["experiment_name"],
            exp["model_name"],
            exp["status"],
            f"{score:.4f}",
            exp["created_at"],
        ]))
    print("=" * 110 + "\n")


def cmd_show(inspector: Any, run_id: str, as_json: bool) -> None:
    detail = inspector.show_experiment(run_id)
    if detail is None:
        print(f"[ERROR] No experiment found with run_id='{run_id}'")
        sys.exit(1)
    if as_json:
        print(json.dumps(detail, indent=2, default=str))
        return

    print("\n" + "=" * 70)
    print(f"  Experiment: {detail['experiment_name']}")
    print("=" * 70)
    print(f"  Run ID    : {detail['run_id']}")
    print(f"  Model     : {detail['model_name']}")
    print(f"  Status    : {detail['status']}")
    print(f"  Best Score: {detail['best_score']:.4f}")

    print("\n  Generation Scores")
    print("  " + "-" * 50)
    rh = detail.get("reward_history", {})
    for gen, score in sorted(rh.items(), key=lambda kv: int(kv[0])):
        score = float(score) if score is not None else 0.0
        print(f"    Gen {int(gen):>3}  {score:>8.4f}  {_bar(score)}")
    print()


def cmd_best(inspector: Any, run_id: str, as_json: bool) -> None:
    result = inspector.get_best_prompt(run_id)
    if result is None:
        print(f"[ERROR] No experiment found with run_id='{run_id}'")
        sys.exit(1)
    if as_json:
        print(json.dumps(result, indent=2, default=str))
        return

    print("\n" + "=" * 70)
    print(f"  Best Prompt — run_id={run_id}  score={result['best_score']:.4f}")
    print("=" * 70)
    print(result["best_prompt"])
    print()


def cmd_compare(inspector: Any, as_json: bool) -> None:
    experiments = inspector.list_experiments()
    if not experiments:
        print("No experiments found.")
        return

    # Group by model — keep best score per model
    by_model: dict[str, dict] = {}
    for exp in experiments:
        model = exp["model_name"]
        score = exp.get("best_score") or 0.0
        if model not in by_model or score > (by_model[model].get("best_score") or 0.0):
            by_model[model] = exp

    ranked = sorted(by_model.values(), key=lambda e: e.get("best_score") or 0.0, reverse=True)

    if as_json:
        print(json.dumps(ranked, indent=2, default=str))
        return

    widths = [25, 12, 30]
    headers = ["Model", "Best Score", "Experiment"]
    print("\n" + "=" * 75)
    print("  Model Comparison (best score per model)")
    print("=" * 75)
    print(_row(widths, headers))
    print(_sep(widths))
    for exp in ranked:
        score = exp.get("best_score") or 0.0
        print(_row(widths, [
            exp["model_name"],
            f"{score:.4f}",
            exp["experiment_name"],
        ]))
    print("=" * 75 + "\n")


# ======================================================================= #
#  Entry point                                                             #
# ======================================================================= #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect Synthenexa experiment results (file or sqlite backend).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        choices=["file", "sqlite"],
        help="Storage backend (overrides STORAGE_BACKEND env var).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=os.path.join(_PIPELINE, "experiments", "artifacts"),
        help="Artifacts root directory (file backend only).",
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="Output as JSON.")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List all experiments.")
    show_p = sub.add_parser("show", help="Show full details for one experiment.")
    show_p.add_argument("run_id", help="Run ID to inspect.")
    best_p = sub.add_parser("best", help="Print best prompt text + score for one experiment.")
    best_p.add_argument("run_id", help="Run ID to inspect.")
    sub.add_parser("compare", help="Cross-experiment comparison table (best score per model).")

    args = parser.parse_args()
    backend = (args.storage or STORAGE_BACKEND).lower()

    if backend == "sqlite":
        try:
            inspector: Any = SQLiteInspector()
        except Exception as exc:
            print(f"[ERROR] Could not connect to SQLite: {exc}")
            sys.exit(1)
    else:
        inspector = FileInspector(args.artifacts_dir)

    if args.command == "list":
        cmd_list(inspector, args.as_json)
    elif args.command == "show":
        cmd_show(inspector, args.run_id, args.as_json)
    elif args.command == "best":
        cmd_best(inspector, args.run_id, args.as_json)
    elif args.command == "compare":
        cmd_compare(inspector, args.as_json)


if __name__ == "__main__":
    main()
