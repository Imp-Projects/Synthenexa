"""
scripts/migrate_json_to_postgres.py — Migrate historical JSON artifacts to PostgreSQL.

Reads every experiment directory under experiments/artifacts/ and bulk-inserts
all historical data into the configured PostgreSQL database.

Usage (run from synthenexa_pipeline/):
    python scripts/migrate_json_to_postgres.py
    python scripts/migrate_json_to_postgres.py --artifacts-dir path/to/artifacts
    python scripts/migrate_json_to_postgres.py --dry-run
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ── Ensure synthenexa_pipeline/ is on sys.path ──────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))      # scripts/
_PIPELINE = os.path.dirname(_HERE)                       # synthenexa_pipeline/
if _PIPELINE not in sys.path:
    sys.path.insert(0, _PIPELINE)
# ─────────────────────────────────────────────────────────────────────────────

from core.config import DatabaseConfig
from core.db_models import Evaluation, Experiment, Prompt, RunLog, Snapshot
from services.database import get_database_manager


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def migrate_experiment(session, exp_dir: Path, dry_run: bool) -> dict:
    """Migrate one experiment directory. Returns a summary dict."""
    name = exp_dir.name
    best_prompt_file = exp_dir / "best_prompt.json"
    run_logs_file = exp_dir / "run_logs.jsonl"

    if not best_prompt_file.exists():
        return {"experiment": name, "status": "skipped", "reason": "no best_prompt.json"}

    best = _load_json(best_prompt_file)
    run_id = best.get("run_id", name[:8])
    model_name = best.get("model", "unknown")
    reward_history = best.get("reward_history", {})
    best_prompt_id = best.get("prompt_id", "")
    best_prompt_text = best.get("text", "")

    counts = {"prompts": 0, "evaluations": 0, "snapshots": 0, "run_logs": 0}

    if dry_run:
        # Count what *would* be inserted without touching the DB
        for gen_file in sorted(exp_dir.glob("prompts_gen_*.jsonl")):
            counts["prompts"] += len(_load_jsonl(gen_file))
        for gen_file in sorted(exp_dir.glob("evaluations_gen_*.jsonl")):
            counts["evaluations"] += len(_load_jsonl(gen_file))
        counts["snapshots"] = len(list(exp_dir.glob("snapshot_gen_*.json")))
        if run_logs_file.exists():
            counts["run_logs"] = len(_load_jsonl(run_logs_file))
        return {"experiment": name, "status": "dry-run", **counts}

    # Check if already migrated (skip to avoid duplicate run_id)
    existing = session.query(Experiment).filter_by(run_id=run_id).first()
    if existing:
        return {"experiment": name, "status": "already_exists", "experiment_id": existing.id}

    # Create Experiment row
    experiment = Experiment(
        run_id=run_id,
        experiment_name=name,
        model_name=model_name,
        config={"migrated": True, "reward_history": reward_history},
        status="completed",
    )
    session.add(experiment)
    session.flush()  # get experiment.id
    exp_id = experiment.id

    # Insert Prompts
    for gen_file in sorted(exp_dir.glob("prompts_gen_*.jsonl")):
        gen_num = int(gen_file.stem.split("_")[-1])
        for row in _load_jsonl(gen_file):
            pid = row.get("id", "")
            session.add(Prompt(
                prompt_id=pid,
                experiment_id=exp_id,
                generation=gen_num,
                text=row.get("text", ""),
                is_best=(pid == best_prompt_id),
            ))
            counts["prompts"] += 1

    # Insert Evaluations
    for gen_file in sorted(exp_dir.glob("evaluations_gen_*.jsonl")):
        gen_num = int(gen_file.stem.split("_")[-1])
        for row in _load_jsonl(gen_file):
            session.add(Evaluation(
                prompt_id=row.get("prompt_id", ""),
                experiment_id=exp_id,
                generation=gen_num,
                passed=row.get("passed", 0),
                failed=row.get("failed", 0),
                total=row.get("total", 0),
                syntax_errors=row.get("syntax_errors", 0),
                errors=row.get("errors"),
            ))
            counts["evaluations"] += 1

    # Insert Snapshots
    for snap_file in sorted(exp_dir.glob("snapshot_gen_*.json")):
        gen_num = int(snap_file.stem.split("_")[-1])
        snap = _load_json(snap_file)
        session.add(Snapshot(
            experiment_id=exp_id,
            generation=gen_num,
            best_score=snap.get("best_score"),
            best_prompt_id=snap.get("best_prompt_id"),
            best_prompt_text=snap.get("best_prompt_text"),
            scores=snap.get("scores"),
            reward_history=snap.get("reward_history"),
        ))
        counts["snapshots"] += 1

    # Insert RunLogs
    if run_logs_file.exists():
        for entry in _load_jsonl(run_logs_file):
            session.add(RunLog(experiment_id=exp_id, log_entry=entry))
            counts["run_logs"] += 1

    return {"experiment": name, "status": "migrated", "experiment_id": exp_id, **counts}


def _print_summary(results: list[dict]) -> None:
    col_w = [30, 12, 8, 12, 11, 9]
    header = ["Experiment", "Status", "Prompts", "Evaluations", "Snapshots", "RunLogs"]
    sep = "  ".join("-" * w for w in col_w)
    row_fmt = "  ".join(f"{{:<{w}}}" for w in col_w)

    print("\n" + "=" * 80)
    print("  Migration Summary")
    print("=" * 80)
    print(row_fmt.format(*header))
    print(sep)
    for r in results:
        print(row_fmt.format(
            r["experiment"][:29],
            r["status"],
            str(r.get("prompts", "-")),
            str(r.get("evaluations", "-")),
            str(r.get("snapshots", "-")),
            str(r.get("run_logs", "-")),
        ))
    print("=" * 80)
    migrated = sum(1 for r in results if r["status"] == "migrated")
    skipped  = sum(1 for r in results if r["status"] in ("skipped", "already_exists"))
    print(f"  Total: {len(results)} experiments — {migrated} migrated, {skipped} skipped.\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate historical JSON artifacts to PostgreSQL."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=os.path.join(_PIPELINE, "experiments", "artifacts"),
        help="Path to the artifacts root directory (default: experiments/artifacts/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count rows that would be inserted without writing to the database.",
    )
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts_dir)
    if not artifacts_root.is_dir():
        print(f"[ERROR] Artifacts directory not found: {artifacts_root}")
        sys.exit(1)

    exp_dirs = sorted(
        p for p in artifacts_root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    if not exp_dirs:
        print("[INFO] No experiment directories found.")
        sys.exit(0)

    print(f"[INFO] Found {len(exp_dirs)} experiment director{'y' if len(exp_dirs)==1 else 'ies'}.")
    if args.dry_run:
        print("[INFO] Dry-run mode -- no data will be written to the database.\n")
        results = []
        for exp_dir in exp_dirs:
            result = migrate_experiment(None, exp_dir, dry_run=True)
            results.append(result)
        _print_summary(results)
        return

    db = get_database_manager(DatabaseConfig())
    results = []

    for exp_dir in exp_dirs:
        if args.dry_run:
            result = migrate_experiment(None, exp_dir, dry_run=True)
            results.append(result)
        else:
            try:
                with db.get_session() as session:
                    result = migrate_experiment(session, exp_dir, dry_run=False)
                    results.append(result)
                    status_str = result["status"]
                    if status_str == "migrated":
                        print(f"  ✓ {exp_dir.name} — migrated (id={result.get('experiment_id')})")
                    elif status_str == "already_exists":
                        print(f"  ~ {exp_dir.name} — already in DB (id={result.get('experiment_id')}), skipped.")
                    else:
                        print(f"  ! {exp_dir.name} — skipped ({result.get('reason', '')})")
            except Exception as exc:
                print(f"  ✗ {exp_dir.name} — ERROR: {exc}")
                results.append({"experiment": exp_dir.name, "status": "error", "reason": str(exc)})

    _print_summary(results)


if __name__ == "__main__":
    main()
