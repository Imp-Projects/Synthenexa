"""
services/storage.py — Persist pipeline artifacts to disk or SQLite.

Architecture (dual-backend router):
    StorageService(backend="file" | "local" | "sqlite")
        ├── FileStorageBackend   — JSON/JSONL files (default)
        └── SQLiteStorageBackend — SQLAlchemy + SQLite

The backend is selected at construction time via the ``backend=`` kwarg
(which defaults to the ``STORAGE_BACKEND`` env var, which itself defaults to
"file"). The public API is identical for both backends.

``"local"`` is accepted as an alias for ``"file"`` for convenience.
"""
import json
import os
from typing import Any, Dict, List, Optional

from core.config import STORAGE_BACKEND
from core.schema import EvalResult, PromptCandidate


# ======================================================================= #
#  Phase 1 — File-based backend (original behaviour, fully preserved)     #
# ======================================================================= #


class FileStorageBackend:
    """Stores pipeline artifacts as JSON/JSONL files under a base directory."""

    def __init__(self, base_dir: str = "experiments/artifacts"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    # ---------------------------------------------------------------- #
    #  Low-level JSONL helpers                                          #
    # ---------------------------------------------------------------- #

    def save_jsonl(self, filename: str, data: List[Dict[str, Any]]) -> None:
        """Append rows to a JSONL file under base_dir."""
        path = os.path.join(self.base_dir, filename)
        with open(path, "a", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def load_jsonl(self, filename: str) -> List[Dict[str, Any]]:
        """Load all rows from a JSONL file under base_dir (empty list if missing)."""
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
    #  Domain-specific save methods (public API)                        #
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
        """Persist a full generation-state snapshot to a JSON file."""
        snapshot: Dict[str, Any] = {
            "generation":       generation,
            "best_score":       context.best_score,
            "best_prompt_id":   context.best_prompt.id if context.best_prompt else None,
            "best_prompt_text": context.best_prompt.text if context.best_prompt else None,
            "scores":           context.scores,
            "reward_history":   context.reward_history,
        }
        path = os.path.join(self.base_dir, f"snapshot_gen_{generation}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False, default=str)

    def save_run_log(self, log_entry: Dict[str, Any]) -> None:
        """Append a single log entry to the run log."""
        self.save_jsonl("run_logs.jsonl", [log_entry])

    # Lifecycle no-ops (file backend needs no experiment record)
    def create_experiment_record(self, run_id: str, model_name: str,
                                  experiment_name: str, config_dict: Dict[str, Any]) -> None:
        """No-op for the file backend."""

    def finalize_experiment(self, best_prompt: Optional[Any], best_score: float) -> None:
        """No-op for the file backend."""


# ======================================================================= #
#  SQLite backend                                                          #
# ======================================================================= #


class SQLiteStorageBackend:
    """Stores pipeline artifacts as rows in an SQLite database."""

    def __init__(self) -> None:
        # Import lazily so the file backend never requires SQLAlchemy
        from services.database import get_database_manager
        from core.config import DatabaseConfig
        self._db = get_database_manager(DatabaseConfig())
        # Auto-create tables on first use (replaces Alembic for local SQLite)
        self._db.create_all_tables()
        self._experiment_id: Optional[int] = None

    # ---------------------------------------------------------------- #
    #  Lifecycle                                                        #
    # ---------------------------------------------------------------- #

    def create_experiment_record(self, run_id: str, model_name: str,
                                  experiment_name: str, config_dict: Dict[str, Any]) -> None:
        """Insert an Experiment row and store its PK for subsequent saves."""
        from core.db_models import Experiment
        with self._db.get_session() as session:
            experiment = Experiment(
                run_id=run_id,
                experiment_name=experiment_name,
                model_name=model_name,
                config=config_dict,
                status="running",
            )
            session.add(experiment)
            session.flush()  # populate experiment.id before commit
            self._experiment_id = experiment.id
        print(f"[storage/sqlite] Experiment record created (id={self._experiment_id}, run_id={run_id})")

    def finalize_experiment(self, best_prompt: Optional[Any], best_score: float) -> None:
        """Mark the experiment as completed and flag the best prompt."""
        if self._experiment_id is None:
            return
        from core.db_models import Experiment, Prompt
        with self._db.get_session() as session:
            experiment = session.get(Experiment, self._experiment_id)
            if experiment:
                experiment.status = "completed"
            if best_prompt is not None:
                # Mark the best prompt row
                best_row = (
                    session.query(Prompt)
                    .filter_by(experiment_id=self._experiment_id, prompt_id=best_prompt.id)
                    .first()
                )
                if best_row:
                    best_row.is_best = True
        print(f"[storage/sqlite] Experiment {self._experiment_id} finalised (best_score={best_score:.4f})")

    # ---------------------------------------------------------------- #
    #  Domain-specific save methods (public API)                        #
    # ---------------------------------------------------------------- #

    def save_prompts(self, prompts: List[PromptCandidate], generation: int) -> None:
        """Insert Prompt rows for a given generation."""
        if self._experiment_id is None:
            raise RuntimeError("call create_experiment_record() before saving data")
        from core.db_models import Prompt
        with self._db.get_session() as session:
            for p in prompts:
                session.add(Prompt(
                    prompt_id=p.id,
                    experiment_id=self._experiment_id,
                    generation=generation,
                    text=p.text,
                ))

    def save_evaluations(self, results: List[EvalResult], generation: int) -> None:
        """Insert Evaluation rows for a given generation."""
        if self._experiment_id is None:
            raise RuntimeError("call create_experiment_record() before saving data")
        from core.db_models import Evaluation
        with self._db.get_session() as session:
            for r in results:
                session.add(Evaluation(
                    prompt_id=r.prompt_id,
                    experiment_id=self._experiment_id,
                    generation=generation,
                    passed=r.passed,
                    failed=r.failed,
                    total=r.total,
                    syntax_errors=getattr(r, "syntax_errors", 0),
                    errors=getattr(r, "errors", None),
                ))

    def save_generation_snapshot(self, context: Any, generation: int) -> None:
        """Insert a Snapshot row for a given generation."""
        if self._experiment_id is None:
            raise RuntimeError("call create_experiment_record() before saving data")
        from core.db_models import Snapshot
        with self._db.get_session() as session:
            session.add(Snapshot(
                experiment_id=self._experiment_id,
                generation=generation,
                best_score=context.best_score,
                best_prompt_id=context.best_prompt.id if context.best_prompt else None,
                best_prompt_text=context.best_prompt.text if context.best_prompt else None,
                scores=context.scores,
                reward_history=context.reward_history,
            ))

    def save_run_log(self, log_entry: Dict[str, Any]) -> None:
        """Insert a RunLog row."""
        if self._experiment_id is None:
            raise RuntimeError("call create_experiment_record() before saving data")
        from core.db_models import RunLog
        with self._db.get_session() as session:
            session.add(RunLog(
                experiment_id=self._experiment_id,
                log_entry=log_entry,
            ))


# ======================================================================= #
#  Router — public interface (backend-agnostic)                           #
# ======================================================================= #


class StorageService:
    """
    Backend-agnostic storage router.

    Delegates every save_* call to either ``FileStorageBackend`` or
    ``SQLiteStorageBackend`` depending on the ``backend`` argument (or
    the ``STORAGE_BACKEND`` env var when ``backend`` is not given).

    Accepted backend values:
      - ``"file"``   — JSON/JSONL artifacts on disk (default)
      - ``"local"``  — alias for ``"file"``; useful as an explicit fallback
      - ``"sqlite"`` — SQLite database at the configured db_path

    The public API is identical regardless of backend, so workflow.py and
    all other callers require no changes.

    Usage
    -----
    # File/local backend (default):
        storage = StorageService(base_dir="experiments/artifacts/run_123")

    # SQLite backend:
        storage = StorageService(backend="sqlite")
        storage.create_experiment_record(run_id, model_name, exp_name, cfg)
        # … run pipeline …
        storage.finalize_experiment(best_prompt, best_score)

    # Forced via CLI flag:
        storage = StorageService(backend=args.storage, base_dir=…)
    """

    def __init__(
        self,
        base_dir: str = "experiments/artifacts",
        backend: Optional[str] = None,
    ) -> None:
        resolved = (backend or STORAGE_BACKEND).lower()
        if resolved == "sqlite":
            self._backend: Any = SQLiteStorageBackend()
            print("[storage] Using SQLite backend.")
        else:
            # "file" and "local" both use the file backend
            self._backend = FileStorageBackend(base_dir=base_dir)
            if resolved not in ("file", "local"):
                print(f"[storage] Unknown backend '{resolved}', falling back to file.")

    # ---------------------------------------------------------------- #
    #  Lifecycle (no-op for file backend)                               #
    # ---------------------------------------------------------------- #

    def create_experiment_record(
        self,
        run_id: str,
        model_name: str,
        experiment_name: str,
        config_dict: Dict[str, Any],
    ) -> None:
        """Create a DB experiment record (sqlite) or do nothing (file/local)."""
        self._backend.create_experiment_record(run_id, model_name, experiment_name, config_dict)

    def finalize_experiment(
        self,
        best_prompt: Optional[Any],
        best_score: float,
    ) -> None:
        """Finalise the experiment record (sqlite) or do nothing (file/local)."""
        self._backend.finalize_experiment(best_prompt, best_score)

    # ---------------------------------------------------------------- #
    #  Public save API (identical for both backends)                    #
    # ---------------------------------------------------------------- #

    def save_prompts(self, prompts: List[PromptCandidate], generation: int) -> None:
        """Save candidate prompts for a given generation."""
        self._backend.save_prompts(prompts, generation)

    def save_evaluations(self, results: List[EvalResult], generation: int) -> None:
        """Save evaluation results for a given generation."""
        self._backend.save_evaluations(results, generation)

    def save_generation_snapshot(self, context: Any, generation: int) -> None:
        """Persist a full generation-state snapshot."""
        self._backend.save_generation_snapshot(context, generation)

    def save_run_log(self, log_entry: Dict[str, Any]) -> None:
        """Append a single log entry to the run log."""
        self._backend.save_run_log(log_entry)

    # ---------------------------------------------------------------- #
    #  File-backend helpers (pass-through; no-op on sqlite)              #
    # ---------------------------------------------------------------- #

    def save_jsonl(self, filename: str, data: List[Dict[str, Any]]) -> None:
        """Low-level JSONL append (file/local backend only; silently skipped on sqlite)."""
        if isinstance(self._backend, FileStorageBackend):
            self._backend.save_jsonl(filename, data)

    def load_jsonl(self, filename: str) -> List[Dict[str, Any]]:
        """Low-level JSONL read (file/local backend only; returns [] on sqlite)."""
        if isinstance(self._backend, FileStorageBackend):
            return self._backend.load_jsonl(filename)
        return []
