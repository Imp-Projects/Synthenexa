# PostgreSQL Integration — Full Implementation Plan

## Where Are the JSON Artifact Files?

**Mystery solved.** The files are being written — all 40+ past runs exist in:
```
synthenexa_pipeline/experiments/artifacts/<experiment_name>/
```

The reason you can't see them in VS Code is a **devcontainer setting** in [devcontainer.json](file:///workspaces/Synthenexa/.devcontainer/devcontainer.json#L37):

```json
"files.exclude": {
    "**/experiments/artifacts": true
}
```

This deliberately hides the folder from the VS Code Explorer. The files are there, the terminal correctly writes them, but the IDE suppresses the view. A single run produces:
- `best_prompt.json` — final best prompt and reward history
- `prompts_gen_{N}.jsonl` — all candidate prompts per generation
- `evaluations_gen_{N}.jsonl` — pass/fail results per generation
- `snapshot_gen_{N}.json` — full generation state (scores, history)

---

## Current Architecture Summary

| Layer | Status |
|---|---|
| `docker-compose.yml` (devcontainer) | ✅ Postgres `db` service configured |
| `core/config.py` → `DatabaseConfig` | ✅ Reads env vars, defaults set |
| `services/database.py` → `DatabaseManager` | ✅ SQLAlchemy engine + session pool |
| `core/db_models.py` | ✅ ORM models for all 5 tables |
| `migrations/` (Alembic) | ✅ Schema migration `001_initial_schema.py` ready |
| `.env` → `STORAGE_BACKEND=file` | ✅ Env var exists but is not yet read by any code |
| `services/storage.py` → `StorageService` | ❌ File-only; ignores `STORAGE_BACKEND` entirely |
| `scripts/run_optimization.py` | ❌ Never creates an `Experiment` DB record; never passes a DB session to storage |
| Inspection / query CLI | ❌ Does not exist |

**Key insight:** `STORAGE_BACKEND=file` is already in `.env` — the intent for a dual backend was always there, it just never got wired up.

---

## Design: Dual-Backend `StorageService`

The existing `StorageService` will become a **router** that delegates to one of two backends:

```
StorageService(backend="file" | "postgres")
    ├── FileStorageBackend   — current JSON/JSONL logic (unchanged)
    └── PostgresStorageBackend — new SQLAlchemy-based backend
```

The backend is selected at construction time via the `STORAGE_BACKEND` env var or an explicit CLI flag.

---

## Proposed Changes

---

### 1. Refactor `StorageService` into a Backend-Pluggable Router

#### [MODIFY] [storage.py](file:///workspaces/Synthenexa/synthenexa_pipeline/services/storage.py)

Split the file into three classes in the same module:

- **`FileStorageBackend`** — Extracts the current file-write logic with zero behavioral changes. Preserves full backward compatibility.
- **`PostgresStorageBackend`** — New backend. Uses `DatabaseManager.get_session()` to insert rows into `prompts`, `evaluations`, `snapshots`, and `run_logs` tables. Requires an `experiment_id` (the PK of the `experiments` row for this run).
- **`StorageService`** — Thin router. Constructor reads `STORAGE_BACKEND` env var (or accepts an explicit `backend=` argument). Delegates every `save_*` call to the active backend. Also exposes `create_experiment_record()` which is a no-op for the file backend and inserts an `Experiment` row for the Postgres backend.

The public API (`save_prompts`, `save_evaluations`, `save_generation_snapshot`, `save_run_log`) stays **100% identical** so `workflow.py` needs no changes at all.

---

### 2. Wire the Experiment Record Into the Pipeline Entry Point

#### [MODIFY] [run_optimization.py](file:///workspaces/Synthenexa/synthenexa_pipeline/scripts/run_optimization.py)

In `_run_single_experiment()`, after building the `StorageService` instance:

1. Call `storage.create_experiment_record(run_id, model_name, experiment_name, config_dict)`.
   - For the file backend: no-op.
   - For the Postgres backend: inserts an `Experiment` row and stores the returned `experiment_id` in the backend for subsequent saves.
2. At the end of the run, call `storage.finalize_experiment(best_prompt, best_score)` to update `status="completed"` and mark the best prompt in the `prompts` table.

This is the **only change** to `run_optimization.py` — the rest of the workflow is unaffected.

---

### 3. CLI Flag for Backend Selection

#### [MODIFY] [run_optimization.py](file:///workspaces/Synthenexa/synthenexa_pipeline/scripts/run_optimization.py)

Add `--storage` argument:

```bash
python scripts/run_optimization.py --storage postgres --generations 10 --mutations 5
python scripts/run_optimization.py --storage file --generations 1 --mutations 1  # default
```

If `--storage` is not passed, fall back to the `STORAGE_BACKEND` env var (default: `file`).

---

### 4. Migration Script for Existing JSON Artifacts

#### [NEW] [scripts/migrate_json_to_postgres.py](file:///workspaces/Synthenexa/synthenexa_pipeline/scripts/migrate_json_to_postgres.py)

A one-off script that walks `experiments/artifacts/`, reads each run's files, and inserts all historical data into PostgreSQL:

1. For each experiment directory, read `best_prompt.json` to extract `run_id`, `model`, and `reward_history`.
2. Create an `Experiment` row.
3. For each generation `N`:
   - Read `prompts_gen_{N}.jsonl` → insert `Prompt` rows.
   - Read `evaluations_gen_{N}.jsonl` → insert `Evaluation` rows.
   - Read `snapshot_gen_{N}.json` → insert a `Snapshot` row.
4. Mark the best prompt's `is_best=True`.
5. Print a summary of rows inserted per experiment.

Usage:
```bash
cd synthenexa_pipeline
python scripts/migrate_json_to_postgres.py
```

---

### 5. Developer Inspection Workflow (CLI Tool)

#### [NEW] [scripts/inspect_experiments.py](file:///workspaces/Synthenexa/synthenexa_pipeline/scripts/inspect_experiments.py)

A lightweight CLI tool to replace opening JSON files manually, with these sub-commands:

| Command | What it shows |
|---|---|
| `list` | Table of all experiments: run_id, model, status, best_score, date |
| `show <run_id>` | Full details for one run: all generations, scores, reward history |
| `best <run_id>` | Just the best prompt text + score |
| `compare` | Cross-experiment comparison table (best score per model) |

Example:
```bash
# List all experiments
python scripts/inspect_experiments.py list

# Show detailed results for a run
python scripts/inspect_experiments.py show a1915724

# Print best prompt text for a run
python scripts/inspect_experiments.py best a1915724

# Compare all models across experiments
python scripts/inspect_experiments.py compare
```

Output is rendered as formatted tables to the terminal (no external dependencies — pure Python `texttable` or manual formatting). Optionally `--json` flag for machine-readable output.

> [!NOTE]
> This covers the "developer workflow" question directly: you won't need a database client or pgAdmin to inspect results. The CLI tool queries the DB and formats the output in the same style as the existing `_print_reward_table` output in `run_optimization.py`.

---

### 6. `.env` & Config Alignment

#### [MODIFY] [.env](file:///workspaces/Synthenexa/.env)
- `STORAGE_BACKEND=file` is already there — no change needed for the default.
- Add a comment documenting `STORAGE_BACKEND=postgres` as the alternative.

#### [MODIFY] [core/config.py](file:///workspaces/Synthenexa/synthenexa_pipeline/core/config.py)
- Add `STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "file")` as a module-level constant so both `StorageService` and the CLI can read it cleanly.

---

## What Changes vs. What Stays the Same

| File | Change Type | Summary |
|---|---|---|
| `services/storage.py` | Refactor + extend | Add `FileStorageBackend`, `PostgresStorageBackend`, router logic |
| `scripts/run_optimization.py` | Minor extend | `create_experiment_record()` + `--storage` flag |
| `core/config.py` | 1-line add | `STORAGE_BACKEND` constant |
| `.env` | Comment add | Document `postgres` option |
| `scripts/migrate_json_to_postgres.py` | **New file** | One-off historical data migration |
| `scripts/inspect_experiments.py` | **New file** | CLI inspection tool |
| `workflow.py` | **No change** | Public API stays identical |
| `core/db_models.py` | **No change** | Schema already correct |
| `services/database.py` | **No change** | `DatabaseManager` already correct |
| `migrations/` | **No change** | Alembic already ready |

---

## Verification Plan

### Automated
- Run existing tests: `pytest tests/` — should pass unchanged (file backend is preserved).

### Manual — File Backend
```bash
python scripts/run_optimization.py --storage file --generations 1 --mutations 2
# Verify: experiments/artifacts/<name>/*.jsonl files appear as before
```

### Manual — Postgres Backend
```bash
# 1. Start postgres (via devcontainer — it starts automatically with docker compose)
# 2. Apply schema
python scripts/run_optimization.py --init-db
# 3. Run with postgres backend
python scripts/run_optimization.py --storage postgres --generations 1 --mutations 2
# 4. Inspect via CLI
python scripts/inspect_experiments.py list
python scripts/inspect_experiments.py show <run_id>
```

### Manual — Migration
```bash
python scripts/migrate_json_to_postgres.py
python scripts/inspect_experiments.py list   # all 40 historical runs should appear
```
