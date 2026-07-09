"""
scripts/run_optimization.py — CLI entrypoint for the Synthenexa RL pipeline.

Usage (run from the synthenexa_pipeline/ directory):
    python scripts/run_optimization.py
    python scripts/run_optimization.py --generations 5 --mutations 3
    python scripts/run_optimization.py --policy epsilon_greedy --epsilon 0.2
    python scripts/run_optimization.py --model cogito:3b --generations 2

Model priority order (highest capability first):
    Tier 1 — OpenAI  : openai:gpt-4o, openai:gpt-4o-mini, openai:o3-mini
    Tier 2 — Gemini  : gemini:gemini-2.0-flash, gemini:gemini-1.5-pro
    Tier 3 — Nemotron: nemotron-3-super:cloud  (primary Ollama baseline)
    Tier 4 — Ollama  : cogito:3b, qwen2.5:3b, qwen:latest, gemma4:latest

Model name conventions:
    openai:<name>   — Routes to OpenAI Chat Completions API
    gemini:<name>   — Routes to Google Gemini API
    <name>          — Routes to local Ollama instance
"""

import argparse
import json
import os
import re
import sys
import uuid

# ── Ensure synthenexa_pipeline/ is on sys.path so bare imports work ─────────
_HERE = os.path.dirname(os.path.abspath(__file__))  # scripts/
_PIPELINE = os.path.dirname(_HERE)  # synthenexa_pipeline/
if _PIPELINE not in sys.path:
    sys.path.insert(0, _PIPELINE)
# ─────────────────────────────────────────────────────────────────────────────

import requests
from alembic.config import Config
from alembic import command
from core.config import DATASET_PATH, MODEL_NAME, NUM_MUTATIONS, RLConfig, STORAGE_BACKEND
from core.context import PipelineContext
from core.schema import PromptCandidate
from pipeline.prompt_evolution.policies.registry import PolicyRegistry
from pipeline.prompt_evolution.stages.evaluation import EvaluationStage
from pipeline.prompt_evolution.stages.execution import ExecutionStage
from pipeline.prompt_evolution.stages.mutation import MutationStage
from pipeline.prompt_evolution.stages.scoring import ScoringStage
from pipeline.prompt_evolution.stages.selection import SelectionStage
from pipeline.prompt_evolution.workflow import PromptEvolutionWorkflow
from services.dataset_service import DatasetService
from services.eval_service import EvalService
from services.llm.registry import ModelRegistry, _infer_priority, normalize_model_name
from services.llm_service import LLMService
from services.storage import StorageService
from services.tracker import RLTracker

# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #


def init_database():
    """Run Alembic migrations to initialize database schema."""
    migrations_dir = os.path.join(_PIPELINE, "migrations")
    alembic_ini = os.path.join(migrations_dir, "alembic.ini")
    
    if not os.path.exists(alembic_ini):
        print(f"[ERROR] Alembic configuration not found at {alembic_ini}")
        return False
    
    try:
        alembic_cfg = Config(alembic_ini)
        # Set the script location relative to the migrations directory
        alembic_cfg.set_main_option("script_location", migrations_dir)
        
        print("[INFO] Running database migrations...")
        command.upgrade(alembic_cfg, "head")
        print("[INFO] Database migrations completed successfully.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to run database migrations: {e}")
        return False


def _load_base_prompt(path: str) -> PromptCandidate:
    """Load the base prompt from a JSONL file (first record wins)."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                return PromptCandidate(
                    id="prompt_0",
                    text=data.get("prompt", "Write a Python function to solve the given task."),
                )
    # Fallback default
    return PromptCandidate(
        id="prompt_0",
        text="Write a Python function to solve the given task.",
    )


def _print_reward_table(reward_history: dict) -> None:
    """Print a formatted reward-per-generation table."""
    print("\n" + "=" * 50)
    print("  Reward History")
    print("=" * 50)
    print(f"  {'Gen':>5}  {'Best Score':>12}")
    print(f"  {'-' * 5}  {'-' * 12}")
    for gen, score in sorted(reward_history.items()):
        bar = "█" * int(score * 20)
        print(f"  {gen:>5}  {score:>12.4f}  {bar}")
    print("=" * 50)


def _available_ollama_models(base_url: str = None) -> list[str]:
    """Return the list of installed Ollama models discovered from the local API."""
    candidate_urls = []
    if base_url:
        candidate_urls.append(base_url)
    for env_name in ("OLLAMA_BASE_URL", "OLLAMA_API_BASE"):
        value = os.getenv(env_name)
        if value and value not in candidate_urls:
            candidate_urls.append(value)
    for fallback in ("http://localhost:11434", "http://127.0.0.1:11434", "http://host.docker.internal:11434"):
        if fallback not in candidate_urls:
            candidate_urls.append(fallback)

    for url in candidate_urls:
        try:
            response = requests.get(f"{url.rstrip('/')}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            names = []
            for model in models:
                name = (model.get("name") or model.get("model") or "").strip()
                if name:
                    names.append(name)
            if names:
                return names
        except Exception as exc:
            print(f"[run_optimization] Could not query Ollama at {url}: {exc}")
    return []


def _priority_sorted_models(
    models: list[str],
    excluded: set[str],
    registry: ModelRegistry,
) -> list[str]:
    """Sort a model list by priority tier (1=highest) then by original order within each tier.

    Priority tiers:
        1 — OpenAI models     (openai:*)
        2 — Gemini models     (gemini:*)
        3 — Nemotron cloud    (nemotron-3-super:cloud)
        4 — Other Ollama      (all remaining local models)
    """
    seen: set[str] = set()
    result: list[str] = []
    for name in models:
        norm = normalize_model_name(name)
        if norm in excluded or norm in seen:
            continue
        seen.add(norm)
        result.append(norm)
    # Capture original insertion order before sorting for stable secondary key
    original_order = {name: idx for idx, name in enumerate(result)}
    result.sort(key=lambda m: (registry.get(m).priority, original_order[m]))
    return result



def _safe_name(model_name: str) -> str:
    """Create a filesystem-safe experiment name from a model identifier."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("._-")


def _resolve_model_list(args) -> list[str]:
    """Resolve a priority-sorted list of models from CLI flags.

    Priority order: OpenAI (1) → Gemini (2) → Nemotron (3) → Ollama (4).

    When --available-models is used:
      - Cloud API models (OpenAI, Gemini) are included if their API keys are set
        and --no-openai / --no-gemini flags are absent
      - Locally available Ollama models are appended after cloud models
      - The full list is then priority-sorted
    """
    excluded: set[str] = {item.strip() for item in (getattr(args, "exclude_models", "") or "").split(",") if item.strip()}
    registry = ModelRegistry()

    if getattr(args, "available_models", False):
        candidates: list[str] = []

        # Tier 1 — OpenAI (include if API key present and not suppressed)
        if not getattr(args, "no_openai", False) and os.getenv("OPENAI_API_KEY"):
            openai_models = [
                m for m in os.getenv(
                    "OPENAI_MODEL_LIST",
                    "openai:gpt-4o,openai:gpt-4o-mini",
                ).split(",")
                if m.strip()
            ]
            candidates.extend(openai_models)
            print(f"[run_optimization] OpenAI key found — adding {len(openai_models)} OpenAI model(s).")
        else:
            print("[run_optimization] OpenAI skipped (no OPENAI_API_KEY or --no-openai set).")

        # Tier 2 — Gemini (include if API key present and not suppressed)
        if not getattr(args, "no_gemini", False) and os.getenv("GEMINI_API_KEY"):
            gemini_models = [
                m for m in os.getenv(
                    "GEMINI_MODEL_LIST",
                    "gemini:gemini-2.0-flash,gemini:gemini-1.5-pro",
                ).split(",")
                if m.strip()
            ]
            candidates.extend(gemini_models)
            print(f"[run_optimization] Gemini key found — adding {len(gemini_models)} Gemini model(s).")
        else:
            print("[run_optimization] Gemini skipped (no GEMINI_API_KEY or --no-gemini set).")

        # Tiers 3 & 4 — Ollama local models
        ollama_models = _available_ollama_models(getattr(args, "ollama_base_url", None))
        if ollama_models:
            print(f"[run_optimization] Ollama available — adding {len(ollama_models)} local model(s).")
        else:
            print("[run_optimization] Ollama unavailable or no models found.")
        candidates.extend(ollama_models)

        sorted_models = _priority_sorted_models(candidates, excluded, registry)
        return sorted_models or [getattr(args, "model", MODEL_NAME)]

    if getattr(args, "models", "").strip():
        candidates = [m.strip() for m in getattr(args, "models", "").split(",") if m.strip()]
        return _priority_sorted_models(candidates, excluded, registry)

    return [normalize_model_name(getattr(args, "model", MODEL_NAME))]


def _print_comparison_table(results: list[dict]) -> None:
    """Print a simple comparison table for multi-model runs."""
    print("\n" + "=" * 80)
    print("  Model Comparison")
    print("=" * 80)
    print(f"  {'Model':<20} {'Best Score':>12} {'Experiment':<30}")
    print(f"  {'-' * 20} {'-' * 12} {'-' * 30}")
    for entry in sorted(results, key=lambda item: item.get("best_score", 0.0), reverse=True):
        print(
            f"  {entry.get('model', 'unknown'):<20} "
            f"{entry.get('best_score', 0.0):>12.4f} {entry.get('experiment', 'unknown'):<30}"
        )
    print("=" * 80)


def _save_comparison_summary(name: str, results: list[dict], base_dir: str = None) -> str:
    """Persist a JSON summary of multi-model comparison runs."""
    base_dir = base_dir or os.path.join(_PIPELINE, "experiments", "artifacts")
    target_dir = os.path.join(base_dir, name)
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            sorted(results, key=lambda item: item.get("best_score", 0.0), reverse=True),
            handle,
            indent=2,
        )
    return path


def _run_single_experiment(args, model_name: str, run_id: str, experiment_name: str) -> dict:
    """Run the prompt-evolution workflow for one model and return a summary record."""
    print(f"\n{'=' * 60}")
    print("  Synthenexa — Prompt Evolution Pipeline")
    print(f"{'=' * 60}")
    print(f"  Run ID:       {run_id}")
    print(f"  Model:        {model_name}")
    print(f"  Generations:  {args.generations}")
    print(f"  Mutations:    {args.mutations}")
    print(f"  Policy:       {args.policy}" + (f" (ε={args.epsilon})" if args.policy == "epsilon_greedy" else ""))
    print(f"  Dataset:      {args.dataset}")
    print(f"{'=' * 60}\n")

    dataset_path = args.dataset
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(_PIPELINE, dataset_path)

    tasks = DatasetService.load_dataset(dataset_path)
    if not tasks:
        raise RuntimeError(f"No tasks loaded from '{dataset_path}'.")
    print(f"Loaded {len(tasks)} tasks from dataset.\n")

    base_prompt_path = os.path.join(_PIPELINE, "Dataset", "base_prompt.jsonl")
    base_prompt = _load_base_prompt(base_prompt_path)
    print(f"Base prompt: {base_prompt.text!r}\n")

    llm_service = LLMService(model=model_name, base_url=args.ollama_base_url)
    eval_service = EvalService()
    # Resolve storage backend: CLI flag → env var → default "file"
    storage_backend = getattr(args, "storage", None) or STORAGE_BACKEND
    storage = StorageService(
        base_dir=os.path.join(_PIPELINE, "experiments", "artifacts", experiment_name),
        backend=storage_backend,
    )
    # Create the experiment record (no-op for file backend)
    storage.create_experiment_record(
        run_id=run_id,
        model_name=model_name,
        experiment_name=experiment_name,
        config_dict={"generations": args.generations, "mutations": args.mutations, "policy": args.policy},
    )
    tracker = RLTracker()

    rl_config = RLConfig(policy_type=args.policy, epsilon=args.epsilon)
    policy = PolicyRegistry.get_policy(rl_config)

    stages = [
        MutationStage(llm_service=llm_service, num_mutations=args.mutations),
        ExecutionStage(llm_service=llm_service),
        EvaluationStage(eval_service=eval_service),
        ScoringStage(),
        SelectionStage(policy=policy),
    ]

    context = PipelineContext(
        model_name=model_name,
        run_id=run_id,
        base_prompt=base_prompt,
        tasks=tasks,
    )

    workflow = PromptEvolutionWorkflow(
        context=context,
        stages=stages,
        policy=policy,
        storage=storage,
        tracker=tracker,
    )

    final_context = workflow.run(generations=args.generations)

    # Finalise the experiment record (no-op for file backend)
    storage.finalize_experiment(
        best_prompt=final_context.best_prompt,
        best_score=final_context.best_score,
    )

    print("\n" + "=" * 60)
    print("  Optimization Complete!")
    print("=" * 60)
    if final_context.best_prompt:
        print(f"\n  Best Prompt (score={final_context.best_score:.4f}):")
        print(f"  {final_context.best_prompt.text}")
    else:
        print("  No best prompt found (all generations may have failed).")

    _print_reward_table(final_context.reward_history)

    should_export = getattr(args, "export_artifacts", False) or storage_backend == "file"

    best_path = os.path.join(_PIPELINE, "experiments", "artifacts", experiment_name, "best_prompt.json")
    if final_context.best_prompt and should_export:
        os.makedirs(os.path.dirname(best_path), exist_ok=True)
        with open(best_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "run_id": run_id,
                    "model": model_name,
                    "prompt_id": final_context.best_prompt.id,
                    "text": final_context.best_prompt.text,
                    "score": final_context.best_score,
                    "reward_history": final_context.reward_history,
                },
                handle,
                indent=2,
            )
        print(f"\n  Best prompt saved → {best_path}")

    return {
        "model": model_name,
        "experiment": experiment_name,
        "best_score": final_context.best_score,
        "best_prompt": final_context.best_prompt.text if final_context.best_prompt else "",
        "best_prompt_id": final_context.best_prompt.id if final_context.best_prompt else "",
        "reward_history": final_context.reward_history,
    }


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #


def main():
    parser = argparse.ArgumentParser(description="Synthenexa — RL-based prompt optimization pipeline.")
    parser.add_argument(
        "--generations", type=int, default=3, help="Number of evolution generations to run (default: 3)."
    )
    parser.add_argument(
        "--mutations",
        type=int,
        default=NUM_MUTATIONS,
        help=f"Number of prompt mutations per generation (default: {NUM_MUTATIONS}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=(
            f"Model to use (default: {MODEL_NAME}). "
            "Use 'openai:<name>' for OpenAI, 'gemini:<name>' for Gemini, "
            "or a bare name for Ollama."
        ),
    )
    parser.add_argument(
        "--models", type=str, default="", help="Comma-separated list of models to evaluate sequentially."
    )
    parser.add_argument(
        "--available-models",
        action="store_true",
        help=(
            "Run the workflow against all available models in priority order: "
            "OpenAI (if OPENAI_API_KEY set) → Gemini (if GEMINI_API_KEY set) → Ollama local models."
        ),
    )
    parser.add_argument("--exclude-models", type=str, default="", help="Comma-separated list of models to skip.")
    parser.add_argument(
        "--no-openai",
        action="store_true",
        help="Exclude OpenAI models even when OPENAI_API_KEY is set (useful with --available-models).",
    )
    parser.add_argument(
        "--no-gemini",
        action="store_true",
        help="Exclude Gemini models even when GEMINI_API_KEY is set (useful with --available-models).",
    )
    parser.add_argument(
        "--ollama-base-url",
        type=str,
        default=os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_API_BASE") or "http://localhost:11434",
        help="Ollama base URL (default: env or localhost).",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="greedy",
        choices=["greedy", "epsilon_greedy"],
        help="Selection policy (default: greedy).",
    )
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for epsilon-greedy policy (default: 0.1).")
    parser.add_argument(
        "--dataset", type=str, default=DATASET_PATH, help=f"Path to MBPP micro dataset (default: {DATASET_PATH})."
    )
    parser.add_argument(
        "--experiment", type=str, default="", help="Experiment name (used for artifact directory naming)."
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        choices=["file", "postgres"],
        help=(
            "Storage backend to use: 'file' (default) writes JSON/JSONL artifacts; "
            "'postgres' writes to the configured PostgreSQL database. "
            "Overrides the STORAGE_BACKEND env var."
        ),
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize database schema by running Alembic migrations."
    )
    parser.add_argument(
        "--export-artifacts",
        action="store_true",
        help="Export JSON artifact reports (e.g., best_prompt.json, summary.json). Default is False unless using 'file' storage."
    )
    args = parser.parse_args()

    # Handle database initialization
    if args.init_db:
        if init_database():
            print("[INFO] Database initialization completed.")
            sys.exit(0)
        else:
            print("[ERROR] Database initialization failed.")
            sys.exit(1)

    run_id = str(uuid.uuid4())[:8]
    model_names = _resolve_model_list(args)
    if not model_names:
        model_names = [args.model]

    comparison_results = []
    for index, model_name in enumerate(model_names):
        experiment_name = args.experiment or f"run_{run_id}"
        if len(model_names) > 1:
            experiment_name = f"{experiment_name}_{_safe_name(model_name)}"
        if index > 0:
            run_id = str(uuid.uuid4())[:8]
        result = _run_single_experiment(args, model_name, run_id, experiment_name)
        comparison_results.append(result)

    if len(comparison_results) > 1:
        _print_comparison_table(comparison_results)
        storage_backend = getattr(args, "storage", None) or STORAGE_BACKEND
        should_export = getattr(args, "export_artifacts", False) or storage_backend == "file"
        if should_export:
            summary_path = _save_comparison_summary(args.experiment or f"run_{run_id}", comparison_results)
            print(f"\nComparison summary written to {summary_path}")


if __name__ == "__main__":
    main()
