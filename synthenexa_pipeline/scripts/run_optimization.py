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
import time
import uuid
from datetime import datetime, timezone
from copy import copy
from typing import cast

import requests
from core.config import DATASET_PATH, MODEL_NAME, NUM_MUTATIONS, STORAGE_BACKEND, RLConfig
from core.context import PipelineContext
from core.schema import PromptCandidate
from pipeline.prompt_evolution.policies.registry import PolicyRegistry
from pipeline.prompt_evolution.stages.ensemble_execution import EnsembleExecutionStage
from pipeline.prompt_evolution.stages.ensemble_scoring import AggregationStrategy, EnsembleScoringStage
from pipeline.prompt_evolution.stages.evaluation import EvaluationStage
from pipeline.prompt_evolution.stages.execution import ExecutionStage
from pipeline.prompt_evolution.stages.mutation import MutationStage
from pipeline.prompt_evolution.stages.scoring import ScoringStage
from pipeline.prompt_evolution.stages.selection import SelectionStage
from pipeline.prompt_evolution.workflow import PromptEvolutionWorkflow
from services.dataset_service import DatasetService
from services.eval_service import EvalService
from services.llm.registry import ModelRegistry, _infer_provider, normalize_model_name
from services.llm_service import LLMService
from services.llm_cache import LLMResponseCache
from services.storage import StorageService
from services.tracker import RLTracker

# ── Ensure synthenexa_pipeline/ is on sys.path so bare imports work ─────────
_HERE = os.path.dirname(os.path.abspath(__file__))  # scripts/
_PIPELINE = os.path.dirname(_HERE)  # synthenexa_pipeline/
if _PIPELINE not in sys.path:
    sys.path.insert(0, _PIPELINE)
# ─────────────────────────────────────────────────────────────────────────────

MAX_GENERATIONS = 3
MAX_MUTATIONS = 5

# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #


def init_database() -> bool:
    """Create all SQLite tables defined in db_models.py (idempotent)."""
    try:
        from services.database import get_database_manager

        db = get_database_manager()
        db.create_all_tables()
        print("[INFO] SQLite schema initialised successfully.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to initialise database: {e}")
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
                    is_base=True,
                )
    # Fallback default
    return PromptCandidate(
        id="prompt_0",
        text="Write a Python function to solve the given task.",
        is_base=True,
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


def _available_ollama_models(base_url: str | None = None) -> list[str]:
    """Return the list of installed Ollama models discovered from the local API."""
    candidate_urls: list[str] = []
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


def _load_optimization_tasks(args) -> tuple[list[dict], list[dict], dict]:
    """Load selection tasks and a disjoint held-out test set."""
    dataset_path = args.dataset
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(_PIPELINE, dataset_path)
    tasks = DatasetService.load_dataset(dataset_path)
    if not tasks:
        raise RuntimeError(f"No tasks loaded from '{dataset_path}'.")

    if getattr(args, "no_partition", False):
        return (
            tasks,
            [],
            {
                "partitioned": False,
                "dataset_hash": DatasetService.dataset_hash(tasks),
                "task_ids": {"all": [task["task_id"] for task in tasks]},
            },
        )

    partitioned = DatasetService.partition_dataset(
        tasks,
        seed=getattr(args, "seed", None) or 0,
        train_ratio=getattr(args, "train_ratio", 0.6),
        validation_ratio=getattr(args, "validation_ratio", 0.2),
        test_ratio=getattr(args, "test_ratio", 0.2),
    )
    selection_tasks = partitioned["train"] + partitioned["validation"]
    return (
        selection_tasks,
        partitioned["test"],
        {
            "partitioned": True,
            "dataset_hash": partitioned["dataset_hash"],
            "seed": partitioned["seed"],
            "task_ids": partitioned["task_ids"],
            "ratios": {
                "train": getattr(args, "train_ratio", 0.6),
                "validation": getattr(args, "validation_ratio", 0.2),
                "test": getattr(args, "test_ratio", 0.2),
            },
        },
    )


def _interactive_model_selection(args) -> list[str]:
    from utils.ollama_client import get_available_ollama_models

    print("\n*** Dynamic Ollama Model Selection ***")
    print("Fetching available local Ollama models...\n")
    models = get_available_ollama_models()

    if not models:
        print("[ERROR] No models found or Ollama is unreachable. Falling back to default.")
        return [MODEL_NAME]

    print("Available Ollama models:\n")
    for idx, m in enumerate(models, 1):
        print(f"{idx}. {m}")

    print("\nChoose a model number, or enter multiple numbers (e.g. 1,2) to run multiple models sequentially.")
    print("You can also enter '0' or 'all' to run all available models.")

    try:
        user_input = input("Selection: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[ERROR] Selection aborted. Falling back to default.")
        return [MODEL_NAME]

    if not user_input:
        print("[ERROR] Empty selection. Falling back to default.")
        return [MODEL_NAME]

    if user_input.lower() in ("0", "all"):
        return models

    selected_models = []
    for part in user_input.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part)
            if 1 <= idx <= len(models):
                selected_models.append(models[idx - 1])
            else:
                print(f"[warn] Ignored invalid index: {idx}")
        else:
            print(f"[warn] Ignored invalid input: {part}")

    if not selected_models:
        print("[ERROR] No valid selection made. Falling back to default.")
        return [MODEL_NAME]

    return selected_models


def _resolve_model_list(args) -> list[str]:
    """Resolve a priority-sorted list of models from CLI flags.

    Priority order: OpenAI (1) → Gemini (2) → Nemotron (3) → Ollama (4).

    When --available-models is used:
      - Cloud API models (OpenAI, Gemini) are included if their API keys are set
        and --no-openai / --no-gemini flags are absent
      - Locally available Ollama models are appended after cloud models
      - The full list is then priority-sorted
    """
    excluded: set[str] = {
        item.strip() for item in (getattr(args, "exclude_models", "") or "").split(",") if item.strip()
    }
    registry = ModelRegistry()

    if getattr(args, "interactive", False):
        return _interactive_model_selection(args)

    if getattr(args, "available_models", False):
        candidates: list[str] = []

        # Tier 1 — OpenAI (include if API key present and not suppressed)
        if not getattr(args, "no_openai", False) and os.getenv("OPENAI_API_KEY"):
            openai_models = [
                m
                for m in os.getenv(
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
                m
                for m in os.getenv(
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


def _validate_ollama_experiment(args, model_names: list[str]) -> None:
    """Enforce the small-budget, Ollama-only prototype execution policy."""
    if not 1 <= args.generations <= MAX_GENERATIONS:
        raise ValueError(f"--generations must be between 1 and {MAX_GENERATIONS}.")
    if not 1 <= args.mutations <= MAX_MUTATIONS:
        raise ValueError(f"--mutations must be between 1 and {MAX_MUTATIONS}.")

    non_ollama = [
        model for model in model_names
        if _infer_provider(normalize_model_name(model)) != "ollama"
    ]
    if non_ollama:
        raise ValueError(
            "This prototype is configured for Ollama-only experiments; "
            f"unsupported model(s): {', '.join(non_ollama)}"
        )


def _estimate_llm_calls(model_count: int, generations: int, mutations: int, selection_tasks: int, test_tasks: int) -> int:
    """Estimate worst-case Ollama calls before an expensive run starts."""
    baseline = model_count * selection_tasks
    mutation_calls = generations * model_count * mutations
    candidates_per_generation = model_count * mutations
    execution_calls = generations * model_count * candidates_per_generation * selection_tasks
    final_test = 2 * model_count * test_tasks
    return baseline + mutation_calls + execution_calls + final_test


def _historical_avg_call_seconds() -> float:
    """Infer an average Ollama latency from completed local artifact manifests."""
    default_seconds = 8.0
    artifact_root = os.path.join(_PIPELINE, "experiments", "artifacts")
    samples = []
    if not os.path.isdir(artifact_root):
        return default_seconds
    for experiment_name in os.listdir(artifact_root):
        manifest_path = os.path.join(artifact_root, experiment_name, "manifest.json")
        if not os.path.isfile(manifest_path):
            continue
        try:
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            elapsed = float(manifest.get("elapsed_seconds", 0.0))
            stats = manifest.get("call_stats", {})
            calls = sum(float(stats.get(key, 0)) for key in ("mutation_calls", "generation_calls"))
            if elapsed > 0 and calls > 0:
                samples.append(elapsed / calls)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    # Ignore implausible values caused by interrupted runs or tiny smoke tests.
    valid = [sample for sample in samples if 0.5 <= sample <= 300.0]
    return sum(valid) / len(valid) if valid else default_seconds


def _check_call_budget(args, model_count: int, selection_tasks: int, test_tasks: int) -> int:
    estimate = _estimate_llm_calls(model_count, args.generations, args.mutations, selection_tasks, test_tasks)
    limit = getattr(args, "max_llm_calls", 1200)
    print(f"Estimated worst-case Ollama calls: {estimate} (limit={limit})")
    avg_seconds = _historical_avg_call_seconds()
    estimated_minutes = estimate * avg_seconds / 60.0
    print(
        f"Estimated time to complete the entire workflow: ~{estimated_minutes:.1f} minutes "
        f"({avg_seconds:.1f}s average Ollama call inferred automatically; local evaluation overhead excluded)."
    )
    if estimate > limit:
        raise ValueError(f"Estimated Ollama calls ({estimate}) exceed --max-llm-calls ({limit}).")
    return estimate


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


def _save_comparison_summary(name: str, results: list[dict], base_dir: str | None = None) -> str:
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
    started_monotonic = time.monotonic()
    print(f"\n{'=' * 60}")
    print("  Synthenexa — Prompt Evolution Pipeline")
    print(f"{'=' * 60}")
    eval_backend = getattr(args, "eval_backend", "local")
    print(f"  Run ID:       {run_id}")
    print(f"  Model:        {model_name}")
    print(f"  Generations:  {args.generations}")
    print(f"  Mutations:    {args.mutations}")
    print(f"  Policy:       {args.policy}" + (f" (ε={args.epsilon})" if args.policy == "epsilon_greedy" else ""))
    print(f"  Dataset:      {args.dataset}")
    print(f"  Eval backend: {eval_backend}")
    print(f"{'=' * 60}\n")

    tasks, test_tasks, partition_metadata = _load_optimization_tasks(args)
    print(f"Loaded {len(tasks)} selection tasks and {len(test_tasks)} held-out test tasks.\n")
    _check_call_budget(args, 1, len(tasks), len(test_tasks))

    base_prompt_path = os.path.join(_PIPELINE, "Dataset", "base_prompt.jsonl")
    base_prompt = _load_base_prompt(base_prompt_path)
    print(f"Base prompt: {base_prompt.text!r}\n")

    llm_service = LLMService(model=model_name, base_url=args.ollama_base_url)
    cache = LLMResponseCache(os.path.join(_PIPELINE, "experiments", "cache"))

    eval_service = EvalService()
    print("[EvalBackend] Using local exec() evaluator")
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
        config_dict={
            "generations": args.generations,
            "mutations": args.mutations,
            "policy": args.policy,
            "eval_backend": getattr(args, "eval_backend", "local"),
            "dataset_split": getattr(args, "dataset_split", "all"),
            "seed": getattr(args, "seed", None),
            "partition_metadata": partition_metadata,
            "base_prompt": {"id": base_prompt.id, "text": base_prompt.text},
        },
    )
    tracker = RLTracker()

    rl_config = RLConfig(policy_type=args.policy, epsilon=args.epsilon)
    policy = PolicyRegistry.get_policy(rl_config)

    # Mutation uses a higher temperature inside MutationStage so candidates
    # are diverse; code generation stays deterministic at temperature=0.0.
    # The stages form the per-generation DAG:
    # mutation → execution → evaluation → scoring → selection.
    stages = [
        MutationStage(llm_service=llm_service, num_mutations=args.mutations, cache=cache),
        ExecutionStage(llm_service=llm_service, cache=cache),
        EvaluationStage(eval_service=eval_service),
        ScoringStage(),
        SelectionStage(policy=policy),
    ]

    context = PipelineContext(
        model_name=model_name,
        run_id=run_id,
        base_prompt=base_prompt,
        tasks=tasks,
        test_tasks=test_tasks,
        dataset_split="train_validation" if test_tasks else "all",
        seed=getattr(args, "seed", None),
        partition_metadata=partition_metadata,
    )

    workflow = PromptEvolutionWorkflow(
        context=context,
        stages=stages,
        policy=policy,
        storage=storage,
        tracker=tracker,
    )

    final_context = workflow.run(generations=args.generations)

    storage.update_run_metadata({
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started_monotonic, 2),
        "call_stats": final_context.call_stats,
        "selected_prompt_id": final_context.best_prompt.id if final_context.best_prompt else None,
        "selected_prompt_hash": final_context.best_prompt.prompt_hash if final_context.best_prompt else None,
    })

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
                    "origin_model": final_context.best_prompt.origin_model,
                    "generation": final_context.best_prompt.generation,
                    "mutation_idx": final_context.best_prompt.mutation_idx,
                    "parent_id": final_context.best_prompt.parent_id,
                    "mutation_role": final_context.best_prompt.mutation_role,
                    "prompt_hash": final_context.best_prompt.prompt_hash,
                    "call_stats": final_context.call_stats,
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


def _run_ensemble_experiment(args, model_names: list[str], run_id: str, experiment_name: str) -> dict:
    """Run a cross-model ensemble optimization cycle.

    Each model runs its own Mutation + Execution + Evaluation sub-pipeline
    independently. EnsembleScoringStage then aggregates the per-model scores
    into a single combined reward used to select the next-generation prompt.
    """
    strategy_name = str(getattr(args, "ensemble_strategy", "mean"))
    valid_strategies: set[AggregationStrategy] = {"mean", "min", "median", "max", "vote", "robust"}
    if strategy_name not in valid_strategies:
        raise ValueError(f"Unknown ensemble aggregation strategy: {strategy_name!r}")
    strategy = cast(AggregationStrategy, strategy_name)
    started_monotonic = time.monotonic()
    print(f"\n{'=' * 60}")
    print("  Synthenexa — Ensemble Prompt Evolution Pipeline")
    print(f"{'=' * 60}")
    print(f"  Run ID:       {run_id}")
    print(f"  Models:       {', '.join(model_names)}")
    print(f"  Generations:  {args.generations}")
    print(f"  Mutations:    {args.mutations} per model")
    print(f"  Policy:       {args.policy}" + (f" (ε={args.epsilon})" if args.policy == "epsilon_greedy" else ""))
    print(f"  Dataset:      {args.dataset}")
    print(f"  Strategy:     {strategy}")
    print(f"{'=' * 60}\n")

    tasks, test_tasks, partition_metadata = _load_optimization_tasks(args)
    print(f"Loaded {len(tasks)} selection tasks and {len(test_tasks)} held-out test tasks.\n")
    _check_call_budget(args, len(model_names), len(tasks), len(test_tasks))

    base_prompt_path = os.path.join(_PIPELINE, "Dataset", "base_prompt.jsonl")
    base_prompt = _load_base_prompt(base_prompt_path)
    print(f"Base prompt: {base_prompt.text!r}\n")

    # Build the evaluation backend (shared across all models)
    eval_service = EvalService()
    print("[EvalBackend] Using local exec() evaluator")

    # Build per-model sub-pipelines
    model_pipelines = []
    cache = LLMResponseCache(os.path.join(_PIPELINE, "experiments", "cache"))
    for model_name in model_names:
        llm = LLMService(model=model_name, base_url=args.ollama_base_url)
        model_pipelines.append(
            (
                model_name,
                MutationStage(llm_service=llm, num_mutations=args.mutations, cache=cache),
                ExecutionStage(llm_service=llm, cache=cache),
                EvaluationStage(eval_service=eval_service),
            )
        )

    # Storage + tracker
    storage_backend = getattr(args, "storage", None) or STORAGE_BACKEND
    storage = StorageService(
        base_dir=os.path.join(_PIPELINE, "experiments", "artifacts", experiment_name),
        backend=storage_backend,
    )
    storage.create_experiment_record(
        run_id=run_id,
        model_name="ensemble:" + ",".join(model_names),
        experiment_name=experiment_name,
        config_dict={
            "generations": args.generations,
            "mutations": args.mutations,
            "policy": args.policy,
            "eval_backend": getattr(args, "eval_backend", "local"),
            "dataset_split": getattr(args, "dataset_split", "all"),
            "seed": getattr(args, "seed", None),
            "partition_metadata": partition_metadata,
            "base_prompt": {"id": base_prompt.id, "text": base_prompt.text},
            "ensemble_models": model_names,
            "ensemble_strategy": strategy,
        },
    )
    tracker = RLTracker()

    rl_config = RLConfig(policy_type=args.policy, epsilon=args.epsilon)
    policy = PolicyRegistry.get_policy(rl_config)

    # Ensemble DAG: mutate and cross-evaluate candidates on every Ollama model,
    # aggregate their scores, then select the next parent prompt.
    stages = [
        EnsembleExecutionStage(model_pipelines=model_pipelines),
        EnsembleScoringStage(strategy=strategy),
        SelectionStage(policy=policy),
    ]

    context = PipelineContext(
        model_name="ensemble",
        run_id=run_id,
        base_prompt=base_prompt,
        tasks=tasks,
        test_tasks=test_tasks,
        dataset_split="train_validation" if test_tasks else "all",
        seed=getattr(args, "seed", None),
        partition_metadata=partition_metadata,
    )

    workflow = PromptEvolutionWorkflow(
        context=context,
        stages=stages,
        policy=policy,
        storage=storage,
        tracker=tracker,
    )

    final_context = workflow.run(generations=args.generations)

    storage.update_run_metadata({
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started_monotonic, 2),
        "call_stats": final_context.call_stats,
        "selected_prompt_id": final_context.best_prompt.id if final_context.best_prompt else None,
        "selected_prompt_hash": final_context.best_prompt.prompt_hash if final_context.best_prompt else None,
    })

    storage.finalize_experiment(
        best_prompt=final_context.best_prompt,
        best_score=final_context.best_score,
    )

    print("\n" + "=" * 60)
    print("  Optimization Complete!")
    print("=" * 60)

    print("\n[SelectionStage] Final winner selected.")
    if final_context.best_prompt:
        print(f"\nBest Universal Prompt (score={final_context.best_score:.4f})")
        print("\nOrigin:")
        print(f"  Model: {final_context.best_prompt.origin_model or 'base'}")
        print(f"  Generation: {final_context.best_prompt.generation}")
        print(f"  Mutation: {final_context.best_prompt.mutation_idx}")

        print("\nEvaluated Across:")
        for m, s in sorted(final_context.best_ensemble_scores.items()):
            print(f"  ✓ {m:<22} {s:.4f}")

        print("\n" + "-" * 60)
        print(final_context.best_prompt.text)
        print("-" * 60)

        print("\n============================================================")
        print(" Model Performance Summary")
        print("============================================================")
        print(f"{'Model':<30} Best Score")
        print("-" * 60)
        for m, s in sorted(final_context.model_leaderboard.items(), key=lambda x: x[1], reverse=True):
            print(f"{m:<30} {s:.4f}")
        print("============================================================\n")

        print("Best Prompt Contribution:\n")
        print("Prompt generated by:")
        print(f"{final_context.best_prompt.origin_model or 'base'}\n")
        print("Performance:\n")

        for m, s in sorted(final_context.best_ensemble_scores.items()):
            print(f"{m}:")
            filled = int(s * 10)
            bar = "█" * filled + "░" * (10 - filled)
            print(f"{bar} {s * 100:.0f}%\n")

        print("Mean Reward:")
        print(f"{final_context.best_score:.4f}\n")

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
                    "ensemble_models": model_names,
                    "ensemble_strategy": strategy,
                    "prompt_id": final_context.best_prompt.id,
                    "text": final_context.best_prompt.text,
                    "score": final_context.best_score,
                    "reward_history": final_context.reward_history,
                    "origin_model": final_context.best_prompt.origin_model,
                    "generation": final_context.best_prompt.generation,
                    "mutation_idx": final_context.best_prompt.mutation_idx,
                    "parent_id": final_context.best_prompt.parent_id,
                    "mutation_role": final_context.best_prompt.mutation_role,
                    "prompt_hash": final_context.best_prompt.prompt_hash,
                    "call_stats": final_context.call_stats,
                },
                handle,
                indent=2,
            )
        print(f"\n  Best prompt saved → {best_path}")

    return {
        "model": "ensemble:" + ",".join(model_names),
        "experiment": experiment_name,
        "best_score": final_context.best_score,
        "best_prompt": final_context.best_prompt.text if final_context.best_prompt else "",
        "best_prompt_id": final_context.best_prompt.id if final_context.best_prompt else "",
        "reward_history": final_context.reward_history,
    }


def _run_repeated_experiments(args, model_names: list[str]) -> list[dict]:
    """Run independent repetitions with unique artifact names and stable seeds."""
    base_name = args.experiment or "repeated_experiment"
    base_seed = args.seed if args.seed is not None else 0
    all_results: list[dict] = []

    for repetition in range(1, args.repetitions + 1):
        repetition_args = copy(args)
        repetition_args.seed = base_seed + repetition - 1
        repetition_name = f"{base_name}_rep{repetition}"
        print(f"\n### Repetition {repetition}/{args.repetitions} (seed={repetition_args.seed}) ###")

        if args.ensemble:
            if len(model_names) < 2:
                raise ValueError("--ensemble repetitions require at least two models.")
            all_results.append(
                _run_ensemble_experiment(
                    repetition_args,
                    model_names,
                    str(uuid.uuid4())[:8],
                    repetition_name,
                )
            )
            continue

        for model_name in model_names:
            model_experiment_name = (
                f"{repetition_name}_{_safe_name(model_name)}" if len(model_names) > 1 else repetition_name
            )
            all_results.append(
                _run_single_experiment(
                    repetition_args,
                    model_name,
                    str(uuid.uuid4())[:8],
                    model_experiment_name,
                )
            )

    summary_dir = os.path.join(_PIPELINE, "experiments", "artifacts", base_name)
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "repetitions_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2, default=str)
    print(f"\nRepetition summary written to {summary_path}")
    return all_results


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #


def main():
    parser = argparse.ArgumentParser(description="Synthenexa — RL-based prompt optimization pipeline.")
    parser.add_argument(
        "--generations", type=int, default=3, help="Number of evolution generations to run (1–3; default: 3)."
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of independent runs with incremented seeds; useful for checking consistency (default: 1).",
    )
    parser.add_argument(
        "--mutations",
        type=int,
        default=NUM_MUTATIONS,
        help=(
            f"Number of candidate prompt mutations created from the current parent per generation "
            f"(1–5; default: {NUM_MUTATIONS})."
        ),
    )
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=1200,
        help="Maximum estimated Ollama calls allowed before starting (default: 1200).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=(
            f"Ollama model to use (default: {MODEL_NAME}). Only Ollama models are permitted."
        ),
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated Ollama models; ensemble mode cross-evaluates every candidate on each model.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactively select one or more Ollama models before running.",
    )
    parser.add_argument(
        "--available-models",
        action="store_true",
        help=(
            "Run the workflow against available Ollama models only."
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
        help=(
            "How the next parent prompt is selected. 'greedy' always chooses the highest-scoring candidate; "
            "'epsilon_greedy' sometimes explores a random candidate (default: greedy)."
        ),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help=(
            "Exploration probability for epsilon-greedy selection: 0.0 always exploits and 1.0 always explores "
            "(default: 0.1)."
        ),
    )
    parser.add_argument(
        "--dataset", type=str, default=DATASET_PATH, help=f"Path to MBPP micro dataset (default: {DATASET_PATH})."
    )
    parser.add_argument(
        "--experiment", type=str, default="", help="Experiment name (used for artifact directory naming)."
    )
    parser.add_argument(
        "--eval-backend",
        type=str,
        default="local",
        choices=["local"],
        help="Evaluation backend for the current workflow. The ensemble pipeline currently uses the local exec() evaluator.",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="all",
        choices=["all", "train", "validation", "test"],
        help="Dataset partition label saved with experiment results (default: all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed used to create deterministic dataset partitions and recorded for reproducibility.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="Fraction of tasks assigned to training (default: 0.6).",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
        help="Fraction of tasks assigned to validation (default: 0.2).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of tasks reserved for final testing (default: 0.2).",
    )
    parser.add_argument(
        "--no-partition",
        action="store_true",
        help="Disable train/validation/test partitioning for legacy smoke runs.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        choices=["file", "local", "sqlite"],
        help=(
            "Storage backend to use: "
            "'file' / 'local' (default) writes JSON/JSONL artifacts to disk; "
            "'sqlite' writes structured data to experiments/synthenexa.db for "
            "easy querying with Pandas and SQL. "
            "Overrides the STORAGE_BACKEND env var."
        ),
    )

    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Create the SQLite schema (experiments/synthenexa.db). Safe to run multiple times.",
    )
    parser.add_argument(
        "--export-artifacts",
        action="store_true",
        help="Export JSON artifact reports (e.g., best_prompt.json, summary.json). Default is False unless using 'file' storage.",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help=(
            "Run cross-model ensemble optimization. Each selected model independently "
            "mutates and evaluates prompts; scores are combined to select the prompt that "
            "generalizes best across all models. Requires --models or --interactive with "
            "2+ models."
        ),
    )
    parser.add_argument(
        "--ensemble-strategy",
        type=str,
        default="mean",
        choices=["mean", "min", "median", "max", "vote", "robust"],
        help=(
            "How to aggregate per-model scores in ensemble mode: "
            "'mean' (default) rewards consistent cross-model performance; "
            "'min' rewards prompts that never fail on any model; "
            "'median' rewards the typical model response while ignoring outliers; "
            "'max' rewards prompts that excel on at least one model; "
            "'vote' is ranking-based and counts the fraction of models that picked the prompt as best; "
            "'robust' combines mean performance (70%%) with the weakest model (30%%)."
        ),
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

    try:
        _validate_ollama_experiment(args, model_names)
    except ValueError as exc:
        parser.error(str(exc))

    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")

    if args.repetitions > 1:
        _run_repeated_experiments(args, model_names)
        return

    # ── Ensemble mode: cross-model joint optimization ──────────────────
    if getattr(args, "ensemble", False):
        if len(model_names) < 2:
            print(
                "[ERROR] --ensemble requires at least 2 models. "
                "Use --models 'a,b' or --interactive to select multiple models."
            )
            sys.exit(1)
        experiment_name = args.experiment or f"ensemble_{run_id}"
        result = _run_ensemble_experiment(args, model_names, run_id, experiment_name)
        return

    # ── Standard per-model independent optimization ────────────────────
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
