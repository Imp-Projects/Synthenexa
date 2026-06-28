"""
scripts/run_optimization.py — CLI entrypoint for the Synthenexa RL pipeline.

Usage (run from the synthenexa_pipeline/ directory):
    python scripts/run_optimization.py
    python scripts/run_optimization.py --generations 5 --mutations 3
    python scripts/run_optimization.py --policy epsilon_greedy --epsilon 0.2
    python scripts/run_optimization.py --model cogito:3b --generations 2
"""
import argparse
import json
import os
import sys
import uuid

# ── Ensure synthenexa_pipeline/ is on sys.path so bare imports work ─────────
_HERE       = os.path.dirname(os.path.abspath(__file__))          # scripts/
_PIPELINE   = os.path.dirname(_HERE)                               # synthenexa_pipeline/
if _PIPELINE not in sys.path:
    sys.path.insert(0, _PIPELINE)
# ─────────────────────────────────────────────────────────────────────────────

from core.config import (
    MODEL_NAME, NUM_MUTATIONS, MAX_WORKERS, DATASET_PATH, ExperimentConfig, RLConfig
)
from core.context import PipelineContext
from core.schema import PromptCandidate

from services.dataset_service import DatasetService
from services.eval_service import EvalService
from services.llm_service import LLMService
from services.storage import StorageService
from services.tracker import RLTracker

from pipeline.prompt_evolution.workflow import PromptEvolutionWorkflow
from pipeline.prompt_evolution.stages.mutation import MutationStage
from pipeline.prompt_evolution.stages.execution import ExecutionStage
from pipeline.prompt_evolution.stages.evaluation import EvaluationStage
from pipeline.prompt_evolution.stages.scoring import ScoringStage
from pipeline.prompt_evolution.stages.selection import SelectionStage
from pipeline.prompt_evolution.policies.registry import PolicyRegistry


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

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
    print("\n" + "="*50)
    print("  Reward History")
    print("="*50)
    print(f"  {'Gen':>5}  {'Best Score':>12}")
    print(f"  {'-'*5}  {'-'*12}")
    for gen, score in sorted(reward_history.items()):
        bar = "█" * int(score * 20)
        print(f"  {gen:>5}  {score:>12.4f}  {bar}")
    print("="*50)


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Synthenexa — RL-based prompt optimization pipeline."
    )
    parser.add_argument(
        "--generations", type=int, default=3,
        help="Number of evolution generations to run (default: 3)."
    )
    parser.add_argument(
        "--mutations", type=int, default=NUM_MUTATIONS,
        help=f"Number of prompt mutations per generation (default: {NUM_MUTATIONS})."
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_NAME,
        help=f"Ollama model to use (default: {MODEL_NAME})."
    )
    parser.add_argument(
        "--policy", type=str, default="greedy",
        choices=["greedy", "epsilon_greedy"],
        help="Selection policy (default: greedy)."
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.1,
        help="Epsilon for epsilon-greedy policy (default: 0.1)."
    )
    parser.add_argument(
        "--dataset", type=str, default=DATASET_PATH,
        help=f"Path to MBPP micro dataset (default: {DATASET_PATH})."
    )
    parser.add_argument(
        "--experiment", type=str, default="",
        help="Experiment name (used for artifact directory naming)."
    )
    args = parser.parse_args()

    run_id = str(uuid.uuid4())[:8]
    experiment_name = args.experiment or f"run_{run_id}"

    print(f"\n{'='*60}")
    print(f"  Synthenexa — Prompt Evolution Pipeline")
    print(f"{'='*60}")
    print(f"  Run ID:       {run_id}")
    print(f"  Model:        {args.model}")
    print(f"  Generations:  {args.generations}")
    print(f"  Mutations:    {args.mutations}")
    print(f"  Policy:       {args.policy}" + (
        f" (ε={args.epsilon})" if args.policy == "epsilon_greedy" else ""
    ))
    print(f"  Dataset:      {args.dataset}")
    print(f"{'='*60}\n")

    # ── 1. Load dataset ──────────────────────────────────────────────
    dataset_path = args.dataset
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(_PIPELINE, dataset_path)

    tasks = DatasetService.load_dataset(dataset_path)
    if not tasks:
        print(f"ERROR: No tasks loaded from '{dataset_path}'. Aborting.")
        sys.exit(1)
    print(f"Loaded {len(tasks)} tasks from dataset.\n")

    # ── 2. Load base prompt ──────────────────────────────────────────
    base_prompt_path = os.path.join(_PIPELINE, "Dataset", "base_prompt.jsonl")
    base_prompt = _load_base_prompt(base_prompt_path)
    print(f"Base prompt: {base_prompt.text!r}\n")

    # ── 3. Wire up services ──────────────────────────────────────────
    llm_service  = LLMService(model=args.model)
    eval_service = EvalService()
    storage      = StorageService(
        base_dir=os.path.join(_PIPELINE, "experiments", "artifacts", experiment_name)
    )
    tracker      = RLTracker()

    # ── 4. Build selection policy ────────────────────────────────────
    rl_config = RLConfig(policy_type=args.policy, epsilon=args.epsilon)
    policy    = PolicyRegistry.get_policy(rl_config)

    # ── 5. Build DAG stages ──────────────────────────────────────────
    stages = [
        MutationStage(llm_service=llm_service,  num_mutations=args.mutations),
        ExecutionStage(llm_service=llm_service),
        EvaluationStage(eval_service=eval_service),
        ScoringStage(),
        SelectionStage(policy=policy),
    ]

    # ── 6. Initialise context ────────────────────────────────────────
    context = PipelineContext(
        model_name=args.model,
        run_id=run_id,
        base_prompt=base_prompt,
        tasks=tasks,
    )

    # ── 7. Run the workflow ──────────────────────────────────────────
    workflow = PromptEvolutionWorkflow(
        context=context,
        stages=stages,
        policy=policy,
        storage=storage,
        tracker=tracker,
    )

    final_context = workflow.run(generations=args.generations)

    # ── 8. Print results ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Optimization Complete!")
    print("="*60)
    if final_context.best_prompt:
        print(f"\n  Best Prompt (score={final_context.best_score:.4f}):")
        print(f"  {final_context.best_prompt.text}")
    else:
        print("  No best prompt found (all generations may have failed).")

    _print_reward_table(final_context.reward_history)

    # Save the final best prompt to a well-known file
    best_path = os.path.join(
        _PIPELINE, "experiments", "artifacts", experiment_name, "best_prompt.json"
    )
    if final_context.best_prompt:
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_id":    run_id,
                    "model":     args.model,
                    "prompt_id": final_context.best_prompt.id,
                    "text":      final_context.best_prompt.text,
                    "score":     final_context.best_score,
                    "reward_history": final_context.reward_history,
                },
                f,
                indent=2,
            )
        print(f"\n  Best prompt saved → {best_path}")


if __name__ == "__main__":
    main()
