"""
pipeline/prompt_evolution/stages/ensemble_execution.py

Ensemble Execution Stage
========================
Implements TRUE cross-evaluation across all models in the ensemble.

Architecture (Two-Phase)
------------------------
PHASE 1 — Global Mutation:
    Each model in the ensemble generates its own N mutation variants from the
    same base prompt. All variants are pooled into a single master candidate list.
    No evaluation happens in this phase.

PHASE 2 — Cross-Execution & Evaluation:
    Every model evaluates EVERY prompt in the master candidate pool — not just the
    ones it generated itself. This ensures that the ensemble score for each prompt
    reflects genuine cross-model robustness rather than siloed self-evaluation.

Example (2 models × 2 mutations on 5 tasks):
    Phase 1 produces 4 unique prompts (model_A_m0, model_A_m1, model_B_m0, model_B_m1).
    Phase 2 runs 2 × (4 prompts × 5 tasks) = 40 LLM calls total.
    Each prompt then has a per-model score from BOTH models, enabling a meaningful
    universal average in EnsembleScoringStage.

Why this matters
----------------
Under the old silo approach, model_A only evaluated its own mutations and model_B
only evaluated its own. The "cross-model" average was computed over DIFFERENT prompt
strings, making it statistically meaningless. A prompt could score 100% on its
creator but 20% on every other model — yet still be crowned the winner.

With true cross-evaluation, a prompt can only win by being genuinely robust
across the entire ensemble of LLMs, fulfilling the project objective:
"Automatically refine user prompts to achieve better results across LLMs."
"""

from typing import List, Tuple

from core.context import PipelineContext
from core.reward import compute_reward
from core.schema import EvalResult, PromptCandidate

from pipeline.prompt_evolution.stages.base import Stage
from pipeline.prompt_evolution.stages.evaluation import EvaluationStage
from pipeline.prompt_evolution.stages.execution import ExecutionStage
from pipeline.prompt_evolution.stages.mutation import MutationStage, prompt_hash


class EnsembleExecutionStage(Stage):
    """
    Runs a two-phase ensemble: global mutation pooling followed by
    cross-model execution and evaluation for true robustness scoring.

    Args:
        model_pipelines: list of (model_name, MutationStage, ExecutionStage,
                          EvaluationStage) tuples — one entry per ensemble model.
    """

    def __init__(
        self,
        model_pipelines: List[Tuple[str, MutationStage, ExecutionStage, EvaluationStage]],
    ) -> None:
        if not model_pipelines:
            raise ValueError("EnsembleExecutionStage requires at least one model pipeline.")
        self.model_pipelines = model_pipelines

    def run(self, context: PipelineContext) -> PipelineContext:
        model_names = [name for name, *_ in self.model_pipelines]
        num_models = len(self.model_pipelines)
        print(f"[EnsembleExecutionStage] Running {num_models} model pipelines: {', '.join(model_names)}")

        # ── Reset accumulation state for this generation ──────────────────────
        context.candidate_prompts = []
        context.generated_codes = {}
        context.eval_results = {}
        context.eval_results_by_model = {}
        context.ensemble_split_scores = {}
        context.ensemble_scores = {}

        # =====================================================================
        # PHASE 1: GLOBAL MUTATION
        # Each model generates its own N mutations from the current base prompt.
        # All candidates are pooled into master_candidates; no evaluation yet.
        # =====================================================================
        print(f"\n  {'─' * 56}")
        print(f"  PHASE 1: GLOBAL MUTATION (Pooling Prompt Ideas)")
        print(f"  {'─' * 56}")

        # master_candidates: uid → PromptCandidate  (globally unique IDs)
        master_candidates: dict[str, PromptCandidate] = {}
        seen_text_hashes: set[str] = set()

        for model_name, mut_stage, _exec_stage, _eval_stage in self.model_pipelines:
            print(f"\n  ── Mutating with: {model_name} ──")

            # Reset candidate list so MutationStage starts fresh for this model
            context.candidate_prompts = []
            context = mut_stage.run(context)

            # Tag each candidate with the originating model name to guarantee
            # globally unique IDs across all models in the pool.
            safe_model = model_name.replace(":", "_").replace("-", "_")
            for c in context.candidate_prompts:
                text_hash = prompt_hash(c.text)
                if text_hash in seen_text_hashes:
                    print(f"    - Skipped duplicate prompt from {model_name}: {c.id}")
                    continue
                uid = f"{c.id}_{safe_model}"
                if uid not in master_candidates:
                    candidate = PromptCandidate(
                        id=uid,
                        text=c.text,
                        origin_model=c.origin_model,
                        generation=c.generation,
                        mutation_idx=c.mutation_idx,
                        parent_id=c.parent_id,
                        mutation_role=c.mutation_role,
                        prompt_hash=text_hash,
                    )
                    master_candidates[uid] = candidate
                    seen_text_hashes.add(text_hash)
                    print(f"    + Added to pool: {uid}")

        total_candidates = len(master_candidates)
        print(
            f"\n  [Phase 1 Complete] Master pool contains {total_candidates} unique prompts across {num_models} models."
        )

        # =====================================================================
        # PHASE 2: CROSS-EXECUTION & EVALUATION
        # Every model evaluates EVERY prompt in the master candidate pool.
        # This is the key step that makes the ensemble scoring mathematically
        # meaningful — the same prompt strings are tested by every model.
        # =====================================================================
        print(f"\n  {'─' * 56}")
        print(f"  PHASE 2: CROSS-EXECUTION & EVALUATION")
        print(f"  {'─' * 56}")

        # Expose the full master pool to each model's exec/eval stages
        all_candidates_list = list(master_candidates.values())

        for model_name, _mut_stage, exec_stage, eval_stage in self.model_pipelines:
            print(f"\n  ── Cross-Testing on: {model_name} ──")

            # Give this model the complete candidate pool to generate code for
            context.candidate_prompts = all_candidates_list
            context.generated_codes = {}
            context.eval_results = {}

            # ExecutionStage: model generates code for all candidates × all tasks
            context = exec_stage.run(context)

            # EvaluationStage: unit-test harness scores the generated code
            context = eval_stage.run(context)

            missing = [candidate.id for candidate in all_candidates_list if candidate.id not in context.eval_results]
            if missing:
                raise RuntimeError(
                    f"Incomplete ensemble evaluation for {model_name}; "
                    f"missing {len(missing)} candidate result(s)."
                )

            context.eval_results_by_model[model_name] = {}

            # Record this model's reward for every prompt in the pool
            for uid, eval_result in context.eval_results.items():
                eval_result.evaluator_model = model_name
                context.eval_results_by_model[model_name][uid] = eval_result
                context.ensemble_scores.setdefault(uid, {})
                context.ensemble_scores[uid][model_name] = compute_reward(eval_result)
                # Derive split rewards from the same sandbox outcomes. This is
                # deliberately local bookkeeping: no additional Ollama call.
                outcomes = context.task_outcomes.get(uid, [])
                by_task = {item["task_id"]: item for item in outcomes}
                split_values: dict[str, list[bool]] = {}
                for task in context.tasks:
                    task_id = task.get("task_id")
                    outcome = by_task.get(task_id)
                    if outcome is not None:
                        split_values.setdefault(task.get("dataset_split", "train"), []).append(
                            bool(outcome["passed"])
                        )
                for split, passed_flags in split_values.items():
                    if passed_flags:
                        context.ensemble_split_scores.setdefault(uid, {}).setdefault(model_name, {})[
                            split
                        ] = sum(passed_flags) / len(passed_flags)

        incomplete = {
            candidate.id: sorted(
                set(model_names) - set(context.ensemble_scores.get(candidate.id, {}))
            )
            for candidate in all_candidates_list
            if set(context.ensemble_scores.get(candidate.id, {})) != set(model_names)
        }
        if incomplete:
            raise RuntimeError(f"Incomplete ensemble score matrix: {incomplete}")

        # Restore the full candidate list for downstream stages
        context.candidate_prompts = all_candidates_list

        print(
            f"\n[EnsembleExecutionStage] Complete — "
            f"{total_candidates} prompts cross-evaluated across "
            f"{num_models} models "
            f"({total_candidates * num_models} total evaluation runs)."
        )
        return context

    def evaluate_baseline(self, context: PipelineContext) -> PipelineContext:
        """Evaluate the unchanged base prompt once on every ensemble model."""
        if context.base_prompt is None:
            raise ValueError("Cannot evaluate an ensemble baseline without a base prompt.")

        baseline = context.base_prompt.model_copy(update={"id": "prompt_0", "is_base": True})
        context.candidate_prompts = [baseline]
        context.generated_codes = {}
        context.eval_results = {}
        context.eval_results_by_model = {}
        context.baseline_eval_results_by_model = {}
        context.ensemble_scores = {}

        for model_name, _mut_stage, exec_stage, eval_stage in self.model_pipelines:
            context.candidate_prompts = [baseline]
            context.generated_codes = {}
            context.eval_results = {}
            context = exec_stage.run(context)
            context = eval_stage.run(context)
            result = context.eval_results.get(baseline.id)
            if result is None:
                raise RuntimeError(f"Baseline evaluation missing for model {model_name}.")
            result.evaluator_model = model_name
            result.is_base = True
            context.eval_results_by_model[model_name] = {baseline.id: result}
            context.baseline_eval_results_by_model[model_name] = {baseline.id: result}
            context.ensemble_scores.setdefault(baseline.id, {})[model_name] = compute_reward(result)

        context.candidate_prompts = [baseline]
        context.eval_results = {
            baseline.id: result
            for model_results in context.baseline_eval_results_by_model.values()
            for result in model_results.values()
        }
        return context

    def evaluate_candidate(
        self, context: PipelineContext, candidate: PromptCandidate
    ) -> dict[str, dict[str, EvalResult]]:
        """Evaluate one frozen candidate across all models on the current task set."""
        results_by_model: dict[str, dict[str, EvalResult]] = {}
        for model_name, _mut_stage, exec_stage, eval_stage in self.model_pipelines:
            context.candidate_prompts = [candidate]
            context.generated_codes = {}
            context.eval_results = {}
            context = exec_stage.run(context)
            context = eval_stage.run(context)
            result = context.eval_results.get(candidate.id)
            if result is None:
                raise RuntimeError(f"Evaluation missing for prompt {candidate.id} on model {model_name}.")
            result.evaluator_model = model_name
            results_by_model[model_name] = {candidate.id: result}
        return results_by_model
