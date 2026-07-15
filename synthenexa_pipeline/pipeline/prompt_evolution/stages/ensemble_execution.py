"""
pipeline/prompt_evolution/stages/ensemble_execution.py

Ensemble Execution Stage
========================
Runs each model's full sub-pipeline (Mutation → Execution → Evaluation)
independently for every candidate prompt, then stores the per-model EvalResult
on the context so that EnsembleScoringStage can aggregate them.

Architecture
------------
Unlike the standard single-model flow where MutationStage produces candidates
once and ExecutionStage generates code with one LLM, this stage:

  For each model in the ensemble:
    1. Re-mutates the current_prompt using that model's own MutationStage.
    2. Runs ExecutionStage with that model to generate code solutions.
    3. Runs EvaluationStage to score the generated code.
    4. Stores the scalar reward per (prompt_candidate_id, model_name) pair
       in context.ensemble_scores.

After all models have run, context.candidate_prompts holds the full union of
model-specific candidates so the downstream EnsembleScoringStage and
SelectionStage can operate over the complete cross-model pool.
"""

from typing import List, Tuple

from core.context import PipelineContext
from core.reward import compute_reward
from core.schema import PromptCandidate
from pipeline.prompt_evolution.stages.base import Stage
from pipeline.prompt_evolution.stages.mutation import MutationStage
from pipeline.prompt_evolution.stages.execution import ExecutionStage
from pipeline.prompt_evolution.stages.evaluation import EvaluationStage


class EnsembleExecutionStage(Stage):
    """
    Runs independent Mutation + Execution + Evaluation for every model
    in the ensemble and accumulates per-model scores in context.ensemble_scores.

    Args:
        model_pipelines: list of (model_name, MutationStage, ExecutionStage,
                          EvaluationStage) tuples — one entry per ensemble model.
    """

    def __init__(
        self,
        model_pipelines: List[Tuple[str, MutationStage, ExecutionStage, EvaluationStage]],
    ) -> None:
        if not model_pipelines:
            raise ValueError(
                "EnsembleExecutionStage requires at least one model pipeline."
            )
        self.model_pipelines = model_pipelines

    def run(self, context: PipelineContext) -> PipelineContext:
        model_names = [name for name, *_ in self.model_pipelines]
        print(
            f"[EnsembleExecutionStage] Running {len(self.model_pipelines)} model "
            f"pipelines: {', '.join(model_names)}"
        )

        # Initialise fresh accumulation state for this generation
        context.candidate_prompts = []
        context.generated_codes = {}
        context.eval_results = {}
        context.ensemble_scores = {}

        # Accumulate all candidates across models (tagged_uid → PromptCandidate)
        # We keep two mappings:
        #   all_candidates: tagged_uid → PromptCandidate   (for candidate_prompts list)
        #   uid_to_base_id: tagged_uid → original_id       (for ensemble_scores keying)
        all_candidates: dict[str, PromptCandidate] = {}
        uid_to_base_id: dict[str, str] = {}

        for model_name, mut_stage, exec_stage, eval_stage in self.model_pipelines:
            print(f"\n  ── Model: {model_name} ──")

            # ── 1. Mutation — this model generates its own prompt variants ──
            context.candidate_prompts = []
            context = mut_stage.run(context)

            # Tag each candidate id with the model name so generated_codes keys are
            # globally unique, but preserve the original id for ensemble_scores grouping.
            safe_model = model_name.replace(":", "_").replace("-", "_")
            tagged: List[PromptCandidate] = []
            for c in context.candidate_prompts:
                uid = f"{c.id}_{safe_model}"
                uid_to_base_id[uid] = c.id
                if uid not in all_candidates:
                    candidate = PromptCandidate(
                        id=uid,
                        text=c.text,
                        origin_model=c.origin_model,
                        generation=c.generation,
                        mutation_idx=c.mutation_idx,
                        parent_id=c.parent_id,
                    )
                    all_candidates[uid] = candidate
                    tagged.append(candidate)
            context.candidate_prompts = tagged

            # ── 2. Execution — generate code solutions using this model ──
            context.generated_codes = {}
            context = exec_stage.run(context)

            # ── 3. Evaluation — run unit-test harness on generated code ──
            context.eval_results = {}
            context = eval_stage.run(context)

            # ── 4. Record per-model rewards keyed by the ORIGINAL (untagged) id ──
            # This ensures that when ModelA evaluates uid "prompt_0_gen1_m0_ModelA" and
            # ModelB evaluates "prompt_0_gen1_m0_ModelB", both scores land under the
            # shared base key "prompt_0_gen1_m0" so EnsembleScoringStage can aggregate
            # across models for the same mutation slot.
            for tagged_uid, eval_result in context.eval_results.items():
                base_id = uid_to_base_id.get(tagged_uid, tagged_uid)
                context.ensemble_scores.setdefault(base_id, {})
                context.ensemble_scores[base_id][model_name] = compute_reward(eval_result)

        # Restore the full union of candidates for downstream stages.
        # Re-key ensemble_scores back onto tagged uids so SelectionStage can look
        # up scores by the same ids used in context.candidate_prompts.
        # We map: tagged_uid → ensemble_scores[base_id]
        keyed_by_uid: dict[str, dict[str, float]] = {}
        for uid, candidate in all_candidates.items():
            base_id = uid_to_base_id.get(uid, uid)
            keyed_by_uid[uid] = context.ensemble_scores.get(base_id, {})
        context.ensemble_scores = keyed_by_uid

        context.candidate_prompts = list(all_candidates.values())

        print(
            f"\n[EnsembleExecutionStage] Complete — "
            f"{len(context.candidate_prompts)} total candidates across "
            f"{len(self.model_pipelines)} models."
        )
        return context

