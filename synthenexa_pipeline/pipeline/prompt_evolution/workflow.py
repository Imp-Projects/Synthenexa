"""
pipeline/prompt_evolution/workflow.py — RL + DAG controller.

This is the heart of the system. It:
  1. Accepts a pre-configured PipelineContext (with tasks, base_prompt, etc.)
  2. Runs N generations, each consisting of the full DAG stage chain:
       mutation → execution → evaluation → scoring → selection
  3. Tracks the reward history and persists per-generation snapshots.
"""

from contextlib import nullcontext
import time
from typing import List

from core.context import PipelineContext

from pipeline.prompt_evolution.policies.base import Policy
from pipeline.prompt_evolution.stages.base import Stage

try:
    from observability import get_pipeline_metrics
except Exception:  # pragma: no cover - keeps the workflow backward compatible
    get_pipeline_metrics = None


class PromptEvolutionWorkflow:
    """
    Orchestrates the full multi-generation prompt evolution loop.

    Args:
        context:  Pre-configured PipelineContext (tasks already loaded).
        stages:   Ordered list of Stage objects forming the DAG.
        policy:   Selection policy instance (greedy / epsilon-greedy / …).
        storage:  Optional StorageService — if provided, snapshots each generation.
        tracker:  Optional RLTracker — if provided, logs per-generation stats.
    """

    def __init__(
        self,
        context: PipelineContext,
        stages: List[Stage],
        policy: Policy,
        storage=None,
        tracker=None,
    ):
        self.context = context
        self.stages = stages
        self.policy = policy
        self.storage = storage
        self.tracker = tracker
        self.metrics = get_pipeline_metrics() if get_pipeline_metrics else None

    def run(self, generations: int) -> PipelineContext:
        """
        Run the full evolution loop for `generations` iterations.

        Returns the final PipelineContext (with best_prompt, reward_history, …).
        """
        ctx = self.context
        ctx.max_generations = generations

        try:
            if ctx.base_prompt is not None:
                base_prompt = ctx.base_prompt.model_copy(update={"id": "prompt_0", "is_base": True})
                ctx.base_prompt = base_prompt
                baseline_scope = self.metrics.track_generation() if self.metrics else nullcontext()
                with baseline_scope:
                    ctx.generation = 0
                    print(f"\n{'=' * 60}")
                    print(f"  Generation 0 / {generations} (Baseline Evaluation)")
                    print(f"{'=' * 60}\n")

                    ctx.candidate_prompts = [base_prompt]

                    start_idx = 0
                    for i, stage in enumerate(self.stages):
                        if "Execution" in stage.__class__.__name__:
                            start_idx = i
                            break

                    execution_stage = self.stages[start_idx]
                    baseline_runner = getattr(execution_stage, "evaluate_baseline", None)
                    if callable(baseline_runner):
                        ctx = self._run_stage_with_method(execution_stage, ctx, "evaluate_baseline")
                        remaining_stages = self.stages[start_idx + 1 :]
                    else:
                        remaining_stages = self.stages[start_idx:]

                    for stage in remaining_stages:
                        if "Selection" not in stage.__class__.__name__:
                            ctx = self._run_stage(stage, ctx)

                    ctx.best_prompt = base_prompt
                    ctx.global_best_prompt = base_prompt
                    ctx.best_score = ctx.scores.get(base_prompt.id, 0.0)
                    ctx.best_ensemble_scores = ctx.ensemble_scores.get(base_prompt.id, {})
                    ctx.reward_history[0] = ctx.best_score
                    self._record_best_score(ctx)

                    if self.storage:
                        self.storage.save_generation_snapshot(ctx, 0)
                        self.storage.save_prompts(ctx.candidate_prompts, 0)
                        eval_list = []
                        if ctx.eval_results_by_model:
                            for model_results in ctx.eval_results_by_model.values():
                                eval_list.extend(model_results.values())
                        else:
                            eval_list = list(ctx.eval_results.values())
                        if eval_list:
                            self.storage.save_evaluations(eval_list, 0)

                    if self.tracker:
                        self.tracker.log_generation(
                            generation=0,
                            avg_reward=ctx.best_score,
                            best_prompt=ctx.best_prompt,
                            stats={"scores": ctx.scores, "best_score": ctx.best_score},
                        )

                    ctx.reset_for_next_generation()

            if ctx.current_prompt is None and ctx.base_prompt is not None:
                ctx.current_prompt = ctx.base_prompt

            for gen in range(1, generations + 1):
                generation_started = time.monotonic()
                generation_scope = self.metrics.track_generation() if self.metrics else nullcontext()
                with generation_scope:
                    ctx.generation = gen
                    print(f"\n{'=' * 60}")
                    print(f"  Generation {gen} / {generations}")
                    print(f"{'=' * 60}\n")

                    if ctx.current_prompt:
                        print("Parent Prompt:")
                        print(f"  Source Model: {ctx.current_prompt.origin_model or 'base'}")
                        if ctx.current_prompt.id in ctx.scores:
                            print(f"  Previous Score: {ctx.scores[ctx.current_prompt.id]:.4f}")
                        print("\nEvolving from:")
                        print(f"  {ctx.current_prompt.text[:120]!r}...")
                        print("-" * 60 + "\n")

                    ctx = self.run_generation(ctx)
                    generation_elapsed = time.monotonic() - generation_started
                    remaining = max(generations - gen, 0)
                    projected_remaining = generation_elapsed * remaining
                    print(
                        f"[Timing] Generation {gen} completed in {generation_elapsed / 60:.1f} min; "
                        f"projected remaining workflow time: ~{projected_remaining / 60:.1f} min"
                    )
                    if ctx.best_prompt is not None:
                        ctx.global_best_prompt = ctx.best_prompt
                    ctx.reward_history[gen] = ctx.best_score
                    self._record_best_score(ctx)

                    if self.storage:
                        self.storage.save_generation_snapshot(ctx, gen)
                        self.storage.save_prompts(ctx.candidate_prompts, gen)
                        eval_list = []
                        if ctx.eval_results_by_model:
                            for model_results in ctx.eval_results_by_model.values():
                                eval_list.extend(model_results.values())
                        else:
                            eval_list = list(ctx.eval_results.values())
                        if eval_list:
                            self.storage.save_evaluations(eval_list, gen)

                    if self.tracker and ctx.best_prompt:
                        avg_reward = sum(ctx.scores.values()) / len(ctx.scores) if ctx.scores else 0.0
                        self.tracker.log_generation(
                            generation=gen,
                            avg_reward=avg_reward,
                            best_prompt=ctx.best_prompt,
                            stats={"scores": ctx.scores, "best_score": ctx.best_score},
                        )

                    ctx.reset_for_next_generation()

            self._evaluate_final_test(ctx)
            if self.storage and ctx.final_test_eval_results_by_model:
                final_test_results = [
                    result
                    for model_results in ctx.final_test_eval_results_by_model.values()
                    for result in model_results.values()
                ]
                self.storage.save_final_test_evaluations(final_test_results)

            self.context = ctx
            if self.metrics:
                self.metrics.record_run("success")
            return ctx
        except Exception:
            if self.metrics:
                self.metrics.record_run("failed")
            raise

    def _evaluate_final_test(self, context: PipelineContext) -> None:
        """Evaluate the frozen winner on held-out tasks after selection ends."""
        if not context.test_tasks or context.best_prompt is None or context.base_prompt is None:
            return

        best_prompt = context.best_prompt
        base_prompt = context.base_prompt

        selection_tasks = context.tasks
        selection_split = context.dataset_split
        selection_candidates = context.candidate_prompts
        selection_generated_codes = context.generated_codes
        selection_eval_results = context.eval_results
        selection_eval_by_model = context.eval_results_by_model

        context.tasks = context.test_tasks
        context.dataset_split = "test"
        context.candidate_prompts = [best_prompt]
        context.generated_codes = {}
        context.eval_results = {}
        context.eval_results_by_model = {}
        context.final_test_eval_results_by_model = {}

        # Compare the original prompt with the final selected mutation. The
        # all-time global best may legitimately remain the base prompt, but a
        # complete evidence package still needs the selected refined prompt.
        refined_prompt = context.current_prompt or best_prompt
        candidates = [base_prompt]
        if refined_prompt.id != base_prompt.id:
            candidates.append(refined_prompt)

        execution_stage = next(
            (stage for stage in self.stages if hasattr(stage, "evaluate_candidate")),
            None,
        )
        if execution_stage is not None:
            evaluate_candidate = getattr(execution_stage, "evaluate_candidate")
            for candidate in candidates:
                results = evaluate_candidate(context, candidate)
                for model_name, model_results in results.items():
                    context.final_test_eval_results_by_model.setdefault(model_name, {}).update(model_results)
        else:
            execution_index = next(
                (index for index, stage in enumerate(self.stages) if "Execution" in stage.__class__.__name__),
                None,
            )
            if execution_index is not None and execution_index + 1 < len(self.stages):
                for candidate in candidates:
                    context.candidate_prompts = [candidate]
                    context.generated_codes = {}
                    context.eval_results = {}
                    context = self._run_stage(self.stages[execution_index], context)
                    context = self._run_stage(self.stages[execution_index + 1], context)
                    context.final_test_eval_results_by_model.setdefault(context.model_name, {}).update(
                        context.eval_results
                    )

        context.tasks = selection_tasks
        context.dataset_split = selection_split
        context.candidate_prompts = selection_candidates
        context.generated_codes = selection_generated_codes
        context.eval_results = selection_eval_results
        context.eval_results_by_model = selection_eval_by_model

    def run_generation(self, context: PipelineContext) -> PipelineContext:
        """
        Execute one full generation: run each stage in the DAG in order,
        threading context through as the shared state carrier.
        """
        for stage in self.stages:
            context = self._run_stage(stage, context)
        return context

    def _run_stage(self, stage: Stage, context: PipelineContext) -> PipelineContext:
        return self._run_stage_with_method(stage, context, "run")

    def _run_stage_with_method(self, stage: Stage, context: PipelineContext, method: str) -> PipelineContext:
        stage_scope = self.metrics.track_stage(stage.__class__.__name__) if self.metrics else nullcontext()
        with stage_scope:
            return getattr(stage, method)(context)

    def _record_best_score(self, context: PipelineContext) -> None:
        if not self.metrics:
            return
        model_name = None
        if context.best_prompt is not None:
            model_name = context.best_prompt.origin_model
        if not model_name:
            model_name = getattr(context, "model_name", None)
        self.metrics.record_best_score(model_name, context.best_score)
