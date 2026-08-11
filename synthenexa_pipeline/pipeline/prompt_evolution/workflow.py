"""
pipeline/prompt_evolution/workflow.py — RL + DAG controller.

This is the heart of the system. It:
  1. Accepts a pre-configured PipelineContext (with tasks, base_prompt, etc.)
  2. Runs N generations, each consisting of the full DAG stage chain:
       mutation → execution → evaluation → scoring → selection
  3. Tracks the reward history and persists per-generation snapshots.
"""
from contextlib import nullcontext
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
                baseline_scope = self.metrics.track_generation() if self.metrics else nullcontext()
                with baseline_scope:
                    ctx.generation = 0
                    print(f"\n{'=' * 60}")
                    print(f"  Generation 0 / {generations} (Baseline Evaluation)")
                    print(f"{'=' * 60}\n")

                    ctx.candidate_prompts = [ctx.base_prompt]

                    start_idx = 0
                    for i, stage in enumerate(self.stages):
                        if "Execution" in stage.__class__.__name__:
                            start_idx = i
                            break

                    for stage in self.stages[start_idx:]:
                        if "Selection" not in stage.__class__.__name__:
                            ctx = self._run_stage(stage, ctx)

                    ctx.best_prompt = ctx.base_prompt
                    ctx.best_score = ctx.scores.get(ctx.base_prompt.id, 0.0)
                    ctx.best_ensemble_scores = (
                        ctx.eval_results_by_model if hasattr(ctx, "eval_results_by_model") else {}
                    )
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

            self.context = ctx
            if self.metrics:
                self.metrics.record_run("success")
            return ctx
        except Exception:
            if self.metrics:
                self.metrics.record_run("failed")
            raise

    def run_generation(self, context: PipelineContext) -> PipelineContext:
        """
        Execute one full generation: run each stage in the DAG in order,
        threading context through as the shared state carrier.
        """
        for stage in self.stages:
            context = self._run_stage(stage, context)
        return context

    def _run_stage(self, stage: Stage, context: PipelineContext) -> PipelineContext:
        stage_scope = self.metrics.track_stage(stage.__class__.__name__) if self.metrics else nullcontext()
        with stage_scope:
            return stage.run(context)

    def _record_best_score(self, context: PipelineContext) -> None:
        if not self.metrics:
            return
        model_name = None
        if context.best_prompt is not None:
            model_name = context.best_prompt.origin_model
        if not model_name:
            model_name = getattr(context, "model_name", None)
        self.metrics.record_best_score(model_name, context.best_score)
