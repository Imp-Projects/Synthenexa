"""
pipeline/prompt_evolution/workflow.py — RL + DAG controller.

This is the heart of the system. It:
  1. Accepts a pre-configured PipelineContext (with tasks, base_prompt, etc.)
  2. Runs N generations, each consisting of the full DAG stage chain:
       mutation → execution → evaluation → scoring → selection
  3. Tracks the reward history and persists per-generation snapshots.
"""
from typing import List, Optional

from core.context import PipelineContext
from pipeline.prompt_evolution.stages.base import Stage
from pipeline.prompt_evolution.policies.base import Policy


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
        self.stages  = stages
        self.policy  = policy
        self.storage = storage
        self.tracker = tracker

    # ---------------------------------------------------------------- #
    #  Public entry point                                               #
    # ---------------------------------------------------------------- #

    def run(self, generations: int) -> PipelineContext:
        """
        Run the full evolution loop for `generations` iterations.

        Returns the final PipelineContext (with best_prompt, reward_history, …).
        """
        ctx = self.context
        ctx.max_generations = generations

        # Seed current_prompt from base_prompt on the very first run
        if ctx.current_prompt is None and ctx.base_prompt is not None:
            ctx.current_prompt = ctx.base_prompt

        for gen in range(1, generations + 1):
            ctx.generation = gen
            print(f"\n{'='*60}")
            print(f"  Generation {gen} / {generations}")
            print(f"{'='*60}")

            ctx = self.run_generation(ctx)

            # Record this generation's best score in the reward history
            ctx.reward_history[gen] = ctx.best_score

            # Persist snapshot if storage is wired up
            if self.storage:
                self.storage.save_generation_snapshot(ctx, gen)
                self.storage.save_prompts(ctx.candidate_prompts, gen)
                eval_list = list(ctx.eval_results.values())
                if eval_list:
                    self.storage.save_evaluations(eval_list, gen)

            # Log to tracker if wired up
            if self.tracker and ctx.best_prompt:
                avg_reward = (
                    sum(ctx.scores.values()) / len(ctx.scores)
                    if ctx.scores else 0.0
                )
                self.tracker.log_generation(
                    generation=gen,
                    avg_reward=avg_reward,
                    best_prompt=ctx.best_prompt,
                    stats={"scores": ctx.scores, "best_score": ctx.best_score},
                )

            # Reset ephemeral per-generation state; keep best + history
            ctx.reset_for_next_generation()

        self.context = ctx
        return ctx

    # ---------------------------------------------------------------- #
    #  Single-generation execution                                      #
    # ---------------------------------------------------------------- #

    def run_generation(self, context: PipelineContext) -> PipelineContext:
        """
        Execute one full generation: run each stage in the DAG in order,
        threading context through as the shared state carrier.
        """
        for stage in self.stages:
            context = stage.run(context)
        return context
