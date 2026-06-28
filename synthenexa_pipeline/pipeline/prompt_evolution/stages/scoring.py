"""
pipeline/prompt_evolution/stages/scoring.py — Convert eval results → reward signal.

Thin stage: delegates entirely to core.reward.compute_reward().
Also calls context.update_best() so the global best is always kept up-to-date.
"""
from core.context import PipelineContext
from core.reward import compute_reward
from pipeline.prompt_evolution.stages.base import Stage


class ScoringStage(Stage):
    """Assigns a scalar RL reward to every evaluated prompt candidate."""

    def run(self, context: PipelineContext) -> PipelineContext:
        print("[ScoringStage] Computing rewards...")
        context.scores = {}

        for prompt_id, eval_result in context.eval_results.items():
            score = compute_reward(eval_result)
            context.scores[prompt_id] = score
            print(f"  ✓ {prompt_id}: reward = {score:.4f}")

        # Update the all-time best across all candidates this generation
        for candidate in context.candidate_prompts:
            score = context.scores.get(candidate.id, 0.0)
            context.update_best(candidate, score)

        return context
