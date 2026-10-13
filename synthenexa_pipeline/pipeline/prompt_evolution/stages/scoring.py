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
            outcomes = context.task_outcomes.get(prompt_id, [])
            split_values: dict[str, list[bool]] = {}
            task_splits = {t.get("task_id"): t.get("dataset_split", "train") for t in context.tasks}
            for outcome in outcomes:
                split_values.setdefault(task_splits.get(outcome.get("task_id"), "train"), []).append(
                    bool(outcome.get("passed"))
                )
            if "validation" in split_values:
                search_flags = split_values.get("train", [])
                search_score = sum(search_flags) / len(search_flags) if search_flags else score
                validation_score = sum(split_values["validation"]) / len(split_values["validation"])
                score = 0.7 * search_score + 0.3 * validation_score
            context.scores[prompt_id] = score
            print(f"  ✓ {prompt_id}: reward = {score:.4f}")

        # Update the all-time best across all candidates this generation
        for candidate in context.candidate_prompts:
            score = context.scores.get(candidate.id, 0.0)
            context.update_best(candidate, score)

        return context
