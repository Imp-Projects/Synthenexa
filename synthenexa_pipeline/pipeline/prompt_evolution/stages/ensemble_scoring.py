"""
pipeline/prompt_evolution/stages/ensemble_scoring.py

Ensemble Scoring Stage
======================
Aggregates per-model scores stored in context.ensemble_scores into a single
combined scalar reward per prompt candidate, then writes it to context.scores
so that the standard SelectionStage can pick the winner without modification.

Aggregation Strategies
----------------------
  mean   (default) — average score across all models.
                     Rewards prompts that perform consistently across the ensemble.
  min              — worst-case model score.
                     Rewards prompts that never fall below a floor on any model.
  max              — best-case model score.
                     Useful for identifying prompts that excel on at least one model.
"""

from typing import Literal

from core.context import PipelineContext
from pipeline.prompt_evolution.stages.base import Stage


AggregationStrategy = Literal["mean", "min", "max"]


class EnsembleScoringStage(Stage):
    """
    Converts per-model scores in context.ensemble_scores into a single
    combined reward written to context.scores.

    Args:
        strategy: How to aggregate per-model scores.
                  One of 'mean' (default), 'min', or 'max'.
    """

    def __init__(self, strategy: AggregationStrategy = "mean") -> None:
        if strategy not in ("mean", "min", "max"):
            raise ValueError(f"Unknown ensemble aggregation strategy: {strategy!r}")
        self.strategy = strategy

    def run(self, context: PipelineContext) -> PipelineContext:
        print(
            f"[EnsembleScoringStage] Aggregating per-model scores "
            f"(strategy={self.strategy})...\n"
        )
        context.scores = {}

        # Build a lookup for candidate lineage info (using any tagged variant)
        candidate_lineage = {}
        for c in context.candidate_prompts:
            base_id = c.parent_id + f"_gen{c.generation}_m{c.mutation_idx}"
            if base_id not in candidate_lineage:
                candidate_lineage[base_id] = c

        for prompt_id, model_scores in context.ensemble_scores.items():
            if not model_scores:
                context.scores[prompt_id] = 0.0
                continue

            values = list(model_scores.values())
            if self.strategy == "mean":
                combined = sum(values) / len(values)
            elif self.strategy == "min":
                combined = min(values)
            else:  # max
                combined = max(values)

            context.scores[prompt_id] = combined
            
            # Update the model leaderboard with the best score any prompt achieved on each model
            for m, s in model_scores.items():
                if s > context.model_leaderboard.get(m, -1.0):
                    context.model_leaderboard[m] = s

            # Human-readable breakdown
            c = candidate_lineage.get(prompt_id)
            if c:
                print("Candidate:")
                print(f"  ID: m{c.mutation_idx}")
                print(f"  Origin Model: {c.origin_model}")
                print(f"  Generation: {c.generation}")
                print(f"  Parent: {c.parent_id}\n")
            else:
                print(f"Candidate: {prompt_id}\n")

            print("Score:")
            print(f"  Ensemble: {combined:.4f}\n")
            
            print("Individual Performance:")
            for m, s in sorted(model_scores.items()):
                print(f"  {m}: {s:.4f}")
            print("-" * 40 + "\n")

        # Update the all-time best across this generation
        for candidate in context.candidate_prompts:
            score = context.scores.get(candidate.id, 0.0)
            if score > context.best_score:
                context.best_ensemble_scores = context.ensemble_scores.get(candidate.id, {})
            context.update_best(candidate, score)

        return context
