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
  median           — middle score after sorting per-model scores.
                     Robust to a single noisy model or extreme outlier.
  max              — best-case model score.
                     Useful for identifying prompts that excel on at least one model.
  vote             — ranking-based vote share.
                     Each model votes for its top-scoring candidate prompt.
                     The final score is the fraction of model votes received.
                     Ties are broken deterministically by prompt id.
"""

from statistics import median
from typing import Literal

from core.context import PipelineContext
from pipeline.prompt_evolution.stages.base import Stage


AggregationStrategy = Literal["mean", "min", "median", "max", "vote"]


class EnsembleScoringStage(Stage):
    """
    Converts per-model scores in context.ensemble_scores into a single
    combined reward written to context.scores.

    Args:
        strategy: How to aggregate per-model scores.
                  One of 'mean' (default), 'min', 'median', 'max', or 'vote'.
    """

    def __init__(self, strategy: AggregationStrategy = "mean") -> None:
        if strategy not in ("mean", "min", "median", "max", "vote"):
            raise ValueError(f"Unknown ensemble aggregation strategy: {strategy!r}")
        self.strategy = strategy

    def run(self, context: PipelineContext) -> PipelineContext:
        print(
            f"[EnsembleScoringStage] Aggregating per-model scores "
            f"(strategy={self.strategy})...\n"
        )
        context.scores = {}

        # Direct lookup by globally-unique UID (set by EnsembleExecutionStage Phase 1)
        candidate_by_id = {c.id: c for c in context.candidate_prompts}
        vote_counts: dict[str, int] = {candidate.id: 0 for candidate in context.candidate_prompts}
        total_models = len({m for scores in context.ensemble_scores.values() for m in scores})

        if self.strategy == "vote":
            # Ranking-based voting: each model casts one vote for the prompt(s) with its best score.
            for model_name in {m for scores in context.ensemble_scores.values() for m in scores}:
                model_candidates = {
                    prompt_id: model_scores[model_name]
                    for prompt_id, model_scores in context.ensemble_scores.items()
                    if model_name in model_scores
                }
                if not model_candidates:
                    continue
                best_score = max(model_candidates.values())
                winners = [prompt_id for prompt_id, score in model_candidates.items() if score == best_score]
                winner_id = sorted(winners)[0]
                vote_counts[winner_id] += 1

        for prompt_id, model_scores in context.ensemble_scores.items():
            if not model_scores:
                context.scores[prompt_id] = 0.0
                continue

            values = list(model_scores.values())
            if self.strategy == "mean":
                combined = sum(values) / len(values)
            elif self.strategy == "min":
                combined = min(values)
            elif self.strategy == "median":
                combined = median(values)
            elif self.strategy == "vote":
                # Ranking-based vote: score is the share of models that selected this prompt.
                combined = vote_counts.get(prompt_id, 0) / total_models if total_models else 0.0
            else:  # max
                combined = max(values)

            context.scores[prompt_id] = combined
            
            # Update the model leaderboard with the best score any prompt achieved on each model
            for m, s in model_scores.items():
                if s > context.model_leaderboard.get(m, -1.0):
                    context.model_leaderboard[m] = s

            # Human-readable breakdown
            c = candidate_by_id.get(prompt_id)
            if c:
                print(f"Candidate: {prompt_id}")
                print(f"  Origin Model: {c.origin_model}")
                print(f"  Generation:   {c.generation}")
                print(f"  Mutation:     {c.mutation_idx}\n")
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
