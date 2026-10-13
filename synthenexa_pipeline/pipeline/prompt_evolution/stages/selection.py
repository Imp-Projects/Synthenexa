from core.context import PipelineContext

from pipeline.prompt_evolution.policies.base import Policy
from pipeline.prompt_evolution.stages.base import Stage


class SelectionStage(Stage):
    """Selects the best prompt to carry into the next generation."""

    def __init__(self, policy: Policy):
        self.policy = policy

    def run(self, context: PipelineContext) -> PipelineContext:
        print("\n[SelectionStage] Selecting next generation parent...")

        # Protect the current parent and the all-time winner from being lost
        # when a later mutation round performs worse. For ensembles, also
        # retain the candidate with the strongest weakest-model score.
        pool = list(context.candidate_prompts)
        protected = []
        if context.current_prompt is not None:
            protected.append(context.current_prompt)
        if context.global_best_prompt is not None:
            protected.append(context.global_best_prompt)
        if context.ensemble_score_summary and context.candidate_prompts:
            weakest = max(
                context.candidate_prompts,
                key=lambda candidate: (
                    context.ensemble_score_summary.get(candidate.id, {}).get("minimum", -1.0),
                    candidate.id,
                ),
            )
            protected.append(weakest)

        known_ids = {candidate.id for candidate in pool}
        for candidate in protected:
            if candidate.id not in known_ids:
                pool.append(candidate)
                known_ids.add(candidate.id)
            # The all-time score is the only reliable score after a reset.
            if (
                context.global_best_prompt is not None
                and candidate.id == context.global_best_prompt.id
                and candidate.id not in context.scores
            ):
                context.scores[candidate.id] = context.best_score

        if not pool:
            return context

        # Determine top scorers for logging
        parent_id = context.current_prompt.id if context.current_prompt else ""
        top_candidates = sorted(
            pool,
            key=lambda p: (-context.scores.get(p.id, 0.0), p.id != parent_id, p.id),
        )

        # The policy determines the *next* prompt (handles epsilon-greedy logic)
        next_prompts = self.policy.select(pool, context.scores)
        if next_prompts:
            context.selected_next_prompt = next_prompts[0]
            context.current_prompt = context.selected_next_prompt
            print(f"  Selected best prompt:")
            print(f"    {context.current_prompt.id}")
            print(f"    score={context.scores.get(context.current_prompt.id, 0.0):.4f}\n")
        elif top_candidates:
            # Fallback to the top-scoring candidate if the policy returns nothing.
            context.selected_next_prompt = top_candidates[0]
            context.current_prompt = context.selected_next_prompt
            print(f"  Policy returned no prompt; falling back to top candidate:")
            print(f"    {context.current_prompt.id}")
            print(f"    score={context.scores.get(context.current_prompt.id, 0.0):.4f}\n")

        return context
