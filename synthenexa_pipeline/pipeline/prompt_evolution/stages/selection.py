from core.context import PipelineContext
from pipeline.prompt_evolution.stages.base import Stage
from pipeline.prompt_evolution.policies.base import Policy

class SelectionStage(Stage):
    """Selects the best prompt to carry into the next generation."""
    
    def __init__(self, policy: Policy):
        self.policy = policy

    def run(self, context: PipelineContext) -> PipelineContext:
        print("Running SelectionStage...")
        
        # We need a fallback if no candidates exist, though they should
        if not context.candidate_prompts:
            return context
            
        # Determine top scorers for logging
        top_candidates = sorted(
            context.candidate_prompts,
            key=lambda p: context.scores.get(p.id, 0.0),
            reverse=True
        )
        
        # Track the absolute best this generation regardless of selection policy's explore/exploit
        if top_candidates:
            context.best_prompt = top_candidates[0]
            context.best_score = context.scores.get(top_candidates[0].id, 0.0)
            
        # The policy determines the *next* prompt (handles epsilon-greedy logic)
        next_prompts = self.policy.select(context.candidate_prompts, context.scores)
        if next_prompts:
            context.current_prompt = next_prompts[0]
            
        return context
