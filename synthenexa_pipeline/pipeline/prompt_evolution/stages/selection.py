from core.context import PipelineContext
from pipeline.prompt_evolution.stages.base import Stage
from pipeline.prompt_evolution.policies.base import Policy

class SelectionStage(Stage):
    """Selects the best prompt to carry into the next generation."""
    
    def __init__(self, policy: Policy):
        self.policy = policy

    def run(self, context: PipelineContext) -> PipelineContext:
        print("\n[SelectionStage] Selecting next generation parent...")
        
        # We need a fallback if no candidates exist, though they should
        if not context.candidate_prompts:
            return context
            
        # Determine top scorers for logging
        top_candidates = sorted(
            context.candidate_prompts,
            key=lambda p: context.scores.get(p.id, 0.0),
            reverse=True
        )
        
        # The policy determines the *next* prompt (handles epsilon-greedy logic)
        next_prompts = self.policy.select(context.candidate_prompts, context.scores)
        if next_prompts:
            context.current_prompt = next_prompts[0]
            print(f"  Selected best prompt:")
            print(f"    {context.current_prompt.id}")
            print(f"    score={context.scores.get(context.current_prompt.id, 0.0):.4f}\n")
        elif top_candidates:
            # Fallback to the top-scoring candidate if the policy returns nothing.
            context.current_prompt = top_candidates[0]
            print(f"  Policy returned no prompt; falling back to top candidate:")
            print(f"    {context.current_prompt.id}")
            print(f"    score={context.scores.get(context.current_prompt.id, 0.0):.4f}\n")
            
        return context
