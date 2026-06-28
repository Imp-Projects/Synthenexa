from typing import List, Dict, Any
from core.schema import PromptCandidate

class RLTracker:
    def __init__(self):
        # Reward curve over generations (generation -> average_reward)
        self.reward_curve: Dict[int, float] = {}
        
        # History of best-performing prompts (generation -> PromptCandidate)
        self.best_prompts: Dict[int, PromptCandidate] = {}
        
        # Per-generation statistics
        self.generation_stats: Dict[int, Dict[str, Any]] = {}

    def log_generation(self, generation: int, avg_reward: float, best_prompt: PromptCandidate, stats: Dict[str, Any]):
        """Tracks RL training progress across generations."""
        self.reward_curve[generation] = avg_reward
        self.best_prompts[generation] = best_prompt
        self.generation_stats[generation] = stats

    def get_reward_curve(self) -> Dict[int, float]:
        return self.reward_curve

    def get_best_prompts_history(self) -> Dict[int, PromptCandidate]:
        return self.best_prompts
