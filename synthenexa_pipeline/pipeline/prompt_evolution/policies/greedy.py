from typing import List, Dict
from core.schema import PromptCandidate
from pipeline.prompt_evolution.policies.base import Policy

class GreedySelectionPolicy(Policy):
    """Always selects the prompt with the highest reward score — pure exploitation, no exploration."""
    
    def select(self, candidates: List[PromptCandidate], rewards: Dict[str, float]) -> List[PromptCandidate]:
        if not candidates:
            return []
            
        sorted_candidates = sorted(
            candidates,
            key=lambda p: rewards.get(p.id, 0.0),
            reverse=True
        )
        return [sorted_candidates[0]]
