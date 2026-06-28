import random
from typing import List, Dict
from core.schema import PromptCandidate
from pipeline.prompt_evolution.policies.base import Policy

class EpsilonGreedySelectionPolicy(Policy):
    """
    Balances exploration and exploitation — picks the top-scoring prompt most of the time, 
    but occasionally selects a random candidate to explore alternatives.
    """
    
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon

    def select(self, candidates: List[PromptCandidate], rewards: Dict[str, float]) -> List[PromptCandidate]:
        if not candidates:
            return []

        if random.random() < self.epsilon:
            # Explore: pick a random candidate
            return [random.choice(candidates)]
        else:
            # Exploit: pick the top scorer
            sorted_candidates = sorted(
                candidates,
                key=lambda p: rewards.get(p.id, 0.0),
                reverse=True
            )
            return [sorted_candidates[0]]
