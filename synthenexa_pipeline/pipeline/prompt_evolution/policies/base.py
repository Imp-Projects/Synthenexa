from abc import ABC, abstractmethod
from typing import List, Dict
from core.schema import PromptCandidate

class Policy(ABC):
    """Defines the base class all selection policies inherit from."""
    
    @abstractmethod
    def select(self, candidates: List[PromptCandidate], rewards: Dict[str, float]) -> List[PromptCandidate]:
        """Override this to implement a custom selection strategy."""
        pass
