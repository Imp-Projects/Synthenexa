from abc import ABC, abstractmethod
from core.context import PipelineContext

class Stage(ABC):
    """Defines the base class all pipeline stages inherit from."""
    
    @abstractmethod
    def run(self, context: PipelineContext) -> PipelineContext:
        """Override this in each stage to implement its logic."""
        pass
