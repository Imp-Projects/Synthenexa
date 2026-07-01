"""Common LLM adapter interface."""

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLMAdapter(ABC):
    """Contract implemented by model provider adapters."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_retries: Optional[int] = None,
        retry_delay: float = 2.0,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate text from a user prompt and optional system prompt."""
