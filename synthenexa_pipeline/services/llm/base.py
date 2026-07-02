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


class ProviderQuotaError(Exception):
    """Raised when a provider rejects a request due to a hard quota/billing limit.
    This implies retries will be futile and the pipeline should immediately fallback.
    """
    def __init__(self, provider: str, message: str):
        self.provider = provider
        super().__init__(f"[{provider}] Quota exhausted: {message}")
