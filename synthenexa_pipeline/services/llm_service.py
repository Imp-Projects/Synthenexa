"""Compatibility wrapper around the lightweight LLM adapter layer."""

import os
from typing import Optional

from services.llm.factory import LLMFactory


class LLMService:
    """Thin wrapper that preserves the old pipeline API while delegating to adapters."""

    def __init__(
        self,
        model: str = "nemotron-3-super:cloud",
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        timeout_seconds: Optional[int] = None,
        max_retries: Optional[int] = None,
        provider: str = "ollama",
    ):
        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_API_BASE") or "http://localhost:11434"
        ).rstrip("/")
        self.temperature = temperature
        self.provider = provider
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.model = model.removeprefix("ollama:")

        self._adapter = LLMFactory().create(
            model=model,
            provider=provider,
            base_url=self.base_url,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        if hasattr(self._adapter, "model") and not isinstance(self._adapter.model, str):
            self._adapter.model = self.model
        if hasattr(self._adapter, "base_url") and not isinstance(self._adapter.base_url, str):
            self._adapter.base_url = self.base_url

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_retries: Optional[int] = None,
        retry_delay: float = 2.0,
        temperature: Optional[float] = None,
    ) -> str:
        """Delegate to the configured provider adapter."""
        return self._adapter.generate(
            prompt=prompt,
            system=system,
            max_retries=max_retries,
            retry_delay=retry_delay,
            temperature=temperature,
        )

    @staticmethod
    def clean_output(text: str) -> str:
        """Expose the adapter output-cleaning helper for compatibility."""
        from services.llm.ollama_adapter import OllamaAdapter

        return OllamaAdapter.clean_output(text)

    _clean_output = clean_output
