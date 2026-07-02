"""Compatibility wrapper around the lightweight LLM adapter layer."""

import os
from typing import Optional

from services.llm.base import ProviderQuotaError
from services.llm.factory import LLMFactory
from services.llm.health import ProviderHealthTracker
from services.llm.registry import ModelRegistry, _infer_provider, normalize_model_name


class LLMService:
    """Thin wrapper that preserves the old pipeline API while delegating to adapters.

    Provider auto-detection from model name prefix:
        openai:<model>  → OpenAI Chat Completions API
        gemini:<model>  → Google Gemini API
        <model>         → Ollama (local or cloud-tagged)
    """

    def __init__(
        self,
        model: str = "nemotron-3-super:cloud",
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        timeout_seconds: Optional[int] = None,
        max_retries: Optional[int] = None,
        provider: Optional[str] = None,
    ):
        self.model = normalize_model_name(model)
        self.temperature = temperature

        # Auto-detect provider from model name if not explicitly given
        self.provider = provider or _infer_provider(self.model)

        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_API_BASE") or "http://localhost:11434"
        ).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

        self._adapter = LLMFactory().create(
            model=self.model,
            provider=self.provider,
            base_url=self.base_url,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_retries: Optional[int] = None,
        retry_delay: float = 2.0,
        temperature: Optional[float] = None,
    ) -> str:
        """Delegate to the configured provider adapter, with fallback routing."""
        try:
            return self._adapter.generate(
                prompt=prompt,
                system=system,
                max_retries=max_retries,
                retry_delay=retry_delay,
                temperature=temperature,
            )
        except ProviderQuotaError as e:
            ProviderHealthTracker.disable(e.provider)
            print(f"[LLMService] Provider '{e.provider}' disabled due to quota exhaustion.")
            
            # Find the next best fallback model
            registry = ModelRegistry()
            available_models = registry.list()
            available_models.sort(key=lambda s: s.priority)
            
            fallback_spec = None
            for spec in available_models:
                if ProviderHealthTracker.is_enabled(spec.provider):
                    fallback_spec = spec
                    break
            
            if not fallback_spec:
                raise RuntimeError("All LLM providers have been exhausted due to quota or hard errors.") from e
                
            print(f"[LLMService] Hot-swapping adapter. Falling back to: {fallback_spec.name} ({fallback_spec.provider})")
            self.model = fallback_spec.name
            self.provider = fallback_spec.provider
            self._adapter = LLMFactory().create(
                model=self.model,
                provider=self.provider,
                base_url=self.base_url,
                temperature=self.temperature,
                timeout_seconds=self.timeout_seconds,
                max_retries=self.max_retries,
            )
            
            # Recursively attempt to generate with the new adapter
            return self.generate(prompt, system, max_retries, retry_delay, temperature)

    @staticmethod
    def clean_output(text: str) -> str:
        """Expose the adapter output-cleaning helper for compatibility."""
        from services.llm.ollama_adapter import OllamaAdapter

        return OllamaAdapter.clean_output(text)

    _clean_output = clean_output
