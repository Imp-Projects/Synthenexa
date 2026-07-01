"""Factory for resolving LLM adapters."""

from typing import Optional

from services.llm.base import BaseLLMAdapter
from services.llm.cloud_adapter import CloudAdapter
from services.llm.ollama_adapter import OllamaAdapter
from services.llm.registry import ModelRegistry, normalize_model_name


class LLMFactory:
    """Create provider adapters while keeping pipeline code provider-neutral."""

    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or ModelRegistry()

    def create(
        self,
        model: str,
        provider: str = "ollama",
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout_seconds: Optional[int] = None,
        max_retries: Optional[int] = None,
    ) -> BaseLLMAdapter:
        provider_name = provider.lower().strip()
        model_name = normalize_model_name(model)

        if provider_name != "ollama":
            raise ValueError(f"Unsupported LLM provider: {provider}")

        spec = self.registry.get(model_name)
        if model_name.endswith(":cloud") or getattr(spec, "resource_profile", "") == "cloud":
            return CloudAdapter(
                model=model_name,
                base_url=base_url,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                model_spec=spec,
            )

        return OllamaAdapter(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            model_spec=spec,
        )
