"""Factory for resolving LLM adapters based on model provider."""

from typing import Optional

from services.llm.base import BaseLLMAdapter
from services.llm.registry import ModelRegistry, _infer_provider, normalize_model_name


class LLMFactory:
    """Create provider adapters while keeping pipeline code provider-neutral.

    Routing logic:
        openai:<model>  → OpenAIAdapter
        gemini:<model>  → GeminiAdapter
        <model>         → OllamaAdapter  (local or cloud-tagged)
    """

    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or ModelRegistry()

    def create(
        self,
        model: str,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout_seconds: Optional[int] = None,
        max_retries: Optional[int] = None,
    ) -> BaseLLMAdapter:
        model_name = normalize_model_name(model)
        spec = self.registry.get(model_name)

        # Determine provider: explicit arg > spec field > inferred from name prefix
        resolved_provider = (
            (provider or "").lower().strip()
            or spec.provider
            or _infer_provider(model_name)
        )

        if resolved_provider == "openai":
            from services.llm.openai_adapter import OpenAIAdapter

            return OpenAIAdapter(
                model=model_name,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                model_spec=spec,
            )

        if resolved_provider == "gemini":
            from services.llm.gemini_adapter import GeminiAdapter

            return GeminiAdapter(
                model=model_name,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                model_spec=spec,
            )

        # Default: Ollama (local or cloud-tagged via :cloud suffix)
        if model_name.endswith(":cloud") or getattr(spec, "resource_profile", "") == "cloud":
            from services.llm.cloud_adapter import CloudAdapter

            return CloudAdapter(
                model=model_name,
                base_url=base_url,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                model_spec=spec,
            )

        from services.llm.ollama_adapter import OllamaAdapter

        return OllamaAdapter(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            model_spec=spec,
        )
