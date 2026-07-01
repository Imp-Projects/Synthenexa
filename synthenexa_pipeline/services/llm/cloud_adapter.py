"""Compatibility adapter for cloud-hosted models such as nemotron-3-super:cloud."""

import os
from typing import Optional

import requests

from services.llm.base import BaseLLMAdapter
from services.llm.registry import ModelSpec, normalize_model_name


class CloudAdapter(BaseLLMAdapter):
    """Send requests to the configured cloud/Ollama host endpoint for cloud-tagged models."""

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout_seconds: Optional[int] = None,
        max_retries: Optional[int] = None,
        model_spec: Optional[ModelSpec] = None,
    ):
        self.model = normalize_model_name(model)
        self.model_spec = model_spec
        self.base_url = self._resolve_base_url(base_url).rstrip("/")
        self.temperature = (
            temperature if temperature is not None else (model_spec.default_temperature if model_spec else 0.7)
        )
        self.timeout_seconds = timeout_seconds or int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "600"))
        self.max_retries = max_retries or int(os.getenv("OLLAMA_MAX_RETRIES", "3"))

    @staticmethod
    def _resolve_base_url(base_url: Optional[str] = None) -> str:
        candidates = []
        if base_url:
            candidates.append(base_url)
        for env_name in ("OLLAMA_BASE_URL", "OLLAMA_API_BASE"):
            value = os.getenv(env_name)
            if value and value not in candidates:
                candidates.append(value)
        for fallback in ("http://host.docker.internal:11434", "http://127.0.0.1:11434", "http://localhost:11434"):
            if fallback not in candidates:
                candidates.append(fallback)

        for candidate in candidates:
            try:
                response = requests.get(f"{candidate.rstrip('/')}/api/tags", timeout=2)
                if response.ok:
                    return candidate.rstrip("/")
            except Exception:
                continue
        return candidates[0].rstrip("/") if candidates else "http://host.docker.internal:11434"

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_retries: Optional[int] = None,
        retry_delay: float = 2.0,
        temperature: Optional[float] = None,
    ) -> str:
        from services.llm.ollama_adapter import OllamaAdapter

        adapter = OllamaAdapter(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature if temperature is None else temperature,
            timeout_seconds=self.timeout_seconds,
            max_retries=max_retries if max_retries is not None else self.max_retries,
            model_spec=self.model_spec,
        )
        return adapter.generate(
            prompt=prompt, system=system, max_retries=max_retries, retry_delay=retry_delay, temperature=temperature
        )
