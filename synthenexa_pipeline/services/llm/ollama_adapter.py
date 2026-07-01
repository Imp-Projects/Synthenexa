"""Ollama adapter used by the prompt-evolution pipeline."""

import os
import time
from typing import Optional

import requests

from services.llm.base import BaseLLMAdapter
from services.llm.registry import ModelSpec, normalize_model_name


class OllamaAdapter(BaseLLMAdapter):
    """Adapter for the local Ollama `/api/chat` endpoint."""

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout_seconds: Optional[int] = None,
        max_retries: Optional[int] = None,
        model_spec: Optional[ModelSpec] = None,
    ):
        self.base_url = self._resolve_base_url(base_url).rstrip("/")
        self.model = normalize_model_name(model)
        self.model_spec = model_spec
        self.temperature = (
            temperature if temperature is not None else (model_spec.default_temperature if model_spec else 0.7)
        )
        self.timeout_seconds = timeout_seconds or int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "600"))
        self.max_retries = max_retries or int(os.getenv("OLLAMA_MAX_RETRIES", "3"))

    @staticmethod
    def _resolve_base_url(base_url: Optional[str] = None) -> str:
        """Choose a reachable Ollama base URL, falling back when a configured host is unavailable."""
        candidates = []
        if base_url:
            candidates.append(base_url)
        for env_name in ("OLLAMA_BASE_URL", "OLLAMA_API_BASE"):
            value = os.getenv(env_name)
            if value and value not in candidates:
                candidates.append(value)
        for fallback in ("http://127.0.0.1:11434", "http://localhost:11434", "http://host.docker.internal:11434"):
            if fallback not in candidates:
                candidates.append(fallback)

        for candidate in candidates:
            try:
                response = requests.get(f"{candidate.rstrip('/')}/api/tags", timeout=2)
                if response.ok:
                    return candidate.rstrip("/")
            except Exception:
                continue

        return candidates[0].rstrip("/") if candidates else "http://localhost:11434"

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_retries: Optional[int] = None,
        retry_delay: float = 2.0,
        temperature: Optional[float] = None,
    ) -> str:
        """Send a prompt to Ollama and return the cleaned text response."""
        url = f"{self.base_url}/api/chat"
        attempts = max_retries if max_retries is not None else self.max_retries

        payload = {
            "model": self.model,
            "messages": self._build_messages(prompt=prompt, system=system),
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_predict": 1024,
            },
        }

        for attempt in range(attempts):
            try:
                response = requests.post(url, json=payload, timeout=self.timeout_seconds)
                if not response.ok:
                    detail = response.text.strip()
                    raise RuntimeError(f"{response.status_code} {response.reason}: {detail[:300]}")
                data = response.json()
                raw = data.get("message", {}).get("content", "")
                return self.clean_output(raw)
            except Exception as exc:
                print(f"[OllamaAdapter] Attempt {attempt + 1}/{attempts} failed: {exc}")
                if attempt < attempts - 1:
                    time.sleep(retry_delay * (2**attempt))
                else:
                    raise RuntimeError(f"OllamaAdapter.generate() failed after {attempts} attempts.") from exc

        return ""

    @staticmethod
    def _build_messages(prompt: str, system: Optional[str] = None) -> list:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def clean_output(text: str) -> str:
        """Strip markdown code fences that LLMs often wrap code in."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text
