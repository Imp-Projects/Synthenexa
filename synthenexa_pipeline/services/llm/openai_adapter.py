"""OpenAI adapter for the Synthenexa prompt-evolution pipeline.

Supports any model available via the OpenAI Chat Completions API, including
GPT-4o, GPT-4o-mini, o1, o3-mini, etc.

Model name convention (used throughout the pipeline):
    openai:gpt-4o          → provider=openai, model=gpt-4o
    openai:gpt-4o-mini     → provider=openai, model=gpt-4o-mini
    openai:o3-mini         → provider=openai, model=o3-mini
"""

import os
import time
from typing import Optional

from services.llm.base import BaseLLMAdapter, ProviderQuotaError
from services.llm.registry import ModelSpec


class OpenAIAdapter(BaseLLMAdapter):
    """Adapter for the OpenAI Chat Completions API."""

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        model: str,
        temperature: Optional[float] = None,
        timeout_seconds: Optional[int] = None,
        max_retries: Optional[int] = None,
        model_spec: Optional[ModelSpec] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model.removeprefix("openai:")
        self.model_spec = model_spec
        self.temperature = temperature if temperature is not None else 0.7
        self.timeout_seconds = timeout_seconds or int(os.getenv("OPENAI_TIMEOUT_SECONDS", "120"))
        self.max_retries = max_retries or int(os.getenv("OPENAI_MAX_RETRIES", "3"))
        self._api_key = api_key or os.getenv("OPENAI_API_KEY") or ""

        if not self._api_key:
            raise ValueError(
                "OpenAI API key is not configured. "
                "Set the OPENAI_API_KEY environment variable."
            )

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_retries: Optional[int] = None,
        retry_delay: float = 2.0,
        temperature: Optional[float] = None,
    ) -> str:
        """Send a prompt to the OpenAI Chat Completions API and return the response."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAIAdapter. "
                "Install it with: pip install openai"
            ) from exc

        client = OpenAI(api_key=self._api_key)
        attempts = max_retries if max_retries is not None else self.max_retries
        effective_temp = temperature if temperature is not None else self.temperature

        messages = self._build_messages(prompt=prompt, system=system)

        for attempt in range(attempts):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=effective_temp,
                    timeout=self.timeout_seconds,
                )
                raw = response.choices[0].message.content or ""
                return self.clean_output(raw)
            except Exception as exc:
                err_msg = str(exc).lower()
                if "insufficient_quota" in err_msg:
                    raise ProviderQuotaError("openai", str(exc)) from exc
                
                print(f"[OpenAIAdapter] Attempt {attempt + 1}/{attempts} failed: {exc}")
                if attempt < attempts - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    raise RuntimeError(
                        f"OpenAIAdapter.generate() failed after {attempts} attempts."
                    ) from exc

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
