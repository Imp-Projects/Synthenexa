"""Gemini adapter for the Synthenexa prompt-evolution pipeline.

Supports any model available via the Google Gemini API (google-genai SDK).

Model name convention (used throughout the pipeline):
    gemini:gemini-2.0-flash     → provider=gemini, model=gemini-2.0-flash
    gemini:gemini-1.5-pro       → provider=gemini, model=gemini-1.5-pro
    gemini:gemini-1.5-flash     → provider=gemini, model=gemini-1.5-flash
"""

import os
import time
from typing import Optional

from services.llm.base import BaseLLMAdapter
from services.llm.registry import ModelSpec


class GeminiAdapter(BaseLLMAdapter):
    """Adapter for the Google Gemini API via the google-genai SDK."""

    DEFAULT_MODEL = "gemini-2.0-flash"

    def __init__(
        self,
        model: str,
        temperature: Optional[float] = None,
        timeout_seconds: Optional[int] = None,
        max_retries: Optional[int] = None,
        model_spec: Optional[ModelSpec] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model.removeprefix("gemini:")
        self.model_spec = model_spec
        self.temperature = temperature if temperature is not None else 0.7
        self.timeout_seconds = timeout_seconds or int(os.getenv("GEMINI_TIMEOUT_SECONDS", "120"))
        self.max_retries = max_retries or int(os.getenv("GEMINI_MAX_RETRIES", "3"))
        self._api_key = api_key or os.getenv("GEMINI_API_KEY") or ""

        if not self._api_key:
            raise ValueError(
                "Gemini API key is not configured. "
                "Set the GEMINI_API_KEY environment variable."
            )

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_retries: Optional[int] = None,
        retry_delay: float = 2.0,
        temperature: Optional[float] = None,
    ) -> str:
        """Send a prompt to the Gemini API and return the response."""
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError(
                "google-genai package is required for GeminiAdapter. "
                "Install it with: pip install google-genai"
            ) from exc

        client = genai.Client(api_key=self._api_key)
        attempts = max_retries if max_retries is not None else self.max_retries
        effective_temp = temperature if temperature is not None else self.temperature

        for attempt in range(attempts):
            try:
                config = types.GenerateContentConfig(
                    temperature=effective_temp,
                    system_instruction=system if system else None,
                )
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                raw = response.text or ""
                return self.clean_output(raw)
            except Exception as exc:
                print(f"[GeminiAdapter] Attempt {attempt + 1}/{attempts} failed: {exc}")
                if attempt < attempts - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    raise RuntimeError(
                        f"GeminiAdapter.generate() failed after {attempts} attempts."
                    ) from exc

        return ""

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
