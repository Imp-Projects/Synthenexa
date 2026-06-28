"""
services/llm_service.py — Single abstraction over Ollama's chat API.

Uses /api/chat (multi-turn capable) instead of the legacy /api/generate.
Mirrors the battle-tested approach in utils/ollama_client.py.
"""
import time
from typing import Optional

import requests


class LLMService:
    """
    Thin wrapper around the local Ollama /api/chat endpoint.

    Responsibilities:
      - Call Ollama API with structured chat messages
      - Handle retries with exponential back-off
      - Strip common LLM output artifacts (markdown fences)
      - Optional temperature + system prompt control
    """

    def __init__(
        self,
        model: str = "nemotron-3-super:cloud",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
    ):
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

        # Normalise model name: strip "ollama:" prefix if present
        self.model = model.removeprefix("ollama:")

    # ---------------------------------------------------------------- #
    #  Public API                                                       #
    # ---------------------------------------------------------------- #

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a prompt to Ollama and return the text response.

        Args:
            prompt:       The user-turn message.
            system:       Optional system persona / instructions.
            max_retries:  How many times to retry on failure.
            retry_delay:  Seconds between retries (doubles each attempt).
            temperature:  Per-call override; falls back to instance default.

        Returns:
            Raw string response from the model (markdown fences stripped).
        """
        url = f"{self.base_url}/api/chat"

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model":   self.model,
            "messages": messages,
            "stream":  False,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_predict": 1024,
            },
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, timeout=600)
                response.raise_for_status()
                data = response.json()
                raw = data.get("message", {}).get("content", "")
                return self._clean_output(raw)
            except Exception as exc:
                print(
                    f"[LLMService] Attempt {attempt + 1}/{max_retries} failed: {exc}"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))   # exponential back-off
                else:
                    raise RuntimeError(
                        f"LLMService.generate() failed after {max_retries} attempts."
                    ) from exc

        return ""  # unreachable, satisfies type checkers

    # ---------------------------------------------------------------- #
    #  Helpers                                                          #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _clean_output(text: str) -> str:
        """Strip markdown code fences that LLMs often wrap code in."""
        text = text.strip()
        # Remove opening fence (```python or ```)
        if text.startswith("```"):
            lines = text.splitlines()
            # Drop first line (the fence) and last line if it's a closing fence
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text
