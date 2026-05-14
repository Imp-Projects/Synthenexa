import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)


def _generate_url() -> str:
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    return f"{base}/api/generate"


def _groq_chat_url() -> str:
    base = os.getenv("GROQ_BASE_URL", "https://api.groq.com").rstrip("/")
    return f"{base}/openai/v1/chat/completions"


def _has_groq() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def _groq_timeout() -> int:
    return int(os.getenv("GROQ_TIMEOUT_SECONDS", "180"))


def _call_ollama(model: str, prompt: str) -> str:
    response = requests.post(
        _generate_url(),
        json={
            "model": model or os.getenv("OLLAMA_MODEL", "qwen:latest"),
            "prompt": prompt,
            "stream": False,
        },
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("response", "")


def _call_groq(model: str, prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing")

    response = requests.post(
        _groq_chat_url(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        },
        timeout=_groq_timeout(),
    )
    response.raise_for_status()
    data = response.json()
    return (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )


def get_default_model() -> str:
    if _has_groq():
        return os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    return os.getenv("OLLAMA_MODEL", "qwen:latest")


def call_model(model: str, prompt: str) -> str:
    selected_model = model or get_default_model()

    # Force Ollama if model is prefixed with "ollama:".
    if selected_model.startswith("ollama:"):
        return _call_ollama(selected_model.split(":", 1)[1], prompt)

    # Force Groq if model is prefixed with "groq:".
    if selected_model.startswith("groq:"):
        groq_model = selected_model.split(":", 1)[1]
        try:
            return _call_groq(groq_model, prompt)
        except Exception as groq_error:
            print(f"[warn] Groq request failed, falling back to Ollama: {groq_error}")
            return _call_ollama(os.getenv("OLLAMA_MODEL", "qwen:latest"), prompt)

    # Default routing: Groq first (if configured), then fallback to Ollama.
    if _has_groq():
        try:
            return _call_groq(selected_model, prompt)
        except Exception as groq_error:
            print(f"[warn] Groq request failed, falling back to Ollama: {groq_error}")

    return _call_ollama(os.getenv("OLLAMA_MODEL", "qwen:latest"), prompt)
