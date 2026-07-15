import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)


def get_ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

def _chat_url() -> str:
    return f"{get_ollama_base_url()}/api/chat"

def get_available_ollama_models() -> list[str]:
    try:
        url = f"{get_ollama_base_url()}/api/tags"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        print(f"[warn] Failed to fetch Ollama models: {e}")
        return []


def _groq_chat_url() -> str:
    base = os.getenv("GROQ_BASE_URL", "https://api.groq.com").rstrip("/")
    return f"{base}/openai/v1/chat/completions"


def _has_groq() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def _groq_timeout() -> int:
    return int(os.getenv("GROQ_TIMEOUT_SECONDS", "180"))

def get_default_model() -> str:
    if _has_groq():
        return os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    return os.getenv("OLLAMA_MODEL", "nemotron-3-super:cloud")


def _call_ollama(
    model: str,
    prompt: str,
    system: str | None = None,
    options: dict | None = None,
) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # Default options for Ollama: low temperature for deterministic code
    default_opts = {
        "temperature": 0.0,
        "top_p": 0.9,
        "stop": ["```"],
        "num_predict": 512,
    }
    if options:
        default_opts.update(options)

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": default_opts,
    }

    response = requests.post(
        _chat_url(),
        json=payload,
        timeout=600,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("message", {}).get("content", "")


def _call_groq(
    model: str,
    prompt: str,
    options: dict | None = None,
) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing")

    # Default options for Groq: deterministic
    default_opts = {
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": 512,
        "stop": ["```"],
    }
    if options:
        default_opts.update(options)

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        **default_opts,
    }

    response = requests.post(
        _groq_chat_url(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=_groq_timeout(),
    )
    response.raise_for_status()
    data = response.json()
    return (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )


def call_model(model: str | None, prompt: str, system: str | None = None, options: dict | None = None) -> str:
    # 1. User explicitly specified a model: NO fallback.
    if model:
        if model.startswith("groq:"):
            return _call_groq(model.split(":", 1)[1], prompt, options)
        elif model.startswith("ollama:"):
            return _call_ollama(model.split(":", 1)[1], prompt, system, options)
        else:
            # By default, assume unprefixed explicitly passed models are Ollama models.
            return _call_ollama(model, prompt, system, options)

    # 2. No model specified: Use the default fallback chain.
    if _has_groq():
        groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        try:
            return _call_groq(groq_model, prompt, options)
        except Exception as groq_error:
            print(f"[warn] Groq request failed, falling back to Ollama: {groq_error}")

    # Fallback / Default to Ollama
    ollama_model = os.getenv("OLLAMA_MODEL", "nemotron-3-super:cloud")
    return _call_ollama(ollama_model, prompt, system, options)