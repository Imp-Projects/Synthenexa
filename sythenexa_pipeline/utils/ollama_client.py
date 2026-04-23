import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)


def _generate_url() -> str:
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    return f"{base}/api/generate"


def get_default_model() -> str:
    return os.getenv("OLLAMA_MODEL", "qwen:latest")


def call_model(model: str, prompt: str) -> str:
    response = requests.post(
        _generate_url(),
        json={
            "model": model or get_default_model(),
            "prompt": prompt,
            "stream": False,
        },
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("response", "")