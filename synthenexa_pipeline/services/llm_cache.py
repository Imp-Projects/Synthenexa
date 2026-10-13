"""Small persistent JSON cache for deterministic prototype LLM requests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional


class LLMResponseCache:
    """File-backed cache shared by experiments without external services."""

    def __init__(self, directory: str | Path = "experiments/cache") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(*parts: Any) -> str:
        material = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[str]:
        path = self.directory / f"{key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("response")
        except (OSError, json.JSONDecodeError):
            return None

    def put(self, key: str, response: str) -> None:
        path = self.directory / f"{key}.json"
        path.write_text(json.dumps({"response": response}, ensure_ascii=False), encoding="utf-8")
