"""Registry of lightweight Ollama models supported by the project."""

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

DEFAULT_SUPPORTED_MODELS = [
    "nemotron-3-super:cloud",
    "qwen2.5:3b",
    "cogito:3b",
    "qwen:latest",
    "gemma4:latest",
]


@dataclass(frozen=True)
class ModelSpec:
    """Metadata used to select and document a supported model."""

    name: str
    provider: str = "ollama"
    role: str = "general"
    default_temperature: float = 0.7
    resource_profile: str = "lightweight"
    description: str = ""


class ModelRegistry:
    """Central registry for lightweight local Ollama model metadata."""

    _DEFAULT_SPECS: Dict[str, ModelSpec] = {
        "nemotron-3-super:cloud": ModelSpec(
            name="nemotron-3-super:cloud",
            role="primary_baseline",
            resource_profile="cloud",
            description="Primary baseline model for prompt evolution experiments.",
        ),
        "qwen2.5:3b": ModelSpec(
            name="qwen2.5:3b",
            role="general_purpose",
            resource_profile="3b",
            description="Primary small general-purpose model.",
        ),
        "cogito:3b": ModelSpec(
            name="cogito:3b",
            role="code_fallback",
            resource_profile="3b",
            description="Compact model useful for code-generation tasks.",
        ),
        "qwen:latest": ModelSpec(
            name="qwen:latest",
            role="secondary_baseline",
            resource_profile="4b",
            description="General-purpose fallback model for local experiments.",
        ),
        "gemma4:latest": ModelSpec(
            name="gemma4:latest",
            role="higher_capacity",
            resource_profile="8b",
            description="Optional higher-capacity comparison model.",
        ),
    }

    def __init__(self, specs: Optional[Iterable[ModelSpec]] = None):
        configured_names = _split_model_list(os.getenv("OLLAMA_MODEL_LIST", ""))
        names = configured_names or DEFAULT_SUPPORTED_MODELS

        self._specs: Dict[str, ModelSpec] = {}
        for name in names:
            clean_name = normalize_model_name(name)
            self._specs[clean_name] = self._DEFAULT_SPECS.get(
                clean_name,
                ModelSpec(
                    name=clean_name,
                    role="custom",
                    resource_profile="unknown",
                    description="Custom Ollama model from configuration.",
                ),
            )

        if specs:
            for spec in specs:
                self._specs[normalize_model_name(spec.name)] = spec

    def get(self, model_name: str) -> ModelSpec:
        """Return a model specification, allowing custom Ollama model names."""
        clean_name = normalize_model_name(model_name)
        if clean_name in self._specs:
            return self._specs[clean_name]
        return ModelSpec(
            name=clean_name,
            role="custom",
            resource_profile="unknown",
            description="Custom Ollama model selected at runtime.",
        )

    def list(self) -> List[ModelSpec]:
        """Return all configured model specifications."""
        return list(self._specs.values())

    def names(self) -> List[str]:
        """Return configured model names."""
        return list(self._specs.keys())

    def is_supported(self, model_name: str) -> bool:
        """Return whether the model is in the configured lightweight set."""
        return normalize_model_name(model_name) in self._specs


def normalize_model_name(model_name: str) -> str:
    """Normalize model names accepted by older and newer call sites."""
    return model_name.strip().removeprefix("ollama:")


def _split_model_list(value: str) -> List[str]:
    return [normalize_model_name(item) for item in value.split(",") if item.strip()]
