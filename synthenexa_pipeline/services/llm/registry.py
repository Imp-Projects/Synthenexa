"""Registry of all models supported by the Synthenexa pipeline.

Model name conventions
-----------------------
  openai:<model>   — OpenAI Chat Completions (e.g. openai:gpt-4o)
  gemini:<model>   — Google Gemini API      (e.g. gemini:gemini-2.0-flash)
  <model>          — Local Ollama model     (e.g. nemotron-3-super:cloud)

Priority tiers (used by _resolve_model_list in run_optimization.py)
---------------------------------------------------------------------
  1 = OpenAI models   (highest capability)
  2 = Gemini models
  3 = Nemotron cloud  (primary Ollama baseline)
  4 = Other Ollama    (local fallbacks)
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


# ------------------------------------------------------------------ #
#  Default ordered model list used when no overrides are set          #
#  Order reflects priority: best capability first.                    #
# ------------------------------------------------------------------ #

DEFAULT_SUPPORTED_MODELS = [
    # Tier 1 — OpenAI
    "openai:gpt-4o",
    "openai:gpt-4o-mini",
    # Tier 2 — Gemini
    "gemini:gemini-2.0-flash",
    "gemini:gemini-1.5-pro",
    # Tier 3 — Cloud
    "nemotron-3-super:cloud",
    "qwen3-coder-next:cloud",
    "gpt-oss:120b-cloud",
    "qwen3-coder:480b-cloud",
]


@dataclass(frozen=True)
class ModelSpec:
    """Metadata used to select and document a supported model."""

    name: str
    provider: str = "ollama"           # "openai" | "gemini" | "ollama"
    role: str = "general"
    priority: int = 4                  # 1=highest capability, 4=lowest
    default_temperature: float = 0.7
    resource_profile: str = "lightweight"
    description: str = ""


class ModelRegistry:
    """Central registry for all supported model metadata."""

    _DEFAULT_SPECS: Dict[str, ModelSpec] = {
        # ── Tier 1: OpenAI ────────────────────────────────────────────────
        "openai:gpt-4o": ModelSpec(
            name="openai:gpt-4o",
            provider="openai",
            role="primary_openai",
            priority=1,
            resource_profile="cloud",
            description="OpenAI GPT-4o — top-tier OpenAI model.",
        ),
        "openai:gpt-4o-mini": ModelSpec(
            name="openai:gpt-4o-mini",
            provider="openai",
            role="efficient_openai",
            priority=1,
            resource_profile="cloud",
            description="OpenAI GPT-4o-mini — fast and cost-effective OpenAI model.",
        ),
        "openai:o3-mini": ModelSpec(
            name="openai:o3-mini",
            provider="openai",
            role="reasoning_openai",
            priority=1,
            resource_profile="cloud",
            description="OpenAI o3-mini — compact reasoning model.",
        ),
        # ── Tier 2: Gemini ────────────────────────────────────────────────
        "gemini:gemini-2.0-flash": ModelSpec(
            name="gemini:gemini-2.0-flash",
            provider="gemini",
            role="primary_gemini",
            priority=2,
            resource_profile="cloud",
            description="Google Gemini 2.0 Flash — fast multimodal model.",
        ),
        "gemini:gemini-1.5-pro": ModelSpec(
            name="gemini:gemini-1.5-pro",
            provider="gemini",
            role="capable_gemini",
            priority=2,
            resource_profile="cloud",
            description="Google Gemini 1.5 Pro — high-capability Gemini model.",
        ),
        "gemini:gemini-1.5-flash": ModelSpec(
            name="gemini:gemini-1.5-flash",
            provider="gemini",
            role="efficient_gemini",
            priority=2,
            resource_profile="cloud",
            description="Google Gemini 1.5 Flash — fast and efficient Gemini model.",
        ),
        # ── Tier 3: Cloud Models (via Ollama) ──────────────
        "nemotron-3-super:cloud": ModelSpec(
            name="nemotron-3-super:cloud",
            provider="ollama",
            role="primary_baseline",
            priority=3,
            resource_profile="cloud",
            description="Primary Ollama baseline model for prompt evolution experiments.",
        ),
        "qwen3-coder-next:cloud": ModelSpec(
            name="qwen3-coder-next:cloud",
            provider="ollama",
            role="coder",
            priority=3,
            resource_profile="cloud",
            description="Advanced Qwen3 Coder model for cloud deployment.",
        ),
        "gpt-oss:120b-cloud": ModelSpec(
            name="gpt-oss:120b-cloud",
            provider="ollama",
            role="large_capacity",
            priority=3,
            resource_profile="cloud",
            description="High-capacity open-source GPT alternative.",
        ),
        "qwen3-coder:480b-cloud": ModelSpec(
            name="qwen3-coder:480b-cloud",
            provider="ollama",
            role="massive_capacity",
            priority=3,
            resource_profile="cloud",
            description="Massive parameter Qwen3 coder model.",
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
                    provider=_infer_provider(clean_name),
                    role="custom",
                    priority=_infer_priority(clean_name),
                    resource_profile="unknown",
                    description="Custom model from configuration.",
                ),
            )

        if specs:
            for spec in specs:
                self._specs[normalize_model_name(spec.name)] = spec

    def get(self, model_name: str) -> ModelSpec:
        """Return a model specification, falling back to a sensible default."""
        clean_name = normalize_model_name(model_name)
        if clean_name in self._specs:
            return self._specs[clean_name]
        return ModelSpec(
            name=clean_name,
            provider=_infer_provider(clean_name),
            role="custom",
            priority=_infer_priority(clean_name),
            resource_profile="unknown",
            description="Custom model selected at runtime.",
        )

    def list(self) -> List[ModelSpec]:
        """Return all configured model specifications."""
        return list(self._specs.values())

    def names(self) -> List[str]:
        """Return configured model names."""
        return list(self._specs.keys())

    def is_supported(self, model_name: str) -> bool:
        """Return whether the model is in the configured set."""
        return normalize_model_name(model_name) in self._specs


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def normalize_model_name(model_name: str) -> str:
    """Normalize model names accepted by older and newer call sites.

    Strips the legacy 'ollama:' prefix but preserves 'openai:' / 'gemini:'
    provider prefixes which carry routing information.
    """
    name = model_name.strip()
    # Only strip the ollama: prefix — keep openai: and gemini: intact.
    if name.startswith("ollama:") and not name.startswith("openai:") and not name.startswith("gemini:"):
        name = name.removeprefix("ollama:")
    return name


def _infer_provider(model_name: str) -> str:
    """Infer the provider from a normalized model name."""
    if model_name.startswith("openai:"):
        return "openai"
    if model_name.startswith("gemini:"):
        return "gemini"
    return "ollama"


def _infer_priority(model_name: str) -> int:
    """Infer the priority tier from a normalized model name."""
    if model_name.startswith("openai:"):
        return 1
    if model_name.startswith("gemini:"):
        return 2
    if "nemotron" in model_name:
        return 3
    return 4


def _split_model_list(value: str) -> List[str]:
    return [normalize_model_name(item) for item in value.split(",") if item.strip()]
