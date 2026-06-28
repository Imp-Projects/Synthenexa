"""
core/models.py — Lightweight runtime objects (NOT storage schemas).

These are used internally at runtime and are NOT serialised to disk.
For serialisation, use core/schema.py (Pydantic models).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Prompt:
    """
    Lightweight runtime representation of a prompt candidate.
    Distinct from core.schema.PromptCandidate (the Pydantic contract).
    """
    prompt_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Task:
    """Runtime wrapper for a benchmark task (MBPP / HumanEval)."""
    task_id: str
    prompt: str
    entry_point: str
    test: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationResult:
    """
    Runtime wrapper for one code-generation output from the LLM.
    Replaces the old CompletionResult name.
    """
    task_id: str
    completion: str
    raw_completion: str | None = None
    prompt_id: str | None = None
    score: float | None = None
    error_category: str = "ok"
    tests_passed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
