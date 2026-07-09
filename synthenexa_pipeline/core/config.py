"""
core/config.py — Central configuration for the entire Synthenexa pipeline.

Provides both:
  1. Module-level constants (simple access anywhere via `from core.config import X`)
  2. Pydantic config classes (for structured experiment configs, serialisation)
"""

import os
from typing import Dict, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sqlalchemy.orm import DeclarativeBase

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(ROOT_DIR, ".env"), override=False)

# ------------------------------------------------------------------ #
#  Module-level constants (quick access)                              #
# ------------------------------------------------------------------ #

MODEL_NAME = os.getenv("OLLAMA_DEFAULT_MODEL") or os.getenv("OLLAMA_MODEL") or "nemotron-3-super:cloud"
TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))

NUM_MUTATIONS = 5
MAX_WORKERS = 4

DATASET_PATH = "Dataset/benchmark_mbpp_micro.jsonl"
DEFAULT_SUPPORTED_MODELS = [
    item.strip()
    for item in os.getenv(
        "OLLAMA_MODEL_LIST",
        "nemotron-3-super:cloud,cogito:3b,qwen2.5:3b,qwen:latest,gemma4:latest",
    ).split(",")
    if item.strip()
]


# PostgreSQL configuration
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "synthenexa")
POSTGRES_USER = os.getenv("POSTGRES_USER", "synthenexa")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "synthenexa")

# Storage backend selection: "file" (default) or "postgres"
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "file")


# ------------------------------------------------------------------ #
#  Pydantic config classes (structured / serialisable)               #
# ------------------------------------------------------------------


# SQLAlchemy Base class for ORM models
class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


class ModelConfig(BaseModel):
    provider: str = "ollama"
    model_name: str = MODEL_NAME
    temperature: float = TEMPERATURE
    max_tokens: Optional[int] = None


class PipelineConfig(BaseModel):
    num_generations: int = 3
    batch_size: int = 1
    max_workers: int = MAX_WORKERS


class MutationConfig(BaseModel):
    num_mutations: int = NUM_MUTATIONS
    mutation_strength: float = 1.0
    strategy_type: str = "default"


class EvaluationConfig(BaseModel):
    pass_at_k: int = 1
    timeout: float = 10.0
    retries: int = 0


class RLConfig(BaseModel):
    policy_type: str = "greedy"
    epsilon: float = 0.1
    reward_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "pass_at_1": 1.0,
            "pass_at_k": 0.0,
            "diversity_bonus": 0.0,
        }
    )


class DatabaseConfig(BaseModel):
    """Configuration for PostgreSQL database connection."""
    host: str = POSTGRES_HOST
    port: int = int(POSTGRES_PORT)
    database: str = POSTGRES_DB
    user: str = POSTGRES_USER
    password: str = POSTGRES_PASSWORD
    pool_size: int = 5
    max_overflow: int = 10


class ExperimentConfig(BaseModel):
    experiment_name: str = "default_run"
    dataset_path: str = DATASET_PATH

    model: ModelConfig = Field(default_factory=ModelConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    mutation: MutationConfig = Field(default_factory=MutationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)

    def to_json(self) -> str:
        return self.model_dump_json(indent=4)

    @classmethod
    def from_json(cls, json_str: str) -> "ExperimentConfig":
        return cls.model_validate_json(json_str)

    def save_to_file(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def load_from_file(cls, filepath: str) -> "ExperimentConfig":
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_json(f.read())
