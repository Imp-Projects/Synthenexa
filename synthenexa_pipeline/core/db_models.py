"""
core/db_models.py — SQLAlchemy ORM models for PostgreSQL database.

Defines database schema for:
  - experiments: Experiment metadata and run information
  - prompts: Prompt candidates across generations
  - evaluations: Evaluation results for prompts
  - snapshots: Generation state snapshots
  - run_logs: Append-only run logs
"""
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.config import Base


class Experiment(Base):
    """
    Stores experiment metadata and run information.
    
    Attributes:
        id: Primary key
        run_id: Unique identifier for the pipeline run
        experiment_name: Human-readable experiment name
        model_name: LLM model used for the experiment
        base_prompt_id: ID of the base prompt used
        created_at: Timestamp when experiment was created
        updated_at: Timestamp when experiment was last updated
        config: Experiment configuration as JSONB
        status: Current status of the experiment (running, completed, failed)
    """
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    experiment_name: Mapped[Optional[str]] = mapped_column(String(255))
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    base_prompt_id: Mapped[Optional[str]] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    config: Mapped[Optional[dict]] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(50), default="running")

    # Relationships
    prompts: Mapped[list["Prompt"]] = relationship(back_populates="experiment", cascade="all, delete-orphan")
    evaluations: Mapped[list["Evaluation"]] = relationship(back_populates="experiment", cascade="all, delete-orphan")
    snapshots: Mapped[list["Snapshot"]] = relationship(back_populates="experiment", cascade="all, delete-orphan")
    run_logs: Mapped[list["RunLog"]] = relationship(back_populates="experiment", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_experiments_created_at", "created_at"),
    )


class Prompt(Base):
    """
    Stores prompt candidates across generations.
    
    Attributes:
        id: Primary key
        prompt_id: Unique identifier for the prompt
        experiment_id: Foreign key to experiments table
        generation: Generation number for this prompt
        text: The actual prompt text
        is_base: Whether this is the base prompt
        is_best: Whether this is the best performing prompt
        created_at: Timestamp when prompt was created
    """
    __tablename__ = "prompts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prompt_id: Mapped[str] = mapped_column(String(50), nullable=False)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), nullable=False)
    generation: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    is_base: Mapped[bool] = mapped_column(Boolean, default=False)
    is_best: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))

    # Relationships
    experiment: Mapped["Experiment"] = relationship(back_populates="prompts")

    __table_args__ = (
        UniqueConstraint("prompt_id", "experiment_id", "generation", name="uq_prompt_experiment_generation"),
        Index("idx_prompts_experiment_generation", "experiment_id", "generation"),
        Index("idx_prompts_is_best", "is_best", postgresql_where=is_best == True),
    )


class Evaluation(Base):
    """
    Stores evaluation results for prompts.
    
    Attributes:
        id: Primary key
        prompt_id: ID of the prompt being evaluated
        experiment_id: Foreign key to experiments table
        generation: Generation number for this evaluation
        passed: Number of test cases passed
        failed: Number of test cases failed
        total: Total number of test cases
        syntax_errors: Number of syntax errors encountered
        errors: Detailed error information as JSONB
        created_at: Timestamp when evaluation was created
    """
    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prompt_id: Mapped[str] = mapped_column(String(50), nullable=False)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), nullable=False)
    generation: Mapped[int] = mapped_column(Integer, nullable=False)
    passed: Mapped[int] = mapped_column(Integer, nullable=False)
    failed: Mapped[int] = mapped_column(Integer, nullable=False)
    total: Mapped[int] = mapped_column(Integer, nullable=False)
    syntax_errors: Mapped[int] = mapped_column(Integer, default=0)
    errors: Mapped[Optional[list]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))

    # Relationships
    experiment: Mapped["Experiment"] = relationship(back_populates="evaluations")

    __table_args__ = (
        Index("idx_evaluations_prompt_generation", "prompt_id", "generation"),
        Index("idx_evaluations_experiment", "experiment_id"),
    )


class Snapshot(Base):
    """
    Stores generation state snapshots.
    
    Attributes:
        id: Primary key
        experiment_id: Foreign key to experiments table
        generation: Generation number for this snapshot
        best_score: Best score achieved in this generation
        best_prompt_id: ID of the best prompt
        best_prompt_text: Text of the best prompt
        scores: All candidate scores as JSONB
        reward_history: Reward history across generations as JSONB
        created_at: Timestamp when snapshot was created
    """
    __tablename__ = "snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), nullable=False)
    generation: Mapped[int] = mapped_column(Integer, nullable=False)
    best_score: Mapped[Optional[float]] = mapped_column(Float)
    best_prompt_id: Mapped[Optional[str]] = mapped_column(String(50))
    best_prompt_text: Mapped[Optional[str]] = mapped_column(Text)
    scores: Mapped[Optional[dict]] = mapped_column(JSONB)
    reward_history: Mapped[Optional[dict]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))

    # Relationships
    experiment: Mapped["Experiment"] = relationship(back_populates="snapshots")

    __table_args__ = (
        UniqueConstraint("experiment_id", "generation", name="uq_snapshot_experiment_generation"),
        Index("idx_snapshots_experiment", "experiment_id"),
    )


class RunLog(Base):
    """
    Stores append-only run logs.
    
    Attributes:
        id: Primary key
        experiment_id: Foreign key to experiments table
        log_entry: Log entry data as JSONB
        created_at: Timestamp when log was created
    """
    __tablename__ = "run_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), nullable=False)
    log_entry: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))

    # Relationships
    experiment: Mapped["Experiment"] = relationship(back_populates="run_logs")

    __table_args__ = (
        Index("idx_run_logs_experiment", "experiment_id"),
    )
