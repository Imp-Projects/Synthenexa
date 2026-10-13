"""
services/database.py — Database connection management for SQLite.

Provides:
  - DatabaseManager class using SQLAlchemy + SQLite
  - Session management with context manager
  - Auto-creates the database file and schema on first use
"""

import os
from contextlib import contextmanager
from typing import Optional

from core.config import DatabaseConfig
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker


class DatabaseManager:
    """
    Manages an SQLite database connection via SQLAlchemy.

    The database file is created automatically when the manager is first
    instantiated.  No server, password, or network configuration is required.

    Usage:
        config = DatabaseConfig()          # defaults to experiments/synthenexa.db
        db = DatabaseManager(config)

        with db.get_session() as session:
            # Use session for database operations
            pass
    """

    def __init__(self, config: DatabaseConfig):
        """
        Initialise the database manager and create the schema if needed.

        Args:
            config: DatabaseConfig instance holding the db_path.
        """
        self.config = config
        self._engine: Optional[Engine] = None
        self._SessionLocal: Optional[sessionmaker] = None
        self._initialize_engine()

    def _initialize_engine(self) -> None:
        """Create a SQLAlchemy engine pointed at the configured SQLite file."""
        # Ensure the parent directory exists (e.g. experiments/)
        db_dir = os.path.dirname(self.config.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        connection_string = f"sqlite:///{self.config.db_path}"

        self._engine = create_engine(
            connection_string,
            # Allow the same connection to be used across threads (safe for our
            # single-writer / multi-reader evaluation pipeline)
            connect_args={"check_same_thread": False},
            echo=False,  # Set to True for SQL query logging
        )

        self._SessionLocal = sessionmaker(
            bind=self._engine,
            autocommit=False,
            autoflush=False,
        )

    def create_all_tables(self) -> None:
        """
        Create all ORM-defined tables that do not yet exist.

        This is idempotent — calling it multiple times is safe.
        Replaces the old Alembic migration step for local development.
        """
        from core.config import Base
        from core.db_models import (  # noqa: F401 — import to register models
            Evaluation,
            Experiment,
            Prompt,
            RunLog,
            Snapshot,
        )

        Base.metadata.create_all(self._engine)
        self._apply_sqlite_compat_migrations()

    def _apply_sqlite_compat_migrations(self) -> None:
        """Backfill analysis-related columns for existing local SQLite files."""
        column_defs = {
            "evaluations": {
                "evaluator_model": "VARCHAR(255)",
                "is_base": "BOOLEAN DEFAULT 0",
                "dataset_split": "VARCHAR(50) DEFAULT 'all'",
                "task_ids": "JSON",
                "total_time_ms": "FLOAT DEFAULT 0.0",
                "min_time_ms": "FLOAT DEFAULT 0.0",
                "max_time_ms": "FLOAT DEFAULT 0.0",
                "avg_time_ms": "FLOAT DEFAULT 0.0",
            },
            "snapshots": {
                "best_ensemble_scores": "JSON",
                "model_leaderboard": "JSON",
                "ensemble_scores": "JSON",
            },
        }

        with self.engine.begin() as connection:
            for table_name, columns in column_defs.items():
                existing = {row[1] for row in connection.execute(text(f"PRAGMA table_info({table_name})")).fetchall()}
                for column_name, column_type in columns.items():
                    if column_name in existing:
                        continue
                    connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"))

    @property
    def engine(self) -> Engine:
        """Get the SQLAlchemy engine."""
        if self._engine is None:
            raise RuntimeError("Database engine not initialized")
        return self._engine

    @contextmanager
    def get_session(self) -> Session:
        """
        Context manager for database sessions.

        Automatically handles:
          - Session creation
          - Transaction commit on success
          - Transaction rollback on error
          - Session cleanup

        Yields:
            Session: SQLAlchemy session object
        """
        if self._SessionLocal is None:
            raise RuntimeError("Database session factory not initialized")

        session = self._SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def check_connection(self) -> bool:
        """
        Check if the database connection is healthy.

        Returns:
            bool: True if connection is healthy, False otherwise.
        """
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def dispose(self) -> None:
        """Dispose of the engine and close all connections."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._SessionLocal = None


# Global database manager instance (initialised on first use)
_global_db_manager: Optional[DatabaseManager] = None


def get_database_manager(config: Optional[DatabaseConfig] = None) -> DatabaseManager:
    """
    Get or create the global database manager instance.

    Args:
        config: Optional DatabaseConfig.  If not provided, uses default from environment.

    Returns:
        DatabaseManager: Global database manager instance.
    """
    global _global_db_manager

    if _global_db_manager is None:
        if config is None:
            config = DatabaseConfig()
        _global_db_manager = DatabaseManager(config)

    return _global_db_manager


def reset_database_manager() -> None:
    """Reset the global database manager instance (useful for testing)."""
    global _global_db_manager
    if _global_db_manager is not None:
        _global_db_manager.dispose()
        _global_db_manager = None
