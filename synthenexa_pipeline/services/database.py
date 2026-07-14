"""
services/database.py — Database connection management for PostgreSQL.

Provides:
  - DatabaseManager class for connection pooling
  - Session management with context manager
  - Connection health checks
"""
from contextlib import contextmanager
from typing import Optional

from sqlalchemy import create_engine, Engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from core.config import DatabaseConfig


class DatabaseManager:
    """
    Manages PostgreSQL database connections with connection pooling.
    
    Usage:
        config = DatabaseConfig()
        db = DatabaseManager(config)
        
        with db.get_session() as session:
            # Use session for database operations
            pass
    """

    def __init__(self, config: DatabaseConfig):
        """
        Initialize database manager with connection pool.
        
        Args:
            config: DatabaseConfig instance with connection parameters
        """
        self.config = config
        self._engine: Optional[Engine] = None
        self._SessionLocal: Optional[sessionmaker] = None
        self._initialize_engine()

    def _initialize_engine(self) -> None:
        """Create SQLAlchemy engine with connection pooling."""
        connection_string = (
            f"postgresql://{self.config.user}:{self.config.password}"
            f"@{self.config.host}:{self.config.port}/{self.config.database}"
        )
        
        self._engine = create_engine(
            connection_string,
            poolclass=QueuePool,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_pre_ping=True,  # Verify connections before using
            echo=False,  # Set to True for SQL query logging
        )
        
        self._SessionLocal = sessionmaker(
            bind=self._engine,
            autocommit=False,
            autoflush=False,
        )

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
        Check if database connection is healthy.
        
        Returns:
            bool: True if connection is healthy, False otherwise
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


# Global database manager instance (initialized on first use)
_global_db_manager: Optional[DatabaseManager] = None


def get_database_manager(config: Optional[DatabaseConfig] = None) -> DatabaseManager:
    """
    Get or create the global database manager instance.
    
    Args:
        config: Optional DatabaseConfig. If not provided, uses default from environment.
    
    Returns:
        DatabaseManager: Global database manager instance
    """
    global _global_db_manager
    
    if _global_db_manager is None:
        if config is None:
            from core.config import DatabaseConfig as DefaultConfig
            config = DefaultConfig()
        _global_db_manager = DatabaseManager(config)
    
    return _global_db_manager


def reset_database_manager() -> None:
    """Reset the global database manager instance (useful for testing)."""
    global _global_db_manager
    if _global_db_manager is not None:
        _global_db_manager.dispose()
        _global_db_manager = None
