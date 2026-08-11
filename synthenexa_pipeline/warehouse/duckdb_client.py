"""DuckDB connection utilities for analytical queries over SQLite-backed runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import duckdb

from core.config import ROOT_DIR, SQLITE_DB_PATH


class DuckDBWarehouseClient:
    """Initialises a local DuckDB warehouse and attaches the SQLite source."""

    def __init__(
        self,
        warehouse_path: str | Path | None = None,
        sqlite_path: str | Path | None = None,
    ) -> None:
        self.sqlite_path = Path(sqlite_path or SQLITE_DB_PATH).resolve()
        default_warehouse = Path(ROOT_DIR) / "synthenexa_pipeline" / "experiments" / "warehouse.duckdb"
        self.warehouse_path = Path(warehouse_path or default_warehouse).resolve()
        self.warehouse_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> duckdb.DuckDBPyConnection:
        connection = duckdb.connect(str(self.warehouse_path))
        self._load_sqlite_extension(connection)
        attach_path = str(self.sqlite_path).replace("'", "''")
        connection.execute(f"ATTACH '{attach_path}' AS sqlite_source (TYPE sqlite, READ_ONLY);")
        return connection

    def query(self, sql: str, params: Iterable[Any] | None = None) -> list[tuple[Any, ...]]:
        with self.connect() as connection:
            result = connection.execute(sql, params or [])
            return result.fetchall()

    def to_df(self, sql: str, params: Iterable[Any] | None = None):
        with self.connect() as connection:
            return connection.execute(sql, params or []).df()

    def metadata(self) -> dict[str, str]:
        return {
            "warehouse_path": os.fspath(self.warehouse_path),
            "sqlite_path": os.fspath(self.sqlite_path),
        }

    @staticmethod
    def _load_sqlite_extension(connection: duckdb.DuckDBPyConnection) -> None:
        try:
            connection.execute("LOAD sqlite;")
        except duckdb.Error:
            connection.execute("INSTALL sqlite;")
            connection.execute("LOAD sqlite;")
