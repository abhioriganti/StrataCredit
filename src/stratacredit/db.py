"""DuckDB connection management for StrataCredit.

Provides thread-safe connection factory and convenience
functions for running SQL against Silver and Gold databases.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

import duckdb

from stratacredit.config import data_config

_LOCAL = threading.local()


def _db_path(layer: str) -> Path:
    """Resolve DuckDB file path for a data layer."""
    key = f"{layer}_db"
    raw = data_config["database"][key]
    path = Path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def get_connection(layer: str = "silver") -> Iterator[duckdb.DuckDBPyConnection]:
    """Context-manager that yields a DuckDB connection.

    Creates the file on first use.  Connections are thread-local
    to avoid concurrent write conflicts.

    Args:
        layer: "silver" or "gold"

    Yields:
        An open DuckDB connection.
    """
    attr = f"_conn_{layer}"
    conn = getattr(_LOCAL, attr, None)
    if conn is None:
        path = _db_path(layer)
        conn = duckdb.connect(str(path))
        # Install and load common extensions
        conn.execute("INSTALL parquet; LOAD parquet;")
        conn.execute("INSTALL json;   LOAD json;")
        # Create schema namespaces
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {layer};")
        setattr(_LOCAL, attr, conn)
    try:
        yield conn
    except Exception:
        # Roll back any pending transaction on error
        with suppress(Exception):
            conn.execute("ROLLBACK")
        raise


def run_sql_file(sql_path: Path, layer: str = "silver") -> None:
    """Execute a SQL file against the specified layer database."""
    sql = sql_path.read_text(encoding="utf-8")
    with get_connection(layer) as conn:
        conn.execute(sql)


def query(sql: str, layer: str = "silver") -> duckdb.DuckDBPyRelation:
    """Run a SELECT query and return a DuckDB relation."""
    with get_connection(layer) as conn:
        return conn.execute(sql)


def table_exists(table: str, schema: str = "silver", layer: str = "silver") -> bool:
    """Check whether a table exists in a DuckDB schema."""
    sql = """
        SELECT COUNT(*) > 0
        FROM information_schema.tables
        WHERE table_schema = ?
          AND table_name   = ?
    """
    with get_connection(layer) as conn:
        result = conn.execute(sql, [schema, table]).fetchone()
    return bool(result[0]) if result else False
