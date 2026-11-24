"""
test_config.py
--------------
Unit tests for centralized configuration resolution, database URLs, and readiness checks.
"""

from pathlib import Path
import pytest
from sqlalchemy import create_engine, text

from src.config import (
    get_db_url,
    get_db_engine,
    verify_database_ready,
    TABLE_NAME,
    SQLITE_DB_PATH,
)


def test_get_db_url_sqlite():
    """Verifies that SQLite URL is correctly constructed."""
    url = get_db_url()
    assert url.startswith("sqlite:///")


def test_get_db_engine():
    """Verifies that SQLAlchemy engine can be instantiated."""
    engine = get_db_engine()
    assert engine is not None


def test_verify_database_ready_missing_table(temp_sqlite_engine):
    """Verifies readiness check fails gracefully if table is absent."""
    is_ready, msg = verify_database_ready(temp_sqlite_engine)
    assert is_ready is False
    assert "does not exist" in msg


def test_verify_database_ready_success(temp_sqlite_engine):
    """Verifies readiness check succeeds when table exists and has rows."""
    with temp_sqlite_engine.begin() as conn:
        conn.execute(text(f"CREATE TABLE {TABLE_NAME} (id INT, name TEXT)"))
        conn.execute(text(f"INSERT INTO {TABLE_NAME} VALUES (1, 'Test Patient')"))

    is_ready, msg = verify_database_ready(temp_sqlite_engine)
    assert is_ready is True
    assert "is ready" in msg
