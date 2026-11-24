"""
test_load_sql.py
----------------
Unit tests for database loading, staging table validation, safe table swap,
and failure resilience (protecting production tables).
"""

from pathlib import Path
import pandas as pd
import pytest
from sqlalchemy import text, inspect

from src.load_sql import (
    load_to_sql,
    validate_staging_table,
    swap_staging_to_production,
    TABLE_NAME,
    STAGING_TABLE_NAME,
)


def test_safe_load_to_sql_success(temp_sqlite_engine, sample_cleaned_dataframe: pd.DataFrame):
    """Verifies that full-refresh load succeeds and creates populated production table."""
    load_to_sql(sample_cleaned_dataframe, engine=temp_sqlite_engine)

    with temp_sqlite_engine.connect() as conn:
        insp = inspect(conn)
        tables = [t.lower() for t in insp.get_table_names()]
        assert TABLE_NAME.lower() in tables
        assert STAGING_TABLE_NAME.lower() not in tables

        count_res = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}"))
        assert count_res.scalar() == len(sample_cleaned_dataframe)


def test_failed_load_preserves_production_table(temp_sqlite_engine, sample_cleaned_dataframe: pd.DataFrame):
    """
    CRITICAL SAFETY TEST:
    Verifies that if a subsequent load fails validation, the previously valid
    production table remains completely untouched and intact!
    """
    # 1. Load initial valid production dataset
    load_to_sql(sample_cleaned_dataframe, engine=temp_sqlite_engine)

    with temp_sqlite_engine.connect() as conn:
        initial_count = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar()
    assert initial_count == len(sample_cleaned_dataframe)

    # 2. Attempt to load an invalid dataset (e.g. missing required columns)
    corrupted_df = pd.DataFrame([{"invalid_col": 1, "bad_data": "corrupted"}])

    with pytest.raises(Exception):
        load_to_sql(corrupted_df, engine=temp_sqlite_engine)

    # 3. Verify production table was NOT destroyed or overwritten!
    with temp_sqlite_engine.connect() as conn:
        post_count = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar()
        assert post_count == initial_count, "Safety violation: production table was modified during failed load!"


def test_validate_staging_table_missing():
    """Verifies that validation raises error if staging table does not exist."""
    from sqlalchemy import create_engine
    eng = create_engine("sqlite:///:memory:")
    with pytest.raises(ValueError) as exc:
        validate_staging_table(eng, expected_count=10, df=pd.DataFrame([{"a": 1}]))
    assert "not found" in str(exc.value)
