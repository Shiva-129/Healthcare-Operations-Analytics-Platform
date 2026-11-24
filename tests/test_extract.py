"""
test_extract.py
---------------
Unit tests for data extraction, fallback generation, and schema validation.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.extract import (
    validate_raw_schema,
    generate_synthetic_fallback,
    extract_from_local_csv,
    REQUIRED_RAW_COLUMNS,
)


def test_validate_raw_schema_success(sample_raw_dataframe: pd.DataFrame):
    """Verifies that schema validation succeeds when all expected columns are present."""
    assert validate_raw_schema(sample_raw_dataframe, source_name="test_valid") is True


def test_validate_raw_schema_missing_columns():
    """Verifies that schema validation fails with an actionable error when columns are missing."""
    incomplete_df = pd.DataFrame([{"Name": "John Doe", "Age": 45}])
    with pytest.raises(ValueError) as excinfo:
        validate_raw_schema(incomplete_df, source_name="test_missing")
    assert "Missing required columns" in str(excinfo.value)
    assert "kanakbaghel" in str(excinfo.value)


def test_validate_raw_schema_empty_df():
    """Verifies that an empty dataframe fails validation."""
    empty_df = pd.DataFrame(columns=REQUIRED_RAW_COLUMNS)
    with pytest.raises(ValueError) as excinfo:
        validate_raw_schema(empty_df, source_name="test_empty")
    assert "empty (0 rows)" in str(excinfo.value)


def test_generate_synthetic_fallback(tmp_path: Path, monkeypatch):
    """Verifies that synthetic fallback produces a non-empty, valid dataset with metadata."""
    synthetic_out = tmp_path / "synthetic_raw.csv"
    metadata_out = tmp_path / "extraction_metadata.json"

    monkeypatch.setattr("src.extract.RAW_CSV_PATH", synthetic_out)
    monkeypatch.setattr("src.extract.METADATA_FILE", metadata_out)

    path = generate_synthetic_fallback(n_rows=50)
    assert path.exists()

    df = pd.read_csv(path)
    assert len(df) >= 50
    for col in REQUIRED_RAW_COLUMNS:
        assert col in df.columns

    assert metadata_out.exists()


def test_extract_from_local_csv(tmp_path: Path, sample_raw_dataframe: pd.DataFrame, monkeypatch):
    """Verifies that local CSV extraction validates schema and copies records."""
    local_csv = tmp_path / "input_local.csv"
    dest_csv = tmp_path / "raw_dest.csv"
    metadata_out = tmp_path / "extraction_metadata.json"

    sample_raw_dataframe.to_csv(local_csv, index=False)

    monkeypatch.setattr("src.extract.RAW_CSV_PATH", dest_csv)
    monkeypatch.setattr("src.extract.METADATA_FILE", metadata_out)

    res = extract_from_local_csv(local_csv)
    assert res == dest_csv
    assert dest_csv.exists()
    loaded = pd.read_csv(dest_csv)
    assert len(loaded) == len(sample_raw_dataframe)


def test_extract_from_nonexistent_local_csv():
    """Verifies that missing local CSV raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        extract_from_local_csv(Path("non_existent_file_path_12345.csv"))
