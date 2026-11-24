"""
conftest.py
-----------
Pytest fixtures for the Healthcare Operations Analytics Platform test suite.
Provides isolated, synthetic test data and temporary test databases.
"""

import sys
from pathlib import Path

# Ensure root repository directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import pytest
from sqlalchemy import create_engine


@pytest.fixture
def sample_raw_dataframe() -> pd.DataFrame:
    """Returns a synthetic sample raw hospital DataFrame."""
    return pd.DataFrame([
        {
            "Name": "John Doe",
            "Age": 45,
            "Gender": "Male",
            "Blood Type": "O+",
            "Medical Condition": "Diabetes",
            "Date of Admission": "2024-01-10",
            "Doctor": "Dr. Sarah Smith",
            "Hospital": "City Memorial Hospital",
            "Insurance Provider": "Medicare",
            "Billing Amount": 12500.50,
            "Room Number": 101,
            "Admission Type": "Emergency",
            "Discharge Date": "2024-01-20",
            "Medication": "Aspirin",
            "Test Results": "Normal",
        },
        {
            "Name": "Jane Roe",
            "Age": 62,
            "Gender": "Female",
            "Blood Type": "A+",
            "Medical Condition": "Hypertension",
            "Date of Admission": "2024-02-01",
            "Doctor": "Dr. Robert Jones",
            "Hospital": "St. Jude Hospital",
            "Insurance Provider": "Aetna",
            "Billing Amount": 8400.00,
            "Room Number": 204,
            "Admission Type": "Elective",
            "Discharge Date": "2024-02-06",
            "Medication": "Ibuprofen",
            "Test Results": "Inconclusive",
        },
        {
            "Name": "Bob Vance",
            "Age": 33,
            "Gender": "Male",
            "Blood Type": "B-",
            "Medical Condition": "Asthma",
            "Date of Admission": "2024-03-05",
            "Doctor": "Dr. Sarah Smith",
            "Hospital": "City Memorial Hospital",
            "Insurance Provider": "Blue Cross",
            "Billing Amount": 5200.75,
            "Room Number": 305,
            "Admission Type": "Urgent",
            "Discharge Date": "2024-03-12",
            "Medication": "Penicillin",
            "Test Results": "Abnormal",
        },
    ])


@pytest.fixture
def sample_cleaned_dataframe(sample_raw_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Returns a transformed, standardized sample DataFrame."""
    from src.transform import (
        remove_duplicates,
        handle_missing_values,
        convert_data_types,
        validate_billing,
        add_department,
        add_length_of_stay,
        add_age_groups,
        add_calendar_fields,
        add_patient_id,
        standardize_column_names,
    )

    df = sample_raw_dataframe.copy()
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = convert_data_types(df)
    df = validate_billing(df)
    df = add_department(df)
    df = add_length_of_stay(df)
    df = add_age_groups(df)
    df = add_calendar_fields(df)
    df = add_patient_id(df)
    df = standardize_column_names(df)
    return df


@pytest.fixture
def temp_sqlite_engine(tmp_path: Path):
    """Provides a temporary, file-backed SQLite database engine."""
    db_file = tmp_path / "test_hospital.db"
    engine = create_engine(f"sqlite:///{db_file.as_posix()}")
    yield engine
    engine.dispose()
