"""
test_transform.py
-----------------
Unit tests for data transformation, deduplication, missing value imputation,
length of stay validation, and snake_case schema standardization.
"""

from pathlib import Path
import pandas as pd
import pytest

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


def test_remove_duplicates(sample_raw_dataframe: pd.DataFrame):
    """Verifies that duplicate rows are cleanly removed."""
    duplicated = pd.concat([sample_raw_dataframe, sample_raw_dataframe.iloc[[0]]], ignore_index=True)
    assert len(duplicated) == len(sample_raw_dataframe) + 1
    cleaned = remove_duplicates(duplicated)
    assert len(cleaned) == len(sample_raw_dataframe)


def test_handle_missing_values():
    """Verifies that missing numericals are imputed and critical missing fields are dropped."""
    df = pd.DataFrame([
        {
            "Name": "Valid Patient",
            "Age": None,
            "Billing Amount": None,
            "Insurance Provider": None,
            "Date of Admission": "2024-01-01",
            "Discharge Date": "2024-01-05",
            "Doctor": "Dr. House",
            "Hospital": "County General",
        },
        {
            "Name": None,  # Critical missing
            "Age": 50,
            "Billing Amount": 1000.0,
            "Date of Admission": "2024-01-01",
            "Discharge Date": "2024-01-05",
            "Doctor": "Dr. House",
            "Hospital": "County General",
        },
    ])
    cleaned = handle_missing_values(df)
    assert len(cleaned) == 1
    assert not pd.isnull(cleaned["Age"].iloc[0])
    assert cleaned["Insurance Provider"].iloc[0] == "Unknown"


def test_validate_billing():
    """Verifies that negative or zero billing records are dropped."""
    df = pd.DataFrame([
        {"Billing Amount": 500.0},
        {"Billing Amount": 0.0},
        {"Billing Amount": -250.0},
    ])
    cleaned = validate_billing(df)
    assert len(cleaned) == 1
    assert cleaned["Billing Amount"].iloc[0] == 500.0


def test_add_department_derived():
    """Verifies that department is derived when absent from raw data."""
    df = pd.DataFrame([
        {"Medical Condition": "Diabetes"},
        {"Medical Condition": "Cancer"},
        {"Medical Condition": "Unknown Condition"},
    ])
    res = add_department(df)
    assert res["Department"].iloc[0] == "Endocrinology"
    assert res["Department"].iloc[1] == "Oncology"
    assert res["Department"].iloc[2] == "General Medicine"


def test_add_length_of_stay_filter_negative():
    """Verifies that negative stays (discharge prior to admission) are filtered."""
    df = pd.DataFrame([
        {"Date of Admission": pd.to_datetime("2024-01-01"), "Discharge Date": pd.to_datetime("2024-01-10")},
        {"Date of Admission": pd.to_datetime("2024-01-10"), "Discharge Date": pd.to_datetime("2024-01-01")},  # Negative stay
    ])
    res = add_length_of_stay(df)
    assert len(res) == 1
    assert res["Length of Stay"].iloc[0] == 9


def test_standardize_column_names(sample_cleaned_dataframe: pd.DataFrame):
    """Verifies that output column names are standardized snake_case."""
    cols = sample_cleaned_dataframe.columns.tolist()
    assert "patient_id" in cols
    assert "billing_amount" in cols
    assert "admission_date" in cols
    assert "length_of_stay" in cols
    assert "medical_condition" in cols
    for c in cols:
        assert " " not in c
        assert c == c.lower()
