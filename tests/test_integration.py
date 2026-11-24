"""
test_integration.py
-------------------
End-to-end integration test:
1. Ingests a synthetic raw dataset through ETL (Extract -> Transform -> Load).
2. Connects using centralized database configuration.
3. Verifies table creation and record preservation.
4. Executes read-only queries via the AI chatbot interface.
5. Verifies query results match the loaded dataset.
6. Runs ML stay risk prediction on the ingested records.
"""

from pathlib import Path
import pandas as pd
import pytest
from sqlalchemy import text, inspect

from src.config import (
    TABLE_NAME,
    get_db_engine,
    verify_database_ready,
)
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
from src.load_sql import load_to_sql
from src.rag.chatbot import execute_safe_query, ask_hospital_ai
from src.ml.predict import predict_patient_stay_risk


def test_end_to_end_etl_and_chatbot_integration(temp_sqlite_engine, monkeypatch):
    """
    Integration Test:
    Verifies that data processed by ETL is immediately queryable by the chatbot
    and that chatbot answers reflect actual loaded values.
    """
    # 1. Prepare synthetic dataset
    raw_records = pd.DataFrame([
        {
            "Name": "Alice Johnson",
            "Age": 52,
            "Gender": "Female",
            "Blood Type": "O+",
            "Medical Condition": "Diabetes",
            "Date of Admission": "2024-04-01",
            "Doctor": "Dr. Meredith Grey",
            "Hospital": "Grey Sloan Memorial",
            "Insurance Provider": "Cigna",
            "Billing Amount": 14200.00,
            "Room Number": 401,
            "Admission Type": "Emergency",
            "Discharge Date": "2024-04-18",
            "Medication": "Metformin",
            "Test Results": "Normal",
        },
        {
            "Name": "Derek Shepherd",
            "Age": 49,
            "Gender": "Male",
            "Blood Type": "AB+",
            "Medical Condition": "Hypertension",
            "Date of Admission": "2024-05-10",
            "Doctor": "Dr. Cristina Yang",
            "Hospital": "Grey Sloan Memorial",
            "Insurance Provider": "UnitedHealthcare",
            "Billing Amount": 22500.00,
            "Room Number": 502,
            "Admission Type": "Urgent",
            "Discharge Date": "2024-05-15",
            "Medication": "Lisinopril",
            "Test Results": "Normal",
        },
    ])

    # 2. Run Transformation steps
    df = remove_duplicates(raw_records)
    df = handle_missing_values(df)
    df = convert_data_types(df)
    df = validate_billing(df)
    df = add_department(df)
    df = add_length_of_stay(df)
    df = add_age_groups(df)
    df = add_calendar_fields(df)
    df = add_patient_id(df)
    df = standardize_column_names(df)

    assert len(df) == 2
    assert "patient_id" in df.columns
    assert "department" in df.columns

    # 3. Load into temporary database using safe full-refresh
    load_to_sql(df, engine=temp_sqlite_engine)

    # 4. Point centralized config to temporary engine for chatbot execution
    monkeypatch.setattr("src.rag.chatbot.get_db_engine", lambda: temp_sqlite_engine)
    monkeypatch.setattr("src.config.get_db_engine", lambda: temp_sqlite_engine)
    monkeypatch.setattr("src.config.verify_database_ready", lambda engine=temp_sqlite_engine: (True, "Ready"))

    # 5. Query via Chatbot safe execution interface
    rows = execute_safe_query(f"SELECT COUNT(*) AS total FROM {TABLE_NAME}")
    assert rows[0]["total"] == 2

    # Verify field values match loaded data
    records = execute_safe_query(f"SELECT name, doctor, billing_amount FROM {TABLE_NAME} ORDER BY billing_amount DESC")
    assert len(records) == 2
    assert records[0]["name"] == "Derek Shepherd"
    assert records[0]["doctor"] == "Dr. Cristina Yang"
    assert records[0]["billing_amount"] == 22500.00

    # 6. Verify Chatbot high-level analytics answer reflects loaded data
    answer = ask_hospital_ai("What is the total number of patients?")
    assert "2" in answer

    # 7. Verify ML Prediction on loaded records
    ml_preds = predict_patient_stay_risk(df)
    assert len(ml_preds) == 2
    assert "predicted_long_stay" in ml_preds.columns
    assert "risk_probability" in ml_preds.columns
