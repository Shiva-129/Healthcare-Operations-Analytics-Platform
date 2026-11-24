"""
transform.py
------------
Step 2 of the ETL pipeline: TRANSFORM

Cleans, enriches, and standardizes the raw hospital dataset:
- Validates required input columns
- Removes duplicate records
- Handles missing values with robust defaults and medians
- Converts data types (dates, numerics)
- Standardizes date formats and calculates length of stay
- Derives or preserves Department based on clinical context
- Validates billing values (non-negative, non-zero)
- Generates clinical age groups and calendar dimensions
- Assigns unique integer patient_id
- Standardizes all columns to clean snake_case identifiers
"""

import logging
import sys
from pathlib import Path

# Ensure root repository directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from src.config import (
    RAW_CSV_PATH,
    CLEANED_CSV_PATH,
    COLUMN_MAPPING,
    STANDARDIZED_COLUMNS,
    REQUIRED_RAW_COLUMNS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("transform")

CONDITION_TO_DEPARTMENT = {
    "Diabetes": "Endocrinology",
    "Hypertension": "Cardiology",
    "Asthma": "Pulmonology",
    "Cancer": "Oncology",
    "Obesity": "General Medicine",
    "Arthritis": "Orthopedics",
}


def load_raw(path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    """Loads raw dataset from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at {path}. Run extract.py first.")
    df = pd.read_csv(path)
    logger.info("Loaded raw data from '%s': shape %s", path, df.shape)
    return df


def validate_raw_input(df: pd.DataFrame) -> pd.DataFrame:
    """Validates that necessary raw columns exist."""
    missing = [col for col in REQUIRED_RAW_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Raw data is missing required columns: {missing}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicate rows while preserving the first occurrence."""
    before = len(df)
    df = df.drop_duplicates()
    logger.info("Removed %d duplicate rows (%d -> %d)", before - len(df), before, len(df))
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Imputes non-critical columns and drops rows missing primary identifiers."""
    df = df.copy()

    # Median imputation for numeric features
    if "Age" in df.columns:
        median_age = df["Age"].median()
        df["Age"] = df["Age"].fillna(median_age)

    if "Billing Amount" in df.columns:
        median_billing = df["Billing Amount"].median()
        df["Billing Amount"] = df["Billing Amount"].fillna(median_billing)

    if "Insurance Provider" in df.columns:
        df["Insurance Provider"] = df["Insurance Provider"].fillna("Unknown")

    if "Room Number" in df.columns:
        df["Room Number"] = df["Room Number"].fillna(0)

    # Drop rows missing critical hospital operations fields
    critical_cols = [c for c in ["Name", "Date of Admission", "Discharge Date", "Doctor", "Hospital"] if c in df.columns]
    before = len(df)
    df = df.dropna(subset=critical_cols)
    logger.info("Dropped %d rows missing critical fields %s", before - len(df), critical_cols)
    return df


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Converts dates to datetime and numbers to appropriate types."""
    df = df.copy()
    df["Date of Admission"] = pd.to_datetime(df["Date of Admission"], errors="coerce")
    df["Discharge Date"] = pd.to_datetime(df["Discharge Date"], errors="coerce")

    # Drop rows where dates could not be parsed
    before = len(df)
    df = df.dropna(subset=["Date of Admission", "Discharge Date"])
    if before > len(df):
        logger.info("Dropped %d rows with unparseable dates", before - len(df))

    df["Age"] = df["Age"].astype(int)
    df["Billing Amount"] = df["Billing Amount"].astype(float).round(2)
    df["Room Number"] = df["Room Number"].astype(int)
    return df


def validate_billing(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures billing amounts are positive and non-zero."""
    before = len(df)
    df = df[df["Billing Amount"] > 0]
    logger.info("Filtered %d non-positive billing records (%d -> %d)", before - len(df), before, len(df))
    return df


def add_department(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures Department is present. If missing from the raw dataset
    (e.g., in canonical prasad22/healthcare-dataset), derives it from Medical Condition.
    """
    df = df.copy()
    if "Department" not in df.columns or df["Department"].isnull().all():
        logger.info("Deriving 'Department' column from 'Medical Condition' mapping...")
        df["Department"] = df["Medical Condition"].map(CONDITION_TO_DEPARTMENT).fillna("General Medicine")
    else:
        df["Department"] = df["Department"].fillna("General Medicine")
    return df


def add_length_of_stay(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Length of Stay in integer days.
    Filters out physically impossible negative stays (discharge prior to admission).
    """
    df = df.copy()
    raw_stay = (df["Discharge Date"] - df["Date of Admission"]).dt.days
    before = len(df)
    # Filter out records where discharge precedes admission
    valid_mask = raw_stay >= 0
    df = df[valid_mask].copy()
    if before > len(df):
        logger.warning("Filtered %d records where discharge date preceded admission date", before - len(df))

    df["Length of Stay"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days
    return df


def add_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Generates standard clinical demographic age groups."""
    df = df.copy()
    bins = [-1, 12, 19, 35, 50, 65, 120]
    labels = [
        "Child (0-12)",
        "Teen (13-19)",
        "Young Adult (20-35)",
        "Adult (36-50)",
        "Middle Age (51-65)",
        "Senior (65+)",
    ]
    df["Age Group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=True).astype(str)
    return df


def add_calendar_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts admission calendar dimensions for time-series analysis."""
    df = df.copy()
    df["Admission Year"] = df["Date of Admission"].dt.year
    df["Admission Month"] = df["Date of Admission"].dt.month
    df["Admission Month Name"] = df["Date of Admission"].dt.strftime("%B")
    return df


def add_patient_id(df: pd.DataFrame) -> pd.DataFrame:
    """Assigns sequential integer patient_id primary key."""
    df = df.reset_index(drop=True)
    if "PatientID" in df.columns:
        df = df.drop(columns=["PatientID"])
    df.insert(0, "PatientID", range(1, len(df) + 1))
    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Renames all columns to standardized snake_case identifiers."""
    df = df.copy()
    rename_dict = {col: COLUMN_MAPPING[col] for col in df.columns if col in COLUMN_MAPPING}
    df = df.rename(columns=rename_dict)

    # Format date columns back to ISO string for CSV/SQL portability
    if "admission_date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["admission_date"]):
        df["admission_date"] = df["admission_date"].dt.strftime("%Y-%m-%d")
    if "discharge_date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["discharge_date"]):
        df["discharge_date"] = df["discharge_date"].dt.strftime("%Y-%m-%d")

    logger.info("Standardized columns to snake_case: %s", list(df.columns))
    return df


def run_transform(raw_path: Path = RAW_CSV_PATH, output_path: Path = CLEANED_CSV_PATH) -> pd.DataFrame:
    """Runs the complete transformation pipeline and saves cleaned CSV."""
    logger.info("Starting transformation pipeline...")
    df = load_raw(raw_path)
    df = validate_raw_input(df)
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Transformation complete. Cleaned dataset saved to '%s' with shape %s", output_path, df.shape)
    return df


if __name__ == "__main__":
    run_transform()
