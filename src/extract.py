"""
extract.py
----------
Step 1 of the ETL pipeline: EXTRACT

Extracts hospital data from one of three sources:
1. Kaggle Dataset via kagglehub API (default canonical source: 'prasad22/healthcare-dataset')
2. Local CSV file (via CLI argument or LOCAL_CSV_PATH environment variable)
3. Synthetic fallback generator (offline development and testing)

Every run validates the raw dataset schema, logs the exact extraction source,
and records extraction metadata so synthetic data is clearly distinguished from real data.
"""

import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure root repository directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from src.config import (
    RAW_CSV_PATH,
    KAGGLE_DATASET,
    DEFAULT_KAGGLE_DATASET,
    LOCAL_CSV_PATH,
    REQUIRED_RAW_COLUMNS,
    DATA_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("extract")

METADATA_FILE = RAW_CSV_PATH.parent / "extraction_metadata.json"


def validate_raw_schema(df: pd.DataFrame, source_name: str) -> bool:
    """
    Validates that the DataFrame contains all required columns for the pipeline.
    Raises ValueError with actionable message if required columns are missing.
    """
    missing_cols = [col for col in REQUIRED_RAW_COLUMNS if col not in df.columns]
    if missing_cols:
        error_msg = (
            f"Schema validation failed for extraction source '{source_name}'.\n"
            f"Missing required columns ({len(missing_cols)}): {missing_cols}\n"
            f"Expected columns ({len(REQUIRED_RAW_COLUMNS)}): {REQUIRED_RAW_COLUMNS}\n"
            f"Found columns ({len(df.columns)}): {list(df.columns)}\n"
            f"Note: This pipeline is designed for the canonical single-table hospital dataset "
            f"('{DEFAULT_KAGGLE_DATASET}'). Incompatible multi-table datasets "
            f"(e.g., 'kanakbaghel/hospital-management-dataset') cannot be ingested directly without pre-joining."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    if len(df) == 0:
        raise ValueError(f"Extracted dataset from '{source_name}' is empty (0 rows).")

    logger.info("Schema validation successful: all %d required columns are present.", len(REQUIRED_RAW_COLUMNS))
    return True


def save_extraction_metadata(source_type: str, source_detail: str, row_count: int, is_synthetic: bool):
    """Saves metadata about the extraction run to clearly identify data provenance."""
    metadata = {
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_type": source_type,
        "source_detail": source_detail,
        "row_count": row_count,
        "is_synthetic": is_synthetic,
        "required_columns_validated": REQUIRED_RAW_COLUMNS,
    }
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Extraction metadata saved to %s", METADATA_FILE)


def extract_from_local_csv(csv_path: str | Path) -> Path:
    """Extracts data from a local CSV file after validating schema."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Local CSV file not found at: {path}")

    logger.info("[EXTRACTION SOURCE: LOCAL_CSV] Loading data from '%s'...", path)
    df = pd.read_csv(path)
    validate_raw_schema(df, source_name=f"local_csv:{path}")

    RAW_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_CSV_PATH, index=False)
    save_extraction_metadata(
        source_type="LOCAL_CSV",
        source_detail=str(path.resolve()),
        row_count=len(df),
        is_synthetic=False,
    )
    logger.info("Successfully copied %d rows from local CSV to %s", len(df), RAW_CSV_PATH)
    return RAW_CSV_PATH


def download_from_kaggle(dataset_name: str | None = None) -> Path | None:
    """
    Attempt to download the dataset using kagglehub.
    Returns path to CSV or None on failure.
    """
    dataset_name = dataset_name or KAGGLE_DATASET
    try:
        import kagglehub

        logger.info("[EXTRACTION SOURCE: KAGGLE] Attempting download of dataset '%s'...", dataset_name)
        dataset_path = kagglehub.dataset_download(dataset_name)
        logger.info("Dataset downloaded by kagglehub to: %s", dataset_path)

        csv_files = list(Path(dataset_path).glob("*.csv"))
        if not csv_files:
            logger.warning("No CSV files found in downloaded Kaggle dataset '%s'.", dataset_name)
            return None

        # Prefer healthcare_dataset.csv or pick the first CSV
        source_csv = csv_files[0]
        for f in csv_files:
            if "healthcare" in f.name.lower() or "hospital" in f.name.lower():
                source_csv = f
                break

        logger.info("Found Kaggle CSV file: %s", source_csv)
        df = pd.read_csv(source_csv)

        # Validate schema before copying
        validate_raw_schema(df, source_name=f"kaggle:{dataset_name}")

        RAW_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_csv, RAW_CSV_PATH)
        save_extraction_metadata(
            source_type="KAGGLE",
            source_detail=dataset_name,
            row_count=len(df),
            is_synthetic=False,
        )
        logger.info("Copied raw data to: %s (%d records)", RAW_CSV_PATH, len(df))
        return RAW_CSV_PATH

    except Exception as e:
        logger.warning("Kaggle download or validation failed: %s", e)
        return None


def generate_synthetic_fallback(n_rows: int = 5000) -> Path:
    """
    Generates a synthetic fallback dataset matching the schema of the canonical dataset
    so the pipeline is fully runnable offline. Clearly flagged as SYNTHETIC.
    """
    import random
    from datetime import timedelta
    import numpy as np

    logger.warning(
        "[EXTRACTION SOURCE: SYNTHETIC_FALLBACK] Falling back to synthetic data generation (%d rows). "
        "THIS DATA IS SYNTHETIC AND MUST NOT BE USED FOR REAL CLINICAL OR OPERATIONAL CLAIMS.",
        n_rows,
    )

    try:
        from faker import Faker
        fake = Faker()
        Faker.seed(42)
    except ImportError:
        logger.error("Faker is not installed. Run 'pip install Faker'.")
        raise

    np.random.seed(42)
    random.seed(42)

    genders = ["Male", "Female"]
    blood_types = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
    conditions = ["Diabetes", "Hypertension", "Asthma", "Cancer", "Obesity", "Arthritis"]
    doctors = [fake.name() for _ in range(40)]
    hospitals = [f"{fake.last_name()} Medical Center" for _ in range(15)]
    insurance_providers = ["Aetna", "Blue Cross", "Cigna", "Medicare", "UnitedHealthcare"]
    admission_types = ["Emergency", "Elective", "Urgent"]
    medications = ["Aspirin", "Ibuprofen", "Penicillin", "Paracetamol", "Lipitor"]
    test_results = ["Normal", "Abnormal", "Inconclusive"]
    departments = [
        "Cardiology",
        "Neurology",
        "Orthopedics",
        "Pediatrics",
        "Oncology",
        "General Medicine",
        "Emergency",
    ]

    start_date = pd.Timestamp("2022-01-01")
    end_date = pd.Timestamp("2024-12-31")
    date_range_days = (end_date - start_date).days

    rows = []
    for _ in range(n_rows):
        admission_date = start_date + timedelta(days=random.randint(0, date_range_days))
        stay_length = random.randint(1, 20)
        discharge_date = admission_date + timedelta(days=stay_length)

        rows.append({
            "Name": fake.name(),
            "Age": random.randint(0, 95),
            "Gender": random.choice(genders),
            "Blood Type": random.choice(blood_types),
            "Medical Condition": random.choice(conditions),
            "Date of Admission": admission_date.strftime("%Y-%m-%d"),
            "Doctor": random.choice(doctors),
            "Hospital": random.choice(hospitals),
            "Department": random.choice(departments),
            "Insurance Provider": random.choice(insurance_providers),
            "Billing Amount": round(random.uniform(500, 55000), 2),
            "Room Number": random.randint(100, 999),
            "Admission Type": random.choice(admission_types),
            "Discharge Date": discharge_date.strftime("%Y-%m-%d"),
            "Medication": random.choice(medications),
            "Test Results": random.choice(test_results),
        })

    df = pd.DataFrame(rows)
    # Intentionally inject duplicates and missing values for transform step to clean
    dup_sample = df.sample(frac=0.02, random_state=1)
    df = pd.concat([df, dup_sample], ignore_index=True)

    for col in ["Billing Amount", "Insurance Provider", "Age"]:
        missing_idx = df.sample(frac=0.01, random_state=2).index
        df.loc[missing_idx, col] = None

    validate_raw_schema(df, source_name="synthetic_fallback")

    RAW_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_CSV_PATH, index=False)
    save_extraction_metadata(
        source_type="SYNTHETIC_FALLBACK",
        source_detail="In-memory synthetic generator (Faker seed=42)",
        row_count=len(df),
        is_synthetic=True,
    )
    logger.info("Synthetic raw dataset written to: %s (%d rows)", RAW_CSV_PATH, len(df))
    return RAW_CSV_PATH


def extract_data(source_path: str | Path | None = None, kaggle_dataset: str | None = None) -> Path:
    """
    Main extraction orchestrator:
    1. If source_path or LOCAL_CSV_PATH is provided, extract from local CSV.
    2. Otherwise, attempt Kaggle download using configured dataset.
    3. If Kaggle download fails or is unavailable, use synthetic fallback generator.
    """
    # 1. Local CSV priority
    local_path = source_path or LOCAL_CSV_PATH
    if local_path:
        logger.info("Attempting extraction from specified local CSV: %s", local_path)
        return extract_from_local_csv(local_path)

    # 2. Kaggle download
    target_kaggle = kaggle_dataset or KAGGLE_DATASET
    downloaded_path = download_from_kaggle(target_kaggle)
    if downloaded_path is not None:
        return downloaded_path

    # 3. Synthetic fallback
    return generate_synthetic_fallback()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract hospital dataset from Kaggle or local source")
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to local raw CSV file")
    parser.add_argument("--dataset", "-d", type=str, default=None, help="Kaggle dataset identifier")
    args = parser.parse_args()

    path = extract_data(source_path=args.input, kaggle_dataset=args.dataset)
    df = pd.read_csv(path)
    logger.info("Extraction complete. Raw dataset shape: %s", df.shape)
    return path


if __name__ == "__main__":
    main()
