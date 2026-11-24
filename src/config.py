"""
config.py
---------
Centralized configuration for the Healthcare Operations Analytics Platform.
Single source of truth for:
- File paths and directories
- Dataset configuration
- Schema column mappings
- Database connection settings
- ML parameters and paths
- LLM/Chatbot parameters
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, Engine, text, inspect

# Load environment variables from .env if present
load_dotenv()

# --- Directory Paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports" / "ml"
DOCS_DIR = ROOT_DIR / "docs"

# Ensure runtime directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --- File Paths ---
RAW_CSV_PATH = RAW_DATA_DIR / "hospital_raw.csv"
CLEANED_CSV_PATH = CLEANED_DATA_DIR / "hospital_cleaned.csv"
DEFAULT_SQLITE_PATH = CLEANED_DATA_DIR / "hospital.db"
MODEL_PATH = MODELS_DIR / "provider_model.pkl"
METRICS_PATH = REPORTS_DIR / "metrics.json"

# --- Dataset Configuration ---
# Canonical dataset source
DEFAULT_KAGGLE_DATASET = "prasad22/healthcare-dataset"
KAGGLE_DATASET = os.getenv("KAGGLE_DATASET", DEFAULT_KAGGLE_DATASET)
LOCAL_CSV_PATH = os.getenv("LOCAL_CSV_PATH", None)

# Required columns in raw dataset
REQUIRED_RAW_COLUMNS = [
    "Name",
    "Age",
    "Gender",
    "Blood Type",
    "Medical Condition",
    "Date of Admission",
    "Doctor",
    "Hospital",
    "Insurance Provider",
    "Billing Amount",
    "Room Number",
    "Admission Type",
    "Discharge Date",
    "Medication",
    "Test Results",
]

# Standardized column name mapping (Raw / Derived -> snake_case)
COLUMN_MAPPING = {
    "PatientID": "patient_id",
    "Patient ID": "patient_id",
    "Name": "name",
    "Age": "age",
    "Gender": "gender",
    "Blood Type": "blood_type",
    "Medical Condition": "medical_condition",
    "Date of Admission": "admission_date",
    "Doctor": "doctor",
    "Hospital": "hospital",
    "Department": "department",
    "Insurance Provider": "insurance_provider",
    "Billing Amount": "billing_amount",
    "Room Number": "room_number",
    "Admission Type": "admission_type",
    "Discharge Date": "discharge_date",
    "Medication": "medication",
    "Test Results": "test_results",
    "Age Group": "age_group",
    "Length of Stay": "length_of_stay",
    "Admission Year": "admission_year",
    "Admission Month": "admission_month",
    "Admission Month Name": "admission_month_name",
}

STANDARDIZED_COLUMNS = list(dict.fromkeys(COLUMN_MAPPING.values()))

# --- Database Configuration ---
DB_ENGINE_TYPE = os.getenv("DB_ENGINE", "sqlite").lower()
DB_SERVER = os.getenv("DB_SERVER", "localhost")
DB_NAME = os.getenv("DB_NAME", "HospitalDB")
DB_USER = os.getenv("DB_USER", "")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_DRIVER = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
SQLITE_DB_PATH = Path(os.getenv("SQLITE_DB_PATH", str(DEFAULT_SQLITE_PATH)))

# Main table name used across ETL, SQL analysis, and Chatbot
TABLE_NAME = os.getenv("DB_TABLE_NAME", "patients")
STAGING_TABLE_NAME = f"{TABLE_NAME}_staging"


def get_db_url() -> str:
    """Return database connection URL based on environment configuration."""
    if DB_ENGINE_TYPE == "sqlserver":
        driver = DB_DRIVER.replace(" ", "+")
        if DB_USER and DB_PASSWORD:
            return f"mssql+pyodbc://{DB_USER}:{DB_PASSWORD}@{DB_SERVER}/{DB_NAME}?driver={driver}"
        return f"mssql+pyodbc://@{DB_SERVER}/{DB_NAME}?driver={driver}&trusted_connection=yes"
    return f"sqlite:///{SQLITE_DB_PATH.as_posix()}"


def get_db_engine() -> Engine:
    """Return SQLAlchemy engine for configured database."""
    url = get_db_url()
    return create_engine(url)


def verify_database_ready(engine: Engine | None = None) -> tuple[bool, str]:
    """
    Verifies that the configured database exists, is reachable, and contains
    the main patients table with records.
    Returns (is_ready, message).
    """
    if engine is None:
        try:
            engine = get_db_engine()
        except Exception as e:
            return False, f"Failed to initialize database engine: {e}"

    if DB_ENGINE_TYPE == "sqlite" and not SQLITE_DB_PATH.exists():
        return (
            False,
            f"Database file not found at '{SQLITE_DB_PATH}'. Please run the ETL pipeline first (python src/main.py).",
        )

    try:
        with engine.connect() as conn:
            insp = inspect(conn)
            table_names = insp.get_table_names()
            
            # Check table existence case-insensitively
            matching_tables = [t for t in table_names if t.lower() == TABLE_NAME.lower()]
            if not matching_tables:
                return (
                    False,
                    f"Table '{TABLE_NAME}' does not exist in database. Available tables: {table_names}. "
                    "Please run the ETL pipeline first (python src/main.py).",
                )
            
            actual_table = matching_tables[0]
            count_res = conn.execute(text(f"SELECT COUNT(*) FROM {actual_table}"))
            count = count_res.scalar()
            if count is None or count == 0:
                return (
                    False,
                    f"Table '{actual_table}' exists but is empty. Please run the ETL pipeline (python src/main.py).",
                )
            return True, f"Database is ready: table '{actual_table}' has {count} records."
    except Exception as e:
        return False, f"Database connection or verification failed: {e}"


# --- ML Configuration ---
ML_RANDOM_STATE = 42
ML_TEST_SIZE = 0.2
ML_TARGET_COL = "is_long_stay"
ML_NUMERICAL_FEATURES = ["age"]
ML_CATEGORICAL_FEATURES = ["gender", "medical_condition", "admission_type"]
ML_ALL_FEATURES = ML_NUMERICAL_FEATURES + ML_CATEGORICAL_FEATURES

# --- RAG / Chatbot Configuration ---
USE_OLLAMA = os.getenv("USE_OLLAMA", "1").lower() in {"1", "true", "yes", "on"}
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
