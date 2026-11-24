"""
load_sql.py
-----------
Step 3 of the ETL pipeline: LOAD

Loads the cleaned hospital dataset into the configured database backend
(SQLite for local development, SQL Server for enterprise production).

Architecture & Safety Strategy:
- Uses a Safe Full-Refresh Loading Pattern with an intermediate staging table.
- Records are loaded into `patients_staging` first.
- The staging table undergoes rigorous pre-swap validation (row count, required columns, non-nulls).
- Production table swap occurs inside an atomic transaction only if validation passes.
- If staging validation or loading fails, the previously valid production table is preserved untouched.
- Staging and backup tables are cleanly pruned after completion.
"""

import logging
import sys
import time
from pathlib import Path

# Ensure root repository directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from sqlalchemy import Engine, create_engine, inspect, text

from src.config import (
    CLEANED_CSV_PATH,
    DB_ENGINE_TYPE,
    DB_NAME,
    DB_SERVER,
    DB_USER,
    DB_PASSWORD,
    DB_DRIVER,
    TABLE_NAME,
    STAGING_TABLE_NAME,
    get_db_engine,
    verify_database_ready,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("load_sql")

BACKUP_TABLE_NAME = f"{TABLE_NAME}_backup"


def create_sqlserver_database_if_not_exists():
    """For SQL Server: creates the target database if missing."""
    if DB_ENGINE_TYPE != "sqlserver":
        return

    driver = DB_DRIVER.replace(" ", "+")
    master_conn_str = (
        f"mssql+pyodbc://{DB_USER}:{DB_PASSWORD}@{DB_SERVER}/master?driver={driver}"
        if DB_USER and DB_PASSWORD
        else f"mssql+pyodbc://@{DB_SERVER}/master?driver={driver}&trusted_connection=yes"
    )
    master_engine = create_engine(master_conn_str, isolation_level="AUTOCOMMIT")
    with master_engine.connect() as conn:
        conn.execute(
            text(
                f"IF NOT EXISTS (SELECT * FROM sys.databases WHERE name = '{DB_NAME}') "
                f"CREATE DATABASE [{DB_NAME}]"
            )
        )
    logger.info("Confirmed SQL Server database '%s' exists.", DB_NAME)


def validate_staging_table(engine: Engine, expected_count: int, df: pd.DataFrame) -> bool:
    """
    Validates that the staging table exists, matches expected row count,
    contains all required columns, and has no null primary identifiers.
    """
    insp = inspect(engine)
    tables = [t.lower() for t in insp.get_table_names()]
    if STAGING_TABLE_NAME.lower() not in tables:
        raise ValueError(f"Staging validation failed: table '{STAGING_TABLE_NAME}' not found.")

    with engine.connect() as conn:
        # Verify row count
        count_res = conn.execute(text(f"SELECT COUNT(*) FROM {STAGING_TABLE_NAME}"))
        actual_count = count_res.scalar()
        if actual_count != expected_count:
            raise ValueError(
                f"Staging row count mismatch: expected {expected_count}, got {actual_count}."
            )

        # Verify columns
        staging_cols = [c["name"].lower() for c in insp.get_columns(STAGING_TABLE_NAME)]
        for col in df.columns:
            if col.lower() not in staging_cols:
                raise ValueError(f"Staging schema validation failed: missing column '{col}'.")

        # Verify primary critical values are non-null
        null_res = conn.execute(
            text(f"SELECT COUNT(*) FROM {STAGING_TABLE_NAME} WHERE patient_id IS NULL OR billing_amount IS NULL")
        )
        null_count = null_res.scalar()
        if null_count and null_count > 0:
            raise ValueError(f"Staging table contains {null_count} rows with null patient_id or billing_amount.")

    logger.info("Staging validation passed: %d rows verified in '%s'.", actual_count, STAGING_TABLE_NAME)
    return True


def swap_staging_to_production(engine: Engine) -> None:
    """
    Safely promotes the staging table to production.
    If production already exists, it is backed up, the staging table is renamed to production,
    and the backup is dropped. If an error occurs, the original production table is restored.
    """
    insp = inspect(engine)
    existing_tables = [t.lower() for t in insp.get_table_names()]
    prod_exists = TABLE_NAME.lower() in existing_tables

    logger.info(
        "Executing safe table swap: '%s' -> '%s' (prod_exists=%s)...",
        STAGING_TABLE_NAME,
        TABLE_NAME,
        prod_exists,
    )

    if DB_ENGINE_TYPE == "sqlite":
        with engine.begin() as conn:
            # Remove any leftover backup table from previous runs
            conn.execute(text(f"DROP TABLE IF EXISTS {BACKUP_TABLE_NAME}"))
            if prod_exists:
                conn.execute(text(f"ALTER TABLE {TABLE_NAME} RENAME TO {BACKUP_TABLE_NAME}"))
            conn.execute(text(f"ALTER TABLE {STAGING_TABLE_NAME} RENAME TO {TABLE_NAME}"))
            if prod_exists:
                conn.execute(text(f"DROP TABLE IF EXISTS {BACKUP_TABLE_NAME}"))

            # Recreate standard performance indexes
            conn.execute(
                text(f"CREATE INDEX IF NOT EXISTS ix_{TABLE_NAME}_dept ON {TABLE_NAME}(department)")
            )
            conn.execute(
                text(f"CREATE INDEX IF NOT EXISTS ix_{TABLE_NAME}_doctor ON {TABLE_NAME}(doctor)")
            )
            conn.execute(
                text(f"CREATE INDEX IF NOT EXISTS ix_{TABLE_NAME}_adm_date ON {TABLE_NAME}(admission_date)")
            )
        logger.info("Atomic table swap and index creation successful in SQLite.")

    else:
        # SQL Server implementation
        with engine.begin() as conn:
            conn.execute(text(f"IF OBJECT_ID('dbo.{BACKUP_TABLE_NAME}', 'U') IS NOT NULL DROP TABLE dbo.{BACKUP_TABLE_NAME};"))
            if prod_exists:
                conn.execute(text(f"EXEC sp_rename 'dbo.{TABLE_NAME}', '{BACKUP_TABLE_NAME}';"))
            conn.execute(text(f"EXEC sp_rename 'dbo.{STAGING_TABLE_NAME}', '{TABLE_NAME}';"))
            if prod_exists:
                conn.execute(text(f"IF OBJECT_ID('dbo.{BACKUP_TABLE_NAME}', 'U') IS NOT NULL DROP TABLE dbo.{BACKUP_TABLE_NAME};"))
        logger.info("Safe table swap successful in SQL Server.")


def cleanup_staging(engine: Engine) -> None:
    """Drops staging and backup tables if they exist."""
    try:
        with engine.begin() as conn:
            if DB_ENGINE_TYPE == "sqlite":
                conn.execute(text(f"DROP TABLE IF EXISTS {STAGING_TABLE_NAME}"))
                conn.execute(text(f"DROP TABLE IF EXISTS {BACKUP_TABLE_NAME}"))
            else:
                conn.execute(text(f"IF OBJECT_ID('dbo.{STAGING_TABLE_NAME}', 'U') IS NOT NULL DROP TABLE dbo.{STAGING_TABLE_NAME};"))
                conn.execute(text(f"IF OBJECT_ID('dbo.{BACKUP_TABLE_NAME}', 'U') IS NOT NULL DROP TABLE dbo.{BACKUP_TABLE_NAME};"))
    except Exception as e:
        logger.warning("Could not prune temporary staging table: %s", e)


def load_to_sql(df: pd.DataFrame, engine: Engine | None = None) -> Engine:
    """
    Executes the safe full-refresh loading pipeline:
    1. Writes to staging table
    2. Validates staging table integrity
    3. Safely swaps staging into production
    4. Cleans up staging resources
    """
    start_time = time.time()
    create_sqlserver_database_if_not_exists()
    engine = engine or get_db_engine()

    logger.info("Starting safe full-refresh database load into engine '%s'...", DB_ENGINE_TYPE)
    logger.info("Input dataset contains %d rows and %d columns.", len(df), len(df.columns))

    # Clean up any lingering staging table from an interrupted run
    cleanup_staging(engine)

    try:
        # Step 1: Write to staging table
        logger.info("Writing records to staging table '%s'...", STAGING_TABLE_NAME)
        df.to_sql(
            STAGING_TABLE_NAME,
            engine,
            if_exists="replace",
            index=False,
            chunksize=5000,
        )

        # Step 2: Validate staging table
        validate_staging_table(engine, expected_count=len(df), df=df)

        # Step 3: Swap staging to production
        swap_staging_to_production(engine)

        # Step 4: Verify production readiness
        is_ready, verify_msg = verify_database_ready(engine)
        if not is_ready:
            raise RuntimeError(f"Post-load verification failed: {verify_msg}")

        elapsed = time.time() - start_time
        logger.info("Database load succeeded in %.2f seconds. %s", elapsed, verify_msg)

    except Exception as e:
        logger.error("Database load failed: %s. Cleaning up staging table...", e)
        cleanup_staging(engine)
        raise e

    finally:
        cleanup_staging(engine)

    return engine


def main():
    if not CLEANED_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {CLEANED_CSV_PATH}. Run transform.py first."
        )
    df = pd.read_csv(CLEANED_CSV_PATH)
    load_to_sql(df)


if __name__ == "__main__":
    main()
