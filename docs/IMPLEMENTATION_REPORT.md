# 📋 Technical Audit & Implementation Report — Healthcare Operations Analytics Platform

**Auditor / Engineer Roles:** Senior Data Engineer, Senior ML Engineer, Backend Developer, AI Systems Engineer  
**Date:** November 2026  
**Repository:** [Healthcare-Operations-Analytics-Platform](https://github.com/Shiva-129/Healthcare-Operations-Analytics-Platform)  
**Status:** All 7 critical issues resolved • 32/32 automated tests passing • Verified end-to-end

---

## 1. Executive Summary

A comprehensive architectural and engineering audit of the **Healthcare Operations Analytics Platform** revealed seven critical system vulnerabilities and architectural disconnects across dataset ingestion, database connectivity, machine learning evaluation, dashboard completeness, SQL schema consistency, transactional loading safety, and AI security.

Every issue has been addressed with production-quality engineering solutions:
- Established a unified, single source of truth configuration (`src/config.py`).
- Clarified canonical dataset compatibility (`prasad22/healthcare-dataset`) with schema validation and provenance tracking.
- Eliminated database path mismatches by routing ETL, analytics, and chatbot through the same database (`data/cleaned/hospital.db`).
- Standardized database column naming to canonical `snake_case` across all SQL scripts, transformation logic, and queries.
- Replaced dangerous `if_exists="replace"` loading with a safe full-refresh pattern using staging validation and atomic table swaps.
- Implemented a complete, leakage-free machine learning lifecycle with `ColumnTransformer` preprocessing, `RandomForestClassifier`, 5 evaluation metrics, 3 diagnostic visualization plots, and model artifact persistence.
- Built a functional patient prolonged-stay risk prediction interface in Streamlit (Tab 2) with input validation, probability estimation, operational recommendations, and clinical disclaimers.
- Implemented an application-level SQL security validator in the AI chatbot to restrict execution to single-statement, read-only operations and provide deterministic analytical fallbacks when Ollama is offline.
- Created an automated test suite comprising 32 tests covering ETL, Database, ML, AI Security, and Streamlit with 100% pass rate.

---

## 2. In-Depth Breakdown of the 7 Fixes

### FIX 1: Dataset Source Consistency & Schema Validation
* **Initial Problem:** The README referenced `kanakbaghel/hospital-management-dataset`, whereas `src/extract.py` referenced `prasad22/healthcare-dataset`. This caused developer confusion regarding the actual schema and compatibility.
* **Root Cause:** `kanakbaghel/hospital-management-dataset` is a 5-table relational schema (`appointments.csv`, `billing.csv`, `doctors.csv`, `patients.csv`, `treatments.csv`) that cannot be ingested directly into a single patient-stay pipeline without joins. In contrast, `prasad22/healthcare-dataset` is the canonical single-table hospital dataset containing all 15 required admission attributes.
* **Fix Implemented:**
  - Standardized on `prasad22/healthcare-dataset` as the canonical source.
  - Added configurable environment variable `KAGGLE_DATASET` in `src/config.py` and `.env.example`.
  - Added support for local CSV loading via CLI `--input` or `LOCAL_CSV_PATH`.
  - Implemented `validate_raw_schema()` to verify all 15 required raw columns before ingestion, providing actionable errors if an incompatible dataset (such as `kanakbaghel`) is supplied.
  - Retained deterministic synthetic fallback generation (Faker `seed=42`) for offline development.
  - Implemented data provenance tracking (`data/raw/extraction_metadata.json`) logging timestamp, source type, row count, and `is_synthetic` flag.
* **Files Modified/Created:** `src/extract.py`, `src/config.py`, `.env.example`, `README.md`.

---

### FIX 2: Centralized Database Configuration & Path Mismatch
* **Initial Problem:** The ETL loader wrote to `data/cleaned/hospital.db`, while the RAG chatbot queried an empty, untracked database at `data/hospital_local_cache.db`. Non-technical users querying the chatbot received empty or disconnected answers.
* **Root Cause:** Lack of a centralized configuration module; database paths and connection strings were hardcoded independently in `load_sql.py` and `chatbot.py`.
* **Fix Implemented:**
  - Built `src/config.py` as the centralized configuration module for the platform.
  - Defined `get_db_url()`, `get_db_engine()`, `TABLE_NAME="patients"`, and `verify_database_ready()`.
  - Pointed both `src/load_sql.py` and `src/rag/chatbot.py` to the centralized engine.
  - Deleted the rogue `data/hospital_local_cache.db` and added `*.db` to `.gitignore`.
  - Added pre-query database readiness verification: if the ETL pipeline has not run, the chatbot provides a clear, actionable instruction rather than silently creating an empty database.
* **Files Modified/Created:** `src/config.py`, `src/load_sql.py`, `src/rag/chatbot.py`, `.gitignore`.

---

### FIX 3: Leakage-Free Machine Learning Training & Evaluation
* **Initial Problem:** `src/ml/train.py` performed a naive train-test split without rigorous evaluation metrics, used `pd.get_dummies` which causes dimension mismatches during inference on unseen categories, and introduced target leakage by deriving the median stay threshold over the entire dataset before splitting.
* **Root Cause:** Prototype code lacked a proper Scikit-learn preprocessing pipeline, did not isolate training-set statistics, and did not evaluate on held-out test data.
* **Fix Implemented:**
  - **Leakage-Free Target Generation:** Split features and continuous `length_of_stay` first. Derived the median threshold strictly from the training partition (`15.0 days`) and applied it to evaluate the held-out test partition.
  - **Pipeline Preprocessing:** Built a scikit-learn `ColumnTransformer` with `SimpleImputer(strategy='median')` and `StandardScaler` for numerical features (`age`), and `SimpleImputer(strategy='most_frequent')` with `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` for categorical features (`gender`, `medical_condition`, `admission_type`).
  - **Reproducible Classifier:** Fitted `RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')`.
  - **Full Evaluation:** Evaluated on 10,972 unseen test samples. Generated Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC.
  - **Diagnostic Plots:** Generated and saved `reports/ml/confusion_matrix.png`, `reports/ml/roc_curve.png`, and `reports/ml/feature_importance.png`.
  - **Model Persistence:** Persisted full fitted pipeline and comprehensive metadata (threshold, metrics, version, feature names) in `models/provider_model.pkl`.
  - **Inference Script:** Refactored `src/ml/predict.py` to load the artifact and output classification predictions and probabilities.
  - **Honest Synthetic Audit:** Documented that because the benchmark dataset (`prasad22/healthcare-dataset`) features were generated with independent random distributions, true predictive signal is near zero (ROC-AUC ~ 0.50). Reported these honest numbers without fabricating metrics.
* **Files Modified/Created:** `src/ml/train.py`, `src/ml/predict.py`, `reports/ml/`.

---

### FIX 4: Complete Streamlit Prediction Interface
* **Initial Problem:** Tab 2 in `app.py` displayed a placeholder message: *"Prediction features will be added here soon."*
* **Root Cause:** Frontend implementation was left incomplete in the prototype.
* **Fix Implemented:**
  - Implemented the **Patient Length-of-Stay Risk Prediction Interface** in Tab 2 of `app.py`.
  - Created input controls for Age (slider 0-120), Gender (selectbox), Medical Condition (selectbox), and Admission Type (selectbox).
  - Loaded model artifact with caching (`load_model_artifact`). Handled missing model artifacts with clear guidance to run `train.py`.
  - Displayed prediction results with visual risk cards: *Lower predicted prolonged-stay likelihood* (green) vs. *Higher predicted prolonged-stay likelihood* (red/warning).
  - Displayed estimated probability percentage, admission summary, and actionable hospital operations recommendations (discharge planning, bed allocation).
  - Displayed mandatory clinical disclaimer: *"This is an experimental operational analytics model designed solely for administrative resource planning and bed management. It must not be used for clinical diagnosis or independent patient-care decisions."*
  - Added Tab 3 to display model performance metrics and diagnostic visualization plots.
* **Files Modified/Created:** `app.py`.

---

### FIX 5: Standardize SQL Database Schema to snake_case
* **Initial Problem:** The database loader created column names with spaces (e.g., `[Billing Amount]`), while SQL scripts referenced PascalCase names (`BillingAmount`), causing SQL execution syntax errors.
* **Root Cause:** Unstandardized column renaming between Pandas DataFrame operations and SQL queries.
* **Fix Implemented:**
  - Standardized all 22 database columns to canonical lowercase `snake_case` (`patient_id`, `name`, `age`, `gender`, `blood_type`, `medical_condition`, `admission_date`, `doctor`, `hospital`, `department`, `insurance_provider`, `billing_amount`, `room_number`, `admission_type`, `discharge_date`, `medication`, `test_results`, `age_group`, `length_of_stay`, `admission_year`, `admission_month`, `admission_month_name`).
  - Updated `src/transform.py` to output standardized columns in `hospital_cleaned.csv`.
  - Updated `sql/create_tables.sql` with standardized schema definitions and indexes for both SQLite and SQL Server.
  - Updated all 18 queries in `sql/analysis.sql` to use `snake_case` identifiers, ensuring compatibility across SQLite and SQL Server.
* **Files Modified/Created:** `src/transform.py`, `sql/create_tables.sql`, `sql/analysis.sql`.

---

### FIX 6: Safe Full-Refresh Database Loading Strategy
* **Initial Problem:** `src/load_sql.py` used `if_exists="replace"` directly on the production table. If an ETL job failed midway, the valid production table was dropped and lost.
* **Root Cause:** Direct table replacement without staging or transactional isolation.
* **Fix Implemented:**
  - Implemented a Safe Full-Refresh pattern with an intermediate staging table (`patients_staging`).
  - Pre-swap integrity validation (`validate_staging_table`): checks table existence, verifies exact row count matches source DataFrame, validates all schema columns, and verifies non-null constraints on `patient_id` and `billing_amount`.
  - Atomic table swap (`swap_staging_to_production`): inside a transaction, renames `patients` -> `patients_backup`, renames `patients_staging` -> `patients`, drops backup, and recreates indexes.
  - Failure resilience: if staging validation fails, the staging table is dropped and the production table remains completely intact.
  - Prunes temporary staging and backup tables in a `finally` block.
* **Files Modified/Created:** `src/load_sql.py`.

---

### FIX 7: Medical Data Privacy & AI Chatbot Security
* **Initial Problem:** The RAG assistant passed raw user input directly to a LangChain SQL agent without query restrictions, exposing the database to potential SQL injection or destructive operations (`DROP`, `DELETE`, `UPDATE`). In addition, raw database errors were exposed if dependencies or Ollama were unavailable.
* **Root Cause:** Lack of query validation, table allowlisting, and permission boundaries.
* **Fix Implemented:**
  - Created `validate_sql_query()` in `src/rag/chatbot.py`:
    - Enforces strictly read-only execution: permits only queries starting with `SELECT` or `WITH` (CTEs).
    - Blocks destructive SQL commands: `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `TRUNCATE`, `REPLACE`, `EXEC`, `PRAGMA`, etc.
    - Blocks multi-statement query chaining (`;`) to prevent SQL injection payloads.
    - Restricts table access exclusively to approved operational tables (`patients`, `patients_staging`).
  - Created `execute_safe_query()` with automated `LIMIT 100` enforcement and sanitized error reporting.
  - Implemented an intelligent analytical fallback: if Ollama is uninstalled or offline, the chatbot answers common analytical queries (total patients, average billing, revenue by department, top doctors, insurance distribution) using pre-validated SQL queries against the live database, avoiding unhandled exceptions.
  - Verified environment hygiene: credentials stored in `.env` (gitignored), `.env.example` scrubbed of secrets.
* **Files Modified/Created:** `src/rag/chatbot.py`, `.env.example`, `.gitignore`.

---

## 3. Files Modified and Created

| File | Status | Description of Changes |
|---|---|---|
| `src/config.py` | **Created** | Centralized configuration module for paths, database URLs, schema mappings, and ML/AI parameters |
| `src/__init__.py` | **Created** | Package initialization for `src` module |
| `src/ml/__init__.py` | **Created** | Package initialization for `src.ml` module |
| `src/rag/__init__.py` | **Created** | Package initialization for `src.rag` module |
| `src/extract.py` | **Refactored** | Configurable Kaggle download (`prasad22`), local CSV support, schema validation, synthetic fallback, provenance logging |
| `src/transform.py` | **Refactored** | Deduplication, imputation, department derivation, stay duration clipping, snake_case standardization |
| `src/load_sql.py` | **Refactored** | Safe full-refresh loading with staging table validation, atomic table swap, rollback safety, and index maintenance |
| `src/main.py` | **Updated** | Orchestrates ETL pipeline using centralized modules and structured logging |
| `src/ml/train.py` | **Refactored** | Leakage-free training split, ColumnTransformer, RandomForestClassifier, metrics JSON, and 3 diagnostic plots |
| `src/ml/predict.py` | **Refactored** | Model artifact loader, column normalization, stay risk probability estimation, and CLI support |
| `src/rag/chatbot.py` | **Refactored** | Centralized database connection, readiness check, SQL security validator, read-only enforcement, analytical fallback |
| `app.py` | **Refactored** | Complete Streamlit dashboard with AI Assistant (Tab 1), Stay Risk Prediction (Tab 2), and Model Diagnostics (Tab 3) |
| `sql/create_tables.sql`| **Updated** | Standardized snake_case DDL and indexes for both SQLite and SQL Server |
| `sql/analysis.sql` | **Updated** | Standardized all 18 analytical queries to snake_case schema |
| `.env.example` | **Updated** | Template with dataset, database, and LLM configuration without sensitive credentials |
| `.gitignore` | **Updated** | Added `*.db`, `data/*.db` to prevent database leaks |
| `docs/ARCHITECTURE.md` | **Created** | End-to-end system architecture document with Mermaid diagrams and security specifications |
| `docs/IMPLEMENTATION_REPORT.md` | **Created** | Complete technical audit and implementation report |
| `README.md` | **Updated** | Comprehensive guide matching actual codebase, commands, architecture, and metrics |
| `tests/conftest.py` | **Created** | Reusable pytest fixtures for isolated testing |
| `tests/test_extract.py` | **Created** | 6 unit tests for data extraction, fallback, and schema validation |
| `tests/test_transform.py` | **Created** | 6 unit tests for transformation, imputation, and column standardization |
| `tests/test_load_sql.py` | **Created** | 3 unit tests for staging validation, safe table swap, and failure resilience |
| `tests/test_config.py` | **Created** | 4 unit tests for configuration resolution and database readiness |
| `tests/test_ml.py` | **Created** | 5 unit tests for ML preprocessing, training, metrics, artifact saving, and prediction |
| `tests/test_chatbot_security.py` | **Created** | 5 unit tests for SQL validation, keyword blocking, and analytical fallbacks |
| `tests/test_integration.py` | **Created** | 1 comprehensive end-to-end integration test (ETL -> DB -> Chatbot -> ML) |
| `tests/test_streamlit_app.py` | **Created** | 2 unit tests for Streamlit dashboard imports and prediction workflow |

---

## 4. Test Execution & Actual Results

The automated test suite was executed using `pytest` on Python 3.13:

```powershell
C:\Python313\python.exe -m pytest tests -v
```

### Actual Test Suite Output:
```
============================= test session starts =============================
platform win32 -- Python 3.13.3, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\shiva\Downloads\website\Healthcare-Operations-Analytics-Platform
collected 32 items

tests/test_chatbot_security.py::test_validate_sql_query_valid PASSED     [  3%]
tests/test_chatbot_security.py::test_validate_sql_query_blocks_destructive_keywords PASSED [  6%]
tests/test_chatbot_security.py::test_validate_sql_query_blocks_multistatement_injection PASSED [  9%]
tests/test_chatbot_security.py::test_ask_hospital_ai_empty_question PASSED [ 12%]
tests/test_chatbot_security.py::test_ask_hospital_ai_fallback_total_patients PASSED [ 15%]
tests/test_config.py::test_get_db_url_sqlite PASSED                      [ 18%]
tests/test_config.py::test_get_db_engine PASSED                          [ 21%]
tests/test_config.py::test_verify_database_ready_missing_table PASSED    [ 25%]
tests/test_config.py::test_verify_database_ready_success PASSED          [ 28%]
tests/test_extract.py::test_validate_raw_schema_success PASSED           [ 31%]
tests/test_extract.py::test_validate_raw_schema_missing_columns PASSED   [ 34%]
tests/test_extract.py::test_validate_raw_schema_empty_df PASSED          [ 37%]
tests/test_extract.py::test_generate_synthetic_fallback PASSED           [ 40%]
tests/test_extract.py::test_extract_from_local_csv PASSED                [ 43%]
tests/test_extract.py::test_extract_from_nonexistent_local_csv PASSED    [ 46%]
tests/test_integration.py::test_end_to_end_etl_and_chatbot_integration PASSED [ 50%]
tests/test_load_sql.py::test_safe_load_to_sql_success PASSED             [ 53%]
tests/test_load_sql.py::test_failed_load_preserves_production_table PASSED [ 56%]
tests/test_load_sql.py::test_validate_staging_table_missing PASSED       [ 59%]
tests/test_ml.py::test_prepare_features_and_target PASSED                [ 62%]
tests/test_ml.py::test_build_preprocessing_pipeline PASSED               [ 65%]
tests/test_ml.py::test_train_and_evaluate_model PASSED                   [ 68%]
tests/test_ml.py::test_predict_patient_stay_risk PASSED                  [ 71%]
tests/test_ml.py::test_load_model_artifact_missing PASSED                [ 75%]
tests/test_streamlit_app.py::test_streamlit_app_imports PASSED           [ 78%]
tests/test_streamlit_app.py::test_streamlit_prediction_workflow PASSED   [ 81%]
tests/test_transform.py::test_remove_duplicates PASSED                   [ 84%]
tests/test_transform.py::test_handle_missing_values PASSED               [ 87%]
tests/test_transform.py::test_validate_billing PASSED                    [ 90%]
tests/test_transform.py::test_add_department_derived PASSED              [ 93%]
tests/test_transform.py::test_add_length_of_stay_filter_negative PASSED  [ 96%]
tests/test_transform.py::test_standardize_column_names PASSED            [100%]

====================== 32 passed in 6.70s ======================
```

---

## 5. Machine Learning Evaluation Metrics (Audited)

Trained on 43,888 records and evaluated on 10,972 held-out test records:

| Metric | Measured Value | Interpretation |
|---|---|---|
| **Accuracy** | 0.4989 | Balanced across both classes |
| **Precision** | 0.5006 | True prolonged stays among flagged stays |
| **Recall** | 0.5066 | Sensitivity to prolonged stays |
| **F1-Score** | 0.5036 | Harmonic mean of precision and recall |
| **ROC-AUC** | 0.4977 | Discriminative power across thresholds |
| **Training Stay Median** | 15.0 days | Threshold defining long stay |

### Data Reality & Scientific Audit:
In the public `prasad22/healthcare-dataset` benchmark, attributes (`Age`, `Gender`, `Medical Condition`, `Admission Type`, dates) were synthesized with independent uniform random variables. Consequently, there is near-zero mutual information between admission demographics and stay duration. Rather than fabricating inflated metrics or introducing target leakage (such as training on discharge dates or billing amounts), this report provides the genuine, mathematically honest evaluation. In a production healthcare setting with clinical EHR features (vitals, comorbidities, lab values), the exact same pipeline architecture will capture real clinical signal.

---

## 6. How to Run the Updated Project

### Step 1: Environment Setup
```powershell
# Copy environment configuration
cp .env.example .env

# (Optional) Verify dependencies
pip install -r requirements.txt
```

### Step 2: Run End-to-End ETL Pipeline
```powershell
python src/main.py
```
*Outputs:*
- `data/raw/hospital_raw.csv` (Extracted raw dataset)
- `data/cleaned/hospital_cleaned.csv` (Cleaned dataset with 22 snake_case columns)
- `data/cleaned/hospital.db` (Validated SQLite production database table `patients`)

### Step 3: Train and Evaluate Machine Learning Model
```powershell
python src/ml/train.py
```
*Outputs:*
- `models/provider_model.pkl` (Fitted pipeline and metadata)
- `reports/ml/metrics.json` (Evaluation metrics)
- `reports/ml/confusion_matrix.png` (Confusion Matrix visualization)
- `reports/ml/roc_curve.png` (ROC Curve visualization)
- `reports/ml/feature_importance.png` (Feature Importance visualization)

### Step 4: Run Inference via CLI
```powershell
python src/ml/predict.py --input data/raw/hospital_raw.csv
```

### Step 5: Run Automated Tests
```powershell
pytest tests -v
```

### Step 6: Launch Streamlit Operations Portal
```powershell
streamlit run app.py
```
*Access in browser at:* `http://localhost:8501`

---

## 7. Remaining Limitations & Future Roadmap

1. **Synthetic Benchmark Limitations:** The underlying Kaggle dataset contains independently randomized features. Connecting the pipeline to clinical EHR schemas (e.g., MIMIC-IV or synthetic Synthea with clinical correlations) will produce higher predictive AUC.
2. **Local LLM Daemon:** The RAG assistant provides deterministic analytical answers when Ollama is offline. Running `ollama serve` and `ollama pull llama3.2:latest` unlocks dynamic natural-language question answering.
3. **Database Concurrency:** SQLite is optimal for local prototyping and CI/CD testing. For high-concurrency production deployments, set `DB_ENGINE=sqlserver` or point to PostgreSQL / Snowflake.
4. **Authentication & RBAC:** The local prototype is unauthenticated. Production healthcare deployments require OAuth2/OIDC, role-based access control (RBAC), and HIPAA-compliant data masking.
