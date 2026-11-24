"""
predict.py
----------
Inference engine for patient prolonged stay risk prediction.

Loads the unified model pipeline artifact and performs classification and
probability estimation on unseen patient admission records.

Features:
- Validates and standardizes input feature columns
- Applies fitted ColumnTransformer preprocessing and Random Forest classifier
- Returns risk class, risk probability, and operational risk category
- Supports both DataFrame API and CLI execution
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure root repository directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import pandas as pd

from src.config import MODEL_PATH, ML_ALL_FEATURES, RAW_CSV_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ml_predict")

COLUMN_ALIAS_MAP = {
    "Age": "age",
    "Gender": "gender",
    "Medical Condition": "medical_condition",
    "Admission Type": "admission_type",
}


def load_model_artifact(model_path: Path | str | None = None) -> tuple[object, dict]:
    """Loads the model pipeline and associated training metadata."""
    path = Path(model_path or MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at '{path}'. Please run 'python src/ml/train.py' first."
        )

    artifact = joblib.load(path)
    if isinstance(artifact, dict) and "pipeline" in artifact:
        return artifact["pipeline"], artifact.get("metadata", {})
    elif isinstance(artifact, dict) and "model" in artifact:
        # Compatibility with earlier versions
        return artifact["model"], artifact.get("metadata", {})
    return artifact, {}


def standardize_input_features(input_df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes column names to match model expectation."""
    df = input_df.copy()
    rename_dict = {orig: target for orig, target in COLUMN_ALIAS_MAP.items() if orig in df.columns}
    df = df.rename(columns=rename_dict)

    missing = [col for col in ML_ALL_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required prediction features: {missing}. Expected features: {ML_ALL_FEATURES}"
        )

    return df[ML_ALL_FEATURES].copy()


def predict_patient_stay_risk(
    input_df: pd.DataFrame,
    model_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Generates prolonged-stay risk predictions for a DataFrame of patients.
    Returns input DataFrame with prediction columns appended:
    - predicted_long_stay (0 = at or below median, 1 = above median)
    - risk_probability (probability of prolonged stay)
    - stay_risk_category (operational label)
    """
    pipeline, metadata = load_model_artifact(model_path)
    X = standardize_input_features(input_df)

    predictions = pipeline.predict(X)
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X)[:, 1]
    else:
        probabilities = [float(p) for p in predictions]

    result = input_df.copy()
    result["predicted_long_stay"] = predictions
    result["risk_probability"] = [round(float(p), 4) for p in probabilities]
    result["stay_risk_category"] = result["predicted_long_stay"].map({
        0: "Lower predicted prolonged-stay likelihood",
        1: "Higher predicted prolonged-stay likelihood",
    })

    if "training_threshold_days" in metadata:
        result["stay_threshold_days"] = metadata["training_threshold_days"]

    return result


def predict_from_csv(
    csv_path: str | Path,
    model_path: Path | str | None = None,
) -> pd.DataFrame:
    """Loads CSV and runs stay risk prediction."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found at: {path}")
    df = pd.read_csv(path)
    return predict_patient_stay_risk(df, model_path=model_path)


def main():
    parser = argparse.ArgumentParser(description="Predict hospital length-of-stay risk from patient features")
    parser.add_argument(
        "--input",
        "-i",
        default=str(RAW_CSV_PATH),
        help="Path to input patient records CSV",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=str(MODEL_PATH),
        help="Path to saved model artifact .pkl",
    )
    args = parser.parse_args()

    results = predict_from_csv(args.input, model_path=args.model)
    display_cols = [c for c in ["name", "Name", "age", "Age", "medical_condition", "Medical Condition", "admission_type", "Admission Type", "risk_probability", "stay_risk_category"] if c in results.columns]
    print("\n--- SAMPLE PREDICTIONS ---")
    print(results[display_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
