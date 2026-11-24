"""
train.py
--------
Step 4 of the analytics platform: Machine Learning Training & Evaluation

Business Problem:
Predict whether a patient is likely to experience a prolonged hospital stay
at the time of admission.

Target Formulation (Leakage-Free):
- Target: Is_Long_Stay
  - 0: Stay duration at or below the training-set median length of stay
  - 1: Stay duration strictly above the training-set median length of stay
- The median threshold is strictly derived from the training set only and
  applied to evaluate test data to eliminate lookahead target leakage.

Features Available at Admission:
- Numerical: Age
- Categorical: Gender, Medical Condition, Admission Type

Pipeline:
- Scikit-learn ColumnTransformer with Imputation, Scaling, and OneHotEncoder
- RandomForestClassifier with balanced class weights and reproducible random seed
- Comprehensive evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
- Visualizations: Confusion Matrix, ROC Curve, Feature Importance
- Single persisted artifact containing full fitted pipeline and comprehensive metadata
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure root repository directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CLEANED_CSV_PATH,
    RAW_CSV_PATH,
    MODEL_PATH,
    REPORTS_DIR,
    METRICS_PATH,
    ML_RANDOM_STATE,
    ML_TEST_SIZE,
    ML_NUMERICAL_FEATURES,
    ML_CATEGORICAL_FEATURES,
    ML_ALL_FEATURES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ml_train")


def load_dataset_for_training(data_path: Path | None = None) -> pd.DataFrame:
    """
    Loads dataset from cleaned CSV if available, otherwise from raw CSV.
    Normalizes column headers to snake_case.
    """
    path = data_path or (CLEANED_CSV_PATH if CLEANED_CSV_PATH.exists() else RAW_CSV_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Training dataset not found at '{path}'. Run the ETL pipeline first (python src/main.py)."
        )

    logger.info("Loading training data from '%s'...", path)
    df = pd.read_csv(path)

    # Normalize column names to snake_case for consistent processing
    col_map = {
        "Age": "age",
        "Gender": "gender",
        "Medical Condition": "medical_condition",
        "Admission Type": "admission_type",
        "Date of Admission": "admission_date",
        "Discharge Date": "discharge_date",
        "Length of Stay": "length_of_stay",
        "Length_of_Stay": "length_of_stay",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    return df


def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Validates input features, computes length of stay if absent, filters invalid stays,
    and returns features X and continuous length_of_stay series for leakage-free thresholding.
    """
    # If length_of_stay is not already precomputed, derive it
    if "length_of_stay" not in df.columns:
        if "admission_date" not in df.columns or "discharge_date" not in df.columns:
            raise ValueError("Dataset must contain either 'length_of_stay' or admission and discharge dates.")
        adm = pd.to_datetime(df["admission_date"], errors="coerce")
        dis = pd.to_datetime(df["discharge_date"], errors="coerce")
        df["length_of_stay"] = (dis - adm).dt.days

    # Filter out invalid records
    initial_len = len(df)
    df = df.dropna(subset=["length_of_stay"] + ML_ALL_FEATURES)
    df = df[df["length_of_stay"] >= 0].copy()
    if len(df) < initial_len:
        logger.info("Filtered %d records with missing features or negative length of stay.", initial_len - len(df))

    X = df[ML_ALL_FEATURES].copy()
    los_series = df["length_of_stay"].copy()
    return X, los_series


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Constructs scikit-learn ColumnTransformer for numerical and categorical features."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, ML_NUMERICAL_FEATURES),
            ("cat", categorical_transformer, ML_CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def evaluate_and_generate_plots(
    model_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> dict:
    """
    Computes performance metrics and generates diagnostic evaluation charts:
    - Confusion Matrix plot
    - ROC Curve plot
    - Feature Importance plot
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model_pipeline.predict(X_test)
    y_prob = model_pipeline.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else None
    cm = confusion_matrix(y_test, y_pred).tolist()
    clf_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "test_samples": int(len(y_test)),
        "class_distribution_test": {str(k): int(v) for k, v in y_test.value_counts().to_dict().items()},
        "confusion_matrix": cm,
        "classification_report": clf_report,
    }

    # 1. Confusion Matrix Visualization
    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=np.array(cm),
        display_labels=["Normal Stay (<=Med)", "Prolonged Stay (>Med)"],
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Patient Stay Risk Prediction")
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close("all")
    logger.info("Saved Confusion Matrix plot to %s", cm_path)

    # 2. ROC Curve Visualization
    if roc_auc is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier (AUC = 0.50)")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC)")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        roc_path = output_dir / "roc_curve.png"
        plt.savefig(roc_path, dpi=300)
        plt.close("all")
        logger.info("Saved ROC Curve plot to %s", roc_path)

    # 3. Feature Importance Visualization
    classifier = model_pipeline.named_steps["classifier"]
    preprocessor = model_pipeline.named_steps["preprocessor"]
    try:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        encoded_cat_cols = list(cat_encoder.get_feature_names_out(ML_CATEGORICAL_FEATURES))
        all_feature_names = ML_NUMERICAL_FEATURES + encoded_cat_cols

        importances = classifier.feature_importances_
        feat_df = pd.DataFrame({"feature": all_feature_names, "importance": importances})
        feat_df = feat_df.sort_values(by="importance", ascending=True)

        plt.figure(figsize=(8, 6))
        plt.barh(feat_df["feature"], feat_df["importance"], color="#2b5c8f")
        plt.xlabel("Random Forest Feature Importance (MDI)")
        plt.title("Feature Importance - Prolonged Hospital Stay Prediction")
        plt.tight_layout()
        fi_path = output_dir / "feature_importance.png"
        plt.savefig(fi_path, dpi=300)
        plt.close("all")
        logger.info("Saved Feature Importance plot to %s", fi_path)
    except Exception as e:
        logger.warning("Could not extract feature importances for visualization: %s", e)

    return metrics


def train_provider_model(
    data_path: Path | None = None,
    output_model_path: Path = MODEL_PATH,
    reports_dir: Path = REPORTS_DIR,
) -> dict:
    """
    Executes the complete machine learning training and evaluation lifecycle:
    1. Loads dataset
    2. Derives train/test split before computing target threshold (avoids leakage)
    3. Fits ColumnTransformer and RandomForestClassifier pipeline
    4. Evaluates comprehensively on unseen test data
    5. Saves single unified model artifact and metric reports
    """
    logger.info("Starting Machine Learning Training Workflow...")
    df = load_dataset_for_training(data_path)
    X, los_series = prepare_features_and_target(df)

    logger.info("Loaded %d clean records. Splitting into train/test (test_size=%.2f)...", len(X), ML_TEST_SIZE)

    # Split features and continuous stay duration first
    X_train, X_test, los_train, los_test = train_test_split(
        X,
        los_series,
        test_size=ML_TEST_SIZE,
        random_state=ML_RANDOM_STATE,
    )

    # Leakage-Free Target Generation: Calculate threshold strictly on training set!
    stay_threshold = float(los_train.median())
    logger.info("Derived training stay median threshold: %.1f days", stay_threshold)

    y_train = (los_train > stay_threshold).astype(int)
    y_test = (los_test > stay_threshold).astype(int)

    logger.info("Training class distribution:\n%s", y_train.value_counts(normalize=True).to_dict())
    logger.info("Testing class distribution:\n%s", y_test.value_counts(normalize=True).to_dict())

    # Build Pipeline
    preprocessor = build_preprocessing_pipeline()
    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=ML_RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    # Fit pipeline exclusively on training data
    logger.info("Fitting preprocessing and Random Forest classifier on %d training samples...", len(X_train))
    model_pipeline.fit(X_train, y_train)

    # Evaluate on held-out test data
    logger.info("Evaluating fitted model pipeline on %d unseen test samples...", len(X_test))
    metrics = evaluate_and_generate_plots(model_pipeline, X_test, y_test, reports_dir)

    # Log key evaluation metrics
    logger.info("Test Evaluation Metrics:")
    logger.info("  Accuracy:  %.4f", metrics["accuracy"])
    logger.info("  Precision: %.4f", metrics["precision"])
    logger.info("  Recall:    %.4f", metrics["recall"])
    logger.info("  F1-Score:  %.4f", metrics["f1_score"])
    logger.info("  ROC-AUC:   %s", f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "N/A")

    # Save metrics to JSON
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_json_path = reports_dir / "metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved evaluation metrics JSON to %s", metrics_json_path)

    # Save metrics summary CSV
    metrics_summary_df = pd.DataFrame([
        {"metric": k, "value": v}
        for k, v in metrics.items()
        if isinstance(v, (int, float, str))
    ])
    metrics_summary_csv = reports_dir / "metrics_summary.csv"
    metrics_summary_df.to_csv(metrics_summary_csv, index=False)

    # Persist Unified Model Artifact
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": model_pipeline,
        "metadata": {
            "model_type": "RandomForestClassifier",
            "model_version": "1.0.0",
            "target_name": "is_long_stay",
            "target_definition": f"Stay strictly greater than training median ({stay_threshold:.1f} days)",
            "training_threshold_days": stay_threshold,
            "feature_names": ML_ALL_FEATURES,
            "numerical_features": ML_NUMERICAL_FEATURES,
            "categorical_features": ML_CATEGORICAL_FEATURES,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "training_date": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
        },
    }
    joblib.dump(artifact, output_model_path)
    logger.info("Model pipeline and metadata successfully persisted to %s", output_model_path)

    return metrics


if __name__ == "__main__":
    train_provider_model()
