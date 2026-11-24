"""
test_ml.py
----------
Unit tests for Machine Learning training, leakage-free thresholding,
preprocessing pipelines, model persistence, and prediction inference.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.ml.train import (
    prepare_features_and_target,
    build_preprocessing_pipeline,
    train_provider_model,
)
from src.ml.predict import (
    predict_patient_stay_risk,
    load_model_artifact,
)


@pytest.fixture
def ml_sample_dataset() -> pd.DataFrame:
    """Generates a synthetic DataFrame for training ML models."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "age": np.random.randint(18, 90, size=n),
        "gender": np.random.choice(["Male", "Female"], size=n),
        "medical_condition": np.random.choice(
            ["Diabetes", "Hypertension", "Asthma", "Cancer", "Obesity", "Arthritis"], size=n
        ),
        "admission_type": np.random.choice(["Emergency", "Elective", "Urgent"], size=n),
        "length_of_stay": np.random.randint(1, 30, size=n),
    })


def test_prepare_features_and_target(ml_sample_dataset: pd.DataFrame):
    """Verifies feature extraction and length of stay isolation."""
    X, los_series = prepare_features_and_target(ml_sample_dataset)
    assert len(X) == len(ml_sample_dataset)
    assert len(los_series) == len(ml_sample_dataset)
    assert "length_of_stay" not in X.columns
    assert list(X.columns) == ["age", "gender", "medical_condition", "admission_type"]


def test_build_preprocessing_pipeline():
    """Verifies that ColumnTransformer handles numerical and categorical transformations."""
    preprocessor = build_preprocessing_pipeline()
    assert preprocessor is not None
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert "num" in transformer_names
    assert "cat" in transformer_names


def test_train_and_evaluate_model(tmp_path: Path, ml_sample_dataset: pd.DataFrame):
    """Verifies that training generates metrics, saves model artifact, and writes plots."""
    data_csv = tmp_path / "test_data.csv"
    model_pkl = tmp_path / "test_model.pkl"
    reports_dir = tmp_path / "reports"

    ml_sample_dataset.to_csv(data_csv, index=False)

    metrics = train_provider_model(
        data_path=data_csv,
        output_model_path=model_pkl,
        reports_dir=reports_dir,
    )

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "confusion_matrix" in metrics

    assert model_pkl.exists()
    assert (reports_dir / "confusion_matrix.png").exists()
    assert (reports_dir / "feature_importance.png").exists()
    assert (reports_dir / "metrics.json").exists()


def test_predict_patient_stay_risk(tmp_path: Path, ml_sample_dataset: pd.DataFrame):
    """Verifies that predict_patient_stay_risk generates valid predictions on unseen data."""
    data_csv = tmp_path / "test_data.csv"
    model_pkl = tmp_path / "test_model.pkl"
    reports_dir = tmp_path / "reports"

    ml_sample_dataset.to_csv(data_csv, index=False)
    train_provider_model(data_path=data_csv, output_model_path=model_pkl, reports_dir=reports_dir)

    unseen_patients = pd.DataFrame([
        {"Age": 55, "Gender": "Male", "Medical Condition": "Diabetes", "Admission Type": "Emergency"},
        {"Age": 72, "Gender": "Female", "Medical Condition": "Cancer", "Admission Type": "Elective"},
    ])

    results = predict_patient_stay_risk(unseen_patients, model_path=model_pkl)
    assert len(results) == 2
    assert "predicted_long_stay" in results.columns
    assert "risk_probability" in results.columns
    assert "stay_risk_category" in results.columns
    assert results["predicted_long_stay"].iloc[0] in [0, 1]
    assert 0.0 <= results["risk_probability"].iloc[0] <= 1.0


def test_load_model_artifact_missing():
    """Verifies that missing model artifact raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_model_artifact(Path("non_existent_model_file_xyz.pkl"))
