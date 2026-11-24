"""
test_streamlit_app.py
---------------------
Tests for Streamlit application imports, model-loading exception handling,
and form prediction execution.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.ml.predict import predict_patient_stay_risk


def test_streamlit_app_imports():
    """Verifies that app.py imports cleanly without syntax errors."""
    import app
    assert hasattr(app, "tab1")
    assert hasattr(app, "tab2")
    assert hasattr(app, "tab3")


def test_streamlit_prediction_workflow(sample_cleaned_dataframe: pd.DataFrame):
    """Verifies that the prediction workflow invoked by Streamlit form generates valid risk outputs."""
    patient_input = pd.DataFrame([{
        "Age": 45,
        "Gender": "Male",
        "Medical Condition": "Diabetes",
        "Admission Type": "Emergency",
    }])

    res = predict_patient_stay_risk(patient_input)
    assert len(res) == 1
    assert "predicted_long_stay" in res.columns
    assert "risk_probability" in res.columns
    assert "stay_risk_category" in res.columns
    assert res["predicted_long_stay"].iloc[0] in [0, 1]
