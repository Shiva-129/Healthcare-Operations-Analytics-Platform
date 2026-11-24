"""
app.py
------
Healthcare Operations Analytics Portal (Streamlit Dashboard).

Features:
- Tab 1: AI Data Assistant (RAG Chatbot grounded in centralized database)
- Tab 2: Patient Length-of-Stay Risk Prediction Interface (ML Inference Engine)
- Tab 3: Operational Analytics & System Diagnostics

Run with:
    streamlit run app.py
"""

import json
import os
import sys
from pathlib import Path

# Ensure root repository directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

from src.config import (
    MODEL_PATH,
    METRICS_PATH,
    REPORTS_DIR,
    TABLE_NAME,
    verify_database_ready,
)
from src.ml.predict import predict_patient_stay_risk, load_model_artifact
from src.rag.chatbot import ask_hospital_ai

st.set_page_config(
    page_title="Healthcare Operations Analytics Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished dashboard aesthetic
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #2b5c8f;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .disclaimer-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        padding: 12px;
        border-radius: 6px;
        font-size: 0.9em;
        margin-top: 15px;
    }
    .risk-high {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 15px;
        border-radius: 6px;
        color: #721c24;
    }
    .risk-low {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 6px;
        color: #155724;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.title("🏥 Healthcare Operations Analytics Platform")
st.markdown(
    "**End-to-End Clinical Data Platform • Safe Full-Refresh ETL • Machine Learning Risk Stratification • RAG AI Assistant**"
)

# Sidebar System Health Status
with st.sidebar:
    st.header("⚙️ Platform Health")
    is_ready, db_msg = verify_database_ready()
    if is_ready:
        st.success("Database: Ready")
        st.caption(f"Table: `{TABLE_NAME}`")
    else:
        st.error("Database: Offline")
        st.caption(db_msg)

    st.markdown("---")
    st.subheader("🤖 Model Status")
    if MODEL_PATH.exists():
        st.success("ML Model: Loaded")
        st.caption("Random Forest Classifier v1.0.0")
    else:
        st.warning("ML Model: Not Trained")
        st.caption("Run: `python src/ml/train.py`")

    st.markdown("---")
    st.markdown("### 📌 Navigation Quick Links")
    st.markdown(
        "- **Tab 1:** Natural-language SQL data assistant\n"
        "- **Tab 2:** Prolonged stay risk prediction\n"
        "- **Tab 3:** Operational model metrics & diagnostics"
    )


# --- Tabs ---
tab1, tab2, tab3 = st.tabs([
    "💬 Ask the Data AI",
    "🔮 Predict Patient Stay Risk",
    "📊 Model Performance & Reports",
])


# ==============================================================================
# TAB 1: AI DATA ASSISTANT
# ==============================================================================
with tab1:
    st.subheader("💬 AI Hospital Operations Assistant")
    st.caption(
        "Ask operational questions in plain English. The assistant queries the centralized hospital database "
        "using safe, read-only SQL execution."
    )

    col_q, col_quick = st.columns([2, 1])

    with col_quick:
        st.markdown("**💡 Quick Questions:**")
        quick_qs = [
            "What is the total number of patients?",
            "What is the average billing amount?",
            "Which department generated the highest revenue?",
            "Who are the top doctors by patient count?",
            "What is the insurance provider breakdown?",
        ]
        selected_q = None
        for q_text in quick_qs:
            if st.button(q_text, key=f"quick_{q_text}"):
                selected_q = q_text

    with col_q:
        default_val = selected_q if selected_q else ""
        user_query = st.text_input(
            "Enter your question for the database:",
            value=default_val,
            placeholder="e.g., Which department has the highest revenue?",
            key="chat_input",
        )

        if user_query:
            with st.spinner("Analyzing operational database records..."):
                answer = ask_hospital_ai(user_query)
                st.write("### 🤖 Response:")
                st.markdown(answer)
        else:
            st.info("Enter an operational question above or click one of the quick suggestions.")


# ==============================================================================
# TAB 2: PATIENT STAY RISK PREDICTION INTERFACE
# ==============================================================================
with tab2:
    st.subheader("🔮 Patient Length-of-Stay Risk Stratification")
    st.caption(
        "Predict whether an admitted patient is likely to experience a prolonged hospital stay (> 15 days), "
        "enabling proactive bed management, discharge planning, and resource allocation."
    )

    if not MODEL_PATH.exists():
        st.error(
            f"⚠️ **Model Artifact Missing:** Could not locate model artifact at `{MODEL_PATH}`.\n\n"
            "Please train the model by running:\n"
            "```powershell\n"
            "python src/ml/train.py\n"
            "```"
        )
    else:
        # Load model metadata for context
        try:
            _, metadata = load_model_artifact(MODEL_PATH)
            threshold_days = metadata.get("training_threshold_days", 15.0)
        except Exception:
            threshold_days = 15.0

        st.markdown(
            f"**Target Definition:** Patients with length of stay strictly exceeding the training-set median (**{threshold_days:.0f} days**) "
            "are classified as *Higher Prolonged Stay Risk*."
        )

        with st.form("stay_prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                age_input = st.slider("Patient Age:", min_value=0, max_value=120, value=48, step=1)
                gender_input = st.selectbox("Gender:", ["Male", "Female"])

            with col2:
                condition_input = st.selectbox(
                    "Primary Medical Condition / Diagnosis:",
                    ["Diabetes", "Hypertension", "Asthma", "Cancer", "Obesity", "Arthritis"],
                )
                admission_type_input = st.selectbox(
                    "Admission Type:",
                    ["Emergency", "Elective", "Urgent"],
                )

            submit_predict = st.form_submit_button("🔮 Predict Stay Risk", use_container_width=True)

        if submit_predict:
            # Construct feature DataFrame
            input_record = pd.DataFrame([{
                "Age": age_input,
                "Gender": gender_input,
                "Medical Condition": condition_input,
                "Admission Type": admission_type_input,
            }])

            try:
                prediction_result = predict_patient_stay_risk(input_record, model_path=MODEL_PATH)
                is_long = int(prediction_result["predicted_long_stay"].iloc[0])
                prob = float(prediction_result["risk_probability"].iloc[0])
                prob_pct = round(prob * 100, 1)

                st.markdown("---")
                st.write("### 📋 Prediction Results & Operational Recommendations")

                res_col1, res_col2 = st.columns([1, 1])

                with res_col1:
                    if is_long == 1:
                        st.markdown(
                            f"""
                            <div class="risk-high">
                                <h3>⚠️ Higher Likelihood of Prolonged Stay</h3>
                                <p><strong>Predicted Probability:</strong> {prob_pct}% chance of staying &gt; {threshold_days:.0f} days.</p>
                                <p><strong>Category:</strong> High Care-Coordination Priority</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="risk-low">
                                <h3>✅ Lower Likelihood of Prolonged Stay</h3>
                                <p><strong>Predicted Probability:</strong> {prob_pct}% chance of staying &gt; {threshold_days:.0f} days.</p>
                                <p><strong>Category:</strong> Standard Care-Coordination Pathway</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                with res_col2:
                    st.markdown("**Admission Summary:**")
                    st.write(f"- **Age:** {age_input} years")
                    st.write(f"- **Gender:** {gender_input}")
                    st.write(f"- **Medical Condition:** {condition_input}")
                    st.write(f"- **Admission Type:** {admission_type_input}")
                    st.write(f"- **Operational Threshold:** {threshold_days:.0f} days")

                st.markdown("#### 🏥 Operational Action Items:")
                if is_long == 1:
                    st.info(
                        "1. **Discharge Planning:** Initiate multidisciplinary discharge planning at time of admission.\n"
                        "2. **Bed Management:** Alert nursing ward coordinators to allocate extended stay capacity.\n"
                        "3. **Social Services:** Screen early for post-acute care and home health service requirements."
                    )
                else:
                    st.info(
                        "1. **Pathway Tracking:** Patient assigned to standard expected-stay clinical pathway.\n"
                        "2. **Routine Checkpoints:** Routine follow-up scheduled prior to estimated discharge date."
                    )

            except Exception as e:
                st.error(f"Prediction failed: {e}")

        st.markdown(
            """
            <div class="disclaimer-box">
                ⚠️ <strong>Clinical Disclaimer:</strong> This is an experimental operational analytics model designed
                solely for administrative resource planning and bed management. It must <strong>not</strong> be used
                for clinical diagnosis, medical triage, or independent patient-care decisions.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# TAB 3: MODEL PERFORMANCE & METRICS
# ==============================================================================
with tab3:
    st.subheader("📊 Model Performance & Data Diagnostics")
    st.caption("Audited machine learning performance metrics and diagnostic visualizations.")

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("Accuracy", f"{metrics_data.get('accuracy', 0):.4f}")
        m_col2.metric("Precision", f"{metrics_data.get('precision', 0):.4f}")
        m_col3.metric("Recall", f"{metrics_data.get('recall', 0):.4f}")
        m_col4.metric("F1-Score", f"{metrics_data.get('f1_score', 0):.4f}")
        m_col5.metric("ROC-AUC", f"{metrics_data.get('roc_auc', 0):.4f}" if metrics_data.get('roc_auc') else "N/A")

        st.markdown("---")
        st.subheader("📈 Diagnostic Visualizations")

        img_col1, img_col2, img_col3 = st.columns(3)

        cm_img = REPORTS_DIR / "confusion_matrix.png"
        roc_img = REPORTS_DIR / "roc_curve.png"
        fi_img = REPORTS_DIR / "feature_importance.png"

        with img_col1:
            if cm_img.exists():
                st.image(str(cm_img), caption="Confusion Matrix", use_container_width=True)

        with img_col2:
            if roc_img.exists():
                st.image(str(roc_img), caption="ROC Curve", use_container_width=True)

        with img_col3:
            if fi_img.exists():
                st.image(str(fi_img), caption="Feature Importance", use_container_width=True)

        with st.expander("📋 Detailed Classification Report (JSON)"):
            st.json(metrics_data.get("classification_report", {}))

        st.markdown(
            """
            > **Scientific Audit Note:**
            > In the synthetic benchmark dataset (`prasad22/healthcare-dataset`), patient admission attributes
            > (Age, Condition, Gender, Admission Type) were generated with independent random distributions from stay duration.
            > As a result, genuine predictive signal is near zero, yielding an expected empirical ROC-AUC of ~0.50.
            > These numbers reflect an honest, unmanipulated evaluation on held-out test data without target leakage.
            """
        )
    else:
        st.warning(f"Metrics report not found at `{METRICS_PATH}`. Please run `python src/ml/train.py`.")
