"""
MAUDE NLP Classifier — Streamlit Demo App

Provides:
  1. Batch inference from openFDA API (live fetch)
  2. Single-record inference on user-entered narrative text
  3. Model training trigger (with progress feedback)
  4. Evaluation metrics dashboard
"""

import os
import sys
import logging
import time
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Add project root to path so imports work from streamlit_app/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.openfda_client import fetch_maude_records
from src.preprocessing.text_cleaner import clean_dataframe, clean_text
from src.model.classifier import (
    build_pipeline,
    split_data,
    train_pipeline,
    evaluate,
    predict_single,
    save_model,
    load_model,
)

matplotlib.use("Agg")

MODEL_PATH = "models/maude_classifier.joblib"
SEVERITY_COLORS = {
    "DEATH":          "#d62728",
    "SERIOUS_INJURY": "#ff7f0e",
    "INJURY":         "#e6c619",
    "MALFUNCTION":    "#1f77b4",
    "UNKNOWN":        "#7f7f7f",
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MAUDE NLP Classifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏥 MAUDE Adverse Event Severity Classifier")
st.markdown(
    "Classify medical device adverse event reports from the "
    "[openFDA MAUDE database](https://open.fda.gov/apis/device/event/) "
    "by severity: **Death · Serious Injury · Injury · Malfunction · Unknown**"
)

# ─────────────────────────────────────────────
# Sidebar — Configuration
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "openFDA API Key (optional)",
        type="password",
        help="Leave blank for public rate limit (~240 req/min with key vs 40 without).",
    )
    num_records = st.slider("Records to Fetch", min_value=100, max_value=5000, value=500, step=100)
    model_type = st.selectbox("Classifier", ["logreg", "svm"], index=0,
                               help="logreg = Logistic Regression, svm = Linear SVM")
    drop_unknown = st.checkbox("Exclude UNKNOWN labels from training", value=True)
    run_tuning = st.checkbox("Hyperparameter Tuning (GridSearchCV)", value=False,
                              help="Slower but may improve accuracy.")

    st.divider()
    st.markdown("**Model status**")
    if os.path.exists(MODEL_PATH):
        st.success("✅ Trained model found")
    else:
        st.warning("⚠️ No model yet — train below")

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab_train, tab_infer, tab_explore = st.tabs(
    ["🔧 Train Model", "🔍 Single Inference", "📊 Data Explorer"]
)

# ══════════════════════════════════════════════
# TAB 1 — Train Model
# ══════════════════════════════════════════════
with tab_train:
    st.subheader("Train the Classifier")
    st.markdown(
        "Fetch records from openFDA, preprocess narrative text, and train a "
        f"**{model_type.upper()}** model with TF-IDF features."
    )

    col_a, col_b = st.columns([1, 2])
    with col_a:
        train_btn = st.button("🚀 Fetch Data & Train", use_container_width=True)

    if train_btn:
        progress = st.progress(0, text="Fetching data from openFDA...")

        # Step 1: Fetch
        with st.spinner("Contacting openFDA API..."):
            df_raw = fetch_maude_records(
                total_records=num_records,
                api_key=api_key or None,
            )
        progress.progress(30, text="Data fetched. Preprocessing...")

        # Step 2: Preprocess
        df = clean_dataframe(df_raw)
        if drop_unknown:
            df = df[df["severity_label"] != "UNKNOWN"].reset_index(drop=True)
        progress.progress(50, text="Preprocessing done. Splitting data...")

        if len(df) < 50:
            st.error("Not enough records after cleaning. Try increasing record count.")
            st.stop()

        # Show label distribution
        st.markdown("**Label Distribution (after cleaning)**")
        dist = df["severity_label"].value_counts().reset_index()
        dist.columns = ["Severity", "Count"]
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = [SEVERITY_COLORS.get(s, "#aaaaaa") for s in dist["Severity"]]
        ax.barh(dist["Severity"], dist["Count"], color=colors)
        ax.set_xlabel("Count")
        ax.set_title("Severity Label Distribution")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Step 3: Train
        progress.progress(60, text="Training model...")
        X_train, X_test, y_train, y_test = split_data(df)
        pipeline = build_pipeline(model_type=model_type)
        if run_tuning:
            from src.model.classifier import tune_pipeline
            pipeline = tune_pipeline(pipeline, X_train, y_train, model_type=model_type)
        else:
            pipeline = train_pipeline(pipeline, X_train, y_train)
        progress.progress(85, text="Evaluating model...")

        # Step 4: Evaluate
        metrics = evaluate(pipeline, X_test, y_test)
        save_model(pipeline, MODEL_PATH)
        progress.progress(100, text="Done!")

        st.success(
            f"✅ Training complete! "
            f"Accuracy: **{metrics['accuracy']:.3f}** | "
            f"Weighted F1: **{metrics['f1_weighted']:.3f}**"
        )

        # Classification report
        st.markdown("**Classification Report**")
        st.code(metrics["classification_report"])

        # Confusion matrix
        st.markdown("**Confusion Matrix**")
        cm = metrics["confusion_matrix"]
        classes = metrics["classes"]
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes, ax=ax2
        )
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        ax2.set_title("Confusion Matrix")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # Store in session state for explore tab
        st.session_state["df_train"] = df

# ══════════════════════════════════════════════
# TAB 2 — Single Inference
# ══════════════════════════════════════════════
with tab_infer:
    st.subheader("Classify a Single Adverse Event Narrative")
    st.markdown(
        "Enter a free-text narrative from a MAUDE report and get an instant severity prediction."
    )

    sample_texts = {
        "Select a sample...": "",
        "Device malfunction (pump)": (
            "The infusion pump alarmed and stopped delivering medication. "
            "No patient injury was reported. The device was returned for analysis."
        ),
        "Patient death": (
            "The patient experienced cardiac arrest following implantation of the pacemaker. "
            "The patient passed away 48 hours post-procedure. The event was considered device-related."
        ),
        "Serious injury — burn": (
            "The patient sustained third degree burns on the left forearm during "
            "electrosurgical procedure due to a grounding pad malfunction."
        ),
    }

    sample_choice = st.selectbox("Load a sample narrative", list(sample_texts.keys()))
    narrative_input = st.text_area(
        "Narrative Text",
        value=sample_texts[sample_choice],
        height=150,
        placeholder="Paste or type the MDR narrative here...",
    )

    if st.button("🔎 Classify", use_container_width=True):
        if not narrative_input.strip():
            st.warning("Please enter a narrative text.")
        elif not os.path.exists(MODEL_PATH):
            st.error("No trained model found. Please train the model first (Train Model tab).")
        else:
            pipeline = load_model(MODEL_PATH)
            cleaned = clean_text(narrative_input)
            result = predict_single(pipeline, cleaned)

            predicted = result["predicted_label"]
            color = SEVERITY_COLORS.get(predicted, "#333333")

            st.markdown(
                f"<div style='background:{color};padding:16px;border-radius:8px;"
                f"color:white;font-size:22px;font-weight:bold;text-align:center'>"
                f"Predicted Severity: {predicted}</div>",
                unsafe_allow_html=True,
            )

            if "probabilities" in result:
                st.markdown("**Class Probabilities**")
                proba_df = pd.DataFrame(
                    result["probabilities"].items(), columns=["Severity", "Probability"]
                ).sort_values("Probability", ascending=False)

                fig3, ax3 = plt.subplots(figsize=(6, 3))
                bar_colors = [SEVERITY_COLORS.get(s, "#aaaaaa") for s in proba_df["Severity"]]
                ax3.barh(proba_df["Severity"], proba_df["Probability"], color=bar_colors)
                ax3.set_xlim(0, 1)
                ax3.set_xlabel("Probability")
                plt.tight_layout()
                st.pyplot(fig3)
                plt.close()

            elif "decision_scores" in result:
                st.markdown("**Decision Scores (SVM)**")
                scores_df = pd.DataFrame(
                    result["decision_scores"].items(), columns=["Severity", "Score"]
                ).sort_values("Score", ascending=False)
                st.dataframe(scores_df, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — Data Explorer
# ══════════════════════════════════════════════
with tab_explore:
    st.subheader("Explore MAUDE Records")

    if "df_train" in st.session_state:
        df_view = st.session_state["df_train"]
    else:
        st.info("Train the model first to populate the Data Explorer, or fetch records directly.")
        if st.button("📥 Fetch Records Only"):
            with st.spinner("Fetching..."):
                df_view = fetch_maude_records(total_records=num_records, api_key=api_key or None)
                df_view = clean_dataframe(df_view)
                st.session_state["df_train"] = df_view
            st.rerun()
        df_view = None

    if df_view is not None:
        st.markdown(f"**{len(df_view):,} records loaded**")

        # Filters
        col1, col2 = st.columns(2)
        with col1:
            severity_filter = st.multiselect(
                "Filter by Severity",
                options=list(df_view["severity_label"].unique()),
                default=list(df_view["severity_label"].unique()),
            )
        with col2:
            search_term = st.text_input("Search narrative text", placeholder="e.g. burn, pump, implant")

        filtered = df_view[df_view["severity_label"].isin(severity_filter)]
        if search_term:
            filtered = filtered[filtered["clean_text"].str.contains(search_term, case=False, na=False)]

        st.markdown(f"Showing **{len(filtered):,}** matching records")
        st.dataframe(
            filtered[["report_number", "date_received", "severity_label", "device_name", "clean_text"]]
            .rename(columns={"clean_text": "narrative (cleaned)"}),
            use_container_width=True,
            height=400,
        )

        # Download CSV
        csv_data = filtered.to_csv(index=False)
        st.download_button(
            "⬇️ Download Filtered Records (CSV)",
            data=csv_data,
            file_name="maude_filtered.csv",
            mime="text/csv",
        )
