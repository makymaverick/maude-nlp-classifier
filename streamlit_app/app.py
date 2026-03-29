"""
MAUDE NLP Classifier — Streamlit Demo App

Tabs:
  1. Train Model         — fetch data, train, view metrics
  2. Single Inference    — classify a free-text narrative
  3. Data Explorer       — filter, search, download records
  4. Pipeline Dashboard  — incremental ingestion status + MLflow run history
"""

import os
import sys
import json
import logging
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path so imports work from streamlit_app/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.openfda_client import fetch_maude_records
from src.ingestion.incremental import (
    run_ingestion,
    load_accumulated,
    ACCUMULATED_DATA_PATH,
)
from src.preprocessing.text_cleaner import clean_dataframe, clean_text
from src.model.classifier import (
    build_pipeline,
    split_data,
    train_pipeline,
    evaluate,
    cross_validate_pipeline,
    dummy_baseline,
    predict_single,
    save_model,
    load_model,
)

matplotlib.use("Agg")

MODEL_PATH = "models/maude_classifier.joblib"
CHAMPION_METRICS_PATH = "models/champion_metrics.json"

# Short-code label colours
SEVERITY_COLORS = {
    "D": "#d62728",    # Death — red
    "I": "#ff7f0e",    # Injury — orange
    "M": "#1f77b4",    # Malfunction — blue
    "O": "#7f7f7f",    # Other — grey
    "UNKNOWN": "#aaaaaa",
}
LABEL_NAMES = {
    "D": "Death",
    "I": "Injury",
    "M": "Malfunction",
    "O": "Other",
    "UNKNOWN": "Unknown",
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
    "by severity: **D**eath · **I**njury · **M**alfunction · **O**ther"
)

# ─────────────────────────────────────────────
# Sidebar — Configuration
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "openFDA API Key (optional)",
        type="password",
        help="Leave blank to use the public rate limit.",
    )
    num_records = st.slider("Records to Fetch", 100, 10_000, 1000, 100)
    model_type = st.selectbox("Classifier", ["logreg", "svm"], index=0)
    drop_unknown = st.checkbox("Exclude UNKNOWN labels", value=True)
    run_tuning = st.checkbox("GridSearchCV Tuning", value=False)
    run_cv = st.checkbox("5-Fold Cross-Validation", value=True,
                          help="Recommended — gives realistic generalisation estimate.")

    st.divider()
    st.markdown("**Model status**")
    if os.path.exists(MODEL_PATH):
        st.success("✅ Trained model found")
        if os.path.exists(CHAMPION_METRICS_PATH):
            with open(CHAMPION_METRICS_PATH) as f:
                champ = json.load(f)
            st.metric("Champion F1", f"{champ.get('f1_weighted', 0):.3f}")
            st.metric("Trained on", f"{champ.get('training_records', '?')} records")
    else:
        st.warning("⚠️ No model yet — train below")

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab_train, tab_infer, tab_explore, tab_pipeline = st.tabs([
    "🔧 Train Model",
    "🔍 Single Inference",
    "📊 Data Explorer",
    "🔄 Pipeline Dashboard",
])

# ══════════════════════════════════════════════
# TAB 1 — Train Model
# ══════════════════════════════════════════════
with tab_train:
    st.subheader("Train the Classifier")
    st.markdown(
        "Fetches records live from the openFDA API, preprocesses narrative text, "
        "runs a dummy baseline sanity check, and trains a "
        f"**{'Logistic Regression' if model_type == 'logreg' else 'Linear SVM'}** model."
    )

    train_btn = st.button("🚀 Fetch Data & Train", use_container_width=True)

    if train_btn:
        progress = st.progress(0, text="Fetching data from openFDA...")

        with st.spinner("Contacting openFDA API..."):
            df_raw = fetch_maude_records(
                total_records=num_records,
                api_key=api_key or None,
            )
        progress.progress(25, text="Preprocessing...")

        df = clean_dataframe(df_raw)
        if drop_unknown:
            df = df[df["severity_label"] != "UNKNOWN"].reset_index(drop=True)

        if len(df) < 50:
            st.error("Not enough records after cleaning. Increase record count.")
            st.stop()

        # Label distribution
        st.markdown("**Label Distribution**")
        dist = df["severity_label"].value_counts().reset_index()
        dist.columns = ["Code", "Count"]
        dist["Label"] = dist["Code"].map(LABEL_NAMES)
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = [SEVERITY_COLORS.get(c, "#aaa") for c in dist["Code"]]
        ax.barh(dist["Label"], dist["Count"], color=colors)
        ax.set_xlabel("Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        progress.progress(40, text="Running dummy baseline...")
        X_train, X_test, y_train, y_test = split_data(df)
        dummy = dummy_baseline(X_train, y_train, X_test, y_test)

        st.info(
            f"**Dummy baseline F1: {dummy['dummy_f1_weighted']:.3f}** — "
            "your model must beat this to be meaningful."
        )

        progress.progress(55, text="Training model...")
        pipeline = build_pipeline(model_type=model_type)
        if run_tuning:
            from src.model.classifier import tune_pipeline
            pipeline = tune_pipeline(pipeline, X_train, y_train, model_type=model_type)
        else:
            pipeline = train_pipeline(pipeline, X_train, y_train)

        # Cross-validation
        cv_result = None
        if run_cv:
            progress.progress(70, text="Running 5-fold cross-validation...")
            fresh = build_pipeline(model_type=model_type)
            cv_result = cross_validate_pipeline(fresh, df["clean_text"], df["severity_label"])

        progress.progress(85, text="Evaluating on held-out test set...")
        metrics = evaluate(pipeline, X_test, y_test)
        save_model(pipeline, MODEL_PATH)
        progress.progress(100, text="Done!")

        # Results
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        col2.metric("Weighted F1", f"{metrics['f1_weighted']:.3f}",
                    delta=f"+{metrics['f1_weighted'] - dummy['dummy_f1_weighted']:.3f} vs dummy")
        if cv_result:
            col3.metric("CV F1 (5-fold)",
                        f"{cv_result['cv_f1_mean']:.3f} ± {cv_result['cv_f1_std']:.3f}")

        st.markdown("**Classification Report**")
        st.code(metrics["classification_report"])

        st.markdown("**Confusion Matrix**")
        cm = metrics["confusion_matrix"]
        classes = metrics["classes"]
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes, ax=ax2)
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        st.session_state["df_train"] = df

# ══════════════════════════════════════════════
# TAB 2 — Single Inference
# ══════════════════════════════════════════════
with tab_infer:
    st.subheader("Classify a Single Adverse Event Narrative")

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
            st.error("No trained model found. Train the model first.")
        else:
            pipeline = load_model(MODEL_PATH)
            cleaned = clean_text(narrative_input)
            result = predict_single(pipeline, cleaned)

            predicted = result["predicted_label"]
            label_name = LABEL_NAMES.get(predicted, predicted)
            color = SEVERITY_COLORS.get(predicted, "#333333")

            st.markdown(
                f"<div style='background:{color};padding:16px;border-radius:8px;"
                f"color:white;font-size:22px;font-weight:bold;text-align:center'>"
                f"Predicted: {label_name} ({predicted})</div>",
                unsafe_allow_html=True,
            )

            if "probabilities" in result:
                st.markdown("**Class Probabilities**")
                proba_df = pd.DataFrame(
                    result["probabilities"].items(), columns=["Code", "Probability"]
                ).sort_values("Probability", ascending=False)
                proba_df["Label"] = proba_df["Code"].map(LABEL_NAMES)

                fig3, ax3 = plt.subplots(figsize=(6, 3))
                bar_colors = [SEVERITY_COLORS.get(c, "#aaa") for c in proba_df["Code"]]
                ax3.barh(proba_df["Label"], proba_df["Probability"], color=bar_colors)
                ax3.set_xlim(0, 1)
                ax3.set_xlabel("Probability")
                plt.tight_layout()
                st.pyplot(fig3)
                plt.close()

            elif "decision_scores" in result:
                st.markdown("**Decision Scores (SVM)**")
                scores_df = pd.DataFrame(
                    result["decision_scores"].items(), columns=["Code", "Score"]
                ).sort_values("Score", ascending=False)
                scores_df["Label"] = scores_df["Code"].map(LABEL_NAMES)
                st.dataframe(scores_df, hide_index=True)

# ══════════════════════════════════════════════
# TAB 3 — Data Explorer
# ══════════════════════════════════════════════
with tab_explore:
    st.subheader("Explore MAUDE Records")

    if "df_train" in st.session_state:
        df_view = st.session_state["df_train"]
    else:
        # Try to load accumulated dataset
        acc = load_accumulated()
        if not acc.empty:
            df_view = acc
        else:
            st.info("Train the model or run ingestion to populate the Data Explorer.")
            df_view = None

    if df_view is not None:
        st.markdown(f"**{len(df_view):,} records loaded**")
        col1, col2 = st.columns(2)
        with col1:
            severity_filter = st.multiselect(
                "Filter by Severity Code",
                options=list(df_view["severity_label"].unique()),
                default=list(df_view["severity_label"].unique()),
            )
        with col2:
            search_term = st.text_input("Search narrative text",
                                         placeholder="e.g. burn, pump, implant")

        filtered = df_view[df_view["severity_label"].isin(severity_filter)]
        if search_term:
            text_col = "clean_text" if "clean_text" in filtered.columns else "narrative_text"
            filtered = filtered[
                filtered[text_col].str.contains(search_term, case=False, na=False)
            ]

        st.markdown(f"Showing **{len(filtered):,}** records")
        display_cols = ["report_number", "date_received", "severity_label", "device_name",
                         "clean_text" if "clean_text" in filtered.columns else "narrative_text"]
        st.dataframe(filtered[display_cols], height=400, hide_index=True)

        st.download_button(
            "⬇️ Download Filtered Records (CSV)",
            data=filtered.to_csv(index=False),
            file_name="maude_filtered.csv",
            mime="text/csv",
        )

# ══════════════════════════════════════════════
# TAB 4 — Pipeline Dashboard
# ══════════════════════════════════════════════
with tab_pipeline:
    st.subheader("🔄 Incremental Pipeline Dashboard")
    st.markdown(
        "The incremental pipeline fetches batches of new records from the openFDA API, "
        "deduplicates them against the accumulated dataset, and retrains the model only "
        "when the new model beats the current champion's F1 score."
    )

    # ── Dataset stats ──────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)

    acc_df = load_accumulated()
    col_a.metric("Accumulated Records", f"{len(acc_df):,}" if not acc_df.empty else "0")

    if os.path.exists(CHAMPION_METRICS_PATH):
        with open(CHAMPION_METRICS_PATH) as f:
            champ = json.load(f)
        col_b.metric("Champion F1", f"{champ.get('f1_weighted', 0):.4f}")
        col_c.metric("Champion trained on", f"{champ.get('training_records', '?')} records")
    else:
        col_b.metric("Champion F1", "—")
        col_c.metric("Champion trained on", "—")

    st.divider()

    # ── Manual ingestion trigger ───────────────────────────────────────────
    st.markdown("**Run Incremental Ingestion Now**")
    batch_size = st.number_input("Batch size (records to fetch)", 500, 50_000, 5000, 500)
    run_retrain = st.checkbox("Retrain after ingestion", value=True)

    if st.button("▶️ Run Ingestion Cycle", use_container_width=True):
        with st.spinner(f"Fetching {batch_size:,} records and deduplicating..."):
            summary = run_ingestion(
                batch_size=int(batch_size),
                api_key=api_key or None,
                retrain=run_retrain,
                cross_validate=run_cv,
                model_type=model_type,
            )

        st.success(
            f"✅ Ingestion complete! "
            f"**{summary['new_records_added']:,}** new records added · "
            f"**{summary['total_accumulated']:,}** total accumulated · "
            f"Retrain: {'✅' if summary['retrain_triggered'] else '⏭️ skipped (no new data)'}"
        )

    st.divider()

    # ── MLflow run history ────────────────────────────────────────────────
    st.markdown("**MLflow Run History**")

    try:
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
        client = mlflow.tracking.MlflowClient()

        experiment = client.get_experiment_by_name("maude-nlp-severity")
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=20,
            )
            if runs:
                run_rows = []
                for r in runs:
                    # CV F1 must stay a pure float column (None for missing runs).
                    # Using a string sentinel like "—" creates a mixed-type column
                    # that PyArrow cannot serialize to Arrow format.
                    cv_raw = r.data.metrics.get("cv_f1_mean")
                    cv_f1_val = round(cv_raw, 4) if cv_raw is not None else None

                    run_rows.append({
                        "Run ID":       r.info.run_id[:8],
                        "Started":      pd.to_datetime(r.info.start_time, unit="ms").strftime("%Y-%m-%d %H:%M"),
                        "F1 (hold-out)": round(r.data.metrics.get("f1_weighted", 0), 4),
                        "CV F1 (mean)": cv_f1_val,
                        "Accuracy":     round(r.data.metrics.get("accuracy", 0), 4),
                        "Dummy F1":     round(r.data.metrics.get("dummy_f1_weighted", 0), 4),
                        "Model":        r.data.params.get("model_type", ""),
                        "Records":      int(r.data.params.get("records_fetched", 0) or 0),
                        "Promoted":     r.data.tags.get("promoted", ""),
                        "Reason":       r.data.tags.get("promotion_reason", ""),
                    })
                runs_df = pd.DataFrame(run_rows)
                # "CV F1 (mean)" is float64 with possible NaN — fully Arrow-compatible
                runs_df["CV F1 (mean)"] = runs_df["CV F1 (mean)"].astype("float64")
                st.dataframe(runs_df, hide_index=True)

                # F1 trend chart
                st.markdown("**F1 Score Trend Across Runs**")
                fig_trend, ax_trend = plt.subplots(figsize=(8, 3))
                # Reverse so oldest run is run #1, newest is last
                f1_vals    = runs_df["F1 (hold-out)"].tolist()[::-1]
                cv_f1_vals = runs_df["CV F1 (mean)"].tolist()[::-1]
                x = range(1, len(f1_vals) + 1)
                ax_trend.plot(x, f1_vals, marker="o", color="#1f77b4",
                              linewidth=2, label="Hold-out F1")
                # Only plot CV F1 line where values are not NaN
                cv_x = [i for i, v in zip(x, cv_f1_vals) if v == v]  # NaN != NaN
                cv_y = [v for v in cv_f1_vals if v == v]
                if cv_x:
                    ax_trend.plot(cv_x, cv_y, marker="s", color="#2ca02c",
                                  linewidth=2, linestyle="--", label="CV F1 (5-fold)")
                ax_trend.axhline(
                    runs_df["Dummy F1"].iloc[0],
                    linestyle=":", color="#d62728", alpha=0.7, label="Dummy baseline"
                )
                ax_trend.set_xlabel("Run #")
                ax_trend.set_ylabel("Weighted F1")
                ax_trend.legend()
                plt.tight_layout()
                st.pyplot(fig_trend)
                plt.close()
            else:
                st.info("No MLflow runs yet. Train the model to create the first run.")
        else:
            st.info("No MLflow experiment found yet. Train the model first.")
    except Exception as e:
        st.warning(f"MLflow not available or no runs logged yet: {e}")

    st.divider()
    st.markdown("**Run Scheduled Pipeline from CLI**")
    st.code(
        "# Run once\n"
        "python -m src.ingestion.incremental --batch 10000\n\n"
        "# Run continuously (daily at 2 AM UTC)\n"
        "python -m src.ingestion.incremental --schedule --cron '0 2 * * *' --batch 10000\n\n"
        "# Run with cross-validation on retrain\n"
        "python -m src.ingestion.incremental --batch 10000 --cross-validate",
        language="bash",
    )
