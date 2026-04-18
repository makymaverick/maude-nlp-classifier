"""
MAUDE NLP Severity Classifier — Streamlit Demo App

Two tabs:
  1. Classify      — paste a MAUDE narrative, get a severity prediction
  2. About         — model card: training data, approach, evaluation metrics
"""

import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

matplotlib.use("Agg")

# Add project root so src.* imports resolve from streamlit_app/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.classifier import load_model, predict_single
from src.preprocessing.text_cleaner import clean_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH            = "models/maude_classifier.joblib"
CHAMPION_METRICS_PATH = "models/champion_metrics.json"

# Severity label display config
SEVERITY_COLORS = {
    "D": "#d62728",
    "I": "#ff7f0e",
    "M": "#1f77b4",
    "O": "#7f7f7f",
    "UNKNOWN": "#aaaaaa",
}
LABEL_NAMES = {
    "D": "Death",
    "I": "Injury",
    "M": "Malfunction",
    "O": "Other",
    "UNKNOWN": "Unknown",
}

# ── Sample narratives (let recruiters/interviewers try the demo immediately) ─
SAMPLE_NARRATIVES = {
    "Select a sample…": "",
    "Device malfunction during procedure": (
        "The catheter failed to navigate to the target site during the procedure. "
        "The device kinked and could not be advanced. The procedure was aborted and "
        "the device was removed. No patient injury was reported."
    ),
    "Serious patient injury": (
        "Patient experienced significant blood loss following device failure during "
        "implantation. Emergency intervention was required. Patient was transferred "
        "to ICU and required transfusion. The device lead had fractured at the "
        "connector site."
    ),
    "Patient death following device use": (
        "Patient was found unresponsive approximately 6 hours after device activation. "
        "Resuscitation attempts were unsuccessful. Autopsy results pending. "
        "The implanted neurostimulator was recovered for analysis. "
        "Cause of death under investigation."
    ),
}


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MAUDE NLP Classifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏥 MAUDE Adverse Event Severity Classifier")
st.markdown(
    "Classifies FDA MAUDE medical device adverse event narratives by severity: "
    "**Death · Injury · Malfunction · Other**. "
    "Built on 9 years of domain expertise in post-market surveillance and adverse "
    "event reporting."
)


# ── Sidebar — model status ────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Status")

    if os.path.exists(MODEL_PATH):
        st.success("✅ Model ready")
        if os.path.exists(CHAMPION_METRICS_PATH):
            with open(CHAMPION_METRICS_PATH) as f:
                champ = json.load(f)
            cv_f1 = champ.get("cv_f1_mean") or champ.get("f1_weighted", 0)
            st.metric("CV F1 (5-fold)", f"{cv_f1:.3f}")
            st.metric("Trained on", f"{champ.get('training_records', '?'):,} records")
    else:
        st.warning(
            "⚠️ No trained model found.\n\n"
            "Run from the project root:\n"
            "```\npython -m src.model.train --records 5000\n```"
        )

    st.divider()
    st.caption(
        "Model: TF-IDF + Logistic Regression\n\n"
        "Data source: openFDA MAUDE API\n\n"
        "Evaluation: StratifiedKFold (5-fold)\n\n"
        "[GitHub](https://github.com/makymaverick/maude-nlp-classifier)"
    )


# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def _load_model_cached():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


pipeline = _load_model_cached()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_classify, tab_about = st.tabs(["🔍 Classify", "ℹ️ About the model"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CLASSIFY
# ══════════════════════════════════════════════════════════════════════════════
with tab_classify:
    st.subheader("Classify an adverse event narrative")
    st.markdown(
        "Paste the free-text narrative from an MDR report below. "
        "The model cleans and vectorises the text and returns a predicted "
        "severity class with confidence scores."
    )

    # Sample narrative picker
    sample_choice = st.selectbox("Or load a sample narrative:", list(SAMPLE_NARRATIVES.keys()))
    sample_text = SAMPLE_NARRATIVES[sample_choice]

    narrative_input = st.text_area(
        "Narrative text",
        value=sample_text,
        height=180,
        placeholder=(
            "Paste the MDR narrative here. Example: 'The device failed to deploy "
            "during the procedure. No patient injury was reported…'"
        ),
    )

    col_classify, col_clear = st.columns([1, 5])
    classify_clicked = col_classify.button("Classify", type="primary", use_container_width=True)
    if col_clear.button("Clear", use_container_width=True):
        narrative_input = ""
        st.rerun()

    if classify_clicked:
        if not narrative_input.strip():
            st.warning("Please enter a narrative to classify.")
        elif pipeline is None:
            st.error(
                "No trained model found. "
                "Run `python -m src.model.train --records 5000` from the project root first."
            )
        else:
            # Clean text before inference (same pipeline as training)
            cleaned = clean_text(narrative_input)

            if not cleaned:
                st.warning("The narrative was empty after cleaning. Try a longer text.")
            else:
                result = predict_single(pipeline, cleaned)
                label_code = result["predicted_label"]
                label_name = LABEL_NAMES.get(label_code, label_code)
                color = SEVERITY_COLORS.get(label_code, "#888888")

                st.divider()
                col_pred, col_scores = st.columns([1, 2])

                with col_pred:
                    st.markdown("**Predicted severity**")
                    st.markdown(
                        f"<div style='background:{color}22; border-left:5px solid {color}; "
                        f"padding:16px 20px; border-radius:6px;'>"
                        f"<span style='font-size:2rem; font-weight:700; color:{color};'>"
                        f"{label_name}</span><br>"
                        f"<span style='color:#555; font-size:0.85rem;'>Code: {label_code}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                with col_scores:
                    if "probabilities" in result:
                        st.markdown("**Confidence scores**")
                        proba = result["probabilities"]
                        proba_df = pd.DataFrame([
                            {
                                "Severity": LABEL_NAMES.get(k, k),
                                "Confidence": v,
                            }
                            for k, v in sorted(proba.items(), key=lambda x: -x[1])
                        ])
                        fig, ax = plt.subplots(figsize=(5, 2.5))
                        bars = ax.barh(
                            proba_df["Severity"],
                            proba_df["Confidence"],
                            color=[
                                SEVERITY_COLORS.get(
                                    [k for k, v in LABEL_NAMES.items() if v == row][0], "#888"
                                )
                                for row in proba_df["Severity"]
                            ],
                        )
                        ax.set_xlim(0, 1)
                        ax.set_xlabel("Confidence")
                        ax.invert_yaxis()
                        for bar, val in zip(bars, proba_df["Confidence"]):
                            ax.text(
                                bar.get_width() + 0.01,
                                bar.get_y() + bar.get_height() / 2,
                                f"{val:.3f}",
                                va="center",
                                fontsize=9,
                            )
                        ax.grid(axis="x", alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    elif "decision_scores" in result:
                        st.markdown("**Decision scores** (LinearSVC — not probabilities)")
                        for k, v in sorted(
                            result["decision_scores"].items(), key=lambda x: -x[1]
                        ):
                            st.write(f"{LABEL_NAMES.get(k, k)}: `{v:.4f}`")

                # Show cleaned text for transparency
                with st.expander("Cleaned text (what the model actually sees)"):
                    st.code(cleaned)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ABOUT THE MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.subheader("About this model")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### What it does")
        st.markdown(
            "Classifies free-text narrative descriptions from FDA MAUDE "
            "(Manufacturer and User Facility Device Experience) adverse event "
            "reports into four severity categories:\n\n"
            "- **D — Death**: patient death reported in connection with device use\n"
            "- **I — Injury**: serious or non-serious patient injury\n"
            "- **M — Malfunction**: device failed to meet specifications; no patient harm\n"
            "- **O — Other**: reports not fitting the above categories\n\n"
            "MAUDE is the FDA database I worked with for 9 years during post-market "
            "surveillance and complaint analytics at TCS. "
            "Manual severity triage of these reports is time-consuming and "
            "inconsistent across analysts — this classifier automates the first-pass "
            "categorisation."
        )

        st.markdown("#### Why it matters")
        st.markdown(
            "FDA receives hundreds of thousands of MDR submissions annually. "
            "Pharmacovigilance teams prioritise investigation queues by severity — "
            "Death and Injury reports must be escalated within 30 days under "
            "21 CFR Part 803. A classifier that reliably separates Death/Injury "
            "from Malfunction/Other reduces analyst review time and improves "
            "signal detection response time."
        )

    with col_r:
        st.markdown("#### Model approach")
        st.markdown(
            "**Algorithm:** TF-IDF vectorisation + Logistic Regression\n\n"
            "**Why TF-IDF + LR as baseline:**\n"
            "Interpretable, fast to train, and well-suited for bag-of-words "
            "classification on short clinical texts. Logistic Regression with "
            "`class_weight='balanced'` explicitly handles the class imbalance "
            "in MAUDE data — Death events are under 10% of records.\n\n"
            "**Key preprocessing steps:**\n"
            "- Medical abbreviation expansion (pt → patient, dx → diagnosis)\n"
            "- MAUDE boilerplate removal ('it was reported that…')\n"
            "- Bigram features to capture clinical phrases\n"
            "- Log-normalised TF (sublinear_tf) to compress high-frequency terms\n\n"
            "**Evaluation:**\n"
            "5-fold StratifiedKFold cross-validation — ensures each fold has "
            "proportional class representation, giving a stable F1 estimate "
            "even on smaller datasets where a single train/test split can "
            "vary by ±0.10 F1."
        )

    st.divider()

    # Live metrics from champion_metrics.json
    st.markdown("#### Current model metrics")

    if os.path.exists(CHAMPION_METRICS_PATH):
        with open(CHAMPION_METRICS_PATH) as f:
            champ = json.load(f)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CV F1 (5-fold, weighted)", f"{champ.get('cv_f1_mean', 0):.3f}")
        m2.metric("CV F1 std", f"±{champ.get('cv_f1_std', 0):.3f}")
        m3.metric("Hold-out accuracy", f"{champ.get('accuracy', 0):.3f}")
        m4.metric("Training records", f"{champ.get('training_records', '?'):,}")
    else:
        st.info("No trained model yet — run `python -m src.model.train --records 5000`")

    st.divider()

    # Class imbalance explanation
    st.markdown("#### The class imbalance challenge")
    st.markdown(
        "MAUDE data reflects the real-world distribution of device adverse events: "
        "most reports are Malfunction, followed by Injury, with Death under 10%. "
        "A naive classifier optimising for accuracy would predict Malfunction for "
        "ambiguous cases and score well overall — but fail on the clinically critical "
        "Death and Injury classes.\n\n"
        "This was identified empirically via confusion matrix analysis: Death cases "
        "were systematically misclassified as Injury. The fix was `class_weight='balanced'` "
        "on the Logistic Regression, which penalises misclassification of rare classes "
        "proportionally to their inverse frequency in the training data."
    )

    st.divider()

    st.markdown("#### Data source")
    st.markdown(
        "Records are fetched from the "
        "[openFDA MAUDE API](https://open.fda.gov/apis/device/event/) "
        "using paginated requests. The API is free and requires no authentication "
        "for the public rate limit. Training uses the `event_type` field as the "
        "classification label and `mdr_text` narrative fields as input text.\n\n"
        "The openFDA API caps pagination at 25,000 records per query. "
        "To collect larger datasets, the ingestion client partitions queries "
        "into yearly date-range windows, resetting the offset for each year."
    )

    st.divider()

    st.markdown("#### Roadmap")
    st.markdown(
        "**v1 (current):** TF-IDF + Logistic Regression baseline, "
        "StratifiedKFold evaluation, MLflow experiment tracking\n\n"
        "**v2 (planned):** Fine-tune `emilyalsentzer/Bio_ClinicalBERT` "
        "on the same dataset to measure F1 uplift on the Death class — "
        "the primary motivation being that TF-IDF cannot capture semantic "
        "similarity between clinical phrases ('cardiac arrest' vs 'heart stopped')"
    )

    st.divider()
    st.caption(
        "Built by Mukund Padmanabha · ISB AMPBA 2025 · "
        "9 years FDA & EU MDR regulatory experience · "
        "[GitHub](https://github.com/makymaverick/maude-nlp-classifier)"
    )
