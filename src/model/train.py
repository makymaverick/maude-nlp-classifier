"""
Training entrypoint for MAUDE NLP Classifier.

Includes:
  - openFDA batch ingestion (paginated, no static downloads)
  - Text preprocessing
  - StratifiedKFold cross-validation (always runs — required for reliable promotion)
  - Dummy baseline comparison
  - MLflow experiment tracking & model versioning
  - Robust model promotion with three safeguards against small-data inflation:
      1. Comparison uses CV F1, not hold-out F1
         (CV F1 is far more stable on small datasets)
      2. Minimum records gate: promotion comparison is skipped until
         MIN_RECORDS_FOR_COMPARISON records are accumulated
         (below this threshold, metrics are unreliable and the first
          trained model is always promoted)
      3. Tolerance band: new model must beat champion CV F1 by at least
         PROMOTION_MIN_DELTA, not just epsilon
         (prevents promoting noise as signal)

Usage:
    python -m src.model.train --records 5000 --model logreg
    python -m src.model.train --records 10000 --model logreg --tune
"""

import argparse
import logging
import os
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from src.ingestion.openfda_client import fetch_maude_records, save_raw_data
from src.preprocessing.text_cleaner import clean_dataframe, get_label_distribution
from src.model.classifier import (
    build_pipeline,
    split_data,
    train_pipeline,
    tune_pipeline,
    evaluate,
    cross_validate_pipeline,
    dummy_baseline,
    save_model,
    load_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_PATH = "data/raw/maude_raw.csv"
MODEL_PATH = "models/maude_classifier.joblib"
CHAMPION_METRICS_PATH = "models/champion_metrics.json"
MLFLOW_EXPERIMENT = "maude-nlp-severity"

# ── Promotion safeguard constants ────────────────────────────────────────────
#
# Below this record count, the dataset is too small for metrics to be reliable.
# The first model trained above this threshold becomes the baseline champion.
# Any model trained with fewer records is always promoted (no comparison).
MIN_RECORDS_FOR_COMPARISON = 10_000

# New model must beat the champion's CV F1 by at least this much.
# Prevents promoting a model that improved by statistical noise (e.g. 0.001).
# Rule of thumb: ~0.5-1% of F1 is meaningful on a real dataset.
PROMOTION_MIN_DELTA = 0.005


# ── Champion metrics helpers ─────────────────────────────────────────────────

def _get_champion_metrics() -> dict:
    """
    Read the current champion's metrics from disk.

    Returns an empty dict if no champion exists yet.
    The key used for promotion comparison is 'cv_f1_mean', not 'f1_weighted',
    because CV F1 is stable across dataset sizes whereas hold-out F1 is not.
    """
    if os.path.exists(CHAMPION_METRICS_PATH):
        with open(CHAMPION_METRICS_PATH) as f:
            return json.load(f)
    return {}


def _get_champion_f1() -> float:
    """Return champion's CV F1 mean (used by tests and Streamlit dashboard)."""
    data = _get_champion_metrics()
    # Prefer cv_f1_mean; fall back to f1_weighted for backwards compatibility
    return data.get("cv_f1_mean") or data.get("f1_weighted", 0.0)


def _save_champion_metrics(metrics: dict) -> None:
    """Persist the promoted model's metrics."""
    os.makedirs(os.path.dirname(CHAMPION_METRICS_PATH), exist_ok=True)
    with open(CHAMPION_METRICS_PATH, "w") as f:
        json.dump(
            {
                "f1_weighted": metrics["f1_weighted"],
                "accuracy": metrics["accuracy"],
                "cv_f1_mean": metrics.get("cv_f1_mean"),
                "cv_f1_std": metrics.get("cv_f1_std"),
                "training_records": metrics.get("training_records"),
            },
            f,
            indent=2,
        )
    logger.info(f"Champion metrics saved to {CHAMPION_METRICS_PATH}")


def _should_promote(
    new_cv_f1: float,
    champion_cv_f1: float,
    training_records: int,
    no_champion_exists: bool,
) -> tuple[bool, str]:
    """
    Decide whether to promote the new model to champion, and explain why.

    Three-gate promotion logic:

    Gate 1 — No existing champion: always promote.
    Gate 2 — Minimum records: below MIN_RECORDS_FOR_COMPARISON, always promote
              because there is not enough data for the metrics to be trustworthy.
    Gate 3 — CV F1 improvement with tolerance band: new CV F1 must exceed
              champion CV F1 by at least PROMOTION_MIN_DELTA.

    Returns:
        Tuple of (should_promote: bool, reason: str)
    """
    if no_champion_exists:
        return True, "no_champion_exists"

    if training_records < MIN_RECORDS_FOR_COMPARISON:
        return True, (
            f"below_min_records_threshold "
            f"({training_records} < {MIN_RECORDS_FOR_COMPARISON}) — "
            f"metrics unreliable on small datasets, always promoting"
        )

    improvement = new_cv_f1 - champion_cv_f1
    if improvement >= PROMOTION_MIN_DELTA:
        return True, (
            f"cv_f1_improved "
            f"({new_cv_f1:.4f} vs {champion_cv_f1:.4f}, "
            f"delta={improvement:+.4f} >= threshold={PROMOTION_MIN_DELTA})"
        )

    return False, (
        f"cv_f1_insufficient_improvement "
        f"({new_cv_f1:.4f} vs {champion_cv_f1:.4f}, "
        f"delta={improvement:+.4f} < threshold={PROMOTION_MIN_DELTA})"
    )


# ── Main training function ───────────────────────────────────────────────────

def main(args):
    # Set up MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run() as run:
        logger.info(f"MLflow run ID: {run.info.run_id}")

        # Log training configuration
        mlflow.log_params({
            "model_type": args.model,
            "records_requested": getattr(args, "records", "N/A"),
            "drop_unknown": args.drop_unknown,
            "hyperparameter_tuning": args.tune,
            "min_records_for_comparison": MIN_RECORDS_FOR_COMPARISON,
            "promotion_min_delta": PROMOTION_MIN_DELTA,
        })

        # ── 1. Ingest ────────────────────────────────────────────────────────
        if args.use_cached and os.path.exists(RAW_DATA_PATH):
            logger.info(f"Loading cached data from {RAW_DATA_PATH}")
            df = pd.read_csv(RAW_DATA_PATH)
        else:
            logger.info(f"Fetching {args.records} records from openFDA MAUDE API...")
            df = fetch_maude_records(total_records=args.records)
            save_raw_data(df, RAW_DATA_PATH)

        logger.info(f"Raw data shape: {df.shape}")
        mlflow.log_param("records_fetched", len(df))

        # ── 2. Preprocess ────────────────────────────────────────────────────
        df = clean_dataframe(df)

        if args.drop_unknown:
            before = len(df)
            df = df[df["severity_label"] != "UNKNOWN"].reset_index(drop=True)
            logger.info(f"Dropped UNKNOWN labels: {before - len(df)} rows removed.")

        label_dist = get_label_distribution(df)
        logger.info(f"Label distribution after cleaning:\n{label_dist}")

        for label, count in label_dist.items():
            mlflow.log_metric(f"class_count_{label}", count)

        mlflow.log_param("training_records", len(df))

        if len(df) < 50:
            logger.error("Not enough records after cleaning. Aborting.")
            return

        # ── 3. Train/test split ──────────────────────────────────────────────
        X_train, X_test, y_train, y_test = split_data(df)

        # ── 4. Dummy baseline (sanity check) ─────────────────────────────────
        dummy = dummy_baseline(X_train, y_train, X_test, y_test)
        mlflow.log_metrics({
            "dummy_accuracy": dummy["dummy_accuracy"],
            "dummy_f1_weighted": dummy["dummy_f1_weighted"],
        })

        # ── 5. Build & train ─────────────────────────────────────────────────
        pipeline = build_pipeline(model_type=args.model)
        if args.tune:
            pipeline = tune_pipeline(pipeline, X_train, y_train, model_type=args.model)
        else:
            pipeline = train_pipeline(pipeline, X_train, y_train)

        # ── 6. StratifiedKFold cross-validation (always runs) ────────────────
        #
        # CV F1 is the single source of truth for the promotion decision.
        # It must always run so that every model has a comparable, stable score.
        # A single hold-out split on a small dataset can vary by ±0.10–0.15 F1
        # depending on which records end up in the test set; CV F1 variance
        # is typically 3–5x lower.
        logger.info("Running 5-fold stratified cross-validation (required for promotion)...")
        fresh_pipe = build_pipeline(model_type=args.model)
        cv_results = cross_validate_pipeline(
            fresh_pipe, df["clean_text"], df["severity_label"], n_splits=5
        )
        cv_metrics = {
            "cv_f1_mean": cv_results["cv_f1_mean"],
            "cv_f1_std": cv_results["cv_f1_std"],
        }
        mlflow.log_metrics(cv_metrics)
        for i, fold_score in enumerate(cv_results["cv_f1_per_fold"]):
            mlflow.log_metric(f"cv_f1_fold_{i+1}", fold_score)

        # ── 7. Hold-out evaluation (for reporting, not for promotion) ────────
        metrics = evaluate(pipeline, X_test, y_test)
        metrics.update(cv_metrics)
        metrics["training_records"] = len(df)

        mlflow.log_metrics({
            "accuracy": metrics["accuracy"],
            "f1_weighted": metrics["f1_weighted"],
        })
        mlflow.log_text(metrics["classification_report"], "classification_report.txt")
        mlflow.sklearn.log_model(pipeline, "model")

        logger.info(
            f"\n{'='*55}\n"
            f"  Records trained on: {len(df):,}\n"
            f"  Hold-out Accuracy:  {metrics['accuracy']:.4f}\n"
            f"  Hold-out F1:        {metrics['f1_weighted']:.4f}  "
            f"({'BEATS' if metrics['f1_weighted'] > dummy['dummy_f1_weighted'] else 'DOES NOT BEAT'} "
            f"dummy={dummy['dummy_f1_weighted']:.4f})\n"
            f"  CV F1 (5-fold):     {cv_metrics['cv_f1_mean']:.4f} "
            f"± {cv_metrics['cv_f1_std']:.4f}  ← used for promotion\n"
            f"{'='*55}"
        )

        # ── 8. Robust model promotion ────────────────────────────────────────
        champion_data = _get_champion_metrics()
        no_champion = not champion_data
        champion_cv_f1 = _get_champion_f1()
        new_cv_f1 = cv_metrics["cv_f1_mean"]

        promote, reason = _should_promote(
            new_cv_f1=new_cv_f1,
            champion_cv_f1=champion_cv_f1,
            training_records=len(df),
            no_champion_exists=no_champion,
        )

        mlflow.log_metric("champion_cv_f1_before", champion_cv_f1)
        mlflow.set_tag("promotion_reason", reason)

        if promote:
            logger.info(f"✅ Promoting new model. Reason: {reason}")
            save_model(pipeline, MODEL_PATH)
            _save_champion_metrics(metrics)
            mlflow.set_tag("promoted", "true")
        else:
            logger.info(f"⏭️  Keeping existing champion. Reason: {reason}")
            mlflow.set_tag("promoted", "false")

        mlflow.set_tag("model_type", args.model)
        logger.info(f"Training complete. MLflow run: {run.info.run_id}")


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Train MAUDE NLP Severity Classifier")
    parser.add_argument("--records", type=int, default=5000,
                        help="Number of MAUDE records to fetch per run")
    parser.add_argument("--model", type=str, default="logreg",
                        choices=["logreg", "svm"],
                        help="Classifier type: logreg or svm")
    parser.add_argument("--tune", action="store_true",
                        help="Run GridSearchCV hyperparameter tuning")
    parser.add_argument("--use-cached", action="store_true",
                        help="Load from cached CSV instead of re-fetching from API")
    parser.add_argument("--drop-unknown", action="store_true",
                        help="Exclude records with UNKNOWN severity from training")
    args = parser.parse_args()
    main(args)
