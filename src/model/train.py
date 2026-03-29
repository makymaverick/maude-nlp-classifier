"""
Training entrypoint for MAUDE NLP Classifier.

Includes:
  - openFDA batch ingestion (paginated, no static downloads)
  - Text preprocessing
  - StratifiedKFold cross-validation
  - Dummy baseline comparison
  - MLflow experiment tracking & model versioning
  - Automatic model promotion (new model only replaces champion if F1 improves)

Usage:
    python -m src.model.train --records 5000 --model logreg --tune
    python -m src.model.train --records 10000 --model logreg --cross-validate
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


# ── MLflow helpers ──────────────────────────────────────────────────────────

def _get_champion_f1() -> float:
    """Read the current champion model's F1 from disk. Returns 0 if none exists."""
    if os.path.exists(CHAMPION_METRICS_PATH):
        with open(CHAMPION_METRICS_PATH) as f:
            data = json.load(f)
        return data.get("f1_weighted", 0.0)
    return 0.0


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


# ── Main training function ──────────────────────────────────────────────────

def main(args):
    # Set up MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run() as run:
        logger.info(f"MLflow run ID: {run.info.run_id}")

        # Log training configuration
        mlflow.log_params({
            "model_type": args.model,
            "records_requested": args.records,
            "drop_unknown": args.drop_unknown,
            "hyperparameter_tuning": args.tune,
            "cross_validate": args.cross_validate,
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

        # Log class distribution to MLflow
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
        logger.info(
            f"Dummy baseline — the model must beat F1={dummy['dummy_f1_weighted']:.4f} "
            "to be worth anything."
        )

        # ── 5. Build & train ─────────────────────────────────────────────────
        pipeline = build_pipeline(model_type=args.model)

        if args.tune:
            pipeline = tune_pipeline(pipeline, X_train, y_train, model_type=args.model)
        else:
            pipeline = train_pipeline(pipeline, X_train, y_train)

        # ── 6. StratifiedKFold cross-validation ──────────────────────────────
        cv_metrics = {}
        if args.cross_validate:
            logger.info("Running 5-fold stratified cross-validation...")
            fresh_pipe = build_pipeline(model_type=args.model)
            X_full = df["clean_text"]
            y_full = df["severity_label"]
            cv_results = cross_validate_pipeline(fresh_pipe, X_full, y_full, n_splits=5)
            cv_metrics = {
                "cv_f1_mean": cv_results["cv_f1_mean"],
                "cv_f1_std": cv_results["cv_f1_std"],
            }
            mlflow.log_metrics(cv_metrics)
            for i, fold_score in enumerate(cv_results["cv_f1_per_fold"]):
                mlflow.log_metric(f"cv_f1_fold_{i+1}", fold_score)

        # ── 7. Hold-out evaluation ───────────────────────────────────────────
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
            f"\n{'='*50}\n"
            f"  Accuracy:       {metrics['accuracy']:.4f}\n"
            f"  Weighted F1:    {metrics['f1_weighted']:.4f}\n"
            f"  Dummy F1:       {dummy['dummy_f1_weighted']:.4f}  "
            f"({'BEATS' if metrics['f1_weighted'] > dummy['dummy_f1_weighted'] else 'DOES NOT BEAT'} baseline)\n"
            + (f"  CV F1 (5-fold): {cv_metrics['cv_f1_mean']:.4f} ± {cv_metrics['cv_f1_std']:.4f}\n"
               if cv_metrics else "")
            + f"{'='*50}"
        )

        # ── 8. Model promotion logic ─────────────────────────────────────────
        champion_f1 = _get_champion_f1()
        new_f1 = metrics["f1_weighted"]

        if new_f1 > champion_f1:
            logger.info(
                f"New model F1 ({new_f1:.4f}) > champion F1 ({champion_f1:.4f}). "
                "Promoting to champion."
            )
            save_model(pipeline, MODEL_PATH)
            _save_champion_metrics(metrics)
            mlflow.set_tag("promoted", "true")
            mlflow.log_metric("champion_f1_before", champion_f1)
        else:
            logger.info(
                f"New model F1 ({new_f1:.4f}) did not improve over champion "
                f"({champion_f1:.4f}). Keeping existing champion."
            )
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
    parser.add_argument("--cross-validate", action="store_true",
                        help="Run 5-fold StratifiedKFold cross-validation")
    parser.add_argument("--use-cached", action="store_true",
                        help="Load from cached CSV instead of re-fetching from API")
    parser.add_argument("--drop-unknown", action="store_true",
                        help="Exclude records with UNKNOWN severity from training")
    args = parser.parse_args()
    main(args)
