"""
Training entrypoint for MAUDE NLP Classifier.

Usage:
    python -m src.model.train --records 5000 --model logreg --tune
"""

import argparse
import logging
import os

import pandas as pd

from src.ingestion.openfda_client import fetch_maude_records, save_raw_data
from src.preprocessing.text_cleaner import clean_dataframe, get_label_distribution
from src.model.classifier import (
    build_pipeline,
    split_data,
    train_pipeline,
    tune_pipeline,
    evaluate,
    save_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_PATH = "data/raw/maude_raw.csv"
MODEL_PATH = "models/maude_classifier.joblib"


def main(args):
    # ── 1. Ingest data ──────────────────────────────────────────────────────
    if args.use_cached and os.path.exists(RAW_DATA_PATH):
        logger.info(f"Loading cached data from {RAW_DATA_PATH}")
        df = pd.read_csv(RAW_DATA_PATH)
    else:
        logger.info(f"Fetching {args.records} records from openFDA MAUDE API...")
        df = fetch_maude_records(total_records=args.records)
        save_raw_data(df, RAW_DATA_PATH)

    logger.info(f"Raw data shape: {df.shape}")
    logger.info(f"Label distribution:\n{get_label_distribution(df)}")

    # ── 2. Preprocess ────────────────────────────────────────────────────────
    df = clean_dataframe(df)

    # Drop UNKNOWN labels for cleaner classification (optional)
    if args.drop_unknown:
        before = len(df)
        df = df[df["severity_label"] != "UNKNOWN"].reset_index(drop=True)
        logger.info(f"Dropped UNKNOWN labels: {before - len(df)} rows removed.")

    logger.info(f"Cleaned data shape: {df.shape}")
    logger.info(f"Label distribution after cleaning:\n{get_label_distribution(df)}")

    # ── 3. Build & train ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(df)
    pipeline = build_pipeline(model_type=args.model)

    if args.tune:
        pipeline = tune_pipeline(pipeline, X_train, y_train, model_type=args.model)
    else:
        pipeline = train_pipeline(pipeline, X_train, y_train)

    # ── 4. Evaluate ──────────────────────────────────────────────────────────
    metrics = evaluate(pipeline, X_test, y_test)
    logger.info(f"\nFinal Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Final Weighted F1: {metrics['f1_weighted']:.4f}")
    logger.info(f"\n{metrics['classification_report']}")

    # ── 5. Save model ────────────────────────────────────────────────────────
    save_model(pipeline, MODEL_PATH)
    logger.info("Training complete.")


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Train MAUDE NLP Severity Classifier")
    parser.add_argument("--records", type=int, default=5000,
                        help="Number of MAUDE records to fetch")
    parser.add_argument("--model", type=str, default="logreg",
                        choices=["logreg", "svm"],
                        help="Classifier type: logreg or svm")
    parser.add_argument("--tune", action="store_true",
                        help="Run GridSearchCV hyperparameter tuning")
    parser.add_argument("--use-cached", action="store_true",
                        help="Use cached raw CSV instead of re-fetching from API")
    parser.add_argument("--drop-unknown", action="store_true",
                        help="Exclude records with UNKNOWN severity from training")
    args = parser.parse_args()
    main(args)
