"""
BERT fine-tuning entrypoint for MAUDE NLP Classifier — Phase 1.

Mirrors src/model/train.py in structure:
  - Same openFDA ingestion pipeline
  - Same 3-gate model promotion logic (CV F1, min records, delta threshold)
  - Same MLflow experiment and metric schema
  - Same champion_metrics.json format (adds bert_checkpoint_path key)

Usage:
    # Fine-tune on 10k records, push to HF Hub
    python -m src.model.train_bert --records 10000 --hub-repo mukundisb/maude-clinicalbert

    # Fine-tune locally only
    python -m src.model.train_bert --records 5000 --epochs 3

    # Use cached CSV
    python -m src.model.train_bert --use-cached --epochs 5 --lr 2e-5
"""

import argparse
import json
import logging
import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import pandas as pd

from src.ingestion.openfda_client import fetch_maude_records, save_raw_data
from src.preprocessing.text_cleaner import clean_dataframe, get_label_distribution
from src.model.bert_classifier import (
    train_bert,
    cross_validate_bert,
    save_bert_checkpoint,
    BERT_MODEL_REF_PATH,
    LABEL_ORDER,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_PATH = "data/raw/maude_raw.csv"
BERT_CHECKPOINT_DIR = "models/bert_checkpoint"
CHAMPION_METRICS_PATH = "models/champion_metrics.json"
MLFLOW_EXPERIMENT = "maude-nlp-severity"

# ── Promotion safeguards (same as train.py) ───────────────────────────────────
MIN_RECORDS_FOR_COMPARISON = 10_000
PROMOTION_MIN_DELTA = 0.005


# ── Champion metrics helpers ──────────────────────────────────────────────────

def _get_champion_metrics() -> dict:
    if os.path.exists(CHAMPION_METRICS_PATH):
        with open(CHAMPION_METRICS_PATH) as f:
            return json.load(f)
    return {}


def _get_champion_f1() -> float:
    data = _get_champion_metrics()
    return data.get("cv_f1_mean") or data.get("f1_weighted", 0.0)


def _save_champion_metrics(metrics: dict) -> None:
    os.makedirs(os.path.dirname(CHAMPION_METRICS_PATH), exist_ok=True)
    with open(CHAMPION_METRICS_PATH, "w") as f:
        json.dump(
            {
                "f1_weighted": metrics.get("val_f1_weighted"),
                "accuracy": metrics.get("accuracy"),
                "cv_f1_mean": metrics.get("cv_f1_mean"),
                "cv_f1_std": metrics.get("cv_f1_std"),
                "training_records": metrics.get("training_records"),
                "model_type": "clinicalbert",
                "bert_checkpoint_path": metrics.get("bert_checkpoint_path"),
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
    if no_champion_exists:
        return True, "no_champion_exists"
    if training_records < MIN_RECORDS_FOR_COMPARISON:
        return True, (
            f"below_min_records_threshold "
            f"({training_records} < {MIN_RECORDS_FOR_COMPARISON}) — always promoting"
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run() as run:
        logger.info(f"MLflow run ID: {run.info.run_id}")

        mlflow.log_params({
            "model_type": "clinicalbert",
            "pretrained_model": "emilyalsentzer/Bio_ClinicalBERT",
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "records_requested": args.records,
            "drop_unknown": args.drop_unknown,
            "hub_repo": args.hub_repo or "none",
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

        # ── 2. Preprocess (preserve digits for BERT) ─────────────────────────
        df = clean_dataframe(df, preserve_digits=True)

        if args.drop_unknown:
            before = len(df)
            df = df[df["severity_label"] != "UNKNOWN"].reset_index(drop=True)
            logger.info(f"Dropped UNKNOWN: {before - len(df)} rows removed.")

        label_dist = get_label_distribution(df)
        logger.info(f"Label distribution:\n{label_dist}")
        for label, count in label_dist.items():
            mlflow.log_metric(f"class_count_{label}", count)

        mlflow.log_param("training_records", len(df))

        if len(df) < 50:
            logger.error("Not enough records after cleaning. Aborting.")
            return

        # ── 3. Fine-tune ─────────────────────────────────────────────────────
        logger.info("Starting ClinicalBERT fine-tuning...")
        model, tokenizer, train_metrics = train_bert(
            df,
            text_col="clean_text",
            label_col="severity_label",
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        mlflow.log_metrics({
            "val_f1_weighted": train_metrics["val_f1_weighted"],
        })

        # ── 4. Cross-validation (promotion gate) ─────────────────────────────
        logger.info("Running BERT cross-validation (required for promotion)...")
        cv_results = cross_validate_bert(
            df,
            text_col="clean_text",
            label_col="severity_label",
            n_splits=3,       # 3 folds (vs 5 for TF-IDF) to keep runtime reasonable
            epochs=2,         # 2 epochs per fold
            lr=args.lr,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        cv_metrics = {
            "cv_f1_mean": cv_results["cv_f1_mean"],
            "cv_f1_std": cv_results["cv_f1_std"],
        }
        mlflow.log_metrics(cv_metrics)
        for i, fold_score in enumerate(cv_results["cv_f1_per_fold"]):
            mlflow.log_metric(f"cv_f1_fold_{i+1}", fold_score)

        # ── 5. Promotion decision ─────────────────────────────────────────────
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

        logger.info(
            f"\n{'='*60}\n"
            f"  Records trained on:  {len(df):,}\n"
            f"  Val F1 (hold-out):   {train_metrics['val_f1_weighted']:.4f}\n"
            f"  CV F1 (3-fold):      {cv_metrics['cv_f1_mean']:.4f} "
            f"± {cv_metrics['cv_f1_std']:.4f}  ← used for promotion\n"
            f"  Champion CV F1:      {champion_cv_f1:.4f}\n"
            f"{'='*60}"
        )

        if promote:
            logger.info(f"✅ Promoting BERT model. Reason: {reason}")
            save_bert_checkpoint(
                model, tokenizer, BERT_CHECKPOINT_DIR, hub_repo=args.hub_repo
            )
            metrics_to_save = {
                **train_metrics,
                **cv_metrics,
                "bert_checkpoint_path": BERT_CHECKPOINT_DIR,
            }
            _save_champion_metrics(metrics_to_save)
            mlflow.set_tag("promoted", "true")
        else:
            logger.info(f"⏭️  Keeping existing champion. Reason: {reason}")
            mlflow.set_tag("promoted", "false")

        mlflow.set_tag("model_type", "clinicalbert")
        logger.info(f"Training complete. MLflow run: {run.info.run_id}")


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Fine-tune ClinicalBERT on MAUDE data")
    parser.add_argument("--records", type=int, default=5000,
                        help="Number of MAUDE records to fetch")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate for AdamW")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Per-device batch size")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Max token length (256 covers 95th pct of MAUDE narratives)")
    parser.add_argument("--hub-repo", type=str, default=None,
                        help="HuggingFace Hub repo ID to push checkpoint (e.g. mukundisb/maude-clinicalbert)")
    parser.add_argument("--use-cached", action="store_true",
                        help="Load from cached CSV instead of re-fetching")
    parser.add_argument("--drop-unknown", action="store_true",
                        help="Exclude UNKNOWN severity labels from training")
    args = parser.parse_args()
    main(args)
