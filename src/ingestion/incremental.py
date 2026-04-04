"""
Incremental Batch Ingestion Pipeline for MAUDE NLP Classifier.

Design philosophy (from project architecture review):
  - No static file downloads. All data comes from openFDA's paginated REST API.
  - Each ingestion run fetches a configurable batch of records (default 10,000).
  - New records are deduplicated against the existing dataset by report_number.
  - After ingestion, the classifier is retrained on the full accumulated dataset
    using MLflow-tracked runs with automatic model promotion.
  - A scheduler (APScheduler) can run this on a cron cadence for continuous
    model improvement without any manual intervention.

Usage (one-off):
    python -m src.ingestion.incremental --batch 10000

Usage (scheduled, runs every day at 2 AM):
    python -m src.ingestion.incremental --schedule --cron "0 2 * * *"
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime

import pandas as pd

from src.ingestion.openfda_client import fetch_maude_records, save_raw_data

logger = logging.getLogger(__name__)

ACCUMULATED_DATA_PATH = "data/accumulated/maude_accumulated.csv"


# ── Deduplication ────────────────────────────────────────────────────────────

def load_accumulated(path: str = ACCUMULATED_DATA_PATH) -> pd.DataFrame:
    """Load the accumulated dataset from disk, or return empty DataFrame."""
    if os.path.exists(path):
        df = pd.read_csv(path, dtype=str)
        logger.info(f"Loaded {len(df):,} accumulated records from {path}")
        return df
    logger.info("No accumulated dataset found — starting fresh.")
    return pd.DataFrame()


def merge_and_dedup(
    existing: pd.DataFrame,
    new_records: pd.DataFrame,
    key: str = "report_number",
) -> tuple[pd.DataFrame, int]:
    """
    Append new records to the existing dataset, deduplicated on report_number.

    Args:
        existing: Previously accumulated DataFrame (may be empty).
        new_records: Freshly fetched DataFrame.
        key: Column to deduplicate on.

    Returns:
        Tuple of (merged DataFrame, count of genuinely new rows added).
    """
    if existing.empty:
        return new_records.copy(), len(new_records)

    existing_keys = set(existing[key].dropna().astype(str))
    new_mask = ~new_records[key].astype(str).isin(existing_keys)
    truly_new = new_records[new_mask]

    n_new = len(truly_new)
    if n_new == 0:
        logger.info("No new records found after deduplication.")
        return existing, 0

    merged = pd.concat([existing, truly_new], ignore_index=True)
    logger.info(
        f"Added {n_new:,} new records. "
        f"Total accumulated: {len(merged):,}"
    )
    return merged, n_new


def save_accumulated(df: pd.DataFrame, path: str = ACCUMULATED_DATA_PATH) -> None:
    """Persist the full accumulated dataset to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved {len(df):,} accumulated records to {path}")


# ── Ingestion run ─────────────────────────────────────────────────────────────

def run_ingestion(
    batch_size: int = 10_000,
    api_key: str | None = None,
    retrain: bool = True,
    cross_validate: bool = False,
    model_type: str = "logreg",
) -> dict:
    """
    Execute one incremental ingestion cycle:
      1. Fetch a batch of records from openFDA API
      2. Deduplicate against accumulated dataset
      3. Persist updated accumulated dataset
      4. (Optionally) trigger a full retrain run via train.py

    Args:
        batch_size: Number of records to request per API run.
                    Use 1,000–50,000 depending on API key tier.
        api_key: openFDA API key. Falls back to OPENFDA_API_KEY env var.
        retrain: Whether to trigger a retrain after ingestion.
        cross_validate: Pass --cross-validate to train.py if retraining.
        model_type: 'logreg', 'svm', or 'bert'.

    Returns:
        Summary dict with ingestion stats.
    """
    started_at = datetime.utcnow().isoformat()
    logger.info(f"=== Incremental ingestion started at {started_at} ===")
    logger.info(f"Batch size: {batch_size:,} records requested")

    # Step 1: Fetch new batch
    new_df = fetch_maude_records(
        total_records=batch_size,
        api_key=api_key or os.getenv("OPENFDA_API_KEY"),
    )
    logger.info(f"API returned {len(new_df):,} raw records.")

    # Step 2: Load existing and merge
    existing_df = load_accumulated()
    merged_df, n_added = merge_and_dedup(existing_df, new_df)

    # Step 3: Persist
    save_accumulated(merged_df)

    # Also keep data/raw/ in sync for the train.py --use-cached path
    save_raw_data(merged_df, "data/raw/maude_raw.csv")

    summary = {
        "started_at": started_at,
        "batch_requested": batch_size,
        "batch_returned": len(new_df),
        "new_records_added": n_added,
        "total_accumulated": len(merged_df),
        "retrain_triggered": False,
    }

    # Step 4: Retrain if we actually got new data (or always if forced)
    if retrain and (n_added > 0 or not os.path.exists("models/maude_classifier.joblib")):
        logger.info("New records detected — triggering retrain pipeline...")
        _trigger_retrain(cross_validate=cross_validate, model_type=model_type)
        summary["retrain_triggered"] = True
    elif retrain and n_added == 0:
        logger.info("No new records — skipping retrain to avoid redundant MLflow run.")

    logger.info(f"=== Ingestion complete. Summary: {summary} ===")
    return summary


def _trigger_retrain(cross_validate: bool = False, model_type: str = "logreg") -> None:
    """
    Invoke the appropriate train script as a subprocess.

    model_type="logreg" or "svm" → src.model.train (TF-IDF, CPU, fast)
    model_type="bert"             → src.model.train_bert (ClinicalBERT, GPU recommended)
    """
    if model_type == "bert":
        cmd = [
            sys.executable, "-m", "src.model.train_bert",
            "--use-cached",
            "--drop-unknown",
            "--records", "0",   # ignored when --use-cached is set
        ]
        hub_repo = os.getenv("HF_HUB_REPO")
        if hub_repo:
            cmd += ["--hub-repo", hub_repo]
    else:
        cmd = [
            sys.executable, "-m", "src.model.train",
            "--use-cached",
            "--drop-unknown",
            "--model", model_type,
        ]
        if cross_validate:
            cmd.append("--cross-validate")

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        logger.error(f"Retrain subprocess exited with code {result.returncode}")
    else:
        logger.info("Retrain completed successfully.")


# ── Scheduler ────────────────────────────────────────────────────────────────

def start_scheduler(
    cron: str = "0 2 * * *",
    batch_size: int = 10_000,
    model_type: str = "logreg",
    cross_validate: bool = False,
) -> None:
    """
    Start an APScheduler process that runs incremental ingestion on a cron schedule.

    The cron expression follows standard 5-field format:
      "0 2 * * *"  → every day at 02:00 UTC
      "0 */6 * * *" → every 6 hours

    Args:
        cron: Cron expression string.
        batch_size: Records per ingestion run.
        model_type: Classifier type.
        cross_validate: Whether to run cross-validation on retrain.
    """
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.error(
            "APScheduler not installed. Run: pip install apscheduler"
        )
        sys.exit(1)

    parts = cron.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Invalid cron expression (need 5 fields): {cron!r}")

    minute, hour, day, month, day_of_week = parts

    scheduler = BlockingScheduler()
    scheduler.add_job(
        run_ingestion,
        trigger=CronTrigger(
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            day_of_week=day_of_week,
        ),
        kwargs={
            "batch_size": batch_size,
            "retrain": True,
            "cross_validate": cross_validate,
            "model_type": model_type,
        },
        id="maude_incremental_ingestion",
        name="MAUDE Incremental Ingestion + Retrain",
        replace_existing=True,
    )

    logger.info(
        f"Scheduler started. Next run at: "
        f"{scheduler.get_jobs()[0].next_run_time}"
    )
    logger.info("Press Ctrl+C to stop.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Incremental MAUDE ingestion + retrain pipeline"
    )
    parser.add_argument(
        "--batch", type=int, default=10_000,
        help="Records to fetch per ingestion run (default: 10,000)"
    )
    parser.add_argument(
        "--model", type=str, default="logreg", choices=["logreg", "svm", "bert"],
        help="Classifier type for retraining (bert requires GPU and transformers)"
    )
    parser.add_argument(
        "--no-retrain", action="store_true",
        help="Ingest data only — skip retrain"
    )
    parser.add_argument(
        "--cross-validate", action="store_true",
        help="Run StratifiedKFold CV during retrain"
    )
    parser.add_argument(
        "--schedule", action="store_true",
        help="Run continuously on a cron schedule instead of once"
    )
    parser.add_argument(
        "--cron", type=str, default="0 2 * * *",
        help="Cron expression for scheduled runs (default: daily at 2 AM UTC)"
    )
    args = parser.parse_args()

    if args.schedule:
        start_scheduler(
            cron=args.cron,
            batch_size=args.batch,
            model_type=args.model,
            cross_validate=args.cross_validate,
        )
    else:
        run_ingestion(
            batch_size=args.batch,
            retrain=not args.no_retrain,
            cross_validate=args.cross_validate,
            model_type=args.model,
        )
