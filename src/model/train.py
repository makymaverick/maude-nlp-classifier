"""
Training entrypoint for MAUDE NLP Severity Classifier.

Fetches adverse event records from the openFDA MAUDE API, preprocesses
narrative text, trains a TF-IDF + classifier pipeline, evaluates with
StratifiedKFold cross-validation, and promotes the new model only if
its CV F1 beats the current champion.

Usage:
    python -m src.model.train --records 5000 --model logreg
    python -m src.model.train --records 5000 --model logreg --tune
    python -m src.model.train --records 5000 --use-cached --drop-unknown
"""

import argparse
import json
import logging
import os

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
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_PATH         = "data/raw/maude_raw.csv"
MODEL_PATH            = "models/maude_classifier.joblib"
CHAMPION_METRICS_PATH = "models/champion_metrics.json"
MLFLOW_EXPERIMENT     = "maude-nlp-severity"


# ── Champion metrics helpers ─────────────────────────────────────────────────

def _load_champion_metrics() -> dict:
    """Read current champion metrics from disk. Returns {} if none exists."""
    if not os.path.exists(CHAMPION_METRICS_PATH):
        return {}
    with open(CHAMPION_METRICS_PATH) as f:
        return json.load(f)


def _save_champion_metrics(metrics: dict) -> None:
    """Persist the promoted model's metrics to disk."""
    os.makedirs(os.path.dirname(CHAMPION_METRICS_PATH), exist_ok=True)
    with open(CHAMPION_METRICS_PATH, "w") as f:
        json.dump(
            {
                "accuracy":          metrics["accuracy"],
                "f1_weighted":       metrics["f1_weighted"],
                "cv_f1_mean":        metrics.get("cv_f1_mean"),
                "cv_f1_std":         metrics.get("cv_f1_std"),
                "training_records":  metrics.get("training_records"),
            },
            f,
            indent=2,
        )
    logger.info(f"Champion metrics saved to {CHAMPION_METRICS_PATH}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run() as run:
        logger.info(f"MLflow run ID: {run.info.run_id}")

        mlflow.log_params({
            "model_type":            args.model,
            "records_requested":     args.records,
            "drop_unknown":          args.drop_unknown,
            "hyperparameter_tuning": args.tune,
        })

        # ── 1. Ingest ────────────────────────────────────────────────────────
        if args.use_cached and os.path.exists(RAW_DATA_PATH):
            logger.info(f"Loading cached data from {RAW_DATA_PATH}")
            df = pd.read_csv(RAW_DATA_PATH)
        else:
            logger.info(f"Fetching {args.records:,} records from openFDA MAUDE API...")
            df = fetch_maude_records(total_records=args.records)
            save_raw_data(df, RAW_DATA_PATH)

        logger.info(f"Raw records: {len(df):,}")
        mlflow.log_param("records_fetched", len(df))

        # ── 2. Preprocess ────────────────────────────────────────────────────
        df = clean_dataframe(df)

        if args.drop_unknown:
            before = len(df)
            df = df[df["severity_label"] != "UNKNOWN"].reset_index(drop=True)
            logger.info(f"Dropped {before - len(df)} UNKNOWN-labelled rows.")

        label_dist = get_label_distribution(df)
        logger.info(f"Label distribution:\n{label_dist}")

        for label, count in label_dist.items():
            mlflow.log_metric(f"class_count_{label}", int(count))

        mlflow.log_param("training_records", len(df))

        if len(df) < 50:
            logger.error("Fewer than 50 records after cleaning — aborting.")
            return

        # ── 3. Train / test split ────────────────────────────────────────────
        X_train, X_test, y_train, y_test = split_data(df)

        # ── 4. Dummy baseline ────────────────────────────────────────────────
        #
        # Sanity check: a model that simply predicts the majority class will
        # score well on accuracy when classes are imbalanced.  The dummy
        # baseline makes this explicit — our model must beat it meaningfully.
        dummy = dummy_baseline(X_train, y_train, X_test, y_test)
        mlflow.log_metrics({
            "dummy_accuracy":    dummy["dummy_accuracy"],
            "dummy_f1_weighted": dummy["dummy_f1_weighted"],
        })

        # ── 5. Build + train ─────────────────────────────────────────────────
        pipeline = build_pipeline(model_type=args.model)
        if args.tune:
            pipeline = tune_pipeline(pipeline, X_train, y_train, model_type=args.model)
        else:
            pipeline = train_pipeline(pipeline, X_train, y_train)

        # ── 6. StratifiedKFold cross-validation ──────────────────────────────
        #
        # We use CV F1 — not hold-out F1 — as the promotion signal because
        # a single train/test split on a small dataset can vary by ±0.10 F1
        # depending on which records land in the test set.  Five-fold CV is
        # 3–5× more stable.
        #
        # A fresh (untrained) pipeline is passed to cross_validate_pipeline
        # so that each fold trains independently — no data leakage.
        logger.info("Running 5-fold stratified cross-validation...")
        cv_pipeline = build_pipeline(model_type=args.model)
        cv_results = cross_validate_pipeline(
            cv_pipeline, df["clean_text"], df["severity_label"], n_splits=5
        )
        cv_f1_mean = cv_results["cv_f1_mean"]
        cv_f1_std  = cv_results["cv_f1_std"]

        mlflow.log_metrics({
            "cv_f1_mean": cv_f1_mean,
            "cv_f1_std":  cv_f1_std,
        })
        for i, score in enumerate(cv_results["cv_f1_per_fold"]):
            mlflow.log_metric(f"cv_f1_fold_{i + 1}", score)

        # ── 7. Hold-out evaluation ───────────────────────────────────────────
        metrics = evaluate(pipeline, X_test, y_test)
        metrics["cv_f1_mean"]       = cv_f1_mean
        metrics["cv_f1_std"]        = cv_f1_std
        metrics["training_records"] = len(df)

        mlflow.log_metrics({
            "accuracy":    metrics["accuracy"],
            "f1_weighted": metrics["f1_weighted"],
        })
        mlflow.log_text(metrics["classification_report"], "classification_report.txt")
        mlflow.sklearn.log_model(pipeline, "model")

        beats_dummy = metrics["f1_weighted"] > dummy["dummy_f1_weighted"]
        logger.info(
            f"\n{'=' * 55}\n"
            f"  Records:          {len(df):,}\n"
            f"  Hold-out F1:      {metrics['f1_weighted']:.4f}"
            f"  ({'BEATS' if beats_dummy else 'DOES NOT BEAT'}"
            f" dummy={dummy['dummy_f1_weighted']:.4f})\n"
            f"  CV F1 (5-fold):   {cv_f1_mean:.4f} ± {cv_f1_std:.4f}  <- promotion signal\n"
            f"{'=' * 55}"
        )

        # ── 8. Model promotion ───────────────────────────────────────────────
        #
        # Promote if CV F1 improves over the current champion.
        # Using CV F1 for the comparison — same reason as above.
        champion = _load_champion_metrics()
        champion_cv_f1 = champion.get("cv_f1_mean") or champion.get("f1_weighted", 0.0)
        no_champion = not champion

        if no_champion:
            promote, reason = True, "no_champion_exists"
        elif cv_f1_mean > champion_cv_f1:
            promote = True
            reason  = f"cv_f1_improved ({cv_f1_mean:.4f} > {champion_cv_f1:.4f})"
        else:
            promote = False
            reason  = f"champion_retained ({champion_cv_f1:.4f} >= {cv_f1_mean:.4f})"

        mlflow.log_metric("champion_cv_f1_before", champion_cv_f1)
        mlflow.set_tag("promoted",         str(promote).lower())
        mlflow.set_tag("promotion_reason", reason)

        if promote:
            logger.info(f"Promoting new model. Reason: {reason}")
            save_model(pipeline, MODEL_PATH)
            _save_champion_metrics(metrics)
        else:
            logger.info(f"Keeping existing champion. Reason: {reason}")

        logger.info(f"Done. MLflow run: {run.info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MAUDE NLP Severity Classifier")
    parser.add_argument("--records",     type=int,  default=5000,
                        help="Records to fetch from openFDA per run (default 5000)")
    parser.add_argument("--model",       type=str,  default="logreg",
                        choices=["logreg", "svm"],
                        help="Classifier backend: logreg (default) or svm")
    parser.add_argument("--tune",        action="store_true",
                        help="Run GridSearchCV hyperparameter tuning")
    parser.add_argument("--use-cached",  action="store_true",
                        help="Load from data/raw/maude_raw.csv instead of re-fetching")
    parser.add_argument("--drop-unknown", action="store_true",
                        help="Exclude UNKNOWN severity records from training")
    args = parser.parse_args()
    main(args)
