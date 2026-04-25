"""
Training entrypoint for the HARDENED TF-IDF + LR pipeline — Phase 2 Step 0.

This is a standalone script. It does NOT touch:
  - src/model/train.py
  - models/maude_classifier.joblib
  - models/champion_metrics.json
  - the deployed HF Space

It writes its own artefacts:
  - models/hardened_pipeline.joblib
  - models/hardened_metrics.json

Usage:

    # Baseline hardened run (uses locked CV splits from data/cv_splits.json)
    python -m src.model.train_hardened --use-cached

    # Grid-search variant (wider hyperparameter sweep, slower)
    python -m src.model.train_hardened --use-cached --tune

    # With extended clinical abbreviation enrichment
    python -m src.model.train_hardened --use-cached --enrich

    # Drop UNKNOWN class (matches the config that likely generated champion F1)
    python -m src.model.train_hardened --use-cached --drop-unknown

    # Promote hardened to champion (only if CV F1 beats existing champion + delta)
    python -m src.model.train_hardened --use-cached --promote-champion

Prerequisites:
  1. data/raw/maude_raw.csv exists (or run without --use-cached to fetch).
  2. data/cv_splits.json exists — generate via:
         python -m src.evaluation.cv_splits generate

MLflow experiment: maude-nlp-severity-phase2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Optional

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from src.ingestion.openfda_client import fetch_maude_records, save_raw_data
from src.preprocessing.text_cleaner import clean_dataframe, get_label_distribution
from src.model.classifier_hardened import (
    build_hardened_pipeline,
    tune_hardened_pipeline,
    cross_validate_hardened,
    evaluate_with_confusion,
    enrich_dataframe,
    DEFAULT_HARDENED_GRID,
    SMOKE_HARDENED_GRID,
)
from src.evaluation.cv_splits import (
    load_cv_splits,
    generate_cv_splits,
    save_cv_splits,
    build_fingerprint,
    DEFAULT_CV_SPLITS_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_PATH = "data/raw/maude_raw.csv"
HARDENED_MODEL_PATH = "models/hardened_pipeline.joblib"
HARDENED_METRICS_PATH = "models/hardened_metrics.json"
CHAMPION_METRICS_PATH = "models/champion_metrics.json"
CHAMPION_MODEL_PATH = "models/maude_classifier.joblib"
MLFLOW_EXPERIMENT = "maude-nlp-severity-phase2"

# Promotion gate: hardened must beat champion by this much to be declared new champion
PROMOTION_MIN_DELTA = 0.005


def _load_champion_metrics() -> dict:
    if not os.path.exists(CHAMPION_METRICS_PATH):
        return {}
    with open(CHAMPION_METRICS_PATH) as f:
        return json.load(f)


def _save_hardened_metrics(metrics: dict) -> None:
    os.makedirs(os.path.dirname(HARDENED_METRICS_PATH), exist_ok=True)
    # Drop non-JSON-serialisable arrays before dumping
    clean = {k: v for k, v in metrics.items() if k != "confusion_matrix_array"}
    with open(HARDENED_METRICS_PATH, "w") as f:
        json.dump(clean, f, indent=2, default=str)
    logger.info(f"Hardened metrics saved to {HARDENED_METRICS_PATH}")


def _print_summary(cv: dict, champion: dict, promoted: bool, reason: str) -> None:
    print("\n" + "=" * 72)
    print("  PHASE 2 STEP 0 — HARDENED TF-IDF + LR RESULTS")
    print("=" * 72)
    champion_f1 = champion.get("cv_f1_mean") or champion.get("f1_weighted", 0.0)
    print(f"  Champion v1 CV F1 (weighted):   {champion_f1:.4f}")
    print(f"  Hardened CV F1 (weighted):      {cv['f1_weighted_mean']:.4f} "
          f"± {cv['f1_weighted_std']:.4f}")
    print(f"  Hardened CV F1 (macro):         {cv['f1_macro_mean']:.4f} "
          f"± {cv['f1_macro_std']:.4f}")
    print("  Per-class F1 (hardened, mean across folds):")
    for key in sorted(k for k in cv if k.startswith("f1_") and k.endswith("_mean")
                      and k not in ("f1_weighted_mean", "f1_macro_mean")):
        label = key.replace("f1_", "").replace("_mean", "")
        std_key = key.replace("_mean", "_std")
        print(f"    {label:<10} {cv[key]:.4f} ± {cv[std_key]:.4f}")
    print("-" * 72)
    delta = cv["f1_weighted_mean"] - champion_f1
    sign = "+" if delta >= 0 else ""
    print(f"  Δ vs champion: {sign}{delta:.4f}  ({reason})")
    if promoted:
        print(f"  -> Hardened model PROMOTED to champion.")
    else:
        print(f"  -> Hardened model NOT promoted (gate: +{PROMOTION_MIN_DELTA}).")
    print("=" * 72 + "\n")


def main(args: argparse.Namespace) -> None:
    # ── 1. Ingest ────────────────────────────────────────────────────────────
    if args.use_cached and os.path.exists(RAW_DATA_PATH):
        logger.info(f"Loading cached data from {RAW_DATA_PATH}")
        df = pd.read_csv(RAW_DATA_PATH)
    else:
        logger.info(f"Fetching {args.records:,} records from openFDA MAUDE API...")
        df = fetch_maude_records(total_records=args.records)
        save_raw_data(df, RAW_DATA_PATH)

    logger.info(f"Raw records: {len(df):,}")

    # ── 2. Preprocess (same cleaner as champion v1) ──────────────────────────
    df = clean_dataframe(df)

    if args.drop_unknown:
        before = len(df)
        df = df[df["severity_label"] != "UNKNOWN"].reset_index(drop=True)
        logger.info(f"Dropped {before - len(df)} UNKNOWN rows.")

    logger.info(f"Post-clean records: {len(df):,}")
    logger.info(f"Label distribution:\n{get_label_distribution(df)}")

    # ── 3. Optional extended abbreviation enrichment ────────────────────────
    text_col = "clean_text"
    if args.enrich:
        logger.info("Applying extended clinical abbreviation expansion...")
        df = enrich_dataframe(df, text_col="clean_text", output_col="enriched_text")
        text_col = "enriched_text"

    # ── 4. Locked CV splits ─────────────────────────────────────────────────
    if os.path.exists(args.cv_splits_path) and not args.regenerate_splits:
        logger.info(f"Loading locked CV splits from {args.cv_splits_path}")
        fold_indices = load_cv_splits(
            args.cv_splits_path,
            expected_n_records=len(df) if args.check_fingerprint else None,
        )
    else:
        logger.warning(
            f"No splits at {args.cv_splits_path} (or --regenerate-splits). "
            f"Generating fresh ones — Phase 2 comparability starts from here."
        )
        payload = generate_cv_splits(
            df,
            label_col="severity_label",
            text_col=text_col,
            n_splits=5,
            random_state=42,
        )
        save_cv_splits(payload, args.cv_splits_path)
        fold_indices = [
            (np.asarray(f["train_idx"]), np.asarray(f["val_idx"]))
            for f in payload["folds"]
        ]

    # ── 5. MLflow experiment ─────────────────────────────────────────────────
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=args.run_name or None) as run:
        logger.info(f"MLflow run ID: {run.info.run_id}")
        mlflow.log_params({
            "mode": "tune" if args.tune else "fixed",
            "enrich": args.enrich,
            "drop_unknown": args.drop_unknown,
            "training_records": len(df),
            "cv_splits_path": args.cv_splits_path,
            "phase": "phase2-step0-hardened",
        })

        # ── 6. Build / tune pipeline ─────────────────────────────────────────
        X = df[text_col]
        y = df["severity_label"]

        if args.tune:
            # GridSearchCV uses its own StratifiedKFold; keeps locked folds for
            # the final apples-to-apples evaluation below.
            grid = SMOKE_HARDENED_GRID if args.smoke else DEFAULT_HARDENED_GRID
            best_pipeline, grid_results = tune_hardened_pipeline(
                X, y, grid=grid, n_splits=5, random_state=42,
            )
            mlflow.log_params({f"best__{k}": str(v) for k, v in grid_results["best_params"].items()})
            mlflow.log_metric("grid_best_cv_f1_weighted", grid_results["best_cv_f1_weighted"])
        else:
            best_pipeline = build_hardened_pipeline()

        # ── 7. CV on locked splits with per-class F1 ────────────────────────
        logger.info("Evaluating on locked CV splits with per-class F1...")
        cv = cross_validate_hardened(
            best_pipeline, X, y,
            fold_indices=fold_indices,
            labels=sorted(y.unique()),
        )

        # Log aggregate and per-class F1 to MLflow
        for key, value in cv.items():
            if key == "per_fold" or key == "n_splits":
                continue
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value))
        for i, fold_m in enumerate(cv["per_fold"], start=1):
            for k, v in fold_m.items():
                if k == "fold":
                    continue
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"fold{i}_{k}", float(v))

        # ── 8. Fit on full dataset and save hardened artefacts ──────────────
        logger.info("Fitting final hardened pipeline on full dataset...")
        best_pipeline.fit(X, y)
        joblib.dump(best_pipeline, HARDENED_MODEL_PATH)
        mlflow.sklearn.log_model(best_pipeline, "hardened_pipeline")
        logger.info(f"Hardened pipeline saved to {HARDENED_MODEL_PATH}")

        # ── 9. Persist metrics (separate file — does NOT overwrite champion) ─
        hardened_metrics = {
            "cv_f1_mean": cv["f1_weighted_mean"],
            "cv_f1_std": cv["f1_weighted_std"],
            "cv_f1_macro_mean": cv["f1_macro_mean"],
            "cv_f1_macro_std": cv["f1_macro_std"],
            "per_class": {
                k: v for k, v in cv.items()
                if k.startswith("f1_") and (k.endswith("_mean") or k.endswith("_std"))
            },
            "training_records": len(df),
            "model_type": "hardened_tfidf_lr",
            "enrich": args.enrich,
            "drop_unknown": args.drop_unknown,
            "cv_splits_path": args.cv_splits_path,
            "mlflow_run_id": run.info.run_id,
        }
        _save_hardened_metrics(hardened_metrics)

        # ── 10. Promotion gate (optional) ───────────────────────────────────
        champion = _load_champion_metrics()
        champion_f1 = champion.get("cv_f1_mean") or champion.get("f1_weighted", 0.0)
        delta = cv["f1_weighted_mean"] - champion_f1

        if delta >= PROMOTION_MIN_DELTA:
            reason = f"cv_f1 +{delta:.4f} >= delta {PROMOTION_MIN_DELTA}"
            promoted = bool(args.promote_champion)
        else:
            reason = f"cv_f1 delta {delta:+.4f} < {PROMOTION_MIN_DELTA}"
            promoted = False

        mlflow.set_tag("phase", "phase2-step0-hardened")
        mlflow.set_tag("promoted_to_champion", str(promoted).lower())
        mlflow.set_tag("promotion_reason", reason)
        mlflow.log_metric("champion_v1_cv_f1", champion_f1)
        mlflow.log_metric("delta_vs_champion_v1", delta)

        if promoted:
            logger.info("Promoting hardened pipeline to champion...")
            joblib.dump(best_pipeline, CHAMPION_MODEL_PATH)
            with open(CHAMPION_METRICS_PATH, "w") as f:
                json.dump({
                    "f1_weighted": cv["f1_weighted_mean"],
                    "accuracy": None,  # hardened uses CV, not hold-out accuracy
                    "cv_f1_mean": cv["f1_weighted_mean"],
                    "cv_f1_std": cv["f1_weighted_std"],
                    "training_records": len(df),
                    "model_type": "hardened_tfidf_lr",
                }, f, indent=2)

        _print_summary(cv, champion, promoted, reason)
        logger.info(f"Done. MLflow run: {run.info.run_id}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train HARDENED TF-IDF + LR — Phase 2 Step 0")
    p.add_argument("--records", type=int, default=5000,
                   help="Records to fetch from openFDA if no cache (default 5000)")
    p.add_argument("--use-cached", action="store_true",
                   help="Load cached data/raw/maude_raw.csv")
    p.add_argument("--drop-unknown", action="store_true",
                   help="Exclude UNKNOWN severity rows")
    p.add_argument("--enrich", action="store_true",
                   help="Apply extended clinical abbreviation map before vectorization")
    p.add_argument("--tune", action="store_true",
                   help="Run GridSearchCV over the hardened hyperparameter grid")
    p.add_argument("--smoke", action="store_true",
                   help="Use the compact smoke-test grid (for quick validation)")
    p.add_argument("--cv-splits-path", default=DEFAULT_CV_SPLITS_PATH,
                   help="Locked CV splits JSON (default data/cv_splits.json)")
    p.add_argument("--regenerate-splits", action="store_true",
                   help="Regenerate CV splits (INVALIDATES all prior Phase 2 runs)")
    p.add_argument("--check-fingerprint", action="store_true", default=True,
                   help="Verify splits fingerprint matches current data (default on)")
    p.add_argument("--promote-champion", action="store_true",
                   help="If CV F1 beats champion by min delta, promote to champion")
    p.add_argument("--run-name", default=None,
                   help="MLflow run name")
    args = p.parse_args()
    main(args)
