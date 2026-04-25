"""
Locked StratifiedKFold splits for Phase 2 evaluation parity.

Why this module exists:
  Phase 2 must compare the hardened TF-IDF+LR baseline against Bio_ClinicalBERT
  on IDENTICAL fold splits. Anything else leaves room for "my test set was
  easier" arguments. This module generates those splits once, saves them to
  data/cv_splits.json, and gives every Phase 2 training script a way to load
  them.

Lock policy:
  The file at data/cv_splits.json is the single source of truth for Phase 2
  folds. Do not regenerate it casually. Any regeneration invalidates every
  prior Phase 2 MLflow run's comparability.

Schema of data/cv_splits.json:
  {
    "version": 1,
    "created_utc": "2026-04-24T...",
    "random_state": 42,
    "n_splits": 5,
    "dataset_fingerprint": {
      "n_records": 154776,
      "text_col": "clean_text",
      "label_col": "severity_label",
      "label_distribution": {"M": ..., "I": ..., "D": ..., "O": ..., "UNKNOWN": ...},
      "sha1_first_100_labels": "...",   # guards against silent data changes
    },
    "folds": [
      {"fold": 1, "train_idx": [...], "val_idx": [...]},
      ...
    ]
  }

Usage:

    # One-time (or on deliberate regeneration):
    python -m src.evaluation.cv_splits generate \\
        --data-path data/raw/maude_raw.csv \\
        --out data/cv_splits.json

    # In every Phase 2 training script:
    from src.evaluation.cv_splits import load_cv_splits
    fold_indices = load_cv_splits("data/cv_splits.json",
                                  expected_n_records=len(df))
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.preprocessing.text_cleaner import clean_dataframe

logger = logging.getLogger(__name__)

DEFAULT_CV_SPLITS_PATH = "data/cv_splits.json"
DEFAULT_RAW_DATA_PATH = "data/raw/maude_raw.csv"

SCHEMA_VERSION = 1
DEFAULT_N_SPLITS = 5
DEFAULT_RANDOM_STATE = 42


# ── Dataset fingerprinting ───────────────────────────────────────────────────

def _sha1_first_n_labels(labels: pd.Series, n: int = 100) -> str:
    """Hash the first N labels — cheap guard against silent dataset drift."""
    sample = "|".join(str(x) for x in labels.head(n).tolist())
    return hashlib.sha1(sample.encode()).hexdigest()


def build_fingerprint(df: pd.DataFrame, label_col: str, text_col: str) -> dict:
    """Describe the dataset well enough to detect accidental changes."""
    return {
        "n_records": int(len(df)),
        "text_col": text_col,
        "label_col": label_col,
        "label_distribution": df[label_col].value_counts().to_dict(),
        "sha1_first_100_labels": _sha1_first_n_labels(df[label_col]),
    }


# ── Generate ─────────────────────────────────────────────────────────────────

def generate_cv_splits(
    df: pd.DataFrame,
    label_col: str = "severity_label",
    text_col: str = "clean_text",
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict:
    """
    Generate StratifiedKFold splits on the cleaned DataFrame.

    Returns the JSON-serialisable dict ready to be written to disk.
    """
    if text_col not in df.columns:
        raise ValueError(
            f"'{text_col}' not in DataFrame. Run clean_dataframe() first "
            f"(or pass text_col='narrative_text' if using raw data)."
        )
    if label_col not in df.columns:
        raise ValueError(f"'{label_col}' not in DataFrame.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fingerprint = build_fingerprint(df, label_col=label_col, text_col=text_col)

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(df[text_col], df[label_col]), start=1):
        folds.append({
            "fold": fold_idx,
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
        })

    payload = {
        "version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_state": random_state,
        "n_splits": n_splits,
        "dataset_fingerprint": fingerprint,
        "folds": folds,
    }
    logger.info(
        f"Generated {n_splits} folds on {fingerprint['n_records']:,} records. "
        f"Label dist: {fingerprint['label_distribution']}"
    )
    return payload


def save_cv_splits(payload: dict, path: str = DEFAULT_CV_SPLITS_PATH) -> None:
    """Write the splits payload to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)
    # Report size for sanity
    size_kb = os.path.getsize(path) / 1024
    logger.info(f"Saved {len(payload['folds'])} folds to {path} ({size_kb:.1f} KB)")


# ── Load & verify ────────────────────────────────────────────────────────────

def load_cv_splits(
    path: str = DEFAULT_CV_SPLITS_PATH,
    expected_n_records: Optional[int] = None,
    expected_fingerprint: Optional[dict] = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Load locked folds. Verifies fingerprint if `expected_fingerprint` is given,
    or just checks record count if `expected_n_records` is given.

    Returns a list of (train_idx, val_idx) numpy array tuples — the same shape
    StratifiedKFold.split yields, so it's a drop-in replacement.
    """
    with open(path) as f:
        payload = json.load(f)

    if payload.get("version") != SCHEMA_VERSION:
        raise ValueError(
            f"CV splits file schema version {payload.get('version')} "
            f"!= expected {SCHEMA_VERSION}. Regenerate via "
            f"`python -m src.evaluation.cv_splits generate`."
        )

    fp = payload["dataset_fingerprint"]

    if expected_n_records is not None and fp["n_records"] != expected_n_records:
        raise ValueError(
            f"CV splits fingerprint records={fp['n_records']} "
            f"!= current dataset records={expected_n_records}. "
            f"Dataset likely changed since splits were locked — "
            f"regenerate only if you intend to invalidate prior Phase 2 runs."
        )

    if expected_fingerprint is not None:
        for key, expected in expected_fingerprint.items():
            if fp.get(key) != expected:
                raise ValueError(
                    f"Fingerprint mismatch on '{key}': "
                    f"locked={fp.get(key)} vs current={expected}."
                )

    fold_indices = [
        (np.asarray(f["train_idx"]), np.asarray(f["val_idx"]))
        for f in payload["folds"]
    ]
    logger.info(
        f"Loaded {len(fold_indices)} locked folds from {path} "
        f"(fingerprint: {fp['n_records']:,} records, "
        f"labels={list(fp['label_distribution'].keys())})"
    )
    return fold_indices


def verify_splits_match_df(payload: dict, df: pd.DataFrame) -> None:
    """Raise if dataset fingerprint in the splits file doesn't match df."""
    current_fp = build_fingerprint(
        df,
        label_col=payload["dataset_fingerprint"]["label_col"],
        text_col=payload["dataset_fingerprint"]["text_col"],
    )
    locked_fp = payload["dataset_fingerprint"]
    for key in ("n_records", "sha1_first_100_labels"):
        if current_fp[key] != locked_fp[key]:
            raise ValueError(
                f"Dataset drift detected — fingerprint '{key}' differs: "
                f"locked={locked_fp[key]}, current={current_fp[key]}."
            )


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli_generate(args: argparse.Namespace) -> None:
    """
    Load raw data, clean it, and generate locked CV splits.

    Uses the same clean_dataframe() as train.py so fingerprints align with the
    v1 champion's training set.
    """
    logger.info(f"Loading raw data from {args.data_path}")
    df = pd.read_csv(args.data_path)

    logger.info("Cleaning narratives (same pipeline as train.py)...")
    df = clean_dataframe(df)

    if args.drop_unknown:
        before = len(df)
        df = df[df["severity_label"] != "UNKNOWN"].reset_index(drop=True)
        logger.info(f"Dropped {before - len(df)} UNKNOWN rows.")

    if args.min_records and len(df) < args.min_records:
        raise SystemExit(
            f"Only {len(df)} rows after cleaning — below --min-records {args.min_records}. "
            f"Fetch more data or lower the threshold."
        )

    payload = generate_cv_splits(
        df,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )
    save_cv_splits(payload, args.out)

    # Print a short summary to stdout for the runbook
    fp = payload["dataset_fingerprint"]
    print("\n" + "=" * 60)
    print(f"  CV splits locked to: {args.out}")
    print(f"  Records:            {fp['n_records']:,}")
    print(f"  Folds:              {args.n_splits} (random_state={args.random_state})")
    print(f"  Label distribution: {fp['label_distribution']}")
    print("  Fingerprint SHA1:   " + fp["sha1_first_100_labels"][:16] + "...")
    print("=" * 60)
    print("  Use load_cv_splits() in every Phase 2 training script.")
    print("=" * 60 + "\n")


def _cli_verify(args: argparse.Namespace) -> None:
    """Re-open the splits file and confirm it matches the current cached dataset."""
    logger.info(f"Loading splits from {args.splits_path}")
    with open(args.splits_path) as f:
        payload = json.load(f)

    logger.info(f"Loading current data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    df = clean_dataframe(df)
    if args.drop_unknown:
        df = df[df["severity_label"] != "UNKNOWN"].reset_index(drop=True)

    verify_splits_match_df(payload, df)
    print("✓ Splits fingerprint matches current cleaned dataset.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Lock StratifiedKFold splits for Phase 2.")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Generate and save locked splits.")
    g.add_argument("--data-path", default=DEFAULT_RAW_DATA_PATH)
    g.add_argument("--out", default=DEFAULT_CV_SPLITS_PATH)
    g.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    g.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    g.add_argument("--drop-unknown", action="store_true")
    g.add_argument("--min-records", type=int, default=0)
    g.set_defaults(func=_cli_generate)

    v = sub.add_parser("verify", help="Verify splits match current data.")
    v.add_argument("--splits-path", default=DEFAULT_CV_SPLITS_PATH)
    v.add_argument("--data-path", default=DEFAULT_RAW_DATA_PATH)
    v.add_argument("--drop-unknown", action="store_true")
    v.set_defaults(func=_cli_verify)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
