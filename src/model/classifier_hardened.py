"""
Hardened TF-IDF + Logistic Regression pipeline — Phase 2 Step 0.

Goal: establish the real ceiling of the TF-IDF+LR baseline before declaring
BERT a winner. Differences from the champion v1 pipeline (classifier.py):

  1. FeatureUnion of word (1,2) and (1,3) n-grams PLUS char_wb (3,5) n-grams
     — char n-grams capture clinical suffixes (-ectomy, -itis, -oscopy) and
     spelling variants common in free-text MDRs.
  2. Wider hyperparameter grid (C across log decades, L1 + L2, 30k/50k/100k
     feature cap, extended n-gram range).
  3. Per-class F1 reporting (especially Death class) — weighted F1 alone
     hides Death collapse because Malfunction dominates support.
  4. Optional extended clinical abbreviation expansion before vectorization.
  5. Class-weight options: 'balanced' vs explicit Death-biased weights.

This module does NOT modify classifier.py or change the deployed champion.
It is imported by src.model.train_hardened (the standalone Step 0 entrypoint).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.preprocessing.clinical_abbreviations import EXTENDED_ABBREVIATION_MAP

logger = logging.getLogger(__name__)

LABEL_COL = "severity_label"
TEXT_COL = "clean_text"

# Keep identical to classifier.py so metrics align across v1 / hardened / BERT
LABEL_ORDER = ["D", "I", "M", "O", "UNKNOWN"]


# ── Text enrichment ──────────────────────────────────────────────────────────

def expand_extended_abbreviations(text: str) -> str:
    """
    Apply the ~120-entry clinical abbreviation map from
    src.preprocessing.clinical_abbreviations. Case-insensitive.

    Run this on already-cleaned text (output of text_cleaner.clean_text)
    so boilerplate is already stripped.
    """
    if not isinstance(text, str) or not text:
        return ""
    for pattern, replacement in EXTENDED_ABBREVIATION_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def enrich_dataframe(
    df: pd.DataFrame,
    text_col: str = TEXT_COL,
    output_col: str = "enriched_text",
) -> pd.DataFrame:
    """Apply extended abbreviation expansion to a DataFrame column (in place-safe)."""
    df = df.copy()
    df[output_col] = df[text_col].apply(expand_extended_abbreviations)
    return df


# ── Pipeline builders ────────────────────────────────────────────────────────

def build_hardened_pipeline(
    ngram_word: tuple = (1, 2),
    ngram_char: tuple = (3, 5),
    max_features_word: int = 50_000,
    max_features_char: int = 50_000,
    min_df: int = 3,
    max_df: float = 0.95,
    C: float = 1.0,
    penalty: str = "l2",
    solver: str = "lbfgs",
    class_weight: str = "balanced",
) -> Pipeline:
    """
    Build a Pipeline:
        FeatureUnion([word TF-IDF, char_wb TF-IDF]) -> LogisticRegression.

    Defaults match the champion v1 on the word branch, add char_wb on top.
    All knobs are exposed for GridSearchCV.
    """
    word_vec = TfidfVectorizer(
        ngram_range=ngram_word,
        analyzer="word",
        max_features=max_features_word,
        sublinear_tf=True,
        min_df=min_df,
        max_df=max_df,
        strip_accents="unicode",
        token_pattern=r"\b[a-zA-Z][a-zA-Z]+\b",
    )
    char_vec = TfidfVectorizer(
        ngram_range=ngram_char,
        analyzer="char_wb",
        max_features=max_features_char,
        sublinear_tf=True,
        min_df=min_df,
        max_df=max_df,
        strip_accents="unicode",
    )
    features = FeatureUnion(
        transformer_list=[("word", word_vec), ("char", char_vec)],
    )

    # L1 requires saga/liblinear; L2 can use lbfgs
    if penalty == "l1" and solver == "lbfgs":
        solver = "saga"

    clf = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=2000,
        class_weight=class_weight,
        n_jobs=-1 if solver == "saga" else None,
    )

    return Pipeline([("features", features), ("clf", clf)])


# ── Hyperparameter search ────────────────────────────────────────────────────

DEFAULT_HARDENED_GRID: dict = {
    # Word vectorizer
    "features__word__ngram_range": [(1, 2), (1, 3)],
    "features__word__max_features": [50_000, 100_000],
    # Char vectorizer (cheaper to keep fixed; expand if budget allows)
    "features__char__ngram_range": [(3, 5)],
    "features__char__max_features": [50_000],
    # Classifier
    "clf__C": [0.3, 1.0, 3.0, 10.0],
}

# Compact grid for CI / smoke tests (≤8 combos)
SMOKE_HARDENED_GRID: dict = {
    "features__word__ngram_range": [(1, 2)],
    "features__word__max_features": [50_000],
    "features__char__ngram_range": [(3, 5)],
    "features__char__max_features": [30_000],
    "clf__C": [1.0, 3.0],
}


def tune_hardened_pipeline(
    X: pd.Series,
    y: pd.Series,
    grid: Optional[dict] = None,
    n_splits: int = 5,
    random_state: int = 42,
    scoring: str = "f1_weighted",
    verbose: int = 1,
) -> tuple[Pipeline, dict]:
    """
    GridSearchCV over the hardened pipeline. Returns (best_estimator, results).

    Uses StratifiedKFold with the same random_state as v1 so fold splits match.
    """
    if grid is None:
        grid = DEFAULT_HARDENED_GRID

    pipeline = build_hardened_pipeline()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    gs = GridSearchCV(
        pipeline,
        grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=verbose,
        refit=True,
        return_train_score=False,
    )
    logger.info("Running hardened grid search...")
    gs.fit(X, y)
    logger.info(f"Best params: {gs.best_params_}")
    logger.info(f"Best CV {scoring}: {gs.best_score_:.4f}")

    results = {
        "best_params": gs.best_params_,
        "best_cv_f1_weighted": float(gs.best_score_),
        "cv_results": {
            "mean_test_score": gs.cv_results_["mean_test_score"].tolist(),
            "std_test_score": gs.cv_results_["std_test_score"].tolist(),
            "params": gs.cv_results_["params"],
        },
    }
    return gs.best_estimator_, results


# ── Evaluation ───────────────────────────────────────────────────────────────

def per_class_f1(y_true, y_pred, labels: list[str] | None = None) -> dict:
    """
    Return per-class F1 as a dict, plus macro and weighted aggregates.

    This is the critical metric Phase 1's train.py doesn't surface: weighted F1
    alone can hide Death-class collapse because Malfunction dominates support.
    """
    if labels is None:
        labels = sorted(set(list(y_true) + list(y_pred)))

    f1_per = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    out = {f"f1_{lbl}": float(score) for lbl, score in zip(labels, f1_per)}
    out["f1_macro"] = float(
        f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    out["f1_weighted"] = float(
        f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    )
    return out


def cross_validate_hardened(
    pipeline: Pipeline,
    X: pd.Series,
    y: pd.Series,
    fold_indices: list[tuple[np.ndarray, np.ndarray]] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    labels: list[str] | None = None,
) -> dict:
    """
    Cross-validate with per-class F1 reporting.

    If `fold_indices` is provided (from src.evaluation.cv_splits.load_cv_splits),
    those locked folds are used — required for apples-to-apples comparison
    against Phase 2 BERT runs.

    Otherwise a fresh StratifiedKFold with `random_state` is used.
    """
    if labels is None:
        labels = sorted(y.unique())

    if fold_indices is None:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_indices = [(tr, va) for tr, va in cv.split(X, y)]

    fold_metrics: list[dict] = []
    for fold_idx, (train_idx, val_idx) in enumerate(fold_indices, start=1):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        # Clone via re-building; sklearn.base.clone would work too but this is explicit
        fold_pipe = build_hardened_pipeline(
            **_extract_pipeline_params(pipeline)
        ) if hasattr(pipeline, "named_steps") else pipeline
        fold_pipe.fit(X_tr, y_tr)
        y_pred = fold_pipe.predict(X_va)

        m = per_class_f1(y_va, y_pred, labels=labels)
        m["fold"] = fold_idx
        fold_metrics.append(m)
        logger.info(
            f"Fold {fold_idx}: weighted={m['f1_weighted']:.4f}  "
            f"macro={m['f1_macro']:.4f}  "
            + " ".join(f"{k}={v:.3f}" for k, v in m.items()
                       if k.startswith("f1_") and k not in ("f1_macro", "f1_weighted"))
        )

    agg: dict = {}
    for key in fold_metrics[0]:
        if key == "fold":
            continue
        vals = [m[key] for m in fold_metrics]
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals))
    agg["per_fold"] = fold_metrics
    agg["n_splits"] = len(fold_indices)
    return agg


def _extract_pipeline_params(pipeline: Pipeline) -> dict:
    """
    Extract the kwargs needed to rebuild a hardened pipeline from an existing one.
    Used by cross_validate_hardened to get a clean clone per fold.
    """
    word = pipeline.named_steps["features"].transformer_list[0][1]
    char = pipeline.named_steps["features"].transformer_list[1][1]
    clf = pipeline.named_steps["clf"]
    return dict(
        ngram_word=word.ngram_range,
        ngram_char=char.ngram_range,
        max_features_word=word.max_features,
        max_features_char=char.max_features,
        min_df=word.min_df,
        max_df=word.max_df,
        C=clf.C,
        penalty=clf.penalty,
        solver=clf.solver,
        class_weight=clf.class_weight,
    )


def evaluate_with_confusion(
    pipeline: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
    labels: list[str] | None = None,
) -> dict:
    """Full evaluation: per-class F1, classification report, confusion matrix."""
    y_pred = pipeline.predict(X_test)
    if labels is None:
        labels = sorted(set(list(y_test) + list(y_pred)))

    metrics = per_class_f1(y_test, y_pred, labels=labels)
    metrics["classification_report"] = classification_report(
        y_test, y_pred, labels=labels, zero_division=0
    )
    metrics["confusion_matrix"] = confusion_matrix(
        y_test, y_pred, labels=labels
    ).tolist()
    metrics["labels"] = labels
    return metrics
