"""
TF-IDF + Classifier Pipeline for MAUDE Adverse Event Severity Classification.

Supports:
  - Logistic Regression (default)
  - Linear SVM (SVC)
  - Grid search hyperparameter tuning
  - Model persistence (joblib)
  - Evaluation metrics (classification report, confusion matrix)
"""

import os
import logging
from typing import Tuple, Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)

LABEL_COL = "severity_label"
TEXT_COL = "clean_text"

# Labels ordered by severity (used for display)
LABEL_ORDER = ["DEATH", "SERIOUS_INJURY", "INJURY", "MALFUNCTION", "UNKNOWN"]


def build_pipeline(model_type: str = "logreg") -> Pipeline:
    """
    Build a scikit-learn Pipeline with TF-IDF vectorizer and a classifier.

    Args:
        model_type: 'logreg' for Logistic Regression, 'svm' for Linear SVM.

    Returns:
        sklearn Pipeline object (untrained).
    """
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),      # unigrams + bigrams
        max_features=50_000,
        sublinear_tf=True,       # apply log normalization to TF
        min_df=3,                # ignore terms appearing in fewer than 3 docs
        max_df=0.95,             # ignore terms appearing in >95% of docs
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-zA-Z][a-zA-Z]+\b",  # skip single chars & numbers
    )

    if model_type == "svm":
        clf = LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
        )
    else:  # default: logreg
        clf = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        )

    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Stratified train/test split."""
    X = df[TEXT_COL]
    y = df[LABEL_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def tune_pipeline(
    pipeline: Pipeline,
    X_train: pd.Series,
    y_train: pd.Series,
    model_type: str = "logreg",
) -> Pipeline:
    """
    Run grid search to find best hyperparameters.

    Args:
        pipeline: Untrained sklearn Pipeline.
        X_train: Training text series.
        y_train: Training label series.
        model_type: 'logreg' or 'svm'.

    Returns:
        Best estimator from GridSearchCV.
    """
    if model_type == "svm":
        param_grid = {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__max_features": [30_000, 50_000],
            "clf__C": [0.1, 1.0, 10.0],
        }
    else:
        param_grid = {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__max_features": [30_000, 50_000],
            "clf__C": [0.1, 1.0, 10.0],
        }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1,
    )
    logger.info("Running grid search...")
    grid.fit(X_train, y_train)
    logger.info(f"Best params: {grid.best_params_}")
    logger.info(f"Best CV F1 (weighted): {grid.best_score_:.4f}")
    return grid.best_estimator_


def train_pipeline(
    pipeline: Pipeline,
    X_train: pd.Series,
    y_train: pd.Series,
) -> Pipeline:
    """Train the pipeline directly (no grid search)."""
    logger.info("Training pipeline...")
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate(
    pipeline: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate trained pipeline and return metrics.

    Returns:
        Dict with accuracy, f1_weighted, classification_report, confusion_matrix.
    """
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)

    logger.info(f"\nAccuracy: {acc:.4f} | Weighted F1: {f1:.4f}")
    logger.info(f"\nClassification Report:\n{report}")

    return {
        "accuracy": acc,
        "f1_weighted": f1,
        "classification_report": report,
        "confusion_matrix": cm,
        "classes": list(pipeline.classes_),
    }


def predict_single(pipeline: Pipeline, text: str) -> dict:
    """
    Run inference on a single narrative text string.

    Returns:
        Dict with predicted label and per-class probabilities (if available).
    """
    prediction = pipeline.predict([text])[0]
    result = {"predicted_label": prediction}

    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        proba = pipeline.predict_proba([text])[0]
        result["probabilities"] = dict(zip(pipeline.classes_, proba.tolist()))
    elif hasattr(clf, "decision_function"):
        scores = pipeline.decision_function([text])[0]
        result["decision_scores"] = dict(zip(pipeline.classes_, scores.tolist()))

    return result


def save_model(pipeline: Pipeline, path: str = "models/maude_classifier.joblib") -> None:
    """Persist trained pipeline to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info(f"Model saved to {path}")


def load_model(path: str = "models/maude_classifier.joblib") -> Pipeline:
    """Load a persisted pipeline from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    pipeline = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return pipeline
