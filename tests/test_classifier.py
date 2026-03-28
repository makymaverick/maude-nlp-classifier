"""Unit tests for the TF-IDF classifier pipeline."""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.model.classifier import (
    build_pipeline,
    split_data,
    tune_pipeline,
    train_pipeline,
    evaluate,
    predict_single,
    save_model,
    load_model,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_data():
    texts = [
        "patient died after cardiac device implant failure serious event",
        "patient deceased following pacemaker malfunction fatal outcome",
        "severe burn injury caused by electrosurgical grounding pad failure",
        "patient sustained broken arm after hospital bed rail collapse",
        "infusion pump alarmed and stopped no patient harm occurred",
        "device stopped working returned to manufacturer for analysis",
        "software error caused incorrect insulin dose delivery injury",
        "catheter disconnected patient reported pain and bruising injury",
    ] * 10  # repeat to have enough samples per class

    labels = [
        "DEATH", "DEATH",
        "SERIOUS_INJURY", "INJURY",
        "MALFUNCTION", "MALFUNCTION",
        "SERIOUS_INJURY", "INJURY",
    ] * 10

    return pd.Series(texts), pd.Series(labels)


@pytest.fixture
def sample_df():
    texts = [
        "patient died after cardiac device implant failure serious event",
        "patient deceased following pacemaker malfunction fatal outcome",
        "severe burn injury caused by electrosurgical grounding pad failure",
        "patient sustained broken arm after hospital bed rail collapse",
        "infusion pump alarmed and stopped no patient harm occurred",
        "device stopped working returned to manufacturer for analysis",
        "software error caused incorrect insulin dose delivery injury",
        "catheter disconnected patient reported pain and bruising injury",
    ] * 10

    labels = [
        "DEATH", "DEATH",
        "SERIOUS_INJURY", "INJURY",
        "MALFUNCTION", "MALFUNCTION",
        "SERIOUS_INJURY", "INJURY",
    ] * 10

    return pd.DataFrame({"clean_text": texts, "severity_label": labels})


# ── build_pipeline tests ──────────────────────────────────────────────────────

def test_build_pipeline_logreg():
    pipe = build_pipeline("logreg")
    assert "tfidf" in pipe.named_steps
    assert "clf" in pipe.named_steps


def test_build_pipeline_svm():
    pipe = build_pipeline("svm")
    assert "tfidf" in pipe.named_steps
    assert "clf" in pipe.named_steps


# ── train_pipeline tests ──────────────────────────────────────────────────────

def test_train_pipeline_fits(sample_data):
    X, y = sample_data
    pipe = build_pipeline("logreg")
    trained = train_pipeline(pipe, X, y)
    # Should be able to predict after training
    preds = trained.predict(X[:5])
    assert len(preds) == 5


# ── evaluate tests ────────────────────────────────────────────────────────────

def test_evaluate_returns_metrics(sample_data):
    X, y = sample_data
    pipe = build_pipeline("logreg")
    pipe = train_pipeline(pipe, X, y)
    metrics = evaluate(pipe, X, y)

    assert "accuracy" in metrics
    assert "f1_weighted" in metrics
    assert "classification_report" in metrics
    assert "confusion_matrix" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


# ── predict_single tests ──────────────────────────────────────────────────────

def test_predict_single_returns_label(sample_data):
    X, y = sample_data
    pipe = build_pipeline("logreg")
    pipe = train_pipeline(pipe, X, y)

    result = predict_single(pipe, "the patient died after device implantation")
    assert "predicted_label" in result
    assert result["predicted_label"] in ["DEATH", "SERIOUS_INJURY", "INJURY", "MALFUNCTION"]


def test_predict_single_returns_probabilities(sample_data):
    X, y = sample_data
    pipe = build_pipeline("logreg")
    pipe = train_pipeline(pipe, X, y)

    result = predict_single(pipe, "the device malfunctioned and stopped working")
    assert "probabilities" in result
    probs = result["probabilities"]
    assert abs(sum(probs.values()) - 1.0) < 1e-5  # probabilities sum to 1


# ── split_data tests ──────────────────────────────────────────────────────────

def test_split_data_returns_correct_shapes(sample_df):
    X_train, X_test, y_train, y_test = split_data(sample_df)
    total = len(sample_df)
    assert len(X_train) + len(X_test) == total
    assert len(y_train) + len(y_test) == total
    # default test_size=0.2
    assert len(X_test) == pytest.approx(total * 0.2, abs=2)


def test_split_data_is_stratified(sample_df):
    _, _, y_train, y_test = split_data(sample_df)
    train_dist = y_train.value_counts(normalize=True).sort_index()
    test_dist = y_test.value_counts(normalize=True).sort_index()
    for label in train_dist.index:
        assert abs(train_dist[label] - test_dist[label]) < 0.1


# ── tune_pipeline tests ───────────────────────────────────────────────────────

@patch("src.model.classifier.GridSearchCV")
def test_tune_pipeline_calls_fit_and_returns_best_estimator(mock_grid_cv, sample_data):
    X, y = sample_data
    pipe = build_pipeline("logreg")

    mock_grid_instance = MagicMock()
    mock_grid_instance.best_estimator_ = pipe
    mock_grid_instance.best_params_ = {"clf__C": 1.0}
    mock_grid_instance.best_score_ = 0.9
    mock_grid_cv.return_value = mock_grid_instance

    result = tune_pipeline(pipe, X, y, model_type="logreg")
    mock_grid_instance.fit.assert_called_once_with(X, y)
    assert result is pipe


@patch("src.model.classifier.GridSearchCV")
def test_tune_pipeline_svm_branch(mock_grid_cv, sample_data):
    X, y = sample_data
    pipe = build_pipeline("svm")

    mock_grid_instance = MagicMock()
    mock_grid_instance.best_estimator_ = pipe
    mock_grid_instance.best_params_ = {}
    mock_grid_instance.best_score_ = 0.85
    mock_grid_cv.return_value = mock_grid_instance

    result = tune_pipeline(pipe, X, y, model_type="svm")
    assert result is pipe


# ── predict_single SVM (decision_function) tests ─────────────────────────────

def test_predict_single_svm_returns_decision_scores(sample_data):
    X, y = sample_data
    pipe = build_pipeline("svm")
    pipe = train_pipeline(pipe, X, y)

    result = predict_single(pipe, "device stopped and malfunctioned")
    assert "predicted_label" in result
    assert "decision_scores" in result


# ── save_model / load_model tests ─────────────────────────────────────────────

def test_save_and_load_model_roundtrip(sample_data):
    X, y = sample_data
    pipe = build_pipeline("logreg")
    pipe = train_pipeline(pipe, X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.joblib")
        save_model(pipe, path)
        assert os.path.exists(path)
        loaded = load_model(path)
        preds = loaded.predict(X[:3])
        assert len(preds) == 3


def test_load_model_raises_if_not_found():
    with pytest.raises(FileNotFoundError):
        load_model("/nonexistent/path/model.joblib")
