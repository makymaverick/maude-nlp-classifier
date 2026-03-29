"""Unit tests for the training entrypoint."""

import argparse
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, call


from src.model.train import main, _get_champion_f1, _save_champion_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def base_args():
    return argparse.Namespace(
        records=10,
        model="logreg",
        tune=False,
        cross_validate=False,   # new field
        use_cached=False,
        drop_unknown=False,
    )


@pytest.fixture
def sample_df():
    # ≥ 50 rows required to pass the early-abort guard in train.main()
    base_texts  = ["text one", "text two", "text three", "text four"]
    base_labels = ["D", "M", "I", "D"]
    n = 15  # repeat to get 60 rows
    return pd.DataFrame({
        "clean_text":     (base_texts  * n)[:60],
        "severity_label": (base_labels * n)[:60],
    })


def _mock_evaluate():
    return {
        "accuracy": 0.9,
        "f1_weighted": 0.88,
        "classification_report": "report",
        "confusion_matrix": [[1, 0], [0, 1]],
        "classes": ["D", "I", "M"],
    }


def _mock_split(df, **kwargs):
    X = df["clean_text"]
    y = df["severity_label"]
    mid = len(df) // 2
    return X.iloc[:mid], X.iloc[mid:], y.iloc[:mid], y.iloc[mid:]


def _dummy_baseline_result():
    return {"dummy_accuracy": 0.5, "dummy_f1_weighted": 0.4}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for patching MLflow context manager
# ─────────────────────────────────────────────────────────────────────────────

def _make_mlflow_run_mock():
    """Return a MagicMock that works as an mlflow.start_run context manager."""
    run_info = MagicMock()
    run_info.run_id = "abc123def456"
    run_mock = MagicMock()
    run_mock.__enter__ = MagicMock(return_value=run_mock)
    run_mock.__exit__ = MagicMock(return_value=False)
    run_mock.info = run_info
    return run_mock


MLFLOW_PATCHES = [
    "src.model.train.mlflow.set_tracking_uri",
    "src.model.train.mlflow.set_experiment",
    "src.model.train.mlflow.start_run",
    "src.model.train.mlflow.log_params",
    "src.model.train.mlflow.log_param",
    "src.model.train.mlflow.log_metric",
    "src.model.train.mlflow.log_metrics",
    "src.model.train.mlflow.log_text",
    "src.model.train.mlflow.set_tag",
    "src.model.train.mlflow.sklearn.log_model",
]


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

@patch("src.model.train.save_model")
@patch("src.model.train.evaluate")
@patch("src.model.train.dummy_baseline")
@patch("src.model.train.train_pipeline")
@patch("src.model.train.build_pipeline")
@patch("src.model.train.split_data")
@patch("src.model.train.clean_dataframe")
@patch("src.model.train.get_label_distribution")
@patch("src.model.train.save_raw_data")
@patch("src.model.train.fetch_maude_records")
@patch("src.model.train.mlflow.sklearn.log_model")
@patch("src.model.train.mlflow.set_tag")
@patch("src.model.train.mlflow.log_text")
@patch("src.model.train.mlflow.log_metrics")
@patch("src.model.train.mlflow.log_metric")
@patch("src.model.train.mlflow.log_param")
@patch("src.model.train.mlflow.log_params")
@patch("src.model.train.mlflow.start_run")
@patch("src.model.train.mlflow.set_experiment")
@patch("src.model.train.mlflow.set_tracking_uri")
def test_main_runs_full_pipeline(
    mock_tracking_uri, mock_set_exp, mock_start_run,
    mock_log_params, mock_log_param, mock_log_metric, mock_log_metrics,
    mock_log_text, mock_set_tag, mock_log_model,
    mock_fetch, mock_save_raw, mock_label_dist, mock_clean,
    mock_split, mock_build, mock_train, mock_dummy, mock_evaluate, mock_save_model,
    base_args, sample_df,
):
    mock_start_run.return_value = _make_mlflow_run_mock()
    mock_fetch.return_value = sample_df
    mock_label_dist.return_value = pd.Series({"D": 2, "M": 1})
    mock_clean.return_value = sample_df
    mock_split.side_effect = _mock_split
    mock_pipeline = MagicMock()
    mock_build.return_value = mock_pipeline
    mock_train.return_value = mock_pipeline
    mock_dummy.return_value = _dummy_baseline_result()
    mock_evaluate.return_value = _mock_evaluate()

    main(base_args)

    mock_fetch.assert_called_once_with(total_records=10)
    mock_save_raw.assert_called_once()
    mock_clean.assert_called_once()
    mock_dummy.assert_called_once()
    mock_train.assert_called_once()
    mock_evaluate.assert_called_once()


@patch("src.model.train.save_model")
@patch("src.model.train.evaluate")
@patch("src.model.train.dummy_baseline")
@patch("src.model.train.tune_pipeline")
@patch("src.model.train.build_pipeline")
@patch("src.model.train.split_data")
@patch("src.model.train.clean_dataframe")
@patch("src.model.train.get_label_distribution")
@patch("src.model.train.save_raw_data")
@patch("src.model.train.fetch_maude_records")
@patch("src.model.train.mlflow.sklearn.log_model")
@patch("src.model.train.mlflow.set_tag")
@patch("src.model.train.mlflow.log_text")
@patch("src.model.train.mlflow.log_metrics")
@patch("src.model.train.mlflow.log_metric")
@patch("src.model.train.mlflow.log_param")
@patch("src.model.train.mlflow.log_params")
@patch("src.model.train.mlflow.start_run")
@patch("src.model.train.mlflow.set_experiment")
@patch("src.model.train.mlflow.set_tracking_uri")
def test_main_uses_tune_pipeline_when_flag_set(
    mock_tracking_uri, mock_set_exp, mock_start_run,
    mock_log_params, mock_log_param, mock_log_metric, mock_log_metrics,
    mock_log_text, mock_set_tag, mock_log_model,
    mock_fetch, mock_save_raw, mock_label_dist, mock_clean,
    mock_split, mock_build, mock_tune, mock_dummy, mock_evaluate, mock_save_model,
    sample_df,
):
    args = argparse.Namespace(
        records=10, model="logreg", tune=True, cross_validate=False,
        use_cached=False, drop_unknown=False,
    )
    mock_start_run.return_value = _make_mlflow_run_mock()
    mock_fetch.return_value = sample_df
    mock_label_dist.return_value = pd.Series({"D": 2})
    mock_clean.return_value = sample_df
    mock_split.side_effect = _mock_split
    mock_pipeline = MagicMock()
    mock_build.return_value = mock_pipeline
    mock_tune.return_value = mock_pipeline
    mock_dummy.return_value = _dummy_baseline_result()
    mock_evaluate.return_value = _mock_evaluate()

    main(args)
    mock_tune.assert_called_once()


@patch("src.model.train.save_model")
@patch("src.model.train.evaluate")
@patch("src.model.train.dummy_baseline")
@patch("src.model.train.train_pipeline")
@patch("src.model.train.build_pipeline")
@patch("src.model.train.split_data")
@patch("src.model.train.clean_dataframe")
@patch("src.model.train.get_label_distribution")
@patch("src.model.train.save_raw_data")
@patch("src.model.train.fetch_maude_records")
@patch("src.model.train.mlflow.sklearn.log_model")
@patch("src.model.train.mlflow.set_tag")
@patch("src.model.train.mlflow.log_text")
@patch("src.model.train.mlflow.log_metrics")
@patch("src.model.train.mlflow.log_metric")
@patch("src.model.train.mlflow.log_param")
@patch("src.model.train.mlflow.log_params")
@patch("src.model.train.mlflow.start_run")
@patch("src.model.train.mlflow.set_experiment")
@patch("src.model.train.mlflow.set_tracking_uri")
def test_main_drops_unknown_labels(
    mock_tracking_uri, mock_set_exp, mock_start_run,
    mock_log_params, mock_log_param, mock_log_metric, mock_log_metrics,
    mock_log_text, mock_set_tag, mock_log_model,
    mock_fetch, mock_save_raw, mock_label_dist, mock_clean,
    mock_split, mock_build, mock_train, mock_dummy, mock_evaluate, mock_save_model,
    sample_df,
):
    args = argparse.Namespace(
        records=10, model="logreg", tune=False, cross_validate=False,
        use_cached=False, drop_unknown=True,
    )
    # Build a df large enough that after removing UNKNOWN rows we still have ≥ 50.
    # 90 total rows: 30 D, 30 M, 30 UNKNOWN → 60 rows remain after drop.
    df_with_unknown = pd.DataFrame({
        "clean_text":     (["text one", "text two", "text unknown"] * 30),
        "severity_label": (["D",        "M",        "UNKNOWN"]      * 30),
    })
    mock_start_run.return_value = _make_mlflow_run_mock()
    mock_fetch.return_value = df_with_unknown
    mock_label_dist.return_value = pd.Series({"D": 30, "M": 30, "UNKNOWN": 30})
    mock_clean.return_value = df_with_unknown
    mock_split.side_effect = _mock_split
    mock_pipeline = MagicMock()
    mock_build.return_value = mock_pipeline
    mock_train.return_value = mock_pipeline
    mock_dummy.return_value = _dummy_baseline_result()
    mock_evaluate.return_value = _mock_evaluate()

    main(args)

    call_df = mock_split.call_args[0][0]
    assert "UNKNOWN" not in call_df["severity_label"].values
    assert len(call_df) == 60   # 90 rows − 30 UNKNOWN rows


@patch("src.model.train.save_model")
@patch("src.model.train.evaluate")
@patch("src.model.train.dummy_baseline")
@patch("src.model.train.train_pipeline")
@patch("src.model.train.build_pipeline")
@patch("src.model.train.split_data")
@patch("src.model.train.clean_dataframe")
@patch("src.model.train.get_label_distribution")
@patch("src.model.train.pd.read_csv")
@patch("src.model.train.os.path.exists")
@patch("src.model.train.mlflow.sklearn.log_model")
@patch("src.model.train.mlflow.set_tag")
@patch("src.model.train.mlflow.log_text")
@patch("src.model.train.mlflow.log_metrics")
@patch("src.model.train.mlflow.log_metric")
@patch("src.model.train.mlflow.log_param")
@patch("src.model.train.mlflow.log_params")
@patch("src.model.train.mlflow.start_run")
@patch("src.model.train.mlflow.set_experiment")
@patch("src.model.train.mlflow.set_tracking_uri")
def test_main_uses_cached_data_when_available(
    mock_tracking_uri, mock_set_exp, mock_start_run,
    mock_log_params, mock_log_param, mock_log_metric, mock_log_metrics,
    mock_log_text, mock_set_tag, mock_log_model,
    mock_exists, mock_read_csv, mock_label_dist, mock_clean,
    mock_split, mock_build, mock_train, mock_dummy, mock_evaluate, mock_save_model,
    sample_df,
):
    args = argparse.Namespace(
        records=10, model="logreg", tune=False, cross_validate=False,
        use_cached=True, drop_unknown=False,
    )
    mock_start_run.return_value = _make_mlflow_run_mock()
    mock_exists.return_value = True
    mock_read_csv.return_value = sample_df
    mock_label_dist.return_value = pd.Series({"D": 2})
    mock_clean.return_value = sample_df
    mock_split.side_effect = _mock_split
    mock_pipeline = MagicMock()
    mock_build.return_value = mock_pipeline
    mock_train.return_value = mock_pipeline
    mock_dummy.return_value = _dummy_baseline_result()
    mock_evaluate.return_value = _mock_evaluate()

    main(args)
    mock_read_csv.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Champion metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def test_get_champion_f1_returns_zero_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr("src.model.train.CHAMPION_METRICS_PATH",
                        str(tmp_path / "nonexistent.json"))
    assert _get_champion_f1() == 0.0


def test_save_and_get_champion_metrics_roundtrip(tmp_path, monkeypatch):
    path = str(tmp_path / "metrics.json")
    monkeypatch.setattr("src.model.train.CHAMPION_METRICS_PATH", path)

    metrics = {"f1_weighted": 0.87, "accuracy": 0.91, "training_records": 5000}
    _save_champion_metrics(metrics)
    assert _get_champion_f1() == pytest.approx(0.87)
