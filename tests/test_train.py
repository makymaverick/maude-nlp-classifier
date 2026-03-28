"""Unit tests for the training entrypoint."""

import argparse
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.model.train import main


@pytest.fixture
def base_args():
    return argparse.Namespace(
        records=10,
        model="logreg",
        tune=False,
        use_cached=False,
        drop_unknown=False,
    )


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "clean_text": ["text one", "text two", "text three", "text four"],
        "severity_label": ["D", "M", "I", "D"],
    })


def _mock_evaluate():
    return {
        "accuracy": 0.9,
        "f1_weighted": 0.88,
        "classification_report": "report",
        "confusion_matrix": [[1, 0], [0, 1]],
    }


def _mock_split(df, **kwargs):
    X = df["clean_text"]
    y = df["severity_label"]
    mid = len(df) // 2
    return X.iloc[:mid], X.iloc[mid:], y.iloc[:mid], y.iloc[mid:]


@patch("src.model.train.save_model")
@patch("src.model.train.evaluate")
@patch("src.model.train.train_pipeline")
@patch("src.model.train.build_pipeline")
@patch("src.model.train.split_data")
@patch("src.model.train.clean_dataframe")
@patch("src.model.train.get_label_distribution")
@patch("src.model.train.save_raw_data")
@patch("src.model.train.fetch_maude_records")
def test_main_runs_full_pipeline(
    mock_fetch, mock_save_raw, mock_label_dist, mock_clean,
    mock_split, mock_build, mock_train, mock_evaluate, mock_save_model,
    base_args, sample_df,
):
    mock_fetch.return_value = sample_df
    mock_label_dist.return_value = "D: 2"
    mock_clean.return_value = sample_df
    mock_split.side_effect = _mock_split
    mock_pipeline = MagicMock()
    mock_build.return_value = mock_pipeline
    mock_train.return_value = mock_pipeline
    mock_evaluate.return_value = _mock_evaluate()

    main(base_args)

    mock_fetch.assert_called_once_with(total_records=10)
    mock_save_raw.assert_called_once()
    mock_clean.assert_called_once()
    mock_train.assert_called_once()
    mock_evaluate.assert_called_once()
    mock_save_model.assert_called_once()


@patch("src.model.train.save_model")
@patch("src.model.train.evaluate")
@patch("src.model.train.tune_pipeline")
@patch("src.model.train.build_pipeline")
@patch("src.model.train.split_data")
@patch("src.model.train.clean_dataframe")
@patch("src.model.train.get_label_distribution")
@patch("src.model.train.save_raw_data")
@patch("src.model.train.fetch_maude_records")
def test_main_uses_tune_pipeline_when_flag_set(
    mock_fetch, mock_save_raw, mock_label_dist, mock_clean,
    mock_split, mock_build, mock_tune, mock_evaluate, mock_save_model,
    sample_df,
):
    args = argparse.Namespace(
        records=10, model="logreg", tune=True, use_cached=False, drop_unknown=False
    )
    mock_fetch.return_value = sample_df
    mock_label_dist.return_value = ""
    mock_clean.return_value = sample_df
    mock_split.side_effect = _mock_split
    mock_pipeline = MagicMock()
    mock_build.return_value = mock_pipeline
    mock_tune.return_value = mock_pipeline
    mock_evaluate.return_value = _mock_evaluate()

    main(args)

    mock_tune.assert_called_once()


@patch("src.model.train.save_model")
@patch("src.model.train.evaluate")
@patch("src.model.train.train_pipeline")
@patch("src.model.train.build_pipeline")
@patch("src.model.train.split_data")
@patch("src.model.train.clean_dataframe")
@patch("src.model.train.get_label_distribution")
@patch("src.model.train.save_raw_data")
@patch("src.model.train.fetch_maude_records")
def test_main_drops_unknown_labels(
    mock_fetch, mock_save_raw, mock_label_dist, mock_clean,
    mock_split, mock_build, mock_train, mock_evaluate, mock_save_model,
    sample_df,
):
    args = argparse.Namespace(
        records=10, model="logreg", tune=False, use_cached=False, drop_unknown=True
    )
    df_with_unknown = pd.DataFrame({
        "clean_text": ["text one", "text two", "text three"],
        "severity_label": ["D", "M", "UNKNOWN"],
    })
    mock_fetch.return_value = df_with_unknown
    mock_label_dist.return_value = ""
    mock_clean.return_value = df_with_unknown
    mock_split.side_effect = _mock_split
    mock_pipeline = MagicMock()
    mock_build.return_value = mock_pipeline
    mock_train.return_value = mock_pipeline
    mock_evaluate.return_value = _mock_evaluate()

    main(args)

    # split_data should receive a df with UNKNOWN rows removed
    call_df = mock_split.call_args[0][0]
    assert "UNKNOWN" not in call_df["severity_label"].values
    assert len(call_df) == 2


@patch("src.model.train.save_model")
@patch("src.model.train.evaluate")
@patch("src.model.train.train_pipeline")
@patch("src.model.train.build_pipeline")
@patch("src.model.train.split_data")
@patch("src.model.train.clean_dataframe")
@patch("src.model.train.get_label_distribution")
@patch("src.model.train.pd.read_csv")
@patch("src.model.train.os.path.exists")
def test_main_uses_cached_data_when_available(
    mock_exists, mock_read_csv, mock_label_dist, mock_clean,
    mock_split, mock_build, mock_train, mock_evaluate, mock_save_model,
    sample_df,
):
    args = argparse.Namespace(
        records=10, model="logreg", tune=False, use_cached=True, drop_unknown=False
    )
    mock_exists.return_value = True
    mock_read_csv.return_value = sample_df
    mock_label_dist.return_value = ""
    mock_clean.return_value = sample_df
    mock_split.side_effect = _mock_split
    mock_pipeline = MagicMock()
    mock_build.return_value = mock_pipeline
    mock_train.return_value = mock_pipeline
    mock_evaluate.return_value = _mock_evaluate()

    main(args)

    mock_read_csv.assert_called_once()
