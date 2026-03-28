"""Unit tests for openFDA MAUDE ingestion module."""

import os
import tempfile
import pytest
import requests
from unittest.mock import patch, MagicMock
import pandas as pd

from src.ingestion.openfda_client import _parse_record, fetch_maude_records, save_raw_data


# ── _parse_record tests ──────────────────────────────────────────────────────

def test_parse_record_returns_none_for_empty_text():
    record = {"mdr_text": [], "event_type": ["Malfunction"]}
    assert _parse_record(record) is None


def test_parse_record_maps_death_severity():
    record = {
        "mdr_text": [{"text": "Patient died after device failure."}],
        "event_type": ["Death"],
        "device": [{"brand_name": "TestDevice"}],
        "report_number": "12345",
        "date_received": "20230101",
    }
    result = _parse_record(record)
    assert result is not None
    assert result["severity_label"] == "D"
    assert result["device_name"] == "TestDevice"


def test_parse_record_maps_malfunction_severity():
    record = {
        "mdr_text": [{"text": "Device stopped working."}],
        "event_type": ["Malfunction"],
        "device": [],
        "report_number": "99999",
        "date_received": "20230601",
    }
    result = _parse_record(record)
    assert result is not None
    assert result["severity_label"] == "M"


def test_parse_record_handles_unknown_event_type():
    record = {
        "mdr_text": [{"text": "Unclear event occurred."}],
        "event_type": ["Other"],
        "device": [],
    }
    result = _parse_record(record)
    assert result["severity_label"] == "O"


# ── fetch_maude_records integration (mocked) ────────────────────────────────

@patch("src.ingestion.openfda_client.requests.get")
def test_fetch_maude_records_returns_dataframe(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {
                "mdr_text": [{"text": "Patient fell and was seriously injured."}],
                "event_type": ["Serious Injury"],
                "device": [{"brand_name": "WheelchairPro"}],
                "report_number": "ABC-001",
                "date_received": "20240101",
            }
        ]
    }
    mock_get.return_value = mock_response

    df = fetch_maude_records(total_records=1)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["severity_label"] == "I"


@patch("src.ingestion.openfda_client.requests.get")
def test_fetch_maude_records_stops_on_empty_results(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"results": []}
    mock_get.return_value = mock_response

    df = fetch_maude_records(total_records=100)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


@patch("src.ingestion.openfda_client.requests.get")
def test_fetch_maude_records_handles_404(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
    mock_get.return_value = mock_response

    df = fetch_maude_records(total_records=10)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


@patch("src.ingestion.openfda_client.time.sleep")
@patch("src.ingestion.openfda_client.requests.get")
def test_fetch_maude_records_retries_on_429(mock_get, mock_sleep):
    rate_limited = MagicMock()
    rate_limited.status_code = 429
    rate_limited.raise_for_status.side_effect = requests.exceptions.HTTPError()

    success = MagicMock()
    success.status_code = 200
    success.raise_for_status = MagicMock()
    success.json.return_value = {
        "results": [
            {
                "mdr_text": [{"text": "Device malfunctioned."}],
                "event_type": ["Malfunction"],
                "device": [],
                "report_number": "001",
                "date_received": "20240101",
            }
        ]
    }
    mock_get.side_effect = [rate_limited, success]

    df = fetch_maude_records(total_records=1)
    mock_sleep.assert_any_call(60)
    assert len(df) == 1


@patch("src.ingestion.openfda_client.requests.get")
def test_fetch_maude_records_raises_on_500(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
    mock_get.return_value = mock_response

    with pytest.raises(requests.exceptions.HTTPError):
        fetch_maude_records(total_records=10)


def test_parse_record_handles_exception_gracefully():
    # Passing None triggers AttributeError inside _parse_record
    result = _parse_record(None)
    assert result is None


def test_save_raw_data_creates_csv():
    df = pd.DataFrame([{"report_number": "1", "severity_label": "D", "narrative_text": "test"}])
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "subdir", "output.csv")
        save_raw_data(df, path)
        assert os.path.exists(path)
        loaded = pd.read_csv(path)
        assert len(loaded) == 1
