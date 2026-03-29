"""
Shared pytest fixtures and configuration.

Key isolation: redirects CHAMPION_METRICS_PATH and MODEL_PATH to a
temporary directory for every test, so tests never read from or write
to the real models/ folder. This prevents cross-test contamination and
makes tests deterministic regardless of local training state.
"""

import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def isolate_model_paths(tmp_path):
    """
    Redirect all file-system paths used by train.py to a temp directory.

    This fixture runs automatically for every test in the suite.
    Without it, tests that call main() would read the real
    models/champion_metrics.json from previous runs, causing flaky tests.
    """
    champion_path = str(tmp_path / "champion_metrics.json")
    model_path    = str(tmp_path / "maude_classifier.joblib")

    with patch("src.model.train.CHAMPION_METRICS_PATH", champion_path), \
         patch("src.model.train.MODEL_PATH",            model_path):
        yield
