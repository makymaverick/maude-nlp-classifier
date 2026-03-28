"""Unit tests for text preprocessing module."""

import pytest
import pandas as pd

from src.preprocessing.text_cleaner import (
    clean_text,
    expand_abbreviations,
    remove_boilerplate,
    clean_dataframe,
    get_label_distribution,
)


def test_clean_text_lowercases():
    assert clean_text("PATIENT INJURY") == "patient injury"


def test_clean_text_removes_numbers():
    result = clean_text("Device model 3000 malfunctioned")
    assert "3000" not in result


def test_clean_text_removes_special_chars():
    result = clean_text("Device: failed! (critical)")
    assert "!" not in result
    assert "(" not in result


def test_clean_text_returns_empty_for_blank():
    assert clean_text("") == ""
    assert clean_text("   ") == ""


def test_expand_abbreviations_patient():
    result = expand_abbreviations("The pt was admitted.")
    assert "patient" in result.lower()


def test_expand_abbreviations_doctor():
    result = expand_abbreviations("The MD reviewed the case.")
    assert "physician" in result.lower()


def test_remove_boilerplate():
    text = "It was reported that the device caused injury."
    result = remove_boilerplate(text)
    assert "it was reported that" not in result.lower()


def test_clean_dataframe_drops_empty_rows():
    df = pd.DataFrame({
        "narrative_text": ["Short text with enough words for classification.", "", "   ", "x"],
        "severity_label": ["DEATH", "MALFUNCTION", "INJURY", "UNKNOWN"],
    })
    df_clean = clean_dataframe(df)
    # Empty and very short rows should be dropped
    assert len(df_clean) < len(df)
    assert all(df_clean["clean_text"].str.len() > 10)


def test_clean_dataframe_adds_clean_text_column():
    df = pd.DataFrame({
        "narrative_text": ["The pump malfunctioned and the patient was injured."],
        "severity_label": ["INJURY"],
    })
    df_clean = clean_dataframe(df)
    assert "clean_text" in df_clean.columns


def test_get_label_distribution_returns_counts():
    df = pd.DataFrame({"severity_label": ["D", "D", "M", "I"]})
    dist = get_label_distribution(df)
    assert dist["D"] == 2
    assert dist["M"] == 1
    assert dist["I"] == 1


def test_get_label_distribution_custom_column():
    df = pd.DataFrame({"category": ["A", "B", "A"]})
    dist = get_label_distribution(df, label_col="category")
    assert dist["A"] == 2
