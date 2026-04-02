"""
Text preprocessing pipeline for MAUDE narrative reports.
Cleans and normalizes raw MDR narrative text for NLP ingestion.
"""

import re
import string
import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

# Common medical abbreviation expansions (extend as needed)
ABBREVIATION_MAP = {
    r"\bpt\b": "patient",
    r"\bpts\b": "patients",
    r"\bmd\b": "physician",
    r"\bdr\b": "doctor",
    r"\bhosp\b": "hospital",
    r"\badm\b": "admitted",
    r"\bdx\b": "diagnosis",
    r"\btx\b": "treatment",
    r"\brx\b": "prescription",
    r"\bs/p\b": "status post",
    r"\bw/\b": "with",
    r"\bh/o\b": "history of",
    r"\bc/o\b": "complaint of",
    r"\bn/v\b": "nausea vomiting",
    r"\bSOB\b": "shortness of breath",
    r"\bUNK\b": "unknown",
}


def expand_abbreviations(text: str) -> str:
    """Replace common medical abbreviations with full forms."""
    for pattern, replacement in ABBREVIATION_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def remove_boilerplate(text: str) -> str:
    """
    Remove common MAUDE boilerplate phrases that carry no signal.
    These often appear at the start/end of narrative sections.
    """
    boilerplate_patterns = [
        r"it was reported that",
        r"the reporter stated",
        r"according to the report",
        r"this is a report",
        r"per the report",
        r"the following information was received",
        r"information has been received",
        r"no further information (is|was) available",
        r"follow.?up (is|will be) requested",
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return text


def clean_text(text: str, lowercase: bool = True, preserve_digits: bool = False) -> str:
    """
    Full cleaning pipeline for a single narrative string.

    Steps:
      1. Strip whitespace
      2. Remove boilerplate
      3. Expand abbreviations
      4. Remove special characters (and optionally digits)
      5. Normalize whitespace
      6. Lowercase (optional)

    Args:
        text:             Raw narrative text string.
        lowercase:        Whether to convert to lowercase.
        preserve_digits:  If True, keep digit tokens (required for BERT — device
                          model numbers and dosages are meaningful to the encoder).
                          If False (default, TF-IDF mode), digits are stripped.

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.strip()
    text = remove_boilerplate(text)
    text = expand_abbreviations(text)

    # Remove non-alphanumeric characters (keep spaces; optionally keep digits)
    if preserve_digits:
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    else:
        text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    if lowercase:
        text = text.lower()

    return text


def clean_dataframe(
    df: pd.DataFrame,
    text_col: str = "narrative_text",
    output_col: str = "clean_text",
    preserve_digits: bool = False,
) -> pd.DataFrame:
    """
    Apply clean_text to an entire DataFrame column.

    Args:
        df:               Input DataFrame with raw narrative text.
        text_col:         Name of the column containing raw text.
        output_col:       Name of the new column to store cleaned text.
        preserve_digits:  Passed through to clean_text; set True for BERT.

    Returns:
        DataFrame with a new cleaned text column, with empty-text rows dropped.
    """
    logger.info(f"Cleaning text in column '{text_col}'...")
    df = df.copy()
    df[output_col] = df[text_col].apply(
        lambda t: clean_text(t, preserve_digits=preserve_digits)
    )

    # Drop rows where cleaning produced empty strings
    before = len(df)
    df = df[df[output_col].str.len() > 10].reset_index(drop=True)
    after = len(df)
    logger.info(f"Dropped {before - after} rows with insufficient text. {after} rows remaining.")
    return df


def get_label_distribution(df: pd.DataFrame, label_col: str = "severity_label") -> pd.Series:
    """Return label counts for inspection."""
    return df[label_col].value_counts()
