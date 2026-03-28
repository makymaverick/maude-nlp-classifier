"""
openFDA MAUDE API Client
Ingests Medical Device Adverse Event (MAUDE) reports from the openFDA API.

Docs: https://open.fda.gov/apis/device/event/
"""

import os
import time
import logging
from typing import Optional

import requests
import pandas as pd

logger = logging.getLogger(__name__)

BASE_URL = "https://api.fda.gov/device/event.json"

# Severity label mapping based on MAUDE event_type field
# D=Death, I=Injury, M=Malfunction, O=Other
EVENT_TYPE_SEVERITY = {
    "Death": "D",
    "D": "D",
    "Serious Injury": "I",
    "Injury": "I",
    "I": "I",
    "Malfunction": "M",
    "M": "M",
    "Other": "O",
    "O": "O",
    "No Answer Provided": "UNKNOWN",
    "*": "UNKNOWN",
}


def _build_params(
    query: str,
    limit: int,
    skip: int,
    api_key: Optional[str] = None,
) -> dict:
    params = {
        "search": query,
        "limit": min(limit, 1000),  # openFDA max per page is 1000
        "skip": skip,
    }
    if api_key:
        params["api_key"] = api_key
    return params


def _fetch_natural(
    total: int,
    api_key: Optional[str],
    delay: float,
    page_size: int = 100,
) -> list[dict]:
    """Fetch records from the natural MAUDE distribution (no per-type constraint)."""
    query = "_exists_:mdr_text"
    records = []
    skip = 0

    while len(records) < total:
        remaining = total - len(records)
        params = _build_params(query, min(page_size, remaining), skip, api_key)

        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                logger.warning("No more results (404). Stopping.")
                break
            elif response.status_code == 429:
                logger.warning("Rate limited. Sleeping 60s...")
                time.sleep(60)
                continue
            else:
                logger.error(f"HTTP error: {e}")
                raise

        results = response.json().get("results", [])
        if not results:
            logger.info(f"Exhausted available records at {len(records)}.")
            break

        for r in results:
            record = _parse_record(r)
            if record:
                records.append(record)

        skip += len(results)
        logger.info(f"  Natural fetch: {len(records)} / {total}")
        time.sleep(delay)

    return records


def fetch_maude_records(
    total_records: int = 5000,
    api_key: Optional[str] = None,
    delay: float = 0.5,
) -> pd.DataFrame:
    """
    Fetch MAUDE adverse event records from openFDA using the natural distribution.

    Records are fetched without event_type filtering so the dataset reflects the
    real MAUDE distribution. Class imbalance is handled at training time via
    ``class_weight='balanced'`` in the classifier.

    Args:
        total_records: Total number of records to retrieve.
        api_key: Optional openFDA API key (higher rate limits with key).
        delay: Seconds to wait between paginated requests.

    Returns:
        DataFrame with columns: report_number, event_type, severity_label,
                                 device_name, narrative_text, date_received
    """
    if api_key is None:
        api_key = os.getenv("OPENFDA_API_KEY")

    logger.info(f"Fetching {total_records} records (natural distribution).")

    all_records = _fetch_natural(total_records, api_key, delay)
    logger.info(f"Natural fetch complete: {len(all_records)} records.")

    df = pd.DataFrame(all_records)
    logger.info(f"Done. Total records fetched: {len(df)}")
    return df


def _parse_record(r: dict) -> Optional[dict]:
    """Extract relevant fields from a single MAUDE result record."""
    try:
        # Narrative text from mdr_text array
        mdr_texts = r.get("mdr_text", [])
        narrative = " ".join(
            item.get("text", "") for item in mdr_texts if isinstance(item, dict)
        ).strip()
        if not narrative:
            return None

        # Device name
        devices = r.get("device", [])
        device_name = ""
        if devices and isinstance(devices, list):
            d = devices[0]
            device_name = d.get("brand_name", "") or d.get("generic_name", "")

        # Event type → severity label
        event_types = r.get("event_type", [])
        event_type_str = event_types[0] if event_types else "Other"
        severity_label = EVENT_TYPE_SEVERITY.get(event_type_str, "UNKNOWN")

        return {
            "report_number": r.get("report_number", ""),
            "date_received": r.get("date_received", ""),
            "event_type": event_type_str,
            "severity_label": severity_label,
            "device_name": device_name,
            "narrative_text": narrative,
        }
    except Exception as e:
        logger.debug(f"Skipping malformed record: {e}")
        return None


def save_raw_data(df: pd.DataFrame, path: str = "data/raw/maude_raw.csv") -> None:
    """Persist raw fetched data to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved {len(df)} records to {path}")


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    df = fetch_maude_records(total_records=5000)
    print(df.head())
    print(df["severity_label"].value_counts())
    save_raw_data(df)
