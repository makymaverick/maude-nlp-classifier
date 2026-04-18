"""
openFDA MAUDE API Client
Ingests Medical Device Adverse Event (MAUDE) reports from the openFDA API.

Docs: https://open.fda.gov/apis/device/event/
"""

import json
import os
import time
import logging
from datetime import datetime
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

    page_size = 1000 if api_key else 100
    all_records = _fetch_natural(total_records, api_key, delay, page_size=page_size)
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


# ── Bulk fetch with checkpoint/resume ────────────────────────────────────────

# openFDA rejects skip >= 25,000 with a 400. Date-range windowing resets
# skip=0 for each year so we can collect far more than 25k records total.
OPENFDA_MAX_SKIP = 25_000


def _date_windows(start_year: int = 2015, end_year: int | None = None) -> list[tuple[str, str]]:
    """Return yearly (start, end) pairs in YYYYMMDD format."""
    if end_year is None:
        end_year = datetime.now().year
    return [(f"{y}0101", f"{y}1231") for y in range(start_year, end_year + 1)]


def _load_checkpoint(path: str) -> int:
    """Return the last saved offset, or 0 if no checkpoint exists."""
    if not os.path.exists(path):
        return 0
    try:
        with open(path) as f:
            return json.load(f).get("offset", 0)
    except (json.JSONDecodeError, OSError):
        return 0


def _save_checkpoint(path: str, offset: int) -> None:
    """Write the current pagination offset to a checkpoint file."""
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"offset": offset}, f)


def _load_bulk_checkpoint(path: str) -> tuple[int, int]:
    """Return (window_idx, offset) from a bulk checkpoint, or (0, 0)."""
    if not os.path.exists(path):
        return 0, 0
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("window_idx", 0), data.get("offset", 0)
    except (json.JSONDecodeError, OSError):
        return 0, 0


def _save_bulk_checkpoint(path: str, window_idx: int, offset: int) -> None:
    """Write current window index and offset to a bulk checkpoint file."""
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"window_idx": window_idx, "offset": offset}, f)


def fetch_maude_bulk(
    total_records: int = 50_000,
    api_key: Optional[str] = None,
    delay: float = 0.5,
    output_path: str = "data/raw/maude_raw.csv",
    checkpoint_every: int = 10,
    start_year: int = 2015,
    end_year: int | None = None,
) -> pd.DataFrame:
    """
    Fetch a large batch of MAUDE records with checkpoint/resume support.

    The openFDA API caps ``skip`` at 25,000, so a single linear scan can
    return at most 25k records.  This function overcomes that limit by
    partitioning the query into yearly date-range windows and resetting
    ``skip=0`` for each window.

    Progress is checkpointed every ``checkpoint_every`` batches so that
    long runs can be resumed after an interruption.

    Args:
        total_records:    Target number of records (default 50,000).
        api_key:          openFDA API key. Falls back to OPENFDA_API_KEY env var.
        delay:            Seconds between paginated requests.
        output_path:      Path for both intermediate and final CSV output.
        checkpoint_every: Flush to disk every N batches (default 10).
        start_year:       First year of the date-range window (default 2015).
        end_year:         Last year of the date-range window (default: current year).

    Returns:
        DataFrame with columns: report_number, date_received, event_type,
        severity_label, device_name, narrative_text.
    """
    if api_key is None:
        api_key = os.getenv("OPENFDA_API_KEY")

    checkpoint_path = output_path.replace(".csv", "_checkpoint.json")
    page_size = 1000  # openFDA max per call

    # Resume from checkpoint
    start_win, start_skip = _load_bulk_checkpoint(checkpoint_path)
    records: list[dict] = []
    if (start_win > 0 or start_skip > 0) and os.path.exists(output_path):
        try:
            records = pd.read_csv(output_path, dtype=str).to_dict("records")
            logger.info(
                f"Resuming from checkpoint: year_window={start_win}, "
                f"offset={start_skip:,}, existing={len(records):,} records"
            )
        except Exception:
            start_win, start_skip = 0, 0

    windows = _date_windows(start_year=start_year, end_year=end_year)
    batch_num = 0
    logger.info(
        f"Bulk fetch started — target={total_records:,}, "
        f"years={start_year}–{end_year or datetime.now().year}, "
        f"windows={len(windows)}, output={output_path}"
    )

    for win_idx, (date_from, date_to) in enumerate(windows):
        if win_idx < start_win:
            continue

        skip = start_skip if win_idx == start_win else 0
        query = f"_exists_:mdr_text AND date_received:[{date_from} TO {date_to}]"

        while len(records) < total_records and skip < OPENFDA_MAX_SKIP:
            remaining = total_records - len(records)
            params = _build_params(query, min(page_size, remaining), skip, api_key)

            try:
                response = requests.get(BASE_URL, params=params, timeout=30)
                response.raise_for_status()
            except requests.exceptions.ConnectionError:
                logger.warning("Connection dropped. Sleeping 15s and retrying...")
                time.sleep(15)
                continue
            except requests.exceptions.HTTPError as e:
                if response.status_code == 404:
                    logger.info(f"No more records in window {date_from[:4]}.")
                    break
                elif response.status_code == 429:
                    logger.warning("Rate limited. Sleeping 60s...")
                    time.sleep(60)
                    continue
                else:
                    raise

            results = response.json().get("results", [])
            if not results:
                logger.info(f"Empty batch at {date_from[:4]} offset {skip}.")
                break

            batch = [r for r in (_parse_record(hit) for hit in results) if r]
            records.extend(batch)
            skip += len(results)
            batch_num += 1

            logger.info(
                f"Batch {batch_num:>3} | {date_from[:4]} | offset {skip:>6,} | "
                f"collected {len(records):>6,}/{total_records:,} | "
                f"kept {len(batch):>4}"
            )

            if batch_num % checkpoint_every == 0:
                _save_bulk_checkpoint(checkpoint_path, win_idx, skip)
                out_dir = os.path.dirname(output_path)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                pd.DataFrame(records).to_csv(output_path, index=False)
                logger.info(f"Checkpoint saved. {len(records):,} records on disk.")

            time.sleep(delay)

        if len(records) >= total_records:
            break

    df = pd.DataFrame(records)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Bulk fetch complete. {len(df):,} records saved to {output_path}")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return df


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    df = fetch_maude_records(total_records=5000)
    print(df.head())
    print(df["severity_label"].value_counts())
    save_raw_data(df)
