"""
Download and load the FDA 510(k) premarket notification dataset from openFDA.

Source: openFDA bulk download (https://open.fda.gov/data/downloads/)
The list of files is read from https://api.fda.gov/download.json, so the
code keeps working when FDA re-partitions or updates the export (weekly).

Usage (command line):
    python src/load_510k.py            # download (if needed) + save parquet/csv
    python src/load_510k.py --force    # re-download even if cached

Usage (Python / notebook):
    from src.load_510k import load_510k
    df = load_510k()
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import requests

DOWNLOAD_INDEX_URL = "https://api.fda.gov/download.json"

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PARQUET_PATH = PROCESSED_DIR / "510k.parquet"
CSV_PATH = PROCESSED_DIR / "510k.csv"

# Columns we keep (in this order). Anything else in the raw JSON is kept at the end.
CORE_COLUMNS = [
    "k_number",
    "device_name",
    "applicant",
    "decision_date",
    "date_received",
    "decision_code",
    "decision_description",
    "product_code",
    "clearance_type",
    "advisory_committee",
    "advisory_committee_description",
    "review_advisory_committee",
    "statement_or_summary",
    "third_party_flag",
    "expedited_review_flag",
    "country_code",
    "state",
    "city",
    "contact",
    # flattened from the nested "openfda" block
    "openfda_device_class",
    "openfda_regulation_number",
    "openfda_medical_specialty_description",
    "openfda_device_name",
]


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def get_partition_urls(timeout: int = 30) -> tuple[list[str], dict]:
    """Return the list of 510(k) zip URLs and the index metadata."""
    resp = requests.get(DOWNLOAD_INDEX_URL, timeout=timeout)
    resp.raise_for_status()
    meta = resp.json()["results"]["device"]["510k"]
    urls = [p["file"] for p in meta["partitions"]]
    return urls, meta


def download_raw(force: bool = False) -> list[Path]:
    """Download every 510(k) partition zip into data/raw/ (skips files already there)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    urls, meta = get_partition_urls()
    print(f"openFDA 510(k) export {meta.get('export_date')}: "
          f"{meta.get('total_records'):,} records in {len(urls)} file(s)")

    paths = []
    for url in urls:
        path = RAW_DIR / url.rsplit("/", 1)[-1]
        if path.exists() and not force:
            print(f"  cached   {path.name}")
        else:
            print(f"  download {path.name} ...")
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
        paths.append(path)

    (RAW_DIR / "export_meta.json").write_text(json.dumps(meta, indent=2))
    return paths


# --------------------------------------------------------------------------- #
# Parse
# --------------------------------------------------------------------------- #
def _join(value):
    """openfda fields are lists; turn them into a '; '-separated string."""
    if isinstance(value, list):
        return "; ".join(str(v) for v in value) if value else None
    return value


def _flatten(record: dict) -> dict:
    rec = {k: v for k, v in record.items() if k != "openfda"}
    for key, value in (record.get("openfda") or {}).items():
        rec[f"openfda_{key}"] = _join(value)
    return rec


def read_zip(path: Path) -> list[dict]:
    """Read one openFDA zip (it contains a single JSON file with a 'results' list)."""
    with zipfile.ZipFile(path) as zf:
        records = []
        for name in zf.namelist():
            if name.endswith(".json"):
                with zf.open(name) as f:
                    payload = json.load(io.TextIOWrapper(f, encoding="utf-8"))
                records.extend(payload["results"])
    return records


def build_dataframe(zip_paths: list[Path]) -> pd.DataFrame:
    records = []
    for path in zip_paths:
        records.extend(_flatten(r) for r in read_zip(path))
    df = pd.DataFrame.from_records(records)

    # Types
    for col in ("decision_date", "date_received"):
        if col in df:
            df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce") \
                .fillna(pd.to_datetime(df[col], format="%Y%m%d", errors="coerce"))
    if "k_number" in df:
        df["k_number"] = df["k_number"].str.strip().str.upper()
        df = df.drop_duplicates("k_number")
    if {"decision_date", "date_received"} <= set(df.columns):
        df["review_days"] = (df["decision_date"] - df["date_received"]).dt.days
    if "decision_date" in df:
        df["decision_year"] = df["decision_date"].dt.year.astype("Int64")

    # Column order: core first, then whatever else came in
    first = [c for c in CORE_COLUMNS + ["review_days", "decision_year"] if c in df]
    rest = [c for c in df.columns if c not in first]
    return df[first + rest].sort_values("decision_date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def load_510k(force: bool = False, save_csv: bool = False) -> pd.DataFrame:
    """
    Return the full 510(k) table as a DataFrame.

    First call downloads from openFDA and caches data/processed/510k.parquet;
    later calls just read the parquet (fast). Use force=True to refresh.
    """
    if PARQUET_PATH.exists() and not force:
        return pd.read_parquet(PARQUET_PATH)

    df = build_dataframe(download_raw(force=force))
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_PATH, index=False)
    if save_csv:
        df.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(df):,} rows x {df.shape[1]} cols -> {PARQUET_PATH.relative_to(ROOT)}")
    return df


def summarize(df: pd.DataFrame) -> None:
    print(f"\nRows: {len(df):,}   Columns: {df.shape[1]}")
    if "decision_date" in df:
        print(f"Decision dates: {df['decision_date'].min():%Y-%m-%d} -> "
              f"{df['decision_date'].max():%Y-%m-%d}")
    for col in ("decision_code", "clearance_type", "advisory_committee_description",
                "openfda_device_class"):
        if col in df:
            print(f"\nTop values of {col}:")
            print(df[col].value_counts(dropna=False).head(8).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download and rebuild")
    ap.add_argument("--csv", action="store_true", help="also write data/processed/510k.csv")
    args = ap.parse_args()
    summarize(load_510k(force=args.force, save_csv=args.csv))
