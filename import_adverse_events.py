"""
Download FDA adverse event reports (MAUDE) for PMA devices, 2020-2025.

Source: openFDA bulk files (https://open.fda.gov/data/downloads/). The full
export is too large to keep (about 10.6 GB zipped for these years), so each
file is downloaded, filtered to PMA devices, saved as a small part, and deleted.

The script can be stopped and restarted: finished parts are skipped.

Usage:
    python import_adverse_events.py              # all files
    python import_adverse_events.py --limit 1    # smallest file only (test)
"""

import argparse
import csv
import gzip
import io
import json
import re
import shutil
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

INDEX_URL = "https://api.fda.gov/download.json"
USER_AGENT = {"User-Agent": "Mozilla/5.0 (DTU course project)"}
FIRST_YEAR, LAST_YEAR = 2020, 2025

PARTS_DIR = Path("data/raw/maude_parts")
MANIFEST_PATH = PARTS_DIR / "manifest.csv"
DEVICES_PATH = Path("data/processed/pma_devices.csv")

MIN_FREE_GB = 3
MAX_ATTEMPTS = 6  # waits add up to 15 minutes, enough to ride out a network drop
# Accepts P030005, N12159, and forms with a supplement: P030005/S138, P100018.S011, P150011S013
PMA_NUMBER = re.compile(r"([PN]\d{4,6})(?:[\s/.\-]+(S?\d*)|(S\d+))?$")

COLUMNS = [
    "mdr_report_key", "report_number", "date_received", "date_of_event",
    "event_type", "pma_number", "supplement_number", "link_source", "product_code",
    "brand_name", "generic_name", "n_devices", "report_source",
    "manufacturer_report_type", "patient_age", "patient_sex",
    "product_problems", "patient_problems", "patient_outcomes",
    "event_description", "manufacturer_narrative",
]


def unique_join(values):
    """Join values in order, dropping blanks and repeats."""
    return "; ".join(dict.fromkeys(v.strip() for v in values if v and v.strip()))


def iter_results(stream, chunk_chars=1 << 24):
    """Yield reports one at a time, so a large JSON file never sits in memory."""
    decoder = json.JSONDecoder()
    buffer = ""
    while '"results"' not in buffer:
        chunk = stream.read(chunk_chars)
        if not chunk:
            return
        buffer += chunk
    position = buffer.index("[", buffer.index('"results"')) + 1

    while True:
        while position < len(buffer) and buffer[position] in " \r\n\t,":
            position += 1
        if buffer[position:position + 1] == "]":
            return
        try:
            report, position = decoder.raw_decode(buffer, position)
        except json.JSONDecodeError:
            chunk = stream.read(chunk_chars)
            if not chunk:
                raise
            buffer = buffer[position:] + chunk
            position = 0
            continue
        yield report


def to_row(report, pma_product_codes):
    """Return one output row, or None if the report is not about a PMA device."""
    date_received = report.get("date_received") or ""
    if not FIRST_YEAR <= int(date_received[:4] or 0) <= LAST_YEAR:
        return None

    devices = report.get("device") or []
    codes = [d.get("device_report_product_code") or "" for d in devices]
    reported_number = (report.get("pma_pmn_number") or "").strip().upper()
    match = PMA_NUMBER.match(reported_number)

    if match:
        link_source = "pma_number"
        pma_number = match.group(1)
        supplement_number = match.group(2) or match.group(3) or ""
    elif not reported_number and any(code in pma_product_codes for code in codes):
        link_source = "product_code"
        pma_number = supplement_number = ""
    else:
        return None

    patients = report.get("patient") or []
    texts = report.get("mdr_text") or []
    first_device = devices[0] if devices else {}
    first_patient = patients[0] if patients else {}

    return {
        "mdr_report_key": report.get("mdr_report_key"),
        "report_number": report.get("report_number"),
        "date_received": date_received,
        "date_of_event": report.get("date_of_event"),
        "event_type": report.get("event_type"),
        "pma_number": pma_number,
        "supplement_number": supplement_number,
        "link_source": link_source,
        "product_code": unique_join(codes),
        "brand_name": first_device.get("brand_name"),
        "generic_name": first_device.get("generic_name"),
        "n_devices": len(devices),
        "report_source": report.get("report_source_code"),
        "manufacturer_report_type": report.get("mfr_report_type"),
        "patient_age": first_patient.get("patient_age"),
        "patient_sex": first_patient.get("patient_sex"),
        "product_problems": unique_join(report.get("product_problems") or []),
        "patient_problems": unique_join(p for patient in patients for p in patient.get("patient_problems") or []),
        "patient_outcomes": unique_join(o for patient in patients for o in patient.get("sequence_number_outcome") or []),
        "event_description": " ".join(
            t.get("text") or "" for t in texts if t.get("text_type_code") == "Description of Event or Problem"
        ),
        "manufacturer_narrative": " ".join(
            t.get("text") or "" for t in texts if t.get("text_type_code") != "Description of Event or Problem"
        ),
    }


def part_name(partition):
    """Turn .../device/event/2020q3/device-event-0002-of-0005.json.zip into a file name."""
    url_parts = partition["file"].split("/")
    number = re.search(r"(\d{4})-of", url_parts[-1]).group(1)
    return f"{url_parts[-2]}-{number}.csv.gz"


def process_partition(partition, pma_product_codes):
    """Download one bulk file, write its PMA reports to a part, and delete the zip."""
    part_path = PARTS_DIR / part_name(partition)
    zip_path = part_path.with_suffix(".zip.tmp")
    tmp_path = part_path.with_suffix(".tmp")

    request = urllib.request.Request(partition["file"], headers=USER_AGENT)
    with urllib.request.urlopen(request, timeout=300) as response, open(zip_path, "wb") as f:
        shutil.copyfileobj(response, f, 1 << 20)

    n_read = n_kept = 0
    with zipfile.ZipFile(zip_path) as zf, gzip.open(tmp_path, "wt", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, COLUMNS)
        writer.writeheader()
        with zf.open(zf.namelist()[0]) as raw:
            for report in iter_results(io.TextIOWrapper(raw, encoding="utf-8")):
                n_read += 1
                row = to_row(report, pma_product_codes)
                if row:
                    writer.writerow(row)
                    n_kept += 1

    if n_read != int(partition["records"]):
        zip_path.unlink()
        tmp_path.unlink()
        raise ValueError(f"{part_path.name}: read {n_read:,}, FDA lists {partition['records']}")

    tmp_path.rename(part_path)
    zip_path.unlink()
    return n_read, n_kept


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="only process this many files, smallest first")
    args = parser.parse_args()

    PARTS_DIR.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(INDEX_URL, headers=USER_AGENT)
    with urllib.request.urlopen(request, timeout=60) as response:
        meta = json.load(response)["results"]["device"]["event"]

    partitions = [
        p for p in meta["partitions"]
        if p["display_name"][:4].isdigit() and FIRST_YEAR <= int(p["display_name"][:4]) <= LAST_YEAR
    ]
    partitions.sort(key=lambda p: float(p["size_mb"].replace(",", "")))
    if args.limit:
        partitions = partitions[:args.limit]

    pma_product_codes = set(pd.read_csv(DEVICES_PATH, usecols=["product_code"], dtype=str)["product_code"].dropna())
    print(f"openFDA export {meta['export_date']}: {len(partitions)} files", flush=True)

    if not MANIFEST_PATH.exists():
        MANIFEST_PATH.write_text("part,fda_export_date,processed_at_utc,records_read,records_kept\n")

    for i, partition in enumerate(partitions, 1):
        part_path = PARTS_DIR / part_name(partition)
        if part_path.exists():
            continue

        free_gb = shutil.disk_usage(PARTS_DIR).free / 1e9
        if free_gb < MIN_FREE_GB:
            raise SystemExit(f"Stopped: only {free_gb:.1f} GB free. Restart once there is more space.")

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                n_read, n_kept = process_partition(partition, pma_product_codes)
                break
            except Exception as error:
                print(f"  {part_path.name}: attempt {attempt} failed: {error}", flush=True)
                if attempt == MAX_ATTEMPTS:
                    raise
                time.sleep(60 * attempt)

        processed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with open(MANIFEST_PATH, "a") as f:
            f.write(f"{part_path.name},{meta['export_date']},{processed_at},{n_read},{n_kept}\n")
        size_mb = part_path.stat().st_size / 1e6
        print(f"[{i}/{len(partitions)}] {part_path.name}: kept {n_kept:,} of {n_read:,} ({size_mb:.0f} MB)", flush=True)
        time.sleep(1)


if __name__ == "__main__":
    main()
