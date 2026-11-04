Data folder — contents are git-ignored.

Run `python src/load_510k.py` from the repo root to fill it:

- `raw/` — the openFDA zip partitions + `export_meta.json` (export date, record count)
- `processed/510k.parquet` — the cleaned, flattened table (one row per K-number)
