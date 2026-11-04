# FDA 510(k) Network — Social Graphs project (DTU)

Exploring the FDA 510(k) premarket notification database as a network.
A 510(k) clears a medical device by showing it is "substantially equivalent"
to an already-cleared **predicate** device, so the clearances form a
family tree (a directed graph) of devices going back decades.

## Setup

```bash
git clone <repo-url>
cd fda-510k-network
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Load the data

```bash
python src/load_510k.py          # first run downloads from openFDA (~a few minutes)
python src/load_510k.py --csv    # also write a CSV
python src/load_510k.py --force  # re-download (openFDA updates weekly)
```

Or from a notebook in `notebooks/`:

```python
import sys; sys.path.append("..")
from src.load_510k import load_510k
df = load_510k()        # cached after the first run
```

The loader reads the file list from `https://api.fda.gov/download.json`,
downloads the 510(k) zip partitions to `data/raw/`, flattens the nested
`openfda` fields, parses dates, adds `review_days` and `decision_year`,
de-duplicates on `k_number`, and caches `data/processed/510k.parquet`.
`data/` is git-ignored, so everyone downloads their own copy.

### Main columns

| column | meaning |
|---|---|
| `k_number` | 510(k) ID, e.g. `K191234` (node ID in the graph) |
| `device_name`, `applicant` | device and company |
| `decision_date`, `date_received`, `review_days` | timeline |
| `decision_code` | e.g. `SESE` = substantially equivalent |
| `product_code` | FDA 3-letter device type (groups similar devices) |
| `advisory_committee_description` | medical specialty panel (Cardiovascular, Radiology, …) |
| `clearance_type` | Traditional / Special / Abbreviated |
| `openfda_device_class` | risk class 1 / 2 / 3 |

## Important: predicates are not in this dataset

The openFDA 510(k) table has no predicate column. Predicate K-numbers
appear in each device's 510(k) summary PDF
(`https://www.accessdata.fda.gov/cdrh_docs/pdfYY/KYYNNNN.pdf`), so building
the family-tree edges needs a second step that downloads those PDFs and
extracts `K\d{6}` references. Without that, edges can be approximated
(e.g. same `product_code`, same `applicant`).

## Structure

```
src/load_510k.py   data loading
notebooks/         exploration
data/              downloaded data (not committed)
```
