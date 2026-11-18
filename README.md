# Do medical devices fail by technology or by medical field?

Course project for Social Graphs and Interactions at DTU. We link FDA device types (product codes) by how similar their adverse event reports read, and test whether the network groups them by the medical field they are used in or by the technology they are built on.

## Data

Two public FDA sources, limited to 2020-2025:

| source | what it is | version used |
|---|---|---|
| [PMA database](https://www.fda.gov/medical-devices/device-approvals-and-clearances/pma-approvals) (`pma.zip`) | every approval and supplement for class III (high-risk) devices | FDA file of 2 Nov 2026, downloaded 4 Nov 2026 |
| [MAUDE via openFDA](https://open.fda.gov/data/downloads/) | adverse event reports for medical devices | openFDA export of 3 Nov 2026 |

The data is not in the repository (`data/` is in `.gitignore`). The steps below download and rebuild it. The download script stops if less than 3 GB is free. The finished data takes about 2 GB, or about 7 GB if you also unzip the two large tables (see below).

## Files

| file | what it does |
|---|---|
| `data_import.ipynb` | Part 1 downloads and cleans the PMA database. Part 2 cleans the adverse event reports and limits everything to 2020-2025. Writes the tables in `data/processed/`. |
| `import_adverse_events.py` | Downloads the openFDA bulk files one at a time and keeps reports linked to PMA devices, in `data/raw/maude_parts/`. Can be stopped and restarted. |
| `data_profiling.ipynb` | Describes the dataset: reports per device, patients, medical fields, severity. Saves nothing. |
| `text_similarity_network.ipynb` | The main analysis: builds the network, tests field against technology, finds communities and runs the robustness checks. |
| `labels/technology_labels.csv` | The technology label for each of the 201 product codes in the network. See the notebook's Limits section for how the labels were made. |
| `report/make_fig1.py` | Draws Fig. 1 of the report from the cached sample and embeddings. |

## How to run

Use Python 3.11. The results were produced with:

```
pandas==3.0.5
numpy==1.26.4
matplotlib==3.11.1
networkx==3.6.1
scikit-learn==1.9.1
sentence-transformers==3.4.1
torch==2.2.2
```

```bash
python -m venv .venv && source .venv/bin/activate
pip install pandas==3.0.5 numpy==1.26.4 matplotlib==3.11.1 networkx==3.6.1 \
    scikit-learn==1.9.1 sentence-transformers==3.4.1 torch==2.2.2 jupyter
```

Run the steps in this order, from the repository root:

1. **`data_import.ipynb`, Part 1 only** (sections 1-6). This writes `data/processed/pma_devices.csv`, which the next step needs.
2. **`python import_adverse_events.py`**. This takes about an hour. Try `--limit 1` first to process only the smallest file.
3. **`data_import.ipynb`, Part 2** (sections 7-11).
4. **`data_profiling.ipynb`**.
5. **`text_similarity_network.ipynb`**. The first run takes about 20 minutes on a laptop CPU, because it embeds 48,450 report texts. The sample and embeddings are then cached in `data/processed/`, so later runs take a few minutes.
6. **`python report/make_fig1.py . report/figures`** to redraw Fig. 1.

`data_import.ipynb` has `REFRESH = False`, so it reuses `data/raw/pma.zip` once downloaded. FDA replaces that file about once a month, so a fresh download can change the numbers.

To make the notebooks load faster, you can unzip `maude_reports.csv.gz` and `maude_text.csv.gz` in `data/processed/` (`gunzip -k`). The notebooks use the uncompressed copy when it exists. The contents are the same.

## Main result

By the decision rules set before looking at the results, the hypothesis that devices group by technology more than by field is **partly supported**. Two device types that share a technology are about twice as similar, and about three times as likely to be linked, as two that share only a medical field. But most technologies are used in only one field, so the network as a whole still holds more same-field links. Each label explains part of the network that the other does not. A second network built from FDA problem terms, without embeddings, gives the same pattern. See section 9 of `text_similarity_network.ipynb`.

## Caveats

- A report records that a harm was reported alongside a device, not that the device caused it. MAUDE has no usage numbers, so report counts are not risk.
- The technology labels were drafted from device names, not by a domain expert, and should be checked by a second labeller.
- Each notebook ends with its own list of caveats.
