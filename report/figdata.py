"""Shared setup for the report figures.

Rebuilds the product-code network exactly as text_similarity_network.ipynb does
(cut 0.65, Louvain with 20 seeds, best modularity) from the cached sample and
embeddings, and sets the plotting style used by every figure.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

SEED, THRESHOLD = 42, 0.65
COMMITTEES = {
    "AN": "Anesthesiology", "CH": "Clinical chemistry", "CV": "Cardiovascular", "DE": "Dental",
    "EN": "Ear, nose, throat", "GU": "Gastro and urology", "HE": "Hematology", "HO": "General hospital",
    "IM": "Immunology", "MG": "Medical genetics", "MI": "Microbiology", "NE": "Neurology",
    "OB": "Obstetrics and gynecology", "OP": "Ophthalmic", "OR": "Orthopedic", "PA": "Pathology",
    "PM": "Physical medicine", "RA": "Radiology", "SU": "General and plastic surgery", "TX": "Clinical toxicology",
}
# Same palette as Fig 1. Field is always slot 1 (blue), technology slot 2 (orange).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
FIELD, TECH = PALETTE[0], PALETTE[1]
OTHER = "#c3c2b7"
INK, MUTED = "#2b2b2b", "#6b6b66"


def use_style():
    """Plain matplotlib text in DejaVu Sans at 7 pt, the same as the current Fig 1."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })


def tex(s):
    """Category names as shown in figures: underscores become spaces."""
    return str(s).replace("_", " ")


def load(project):
    project = Path(project)
    P = project / "data/processed"
    sample = pd.read_csv(P / "embedding_sample.csv.gz", dtype=str).reset_index(drop=True)
    emb = np.load(P / "embedding_sample_minilm.npy")
    assert len(emb) == len(sample)
    devices = pd.read_csv(P / "pma_devices_2020_2025.csv", dtype=str)
    records = pd.read_csv(P / "pma_records.csv", dtype=str, usecols=["product_code", "advisory_committee"]).dropna()
    tech = pd.read_csv(project / "labels/technology_labels.csv", dtype=str).set_index("product_code")

    field_rec = records.groupby("product_code")["advisory_committee"].agg(lambda s: s.mode()[0])
    sample["device_field"] = sample["pma_number"].map(devices.set_index("pma_number")["advisory_committee"])
    field_dev = sample.groupby("product_code")["device_field"].agg(lambda s: s.mode()[0])
    codes = sample.groupby("product_code").agg(n_reports=("mdr_report_key", "size"))
    codes["field"] = field_rec.reindex(codes.index).fillna(field_dev).map(COMMITTEES)
    codes["technology"] = tech["technology"].reindex(codes.index).str.replace("_", " ")
    codes["name"] = tech["fda_name"].reindex(codes.index)

    order = codes.index.to_numpy()
    frame = pd.DataFrame(emb).groupby(sample["product_code"].values).mean().loc[order].to_numpy()
    vec = frame / np.linalg.norm(frame, axis=1, keepdims=True)
    sim = vec @ vec.T
    a, b = np.triu_indices(len(order), 1)
    keep = sim[a, b] >= THRESHOLD
    G = nx.Graph()
    G.add_weighted_edges_from(zip(order[a[keep]], order[b[keep]], sim[a, b][keep]))
    assert G.number_of_edges() == 1709, "network differs from the notebook"
    return sample, codes, sim, G


def communities(G):
    runs = [nx.community.louvain_communities(G, weight="weight", seed=s) for s in range(20)]
    scores = [nx.community.modularity(G, r, weight="weight") for r in runs]
    best = runs[int(np.argmax(scores))]
    return pd.Series({n: c for c, members in enumerate(best) for n in members}), max(scores)


def giant_layout(G):
    giant = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    pos = nx.spring_layout(giant, weight="weight", seed=SEED, iterations=300, k=0.25)
    return giant, pos
