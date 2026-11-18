"""Fig 1: the product-code network, coloured by medical field and by technology.

Rebuilds the network exactly as text_similarity_network.ipynb does (cut 0.65,
same seed) from the cached sample and embeddings, then draws one layout twice.

Usage: python make_fig1.py <project_dir> <out_dir>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

project, out = Path(sys.argv[1]), Path(sys.argv[2])
P = project / "data/processed"
SEED, THRESHOLD = 42, 0.65
COMMITTEES = {
    "AN": "Anesthesiology", "CH": "Clinical chemistry", "CV": "Cardiovascular", "DE": "Dental",
    "EN": "Ear, nose, throat", "GU": "Gastro and urology", "HE": "Hematology", "HO": "General hospital",
    "IM": "Immunology", "MG": "Medical genetics", "MI": "Microbiology", "NE": "Neurology",
    "OB": "Obstetrics and gynecology", "OP": "Ophthalmic", "OR": "Orthopedic", "PA": "Pathology",
    "PM": "Physical medicine", "RA": "Radiology", "SU": "General and plastic surgery", "TX": "Clinical toxicology",
}
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
OTHER = "#c3c2b7"

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

order = codes.index.to_numpy()
frame = pd.DataFrame(emb).groupby(sample["product_code"].values).mean().loc[order].to_numpy()
vec = frame / np.linalg.norm(frame, axis=1, keepdims=True)
sim = vec @ vec.T
a, b = np.triu_indices(len(order), 1)
keep = sim[a, b] >= THRESHOLD
G = nx.Graph()
G.add_weighted_edges_from(zip(order[a[keep]], order[b[keep]], sim[a, b][keep]))
print(f"nodes {G.number_of_nodes()}, edges {G.number_of_edges()}, share of pairs {keep.mean():.3f}")
assert G.number_of_edges() == 1709, "network differs from the notebook"

giant = G.subgraph(max(nx.connected_components(G), key=len)).copy()
print(f"largest component: {giant.number_of_nodes()} codes, {giant.number_of_edges()} edges")
G = giant
pos = nx.spring_layout(G, weight="weight", seed=SEED, iterations=300, k=0.25)
linked = codes.loc[list(G.nodes)]

# Match the PNAS template exactly: text is typeset by LaTeX with the same
# packages pnas-new.cls loads (lmodern + helvet), in sans at 7 pt like captions.
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{lmodern}\usepackage[scaled]{helvet}\renewcommand{\familydefault}{\sfdefault}",
    "font.family": "sans-serif",
    "font.size": 7,
})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6))
for ax, label, letter in zip(axes, ["field", "technology"], "AB"):
    shown = list(linked[label].value_counts().index[:7])
    colour = {v: c for v, c in zip(shown, PALETTE)}
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.3, edge_color="#8a8a85", alpha=0.25)
    groups = linked[label].where(linked[label].isin(shown), "other")
    for value in ["other"] + shown[::-1]:
        nodes = list(groups.index[groups == value])
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, ax=ax, node_size=16,
                               node_color=colour.get(value, OTHER), edgecolors="white", linewidths=0.4)
    handles = [plt.Line2D([], [], marker="o", ls="", color=colour[v], markersize=4,
                          label=f"{v.replace('&', r'\\&')} ({(groups == v).sum()})") for v in shown]
    handles.append(plt.Line2D([], [], marker="o", ls="", color=OTHER, markersize=4,
                              label=f"{linked[label].nunique() - len(shown)} others ({(groups == 'other').sum()})"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False,
              fontsize=6.5, handletextpad=0.2, columnspacing=0.8)
    ax.set_title(f"Coloured by {'medical field' if label == 'field' else 'technology'}", fontsize=7)
    ax.text(0.0, 1.02, rf"\textbf{{{letter}}}", transform=ax.transAxes, fontsize=8)
    ax.axis("off")
plt.tight_layout()
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "fig1_network.pdf", bbox_inches="tight")
fig.savefig(out / "fig1_network.png", dpi=200, bbox_inches="tight")
