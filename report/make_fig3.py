"""Fig 3: the communities, and whether each one is held together by field or by technology.

A: the Fig 1 layout coloured by Louvain community (best of 20 seeds, as in the notebook).
B: for each community with at least 5 codes, the share of its codes in its most common
   medical field and in its most common technology, with the FDA problem term most
   over-represented in its reports.

Usage: python report/make_fig3.py <project_dir> <out_dir>
"""
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from figdata import INK, MUTED, OTHER, PALETTE, communities, giant_layout, load, plt, use_style

MIN_CODES = 5
# Communities are named after what they contain. Each name is attached to a code that
# defines the community, so the names survive a change in Louvain's numbering.
NAMES = {
    "DXY": "Pacing, defibrillation and neurostimulation",
    "LPH": "Joint, breast and surgical implants",
    "MAF": "Stents, occluders and vascular catheters",
    "QBJ": "Diabetes devices and device software",
    "HQL": "Eye lenses and eye surgery",
    "DYE": "Heart valves and ablation catheters",
    "LOM": "Infection and cytology tests",
    "MTF": "Prostate cancer tests",
    "QCD": "Implanted glucose sensors",
    "OTE": "Mammography",
}

project, out = Path(sys.argv[1]), Path(sys.argv[2])
sample, codes, sim, G = load(project)
partition, modularity = communities(G)
codes["community"] = partition
names = {partition[code]: name for code, name in NAMES.items()}
assert len(names) == partition.nunique(), "a community has no name, or two names point to one community"
codes["community_name"] = codes["community"].map(names)


# ---------- problem terms over-represented in each community (as in section 5 of the notebook) ----------
def problem_terms(frame):
    terms = (frame["product_problems"].fillna("") + "; " + frame["patient_problems"].fillna("")).str.split("; ")
    return terms.explode().str.strip().loc[lambda s: s != ""]


sample["community_name"] = sample["product_code"].map(codes["community_name"])
all_terms = problem_terms(sample).value_counts(normalize=True)

rows = []
for name, members in codes.dropna(subset=["community_name"]).groupby("community_name"):
    in_comm = sample[sample["community_name"] == name]
    counts = problem_terms(in_comm).value_counts()
    share = counts / len(in_comm)
    lift = (counts / counts.sum() / all_terms.reindex(counts.index)).where(share >= 0.05).dropna()
    lift = lift[lift > 1]
    field_top, tech_top = members["field"].value_counts(), members["technology"].value_counts()
    rows.append({
        "community": name, "codes": len(members),
        "fields": members["field"].nunique(), "technologies": members["technology"].nunique(),
        "top field": field_top.index[0], "field share": field_top.iloc[0] / len(members),
        "top technology": tech_top.index[0], "technology share": tech_top.iloc[0] / len(members),
        "top problem": lift.idxmax(), "lift": lift.max(),
    })
summary = pd.DataFrame(rows).sort_values("codes", ascending=False).reset_index(drop=True)
print(f"modularity {modularity:.2f}")
print(summary.round(2).to_string())

# ---------- figure ----------
use_style()
fig = plt.figure(figsize=(7.0, 3.9))
# Columns: network | dumbbells | room for the text to the right of the dumbbells
grid = fig.add_gridspec(1, 3, width_ratios=[1.2, 0.8, 0.62], wspace=0.08)


def plural(n, word):
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


# A: network coloured by community. The seven largest get a colour, the rest are grey.
giant, pos = giant_layout(G)
in_giant = codes.loc[list(giant.nodes), "community_name"]
shown = list(summary["community"][:len(PALETTE)])
colour = dict(zip(shown, PALETTE))
ax = fig.add_subplot(grid[0, 0])
nx.draw_networkx_edges(giant, pos, ax=ax, width=0.3, edge_color="#8a8a85", alpha=0.25)
groups = in_giant.where(in_giant.isin(shown), "other")
for value in ["other"] + shown[::-1]:
    members = list(groups.index[groups == value])
    nx.draw_networkx_nodes(giant, pos, nodelist=members, ax=ax, node_size=14,
                           node_color=colour.get(value, OTHER), edgecolors="white", linewidths=0.4)
ax.set_title(f"Communities (Louvain, modularity {modularity:.2f})", fontsize=7, color=INK)
ax.text(0.0, 1.02, "A", transform=ax.transAxes, fontsize=8, fontweight="bold")
ax.axis("off")

# B: how much of each community is one field, and how much is one technology.
# Each row: the name on top (with the colour it has in A), the dumbbell below,
# and on the right how many fields and technologies it spans and its top problem term.
ax = fig.add_subplot(grid[0, 1])
big = summary[summary["codes"] >= MIN_CODES]
y = np.arange(len(big))[::-1].astype(float)
for yy, (_, row) in zip(y, big.iterrows()):
    lo, hi = sorted([row["field share"], row["technology share"]])
    ax.plot([lo, hi], [yy, yy], color=OTHER, lw=1.5, zorder=1)
    ax.plot(row["field share"], yy, "o", markersize=5.5, markerfacecolor="white", markeredgecolor=INK,
            markeredgewidth=1.1, zorder=3)
    ax.plot(row["technology share"], yy, "o", markersize=5.5, color=INK, markeredgecolor="white",
            markeredgewidth=0.6, zorder=3)
    ax.plot(0.0, yy + 0.42, "s", markersize=4.2, color=colour.get(row["community"], OTHER), clip_on=False)
    ax.text(0.035, yy + 0.42, f"{row['community']} ({row['codes']} codes)", fontsize=6.6, color=INK, va="center")
    ax.text(1.1, yy + 0.02, f"{plural(row['fields'], 'field')}, {plural(row['technologies'], 'technology').replace('ys', 'ies')}",
            fontsize=6, color=MUTED, va="center")
    ax.text(1.1, yy - 0.26, f"{row['top problem']} (×{row['lift']:.1f})", fontsize=6, color=INK, va="center")
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlim(0, 1.04)
ax.set_ylim(-0.55, len(big) - 0.05)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"])
ax.set_xlabel("Share of the community's codes in its\nmost common field or technology")
ax.text(1.1, len(big) - 0.12, "Top problem term (lift)", fontsize=6, color=MUTED, style="italic", va="center")
ax.plot([], [], "o", markersize=5.5, markerfacecolor="white", markeredgecolor=INK, ls="", label="Medical field")
ax.plot([], [], "o", markersize=5.5, color=INK, ls="", label="Technology")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=6.3)
ax.text(-0.04, 1.02, "B", transform=ax.transAxes, fontsize=8, ha="right", fontweight="bold")

out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "fig3_communities.pdf", bbox_inches="tight")
fig.savefig(out / "fig3_communities.png", dpi=200, bbox_inches="tight")
print("saved", out / "fig3_communities.pdf")
