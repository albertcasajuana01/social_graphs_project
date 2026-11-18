"""Fig 2: do links follow medical field or technology?

A, B: the Fig 1 layout coloured by field and by technology.
C: assortativity of each label against shuffled labels, freely and within the other label.
D: share of possible pairs that became a link, by whether the two codes share a field and a technology.

Usage: python report/make_fig2.py <project_dir> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from figdata import INK, MUTED, OTHER, PALETTE, SEED, giant_layout, load, plt, tex, use_style

N_SHUFFLES = 1_000
project, out = Path(sys.argv[1]), Path(sys.argv[2])
sample, codes, sim, G = load(project)
rng = np.random.default_rng(SEED)


# ---------- C: shuffle tests (same method as section 4 of the notebook) ----------
def assortativity(labels):
    nx.set_node_attributes(G, labels.to_dict(), "label")
    return nx.attribute_assortativity_coefficient(G, "label")


def shuffled(labels, within=None):
    if within is None:
        return pd.Series(rng.permutation(labels.to_numpy()), index=labels.index)
    out_ = labels.copy()
    for _, idx in labels.groupby(within).groups.items():
        out_.loc[idx] = rng.permutation(labels.loc[idx].to_numpy())
    return out_


nodes = list(G.nodes)
tests = [  # (row label, label, shuffled within)
    ("Medical field", "field", None),
    ("Technology", "technology", None),
    ("Medical field,\nshuffled within technology", "field", "technology"),
    ("Technology,\nshuffled within field", "technology", "field"),
]
rows = []
for name, label, within in tests:
    labels = codes[label].reindex(nodes)
    groups = None if within is None else codes[within].reindex(nodes)
    observed = assortativity(labels)
    null = np.array([assortativity(shuffled(labels, groups)) for _ in range(N_SHUFFLES)])
    rows.append({"test": name, "label": label, "observed": observed, "null_mean": null.mean(),
                 "null_lo": np.quantile(null, 0.005), "null_hi": np.quantile(null, 0.995),
                 "null_max": null.max(), "p": (np.sum(null >= observed) + 1) / (N_SHUFFLES + 1)})
tests_df = pd.DataFrame(rows)
print(tests_df.round(3).to_string())

# ---------- D: link rate by pair type ----------
order = np.array(nodes)
i, j = np.triu_indices(len(order), 1)
f, t = codes["field"].reindex(order).to_numpy(), codes["technology"].reindex(order).to_numpy()
same_f, same_t = f[i] == f[j], t[i] == t[j]
edges = set(map(frozenset, G.edges))
is_link = np.array([frozenset((order[a], order[b])) in edges for a, b in zip(i, j)])
kinds = [  # (label, mask, colour)
    ("Same field and technology", same_f & same_t, OTHER),
    ("Same technology only", ~same_f & same_t, INK),
    ("Same field only", same_f & ~same_t, INK),
    ("Neither", ~same_f & ~same_t, OTHER),
]
rates = pd.DataFrame([{"pair type": k, "links": int(is_link[m].sum()), "pairs": int(m.sum()),
                       "rate": is_link[m].mean(), "colour": c} for k, m, c in kinds])
print(rates.drop(columns="colour").round(3).to_string())

# ---------- figure ----------
use_style()
fig = plt.figure(figsize=(7.0, 6.3))
grid = fig.add_gridspec(2, 2, height_ratios=[1.45, 1], hspace=0.42, wspace=0.62)

giant, pos = giant_layout(G)
linked = codes.loc[list(giant.nodes)]
for col, (label, title, letter) in enumerate([("field", "Coloured by medical field", "A"),
                                              ("technology", "Coloured by technology", "B")]):
    ax = fig.add_subplot(grid[0, col])
    shown = list(linked[label].value_counts().index[:7])
    colour = dict(zip(shown, PALETTE))
    nx.draw_networkx_edges(giant, pos, ax=ax, width=0.3, edge_color="#8a8a85", alpha=0.25)
    groups = linked[label].where(linked[label].isin(shown), "other")
    for value in ["other"] + shown[::-1]:
        members = list(groups.index[groups == value])
        nx.draw_networkx_nodes(giant, pos, nodelist=members, ax=ax, node_size=14,
                               node_color=colour.get(value, OTHER), edgecolors="white", linewidths=0.4)
    handles = [plt.Line2D([], [], marker="o", ls="", color=colour[v], markersize=4,
                          label=f"{tex(v)} ({(groups == v).sum()})") for v in shown]
    handles.append(plt.Line2D([], [], marker="o", ls="", color=OTHER, markersize=4,
                              label=f"{linked[label].nunique() - len(shown)} others ({(groups == 'other').sum()})"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False,
              fontsize=6.3, handletextpad=0.2, columnspacing=0.8)
    ax.set_title(title, fontsize=7, color=INK)
    ax.text(0.0, 1.02, letter, transform=ax.transAxes, fontsize=8, fontweight="bold")
    ax.axis("off")

# C: observed value against the middle 99% of 1,000 shuffles.
# C and D use ink and grey only, so they do not reuse the category colours of A and B.
ax = fig.add_subplot(grid[1, 0])
y = np.array([3.4, 2.4, 1.0, 0.0])
for yy, row in zip(y, tests_df.itertuples()):
    ax.plot([row.null_lo, row.null_hi], [yy, yy], color=OTHER, lw=5, solid_capstyle="round", zorder=1)
    ax.plot(row.observed, yy, "o", color=INK, markersize=5.5, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    ax.text(row.observed + 0.025, yy, f"{row.observed:.2f}", va="center", fontsize=6.5, color=INK)
ax.set_yticks(y, [tex(r) for r in tests_df["test"]])
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_xlim(-0.05, 0.6)
ax.set_ylim(-0.6, 4.2)
ax.axvline(0, color=MUTED, lw=0.5, ls=(0, (2, 2)), zorder=0)
ax.set_xlabel("Assortativity (0 = no more than chance)")
ax.text(-0.05, 3.95, "Shuffled freely", fontsize=6.3, style="italic", color=MUTED, transform=ax.transData)
ax.text(-0.05, 1.55, "Shuffled within the other label", fontsize=6.3, style="italic", color=MUTED, transform=ax.transData)
ax.plot([], [], color=OTHER, lw=5, label=f"Middle 99% of {N_SHUFFLES:,} shuffles")
ax.plot([], [], "o", color=INK, markersize=5, label="Observed")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False, fontsize=6.3, handlelength=1.6)
ax.text(-0.02, 1.04, "C", transform=ax.transAxes, fontsize=8, ha="right", fontweight="bold")

# D: link rate by pair type
ax = fig.add_subplot(grid[1, 1])
yd = np.arange(len(rates))[::-1]
ax.barh(yd, rates["rate"], color=rates["colour"], height=0.62)
for yy, row in zip(yd, rates.itertuples()):
    ax.text(row.rate + 0.015, yy, f"{row.rate:.0%}  ({row.links:,} of {row.pairs:,} pairs)",
            va="center", fontsize=6.5, color=INK)
ax.set_yticks(yd, [tex(k) for k in rates["pair type"]])
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_xlim(0, 1.0)
ax.set_xticks([0, 0.25, 0.5, 0.75], ["0", "25%", "50%", "75%"])
ax.set_xlabel("Share of possible pairs that are linked")
ax.text(-0.02, 1.04, "D", transform=ax.transAxes, fontsize=8, ha="right", fontweight="bold")

out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "fig2_field_technology.pdf", bbox_inches="tight")
fig.savefig(out / "fig2_field_technology.png", dpi=200, bbox_inches="tight")
print("saved", out / "fig2_field_technology.pdf")
