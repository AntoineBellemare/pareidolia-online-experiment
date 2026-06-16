"""Paper Fig 6 — two-panel composite of preferred percept categories.

  (a) Semantic galaxy : UMAP of percept words, coloured by log-odds
                        preference (Low ← blue | red → High).
  (b) Forest plot     : per-cluster log-odds with bootstrap 95 % CI,
                        sorted; stars for non-zero-overlapping CIs.

Reuses helpers from `preferred_categories.py` and
`preferred_categories_viz.py`.
"""
from __future__ import annotations

import argparse
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import embeddings, style
from .preferred_categories import (
    LABELS, _long_words, cluster_distinctive_labels, cluster_preference,
    cluster_words,
)
from .preferred_categories_viz import _bootstrap_log_odds


def _build():
    df = _long_words()
    cnt = Counter(df["word"])
    words = sorted(w for w, n in cnt.items() if n >= 3)
    word2cluster = cluster_words(words, min_cluster_size=12)
    labels = cluster_distinctive_labels(df, word2cluster)
    table = cluster_preference(df, word2cluster, labels)
    return df, words, word2cluster, labels, table


# ─── panel functions, drawing into a supplied axis ───────────────────────────

def _draw_galaxy(ax, words, word2cluster, table) -> None:
    import umap
    from matplotlib.colors import LinearSegmentedColormap
    edict = embeddings.embedding_dict()
    V = np.vstack([edict[w] for w in words])
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    coords = umap.UMAP(
        n_components=2, metric="cosine", random_state=0,
        n_neighbors=15, min_dist=0.1,
    ).fit_transform(Vn)
    cluster_pref = dict(zip(table["cluster"], table["log_odds_HvsL"]))
    pref = np.array([cluster_pref.get(word2cluster[w], 0.0) for w in words])
    vmax = max(0.05, float(np.percentile(np.abs(pref), 95)))

    # Custom diverging cmap matching the DAT tertile bar palette:
    # Low (amber)  →  white  →  High (dark red)
    dat_diverging = LinearSegmentedColormap.from_list(
        "DatLowHigh", [style.COLORS["Low"], "#ffffff", style.COLORS["High"]],
    )
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=pref, cmap=dat_diverging, vmin=-vmax, vmax=vmax,
                    s=7, alpha=0.85, linewidths=0)
    # Inset colorbar in the lower-left corner, very small footprint
    cax = ax.inset_axes([0.02, 0.03, 0.025, 0.22])
    cb = plt.colorbar(sc, cax=cax)
    cb.set_label("log-odds\npref.", fontsize=5)
    cb.ax.tick_params(labelsize=4.5)

    top_high = table.nlargest(5, "log_odds_HvsL")
    top_low = table.nsmallest(5, "log_odds_HvsL")
    word2idx = {w: i for i, w in enumerate(words)}
    for _, r in pd.concat([top_high, top_low]).iterrows():
        cluster_id = r["cluster"]
        member_idx = [word2idx[w] for w, c in word2cluster.items()
                      if c == cluster_id and w in word2idx]
        if not member_idx: continue
        cx, cy = coords[member_idx, 0].mean(), coords[member_idx, 1].mean()
        text_col = (style.COLORS["High"] if r["log_odds_HvsL"] > 0
                    else style.COLORS["Low"])
        ax.text(cx, cy, r["label"].split(",")[0],
                fontsize=6, weight="bold", ha="center", va="center",
                color=text_col,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor=text_col, linewidth=0.4, alpha=0.92))
    ax.set_xticks([]); ax.set_yticks([])
    style.style_axis(ax, title="(a) Semantic map",
                     xlabel="UMAP-1", ylabel="UMAP-2")


def _draw_forest(ax, df, table, word2cluster) -> None:
    tertile_totals = df.groupby("dat_tertile").size().reindex(LABELS)
    tot_low = int(tertile_totals["Low"])
    tot_high = int(tertile_totals["High"])
    rows = []
    for _, r in table.iterrows():
        in_low = int(r["Low"]); in_high = int(r["High"])
        ci_lo, ci_hi = _bootstrap_log_odds(in_low, in_high, tot_low, tot_high)
        rows.append({
            "label": r["label"], "log_odds": r["log_odds_HvsL"],
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "total": int(r["total"]),
            "sig": (ci_lo > 0) or (ci_hi < 0),
        })
    forest = pd.DataFrame(rows).sort_values("log_odds")

    ys = np.arange(len(forest))
    for y, (_, r) in zip(ys, forest.iterrows()):
        col = style.COLORS["High"] if r["log_odds"] > 0 else style.COLORS["Low"]
        alpha = 0.95 if r["sig"] else 0.40
        ax.plot([r["ci_lo"], r["ci_hi"]], [y, y], color=col, lw=1.1, alpha=alpha)
        ax.scatter(r["log_odds"], y, s=14, color=col,
                   edgecolor="black", linewidths=0.3, zorder=3, alpha=alpha)
        if r["sig"]:
            ax.text(r["ci_hi"] + 0.03, y, "*",
                    va="center", ha="left", fontsize=9, weight="bold")
    ax.axvline(0, color="black", lw=0.4)
    ax.set_yticks(ys)
    # Shorter labels: keep first 3 distinctive words; put count separately on right
    short_labels = [", ".join(r["label"].split(", ")[:3])
                    for _, r in forest.iterrows()]
    ax.set_yticklabels(short_labels, fontsize=5.5)
    # Cluster size as a right-side annotation, gray
    ax_right = ax.secondary_yaxis("right")
    ax_right.set_yticks(ys)
    ax_right.set_yticklabels([f"n={r['total']}" for _, r in forest.iterrows()],
                              fontsize=5, color="#888888")
    ax_right.tick_params(length=0)
    ax_right.spines["right"].set_visible(False)
    style.style_axis(
        ax, title="(b) Cluster preference",
        xlabel="log-odds  ←  Low DAT   |   High DAT  →",
        ylabel="",
    )


# ─── composite ───────────────────────────────────────────────────────────────

def main(show: bool = True):
    style.apply()
    df, words, word2cluster, labels, table = _build()
    print(f"Loaded {len(words)} words, {len(table)} clusters")

    # Tighter canvas; galaxy on left, forest on right with shorter labels.
    fig = plt.figure(figsize=(style.COL_2, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 0.95], wspace=0.32)
    ax_g = fig.add_subplot(gs[0, 0])
    ax_f = fig.add_subplot(gs[0, 1])
    _draw_galaxy(ax_g, words, word2cluster, table)
    _draw_forest(ax_f, df, table, word2cluster)
    fig.suptitle("Percept clusters preferred by low vs high creativity",
                 y=1.02)
    if show: plt.show()
    style.savefig(fig, "preferred_categories_combined")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    main(show=not args.no_show)
