"""Four candidate visualisations of the "preferred categories" result.

Reuses the clustering from `preferred_categories.py`:
    A. forest_plot           — log-odds with bootstrap 95 % CI (standard)
    B. slope_plot            — share across L/M/H tertiles per cluster
    C. word_galaxy           — UMAP, words coloured by preference
    D. tertile_word_clouds   — one cloud per tertile

Usage:
    python -m analysis.figures.preferred_categories_viz
    python -m analysis.figures.preferred_categories_viz --only forest slope
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .. import config, dat_helper, embeddings, parse_events, style
from .preferred_categories import (
    cluster_distinctive_labels,
    cluster_preference,
    cluster_words,
    _long_words,
    LABELS,
)


# ─── shared data preparation ─────────────────────────────────────────────────

def _build(min_count: int = 3, min_cluster_size: int = 12):
    df = _long_words()
    cnt = Counter(df["word"])
    words = sorted(w for w, n in cnt.items() if n >= min_count)
    word2cluster = cluster_words(words, min_cluster_size=min_cluster_size)
    labels = cluster_distinctive_labels(df, word2cluster)
    table = cluster_preference(df, word2cluster, labels)
    return df, words, word2cluster, labels, table


# ═════════════════════════════════════════════════════════════════════════════
# A. Forest plot with bootstrap CIs
# ═════════════════════════════════════════════════════════════════════════════

def _bootstrap_log_odds(in_low: int, in_high: int,
                         total_low: int, total_high: int,
                         n_boot: int = 2000, seed: int = 0
                         ) -> tuple[float, float]:
    """Bootstrap 95 % CI for log((p_high+eps)/(p_low+eps)).

    Resamples each tertile's word stream and recomputes the log-odds.
    Returns (CI low, CI high).
    """
    rng = np.random.default_rng(seed)
    p_low = in_low / max(total_low, 1)
    p_high = in_high / max(total_high, 1)
    # Approximate via Binomial draws (faster than full resampling)
    samples_high = rng.binomial(total_high, p_high, size=n_boot) / total_high
    samples_low  = rng.binomial(total_low,  p_low,  size=n_boot) / total_low
    eps = 1e-6
    lo = np.log(samples_high + eps) - np.log(samples_low + eps)
    return float(np.percentile(lo, 2.5)), float(np.percentile(lo, 97.5))


def forest_plot(table: pd.DataFrame, df: pd.DataFrame,
                 word2cluster: dict[str, int], show: bool = True,
                 max_clusters: int = 32) -> None:
    style.apply()
    tertile_totals = df.groupby("dat_tertile").size().reindex(LABELS)
    tot_low  = int(tertile_totals["Low"])
    tot_high = int(tertile_totals["High"])
    rows = []
    for _, r in table.iterrows():
        in_low  = int(r["Low"])
        in_high = int(r["High"])
        ci_lo, ci_hi = _bootstrap_log_odds(in_low, in_high, tot_low, tot_high)
        rows.append({"cluster": r["cluster"], "label": r["label"],
                      "log_odds": r["log_odds_HvsL"],
                      "ci_lo": ci_lo, "ci_hi": ci_hi,
                      "total": int(r["total"]),
                      "sig": (ci_lo > 0) or (ci_hi < 0)})
    forest = pd.DataFrame(rows).sort_values("log_odds")
    forest = forest.tail(max_clusters)        # keep all by default

    fig, ax = plt.subplots(figsize=(style.COL_1_5, 0.22 * len(forest) + 1.0))
    ys = np.arange(len(forest))
    for y, (_, r) in zip(ys, forest.iterrows()):
        col = style.COLORS["High"] if r["log_odds"] > 0 else style.COLORS["Low"]
        alpha = 0.95 if r["sig"] else 0.45
        ax.plot([r["ci_lo"], r["ci_hi"]], [y, y], color=col, lw=1.4, alpha=alpha)
        ax.scatter(r["log_odds"], y, s=22, color=col, edgecolor="black",
                   linewidths=0.4, zorder=3, alpha=alpha)
        if r["sig"]:
            ax.text(r["ci_hi"] + 0.04, y, "*",
                    va="center", ha="left", fontsize=11, weight="bold")
    ax.axvline(0, color="black", lw=0.4)
    ax.set_yticks(ys)
    ax.set_yticklabels(
        [f"{r['label']}  ({r['total']})" for _, r in forest.iterrows()],
        fontsize=6,
    )
    style.style_axis(
        ax, title="",
        xlabel="log-odds preference  ←  Low DAT   |   High DAT  →",
        ylabel="",
    )
    fig.suptitle("Forest plot of cluster preference (bootstrap 95 % CI)",
                 y=1.01)
    if show: plt.show()
    style.savefig(fig, "preferred_categories_forest")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# B. Slope plot — share across tertiles
# ═════════════════════════════════════════════════════════════════════════════

def slope_plot(table: pd.DataFrame, show: bool = True,
               highlight_top: int = 6) -> None:
    style.apply()
    fig, ax = plt.subplots(figsize=(style.COL_1_5, 3.6))
    xs = np.arange(3)
    # Normalise the per-tertile share so the line is in [0,1] for the cluster
    # (which highlights direction; absolute share is also OK).
    for _, r in table.iterrows():
        ys = [r["p_Low"], r["p_Mid"], r["p_High"]] / np.mean([r["p_Low"], r["p_Mid"], r["p_High"]])
        col = style.COLORS["High"] if r["log_odds_HvsL"] > 0 else style.COLORS["Low"]
        # Faded background lines for every cluster
        ax.plot(xs, ys, color=col, lw=0.4 * (1 + 1.5 * np.log1p(r["total"]) / 6),
                alpha=0.35)
    # Highlight the top high- and low-preferred clusters
    top_high = table.nlargest(highlight_top, "log_odds_HvsL")
    top_low  = table.nsmallest(highlight_top, "log_odds_HvsL")
    for _, r in pd.concat([top_high, top_low]).iterrows():
        ys = [r["p_Low"], r["p_Mid"], r["p_High"]] / np.mean([r["p_Low"], r["p_Mid"], r["p_High"]])
        col = style.COLORS["High"] if r["log_odds_HvsL"] > 0 else style.COLORS["Low"]
        ax.plot(xs, ys, color=col, lw=1.4, alpha=0.95, zorder=5)
        ax.scatter(xs, ys, color=col, s=14, zorder=6, edgecolor="black",
                   linewidths=0.3)
        # Label at the relevant end
        if r["log_odds_HvsL"] > 0:
            label_x, label_y, ha = 2.05, ys[2], "left"
        else:
            label_x, label_y, ha = -0.05, ys[0], "right"
        ax.text(label_x, label_y, r["label"].split(",")[0],
                fontsize=6, color=col, va="center", ha=ha,
                weight="bold")
    ax.axhline(1.0, color="black", lw=0.4, ls="--")
    ax.set_xticks(xs); ax.set_xticklabels(["Low", "Mid", "High"])
    style.style_axis(ax, title="",
                     xlabel="DAT tertile",
                     ylabel="Cluster share (normalised by cluster mean)")
    fig.suptitle("Slope of percept-cluster usage across DAT tertiles", y=1.0)
    if show: plt.show()
    style.savefig(fig, "preferred_categories_slope")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# C. Word galaxy — UMAP coloured by preference
# ═════════════════════════════════════════════════════════════════════════════

def word_galaxy(table: pd.DataFrame, words: list[str],
                 word2cluster: dict[str, int], show: bool = True) -> None:
    import umap
    style.apply()
    edict = embeddings.embedding_dict()
    V = np.vstack([edict[w] for w in words])
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    coords = umap.UMAP(n_components=2, metric="cosine", random_state=0,
                       n_neighbors=15, min_dist=0.1).fit_transform(Vn)
    cluster_pref = dict(zip(table["cluster"], table["log_odds_HvsL"]))
    pref = np.array([cluster_pref.get(word2cluster[w], 0.0) for w in words])
    vmax = max(0.05, float(np.percentile(np.abs(pref), 95)))

    fig, ax = plt.subplots(figsize=(style.COL_2, 4.6))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=pref, cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax, s=8, alpha=0.85, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.01)
    cb.set_label("log-odds preference\n← Low   |   High →", fontsize=6)
    # Annotate the top 6 high and top 6 low clusters at their centroid
    top_high = table.nlargest(6, "log_odds_HvsL")
    top_low  = table.nsmallest(6, "log_odds_HvsL")
    word2idx = {w: i for i, w in enumerate(words)}
    for _, r in pd.concat([top_high, top_low]).iterrows():
        cluster_id = r["cluster"]
        member_idx = [word2idx[w] for w, c in word2cluster.items()
                      if c == cluster_id and w in word2idx]
        if not member_idx: continue
        cx, cy = coords[member_idx, 0].mean(), coords[member_idx, 1].mean()
        text_col = style.COLORS["High"] if r["log_odds_HvsL"] > 0 else style.COLORS["Low"]
        ax.text(cx, cy, r["label"].split(",")[0],
                fontsize=6.5, weight="bold", ha="center", va="center",
                color=text_col,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor=text_col, linewidth=0.5, alpha=0.9))
    ax.set_xticks([]); ax.set_yticks([])
    style.style_axis(ax, title="", xlabel="UMAP-1", ylabel="UMAP-2")
    fig.suptitle("Semantic galaxy of percepts, coloured by preference", y=0.99)
    if show: plt.show()
    style.savefig(fig, "preferred_categories_galaxy")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# D. Per-tertile word clouds
# ═════════════════════════════════════════════════════════════════════════════

def tertile_word_clouds(df: pd.DataFrame, show: bool = True) -> None:
    from wordcloud import WordCloud
    style.apply()
    # Per-tertile word frequency
    cnt = (df.groupby(["dat_tertile", "word"]).size()
              .unstack(fill_value=0))
    cnt = cnt.reindex(LABELS, fill_value=0)
    # Per-word: distinctiveness for each tertile = log-odds of being in that
    # tertile vs the others, weighted by within-tertile count
    fig, axes = plt.subplots(1, 3, figsize=(style.COL_2, 2.4),
                             gridspec_kw=dict(wspace=0.05))
    tert_totals = cnt.sum(axis=1)
    for ax, t in zip(axes, LABELS):
        in_t   = cnt.loc[t]
        out_t  = cnt.drop(index=t).sum(axis=0)
        n_in   = float(in_t.sum()); n_out = float(out_t.sum())
        eps = 0.5
        p_in   = (in_t + eps) / (n_in + eps)
        p_out  = (out_t + eps) / (n_out + eps)
        score  = np.log(p_in) - np.log(p_out)
        # Keep words with at least 4 uses in this tertile and positive score
        mask = (in_t >= 4) & (score > 0)
        weights = (score[mask] * np.log1p(in_t[mask])).to_dict()
        if not weights:
            ax.text(0.5, 0.5, "(no distinctive words)",
                    transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            continue
        wc = WordCloud(
            width=600, height=480, background_color="white",
            color_func=lambda *a, **k: style.COLORS[t],
            prefer_horizontal=0.85, max_words=80, relative_scaling=0.65,
        ).generate_from_frequencies(weights)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_axis_off()
        ax.set_title(f"{t} DAT (n words = {int(n_in):,})", fontsize=8)
    fig.suptitle("Most distinctive words per creativity tertile", y=1.03)
    if show: plt.show()
    style.savefig(fig, "preferred_categories_wordclouds")
    plt.close(fig)


# ─── runner ──────────────────────────────────────────────────────────────────

REGISTRY = {
    "forest":  lambda d, w, w2c, lab, tbl, show: forest_plot(tbl, d, w2c, show),
    "slope":   lambda d, w, w2c, lab, tbl, show: slope_plot(tbl, show),
    "galaxy":  lambda d, w, w2c, lab, tbl, show: word_galaxy(tbl, w, w2c, show),
    "cloud":   lambda d, w, w2c, lab, tbl, show: tertile_word_clouds(d, show),
}


def main(only: list[str] | None = None, show: bool = True):
    df, words, word2cluster, labels, table = _build()
    print(f"Loaded {len(words)} words, {len(table)} retained clusters")
    selected = only or list(REGISTRY)
    for name in selected:
        print(f"--- {name} ---")
        REGISTRY[name](df, words, word2cluster, labels, table, show)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(REGISTRY))
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    main(only=args.only, show=not args.no_show)
