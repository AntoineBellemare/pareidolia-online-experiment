"""Paper Fig 7 — two-panel composite of preferred percept categories,
analogous to Fig 6 but contrasting FD16 vs FD12 instead of high vs low DAT.

  (a) Semantic map  : UMAP of percept words, coloured by log-odds
                       preference (FD12 ← blue | red → FD16).
  (b) Forest plot   : per-cluster log-odds (FD16 / FD12) with bootstrap
                       95 % CI, sorted; stars for clusters whose CI
                       excludes zero.

Same UMAP + HDBSCAN clustering as Fig 6; only the contrast variable
changes.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config, dat_helper, embeddings, parse_events, style
from .preferred_categories import (
    cluster_distinctive_labels, cluster_words,
)
from .preferred_categories_viz import _bootstrap_log_odds


ASCII = re.compile(r"^[A-Za-z]+$")
LABELS_FD = config.FD_LEVELS    # ["FD12", "FD14", "FD16"]


# ─── data assembly (per-(word, FD) counts) ───────────────────────────────────

def _long_words_fd() -> pd.DataFrame:
    """One row per (user, word, FD) — analogous to preferred_categories._long_words
    but keeping the FD label rather than the DAT tertile."""
    p = dat_helper.notebook_cohort()
    edict = embeddings.embedding_dict()
    rows = []
    for _, r in p.iterrows():
        for fd in config.FD_LEVELS:
            ws = r.get(f"{fd}_words")
            if ws is None: continue
            for w in list(ws):
                if (ASCII.fullmatch(w or "") and w in edict
                        and w not in config.SEMANTIC_STOPWORDS):
                    rows.append({"user_id": r["user_id"], "word": w,
                                  "fd_level": fd})
    return pd.DataFrame(rows).drop_duplicates(["user_id", "word", "fd_level"])


def cluster_preference_fd(df: pd.DataFrame, word2cluster: dict[str, int],
                           labels: dict[int, str],
                           min_cluster_total: int = 25) -> pd.DataFrame:
    """Per-cluster log-odds(FD16 / FD12) and CSV row stats."""
    df = df.assign(cluster=df["word"].map(word2cluster))
    df = df.dropna(subset=["cluster"])
    df = df[df["fd_level"].isin(LABELS_FD)]

    table = df.pivot_table(index="cluster", columns="fd_level",
                            values="word", aggfunc="size", fill_value=0)
    table = table.reindex(columns=LABELS_FD, fill_value=0)
    table["total"] = table[LABELS_FD].sum(axis=1)
    table = table[table["total"] >= min_cluster_total]

    fd_totals = df.groupby("fd_level").size().reindex(LABELS_FD)
    for fd in LABELS_FD:
        table[f"p_{fd}"] = table[fd] / fd_totals[fd]
    eps = 1e-6
    table["log_odds_FD16vsFD12"] = (
        np.log(table["p_FD16"] + eps) - np.log(table["p_FD12"] + eps)
    )
    table["label"] = [labels.get(c, "?") for c in table.index]
    return table.reset_index().sort_values("log_odds_FD16vsFD12")


# ─── panels ──────────────────────────────────────────────────────────────────

def _draw_map(ax, words, word2cluster, table) -> None:
    import umap
    edict = embeddings.embedding_dict()
    V = np.vstack([edict[w] for w in words])
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    coords = umap.UMAP(
        n_components=2, metric="cosine", random_state=0,
        n_neighbors=15, min_dist=0.1,
    ).fit_transform(Vn)
    cluster_pref = dict(zip(table["cluster"], table["log_odds_FD16vsFD12"]))
    pref = np.array([cluster_pref.get(word2cluster[w], 0.0) for w in words])
    vmax = max(0.05, float(np.percentile(np.abs(pref), 95)))

    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=pref, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    s=7, alpha=0.85, linewidths=0)
    cax = ax.inset_axes([0.02, 0.03, 0.025, 0.22])
    cb = plt.colorbar(sc, cax=cax)
    cb.set_label("log-odds\npref.", fontsize=5)
    cb.ax.tick_params(labelsize=4.5)

    top_high = table.nlargest(5, "log_odds_FD16vsFD12")
    top_low  = table.nsmallest(5, "log_odds_FD16vsFD12")
    word2idx = {w: i for i, w in enumerate(words)}
    for _, r in pd.concat([top_high, top_low]).iterrows():
        cluster_id = r["cluster"]
        member_idx = [word2idx[w] for w, c in word2cluster.items()
                      if c == cluster_id and w in word2idx]
        if not member_idx: continue
        cx, cy = coords[member_idx, 0].mean(), coords[member_idx, 1].mean()
        text_col = (style.COLORS["FD16"] if r["log_odds_FD16vsFD12"] > 0
                    else style.COLORS["FD12"])
        ax.text(cx, cy, r["label"].split(",")[0],
                fontsize=6, weight="bold", ha="center", va="center",
                color=text_col,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor=text_col, linewidth=0.4, alpha=0.92))
    ax.set_xticks([]); ax.set_yticks([])
    style.style_axis(ax, title="(a) Semantic map",
                     xlabel="UMAP-1", ylabel="UMAP-2")


def _draw_forest(ax, df, table, word2cluster) -> None:
    fd_totals = df.groupby("fd_level").size().reindex(LABELS_FD)
    tot_low  = int(fd_totals["FD12"])
    tot_high = int(fd_totals["FD16"])
    rows = []
    for _, r in table.iterrows():
        in_low  = int(r["FD12"]); in_high = int(r["FD16"])
        ci_lo, ci_hi = _bootstrap_log_odds(in_low, in_high, tot_low, tot_high)
        rows.append({
            "label": r["label"], "log_odds": r["log_odds_FD16vsFD12"],
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "total": int(r["total"]),
            "sig": (ci_lo > 0) or (ci_hi < 0),
        })
    forest = pd.DataFrame(rows).sort_values("log_odds")

    ys = np.arange(len(forest))
    for y, (_, r) in zip(ys, forest.iterrows()):
        col = (style.COLORS["FD16"] if r["log_odds"] > 0
                else style.COLORS["FD12"])
        alpha = 0.95 if r["sig"] else 0.40
        ax.plot([r["ci_lo"], r["ci_hi"]], [y, y], color=col, lw=1.1, alpha=alpha)
        ax.scatter(r["log_odds"], y, s=14, color=col,
                   edgecolor="black", linewidths=0.3, zorder=3, alpha=alpha)
        if r["sig"]:
            ax.text(r["ci_hi"] + 0.03, y, "*",
                    va="center", ha="left", fontsize=9, weight="bold")
    ax.axvline(0, color="black", lw=0.4)
    ax.set_yticks(ys)
    short_labels = [", ".join(r["label"].split(", ")[:3])
                    for _, r in forest.iterrows()]
    ax.set_yticklabels(short_labels, fontsize=5.5)
    ax_right = ax.secondary_yaxis("right")
    ax_right.set_yticks(ys)
    ax_right.set_yticklabels([f"n={r['total']}" for _, r in forest.iterrows()],
                              fontsize=5, color="#888888")
    ax_right.tick_params(length=0)
    ax_right.spines["right"].set_visible(False)
    style.style_axis(
        ax, title="(b) Cluster preference",
        xlabel="log-odds  ←  FD12 (low)   |   FD16 (high)  →",
        ylabel="",
    )


# ─── composite ───────────────────────────────────────────────────────────────

def main(show: bool = True):
    style.apply()
    df = _long_words_fd()
    print(f"Long words: {len(df):,} rows")
    cnt = Counter(df["word"])
    words = sorted(w for w, n in cnt.items() if n >= 3)
    word2cluster = cluster_words(words, min_cluster_size=12)
    labels = cluster_distinctive_labels(df, word2cluster)
    table = cluster_preference_fd(df, word2cluster, labels)
    print(f"  {len(table)} clusters retained")

    fig = plt.figure(figsize=(style.COL_2, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 0.95], wspace=0.32)
    ax_g = fig.add_subplot(gs[0, 0])
    ax_f = fig.add_subplot(gs[0, 1])
    _draw_map(ax_g, words, word2cluster, table)
    _draw_forest(ax_f, df, table, word2cluster)
    fig.suptitle("Percept clusters preferred by low vs high stimulus FD",
                 y=1.02)
    if show: plt.show()
    style.savefig(fig, "preferred_categories_fd")
    plt.close(fig)
    table.to_csv(config.OUTPUTS_DIR / "preferred_categories_fd.csv",
                 index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    main(show=not args.no_show)
