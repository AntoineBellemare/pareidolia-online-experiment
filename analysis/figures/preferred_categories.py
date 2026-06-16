"""Which semantic categories do high- vs low-creativity participants prefer?

Strategy
--------
1. Take every unique percept word that appears >= `min_count` times in the
   corpus (default 3) and is in the BERT embedding table.
2. UMAP-reduce the 384-d embeddings to 10 dimensions (cosine metric).
3. HDBSCAN cluster the reduced vectors; outliers (label = -1) are kept as a
   "misc" group so all words have a bucket.
4. Label each cluster by its log-odds most distinctive words (so the label
   is what *defines* the cluster, not what is merely common in it).
5. Tag each (participant, word) row with the participant's DAT tertile.
6. For each cluster, compute:
       p_low(c)  = share of Low-tertile words that fall in cluster c
       p_high(c) = share of High-tertile words that fall in cluster c
       log-odds = log((p_high + ε) / (p_low + ε))
   Positive log-odds = preferred by high-creative; negative = preferred by
   low-creative.
7. Plot horizontal bar chart sorted by log-odds, with cluster labels.

Usage:
    python -m analysis.figures.preferred_categories
    python -m analysis.figures.preferred_categories --min-cluster 12 --min-count 4
"""
from __future__ import annotations

import argparse
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config, dat_helper, embeddings, parse_events, style

ASCII = re.compile(r"^[A-Za-z]+$")
LABELS = ["Low", "Mid", "High"]


# ─── data assembly ────────────────────────────────────────────────────────────

def _long_words() -> pd.DataFrame:
    p = dat_helper.notebook_cohort().dropna(subset=["ref_dat_score"])
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
                                  "fd_level": fd,
                                  "ref_dat_score": r["ref_dat_score"]})
    df = pd.DataFrame(rows)
    df["dat_tertile"] = pd.qcut(df["ref_dat_score"], 3, labels=LABELS,
                                duplicates="drop")
    return df


# ─── clustering ───────────────────────────────────────────────────────────────

def cluster_words(words: list[str], min_cluster_size: int = 10,
                   umap_dim: int = 10) -> dict[str, int]:
    """Return {word: cluster_label}. Outliers labelled -1."""
    import umap
    import hdbscan
    edict = embeddings.embedding_dict()
    V = np.vstack([edict[w] for w in words])
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    print(f"  UMAP({umap_dim}d, cosine) on {len(words)} words…")
    reducer = umap.UMAP(n_components=umap_dim, metric="cosine",
                         random_state=0, n_neighbors=15, min_dist=0.0)
    X_low = reducer.fit_transform(Vn)
    print(f"  HDBSCAN(min_cluster_size={min_cluster_size})…")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                 min_samples=3,
                                 cluster_selection_method="eom")
    labels = clusterer.fit_predict(X_low)
    return dict(zip(words, labels.tolist()))


def cluster_distinctive_labels(df: pd.DataFrame, word2cluster: dict[str, int],
                                top_k: int = 4, min_count_in_cluster: int = 3
                                ) -> dict[int, str]:
    """For each cluster, find the words whose presence is most diagnostic of
    the cluster (log-odds vs the rest)."""
    df = df.assign(cluster=df["word"].map(word2cluster))
    df = df.dropna(subset=["cluster"])
    global_cnt = Counter(df["word"])
    g_total = sum(global_cnt.values())
    out: dict[int, str] = {}
    for c, sub in df.groupby("cluster"):
        c_in: Counter = Counter(sub["word"])
        n_in = sum(c_in.values())
        scored = []
        for w, cw in c_in.items():
            if cw < min_count_in_cluster: continue
            c_out = global_cnt[w] - cw
            n_out = g_total - n_in
            p_in = (cw + 0.5) / (n_in + 0.5)
            p_out = (c_out + 0.5) / (n_out + 0.5)
            scored.append((w, np.log(p_in) - np.log(p_out)))
        scored.sort(key=lambda x: x[1], reverse=True)
        out[c] = ", ".join(w for w, _ in scored[:top_k]) or "misc"
    return out


# ─── per-tertile preference ───────────────────────────────────────────────────

def cluster_preference(df: pd.DataFrame, word2cluster: dict[str, int],
                        labels: dict[int, str],
                        min_cluster_total: int = 25) -> pd.DataFrame:
    df = df.assign(cluster=df["word"].map(word2cluster))
    df = df.dropna(subset=["cluster", "dat_tertile"])

    table = df.pivot_table(index="cluster", columns="dat_tertile",
                            values="word", aggfunc="size", fill_value=0)
    table = table.reindex(columns=LABELS, fill_value=0)
    table["total"] = table[LABELS].sum(axis=1)
    table = table[table["total"] >= min_cluster_total]
    # Normalize within each tertile so column sums = 1 -> share of tertile's
    # words that fall in this cluster.
    tertile_totals = df.groupby("dat_tertile").size().reindex(LABELS)
    for t in LABELS:
        table[f"p_{t}"] = table[t] / tertile_totals[t]
    eps = 1e-6
    table["log_odds_HvsL"] = np.log(table["p_High"] + eps) - \
                              np.log(table["p_Low"] + eps)
    table["label"] = [labels.get(c, "?") for c in table.index]
    table = table.reset_index().sort_values("log_odds_HvsL")
    return table


# ─── plotting ─────────────────────────────────────────────────────────────────

def plot_preference(table: pd.DataFrame, show: bool = True,
                    top_n_each_side: int = 10) -> None:
    style.apply()
    bottom = table.head(top_n_each_side)         # preferred by Low
    top    = table.tail(top_n_each_side)         # preferred by High
    pick   = pd.concat([bottom, top.iloc[::-1]]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(style.COL_1_5, 0.32 * len(pick) + 1.0))
    colours = [style.COLORS["High"] if x > 0 else style.COLORS["Low"]
               for x in pick["log_odds_HvsL"]]
    ax.barh(np.arange(len(pick)), pick["log_odds_HvsL"],
            color=colours, edgecolor="black", lw=0.3, alpha=0.9)
    ax.set_yticks(np.arange(len(pick)))
    ax.set_yticklabels(
        [f"{lab} (n={n})" for lab, n in zip(pick["label"], pick["total"])],
        fontsize=6,
    )
    ax.axvline(0, color="black", lw=0.5)
    style.style_axis(
        ax, title="",
        xlabel="log-odds preference  ←  Low DAT   |   High DAT  →",
        ylabel="",
    )
    fig.suptitle("Percept clusters preferred by low vs high creativity", y=1.01)
    if show: plt.show()
    style.savefig(fig, "preferred_categories")
    plt.close(fig)


def plot_cluster_map(words: list[str], word2cluster: dict[str, int],
                      labels: dict[int, str], show: bool = True) -> None:
    """2-D UMAP overview, points coloured by cluster, labelled with top words."""
    import umap
    edict = embeddings.embedding_dict()
    V = np.vstack([edict[w] for w in words])
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    coords = umap.UMAP(n_components=2, metric="cosine", random_state=0,
                       n_neighbors=15, min_dist=0.1).fit_transform(Vn)
    df = pd.DataFrame({"word": words,
                       "x": coords[:, 0], "y": coords[:, 1],
                       "cluster": [word2cluster[w] for w in words]})
    style.apply()
    fig, ax = plt.subplots(figsize=(style.COL_2, 4.5))
    clusters = sorted(set(df["cluster"]) - {-1})
    palette = plt.cm.tab20(np.linspace(0, 1, max(len(clusters), 1)))
    # outliers first, in light grey
    if -1 in df["cluster"].values:
        out = df[df["cluster"] == -1]
        ax.scatter(out["x"], out["y"], s=2, color="lightgrey", alpha=0.5,
                   linewidths=0, label="misc")
    for i, c in enumerate(clusters):
        sub = df[df["cluster"] == c]
        ax.scatter(sub["x"], sub["y"], s=4, color=palette[i], alpha=0.7,
                   linewidths=0, label=f"C{c}: {labels[c]}")
        cx, cy = sub["x"].mean(), sub["y"].mean()
        ax.text(cx, cy, labels[c].split(",")[0],
                fontsize=6, weight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.85))
    ax.set_xticks([]); ax.set_yticks([])
    style.style_axis(ax, title="", xlabel="UMAP-1", ylabel="UMAP-2")
    fig.suptitle("Percept-word clusters (HDBSCAN on UMAP-cosine)", y=0.98)
    if show: plt.show()
    style.savefig(fig, "preferred_categories_map")
    plt.close(fig)


# ─── runner ──────────────────────────────────────────────────────────────────

def main(min_count: int = 3, min_cluster_size: int = 12,
         show: bool = True):
    df = _long_words()
    cnt = Counter(df["word"])
    words = sorted(w for w, n in cnt.items() if n >= min_count)
    print(f"Clustering {len(words)} words (min count = {min_count})")

    word2cluster = cluster_words(words, min_cluster_size=min_cluster_size)
    labels = cluster_distinctive_labels(df, word2cluster)
    n_clusters = len(labels) - (1 if -1 in labels else 0)
    n_noise = sum(1 for v in word2cluster.values() if v == -1)
    print(f"  → {n_clusters} clusters + {n_noise} noise words "
          f"({100 * n_noise / len(words):.0f}%)")

    table = cluster_preference(df, word2cluster, labels)
    table.to_csv(config.OUTPUTS_DIR / "preferred_categories.csv", index=False)
    print(f"  retained {len(table)} clusters with >=25 word occurrences")
    plot_preference(table, show=show)
    plot_cluster_map(words, word2cluster, labels, show=show)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-count", type=int, default=3,
                    help="word must appear at least this many times in corpus")
    ap.add_argument("--min-cluster", type=int, default=12,
                    help="HDBSCAN min_cluster_size")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    main(min_count=args.min_count, min_cluster_size=args.min_cluster,
         show=not args.no_show)
