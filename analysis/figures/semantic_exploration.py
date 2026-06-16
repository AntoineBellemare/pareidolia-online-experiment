"""Exploring how the semantic space differs across creativity groups and FD levels.

This is a *menu* of approaches — call sub-commands to render each piece.

Directions implemented
----------------------
1. centroid_drift   : where in BERT space does each tertile cluster?
                      Plot pairwise cosine distance between tertile centroids
                      and a t-SNE map coloured by tertile centroid coordinate
                      (already in fig_semantic_territory). Quantify: KL of
                      first-2 t-SNE coords distributions, centroid Euclidean
                      distance, silhouette of tertile clustering.
2. distinctive_words: TF-IDF-style "score" of words for each
                      (creativity tertile) and (FD level) - words said
                      disproportionately by one group vs others. Reports top
                      20 per cell. No figure: writes a CSV + a stacked
                      bar of the top-10 per tertile.
3. category_mix     : assign each word to a coarse semantic category by
                      nearest-neighbour cosine to a small set of seed
                      vectors (animal / human / face / object / nature /
                      abstract). Plot category proportions per tertile and
                      per FD level.
4. vocab_divergence : Jensen–Shannon divergence between Low/Mid/High word
                      frequency distributions overall and per FD.

Directions *not* implemented (left as ideas in the README):
* Topic modelling (LDA / NMF on word co-occurrence per FD).
* Word-embedding axis projection (project each percept onto canonical axes
  like animacy, abstractness, valence — needs labelled probe sets).
* Network analysis of co-occurrence within a participant.

Usage:
    python -m analysis.figures.semantic_exploration            # runs all
    python -m analysis.figures.semantic_exploration --only category_mix
"""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.stats import entropy

from .. import config, dat_helper, embeddings, parse_events, style


ASCII = re.compile(r"^[A-Za-z]+$")
LABELS = ["Low", "Mid", "High"]


# ─── shared: long words table with tertile and FD ────────────────────────────

def long_table(use_main_cohort: bool = False) -> pd.DataFrame:
    """One row per (participant, word, fd), with DAT-before tertile.

    ASCII-letter filter + stopword filter applied. Drops words not in
    the embedding table.
    """
    cohort = "main" if use_main_cohort else "notebook"
    p = dat_helper.select_cohort(cohort, "before")
    p = p.dropna(subset=["ref_dat_score"])
    edict = embeddings.embedding_dict()
    rows = []
    for _, r in p.iterrows():
        for fd in config.FD_LEVELS:
            words = r.get(f"{fd}_words")
            if words is None: continue
            for w in list(words):
                if not (ASCII.fullmatch(w or "") and w in edict
                        and w not in config.SEMANTIC_STOPWORDS):
                    continue
                rows.append({
                    "user_id": r["user_id"], "word": w, "fd_level": fd,
                    "ref_dat_score": r["ref_dat_score"],
                })
    df = pd.DataFrame(rows)
    df["dat_tertile"] = pd.qcut(df["ref_dat_score"], 3, labels=LABELS,
                                duplicates="drop")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 1. CENTROID DRIFT
# ═════════════════════════════════════════════════════════════════════════════

def _centroid_per_group(df: pd.DataFrame, group_col: str) -> dict:
    edict = embeddings.embedding_dict()
    out = {}
    for g, sub in df.groupby(group_col):
        V = np.vstack([edict[w] for w in sub["word"]])
        c = V.mean(0)
        out[g] = c / (np.linalg.norm(c) or 1)
    return out


def centroid_drift(df: pd.DataFrame, show: bool = True):
    style.apply()
    fig, axes = plt.subplots(
        1, 2, figsize=(style.COL_1_5, 2.0),
        gridspec_kw=dict(wspace=0.45),
    )
    # Per tertile
    cents_t = _centroid_per_group(df, "dat_tertile")
    # Per FD
    cents_fd = _centroid_per_group(df, "fd_level")

    def _heatmap(ax, cents, order, title):
        mat = np.zeros((len(order), len(order)))
        for i, a in enumerate(order):
            for j, b in enumerate(order):
                mat[i, j] = 1 - float(cents[a] @ cents[b])  # cosine distance
        sns.heatmap(mat, annot=True, fmt=".3f", cmap="rocket_r",
                    xticklabels=order, yticklabels=order,
                    cbar_kws=dict(label="cosine dist"),
                    annot_kws=dict(fontsize=6), ax=ax,
                    linewidths=0.5, linecolor="white")
        ax.set_title(title)

    _heatmap(axes[0], cents_t, LABELS, "DAT tertile centroids")
    _heatmap(axes[1], cents_fd, config.FD_LEVELS, "FD-level centroids")
    fig.suptitle("Semantic centroid drift between groups", y=1.02)
    if show: plt.show()
    style.savefig(fig, "semantic_centroid_drift")
    plt.close(fig)

    rows = []
    for grp in [("tertile", LABELS, cents_t), ("fd", config.FD_LEVELS, cents_fd)]:
        nm, order, cents = grp
        for i, a in enumerate(order):
            for j, b in enumerate(order):
                if i < j:
                    rows.append({"group": nm, "a": a, "b": b,
                                 "cosine_dist": 1 - float(cents[a] @ cents[b])})
    out = pd.DataFrame(rows)
    out.to_csv(config.OUTPUTS_DIR / "semantic_centroid_drift.csv", index=False)
    print(out.to_string(index=False))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 2. DISTINCTIVE WORDS (TF-IDF-like)
# ═════════════════════════════════════════════════════════════════════════════

def distinctive_words(df: pd.DataFrame, group_col: str = "dat_tertile",
                      top_k: int = 15, min_count: int = 3) -> pd.DataFrame:
    """For each group, score each word's specificity:
            score = log(1 + count_in_group) − log(1 + count_in_other)
    Drops words with count_in_group < min_count.
    """
    levels = list(df[group_col].dropna().unique())
    if group_col == "dat_tertile":
        levels = [l for l in LABELS if l in levels]
    elif group_col == "fd_level":
        levels = [l for l in config.FD_LEVELS if l in levels]
    cnt = df.groupby([group_col, "word"]).size().rename("n").reset_index()
    pivot = cnt.pivot(index="word", columns=group_col, values="n").fillna(0)
    rows = []
    for lvl in levels:
        others = [c for c in pivot.columns if c != lvl]
        score = np.log1p(pivot[lvl]) - np.log1p(pivot[others].sum(axis=1))
        # only words appearing at least min_count times in the group
        mask = pivot[lvl] >= min_count
        top = score[mask].sort_values(ascending=False).head(top_k)
        for w, s in top.items():
            rows.append({"group": lvl, "rank": rows.count({}) + 1,
                         "word": w, "score": float(s),
                         "n_in_group": int(pivot.loc[w, lvl]),
                         "n_in_other": int(pivot.loc[w, others].sum())})
    return pd.DataFrame(rows)


def plot_distinctive(df: pd.DataFrame, group_col: str = "dat_tertile",
                     show: bool = True):
    style.apply()
    dwords = distinctive_words(df, group_col)
    dwords.to_csv(
        config.OUTPUTS_DIR / f"distinctive_words_{group_col}.csv", index=False
    )
    levels = dwords["group"].drop_duplicates().tolist()
    fig, axes = plt.subplots(
        1, len(levels), figsize=(style.COL_2, 2.6), sharex=True,
        gridspec_kw=dict(wspace=0.55),
    )
    if len(levels) == 1: axes = [axes]
    for i, (ax, lvl) in enumerate(zip(axes, levels)):
        sub = dwords[dwords["group"] == lvl].sort_values("score", ascending=True)
        colour = style.COLORS.get(lvl, style.COLORS["dot"])
        ax.barh(sub["word"], sub["score"], color=colour, alpha=0.85,
                edgecolor="black", lw=0.3)
        style.style_axis(
            ax, title=str(lvl),
            xlabel="log-odds score" if i == len(axes) // 2 else "",
            ylabel="",
        )
    nice = {"dat_tertile": "creativity tertile", "fd_level": "FD level"}
    fig.suptitle(f"Most distinctive words by {nice.get(group_col, group_col)}",
                 y=1.02)
    if show: plt.show()
    style.savefig(fig, f"distinctive_words_{group_col}")
    plt.close(fig)
    return dwords


# ═════════════════════════════════════════════════════════════════════════════
# 3. CATEGORY MIX  (semantic seed-based labelling)
# ═════════════════════════════════════════════════════════════════════════════

SEEDS = {
    "face/body":   ["face", "eye", "head", "mouth", "person", "body", "hand"],
    "animal":      ["dog", "cat", "bird", "fish", "horse", "elephant", "lion"],
    "human":       ["man", "woman", "girl", "boy", "baby", "human", "child"],
    "object":      ["bottle", "table", "chair", "car", "tool", "lamp", "cup"],
    "nature":      ["tree", "mountain", "river", "flower", "rock", "forest", "sky"],
    "creature":    ["dragon", "alien", "monster", "ghost", "creature", "demon", "angel"],
    "abstract":    ["shape", "figure", "form", "pattern", "symbol", "blob", "swirl"],
    "food":        ["bread", "fruit", "apple", "meat", "cake", "vegetable", "soup"],
}


def _classify_words(words: list[str]) -> pd.Series:
    edict = embeddings.embedding_dict()
    seed_vecs = {}
    for cat, seeds in SEEDS.items():
        V = np.vstack([edict[w] for w in seeds if w in edict])
        c = V.mean(0)
        seed_vecs[cat] = c / (np.linalg.norm(c) or 1)
    cats = list(seed_vecs.keys())
    W = np.vstack([edict[w] for w in words])
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    S = np.vstack([seed_vecs[c] for c in cats])
    sims = W @ S.T
    idx = sims.argmax(axis=1)
    return pd.Series([cats[i] for i in idx], index=words, name="category")


def category_mix(df: pd.DataFrame, show: bool = True):
    style.apply()
    cats = _classify_words(df["word"].unique().tolist())
    df = df.merge(cats.rename("category"), left_on="word", right_index=True)
    # Stack proportions by group
    fig, axes = plt.subplots(
        1, 2, figsize=(style.COL_2, 3.0),
        gridspec_kw=dict(wspace=0.35),
    )
    cat_order = list(SEEDS.keys())
    palette = sns.color_palette("Set2", len(cat_order))

    for ax, group_col in zip(axes, ["dat_tertile", "fd_level"]):
        levels = LABELS if group_col == "dat_tertile" else config.FD_LEVELS
        pivot = (df.groupby([group_col, "category"]).size()
                   .unstack(fill_value=0).reindex(levels))
        prop = pivot.div(pivot.sum(axis=1), axis=0)[cat_order]
        bottom = np.zeros(len(prop))
        xs = np.arange(len(prop))
        for cat, col in zip(cat_order, palette):
            ax.bar(xs, prop[cat], bottom=bottom, color=col,
                   edgecolor="white", lw=0.4, label=cat)
            bottom += prop[cat].values
        ax.set_xticks(xs); ax.set_xticklabels(prop.index)
        style.style_axis(ax, title=group_col.replace("_", " ").title(),
                         ylabel="Proportion", xlabel="")
        ax.set_ylim(0, 1)
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle("Semantic category mix of percepts", y=1.02)
    if show: plt.show()
    style.savefig(fig, "semantic_category_mix")
    plt.close(fig)

    df_summary = (df.groupby(["dat_tertile", "category"]).size()
                    .unstack(fill_value=0))
    df_summary.to_csv(config.OUTPUTS_DIR / "semantic_category_by_tertile.csv")
    df_fd_summary = (df.groupby(["fd_level", "category"]).size()
                       .unstack(fill_value=0))
    df_fd_summary.to_csv(config.OUTPUTS_DIR / "semantic_category_by_fd.csv")
    return df_summary, df_fd_summary


# ═════════════════════════════════════════════════════════════════════════════
# 4. VOCABULARY DIVERGENCE (Jensen–Shannon)
# ═════════════════════════════════════════════════════════════════════════════

def _jsd(p, q):
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def vocab_divergence(df: pd.DataFrame, show: bool = True):
    style.apply()
    # Word freq distributions per tertile and per FD
    cnt_t = df.groupby(["dat_tertile", "word"]).size().unstack(fill_value=0)
    cnt_fd = df.groupby(["fd_level", "word"]).size().unstack(fill_value=0)

    def _div_matrix(cnt: pd.DataFrame, order: list[str]) -> pd.DataFrame:
        m = pd.DataFrame(index=order, columns=order, dtype=float)
        for a in order:
            for b in order:
                m.loc[a, b] = _jsd(cnt.loc[a] + 1e-9, cnt.loc[b] + 1e-9)
        return m

    m_t = _div_matrix(cnt_t, LABELS)
    m_fd = _div_matrix(cnt_fd, config.FD_LEVELS)

    fig, axes = plt.subplots(
        1, 2, figsize=(style.COL_1_5, 2.0),
        gridspec_kw=dict(wspace=0.45),
    )
    for ax, mat, title in [(axes[0], m_t, "DAT tertile JSD"),
                            (axes[1], m_fd, "FD-level JSD")]:
        sns.heatmap(mat.astype(float), annot=True, fmt=".3f",
                    cmap="mako_r", cbar_kws=dict(label="JSD (bits)"),
                    annot_kws=dict(fontsize=6), ax=ax,
                    linewidths=0.5, linecolor="white")
        ax.set_title(title)
    fig.suptitle("Vocabulary distribution divergence", y=1.02)
    if show: plt.show()
    style.savefig(fig, "semantic_vocab_jsd")
    plt.close(fig)

    rows = []
    for a in LABELS:
        for b in LABELS:
            if a < b:
                rows.append({"group": "tertile", "a": a, "b": b,
                             "jsd": float(m_t.loc[a, b])})
    for a in config.FD_LEVELS:
        for b in config.FD_LEVELS:
            if a < b:
                rows.append({"group": "fd", "a": a, "b": b,
                             "jsd": float(m_fd.loc[a, b])})
    out = pd.DataFrame(rows)
    out.to_csv(config.OUTPUTS_DIR / "semantic_vocab_jsd.csv", index=False)
    print(out.to_string(index=False))
    return out


# ─── runner ──────────────────────────────────────────────────────────────────

ANALYSES = {
    "centroid_drift":     centroid_drift,
    "distinctive_tertile": lambda df, show: plot_distinctive(df, "dat_tertile", show),
    "distinctive_fd":      lambda df, show: plot_distinctive(df, "fd_level", show),
    "category_mix":       category_mix,
    "vocab_divergence":   vocab_divergence,
}


def main(only: list[str] | None = None, show: bool = True):
    df = long_table(use_main_cohort=False)
    df = df.dropna(subset=["dat_tertile"])
    print(f"Long words table: {len(df):,} rows, "
          f"{df['user_id'].nunique()} users, {df['word'].nunique()} unique words")
    selected = only or list(ANALYSES.keys())
    for name in selected:
        print(f"\n--- {name} ---")
        ANALYSES[name](df, show=show)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(ANALYSES))
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    main(only=args.only, show=not args.no_show)
