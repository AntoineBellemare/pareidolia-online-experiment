"""Figure 2 — Semantic territory of collective pareidolia by creativity tertile.

t-SNE of every unique word ever spoken during the pareidolia task. Each word
is assigned to a creativity tertile (Low / Mid / High) by the *majority class*
of participants who used it. Per-tertile KDEs reveal whether high-creativity
participants explore semantically different territory.

Reproduces the notebook cell that begins
`# ╔══════════════════════════════════════════════════════════════════════════╗`
(the t-SNE / KDE / labels figure).

Usage:
    python -m analysis.figures.fig_semantic_territory
    python -m analysis.figures.fig_semantic_territory --dat after
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE

from .. import config, dat, dat_helper, embeddings, parse_events, style
from .fig_diversity_vs_dat import long_words_table  # already ASCII-filtered

# ─── plot config ───────────────────────────────────────────────────────────────

# Use the central DAT-tertile palette (sequential purple) so this figure
# never visually clashes with the FD-coloured semantic map.
PALETTE = {k: style.COLORS[k] for k in ("Low", "Mid", "High")}
LABELS = ["Low", "Mid", "High"]
KDE_LEVELS = (0.01, 0.50)
DOT_SIZE_RANGE = (2, 1500)
LABEL_THRESHOLD = 50
RNG_SEED = 42


# ─── helpers ───────────────────────────────────────────────────────────────────

def per_word_aggregate(words_df: pd.DataFrame, dat_col: str) -> pd.DataFrame:
    """One row per unique word: usage counts per DAT tertile, majority class."""
    # Tertile each (participant, word) row by the participant's DAT.
    words_df = words_df.dropna(subset=[dat_col]).copy()
    words_df["dat_class_row"] = pd.qcut(
        words_df[dat_col], 3, labels=LABELS, duplicates="drop"
    )

    cnt_tbl = (
        words_df
        .pivot_table(index="word", columns="dat_class_row",
                     values="user_id", aggfunc="size", fill_value=0)
        .rename_axis(columns=None)
        .reset_index()
    )

    emb = embeddings.load_word_embeddings()
    agg = (
        words_df.groupby("word", as_index=False)
                .agg(n_occ=("user_id", "size"))
                .merge(cnt_tbl, on="word")
                .merge(emb, on="word")
    )

    agg["dat_class"] = agg[LABELS].idxmax(axis=1)
    for c in LABELS:
        agg[f"prop_{c.lower()}"] = agg[c] / agg["n_occ"]
    return agg


def kde_patch(ax, sub: pd.DataFrame, colour: str,
              levels=KDE_LEVELS, gridsize: int = 200):
    if len(sub) < 6:
        return
    pts = sub[["x", "y"]].to_numpy().T
    kde = gaussian_kde(pts)
    xmin, ymin = pts.min(axis=1) - 0.5
    xmax, ymax = pts.max(axis=1) + 0.5
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, gridsize),
                         np.linspace(ymin, ymax, gridsize))
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    z_sort = np.sort(zz.ravel())[::-1]
    cdf = np.cumsum(z_sort) / z_sort.sum()
    thr = np.sort([z_sort[np.searchsorted(cdf, p)] for p in levels])
    ax.contourf(xx, yy, zz,
                levels=np.append(thr, zz.max()),
                colors=[colour] * len(thr), alpha=0.20, zorder=0)


# ─── plotting ──────────────────────────────────────────────────────────────────

def plot(agg: pd.DataFrame, dat_when: str, out_path=None, show: bool = True):
    style.apply()
    fig = plt.figure(figsize=(style.COL_1_5, style.COL_1_5 * 0.95))
    ax = plt.gca()
    for cls in LABELS:
        sub = agg[agg["dat_class"] == cls]
        ax.scatter(sub["x"], sub["y"], s=sub["dot_size"],
                   alpha=0.7, color=PALETTE[cls], edgecolor="none")
        kde_patch(ax, sub, PALETTE[cls])
    texts = []
    big = agg[agg["n_occ"] >= LABEL_THRESHOLD]
    for _, r in big.iterrows():
        texts.append(ax.text(r["x"], r["y"], r["word"],
                             fontsize=6.5, weight="bold",
                             ha="center", va="center",
                             color="black", zorder=3))
    if texts:
        adjust_text(texts, ax=ax,
                    expand_text=(1.05, 1.10),
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="gray"))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")
    handles = [Patch(facecolor=PALETTE[c], edgecolor="none", label=c) for c in LABELS]
    ax.legend(handles=handles, title="Creativity tertile", frameon=False,
              loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=7)
    fig.suptitle("Semantic territory by creativity tertile", y=0.98)
    if show: plt.show()
    style.savefig(fig, f"semantic_territory_{dat_when}")
    plt.close(fig)


# ─── entry-point ───────────────────────────────────────────────────────────────

def main(dat_when: str = "before", show: bool = True,
         auto_perplexity: bool = True, perplexity: int = 30,
         cohort: str = "notebook"):
    participants = dat_helper.select_cohort(cohort, dat_when)
    if dat_when == "before":
        dat_col = "ref_dat_score"
    else:
        dat_col = dat_helper.dat_column(dat_when)
    participants = participants.dropna(subset=[dat_col])

    words_df = long_words_table(participants)
    if dat_col not in words_df.columns:
        # `long_words_table` adds ref_dat_score + dat_after_score by default;
        # for any other column we need to merge it on user_id.
        words_df = words_df.merge(
            participants[["user_id", dat_col]], on="user_id", how="left",
        )
    agg = per_word_aggregate(words_df, dat_col)
    print(f"{len(agg):,} unique words")

    if auto_perplexity:
        perplexity = int(np.clip(len(agg) / 50, 5, 50))

    X = np.vstack(agg["embedding"].apply(lambda v: np.asarray(v)).to_numpy())
    coords = TSNE(
        2, metric="cosine", perplexity=perplexity,
        init="random", random_state=RNG_SEED, n_jobs=-1,
    ).fit_transform(X)
    agg[["x", "y"]] = coords
    agg["dot_size"] = np.interp(
        agg["n_occ"], (agg["n_occ"].min(), agg["n_occ"].max()), DOT_SIZE_RANGE,
    )

    # Save tabular output (embedding column dropped — heavy).
    out_csv = config.OUTPUTS_DIR / f"semantic_territory_{dat_when}.csv"
    agg.drop(columns=["embedding"]).to_csv(out_csv, index=False)

    out_png = config.OUTPUTS_DIR / f"fig_semantic_territory_{dat_when}.png"
    plot(agg, dat_when, out_path=out_png, show=show)
    return agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dat", dest="dat_when", default="before",
                    choices=["before", "after", "raw_before"])
    ap.add_argument("--cohort", default="notebook", choices=["notebook", "main"])
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    main(dat_when=args.dat_when, cohort=args.cohort, show=not args.no_show)
