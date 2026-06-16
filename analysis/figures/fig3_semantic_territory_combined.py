"""Paper Fig 3, two-panel composite.

    (a) Semantic territory coloured by DAT tertile
    (b) Same projection coloured by stimulus FD level

Both panels share the same t-SNE embedding of every unique percept word
in the notebook cohort, so positions are directly comparable across the
two views; only the colour mapping changes.
"""
from __future__ import annotations

import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.patches import Patch
from sklearn.manifold import TSNE

from .. import config, dat_helper, embeddings, style
from .fig_semantic_territory import (
    DOT_SIZE_RANGE, LABEL_THRESHOLD, RNG_SEED, kde_patch,
)

ASCII = re.compile(r"^[A-Za-z]+$")
LABELS_DAT = ["Low", "Mid", "High"]


def _build_words():
    """Per-word table with majority DAT-tertile and majority FD-level
    labels and a single t-SNE position."""
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
    long = pd.DataFrame(rows)
    long["dat_tertile"] = pd.qcut(long["ref_dat_score"], 3,
                                    labels=LABELS_DAT, duplicates="drop")
    # Per-word: counts per FD + per tertile, majority of each
    cnt_fd = (long.pivot_table(index="word", columns="fd_level",
                                values="user_id", aggfunc="size",
                                fill_value=0)
                  .rename_axis(columns=None))
    cnt_t  = (long.pivot_table(index="word", columns="dat_tertile",
                                values="user_id", aggfunc="size",
                                fill_value=0)
                  .rename_axis(columns=None))
    agg = (long.groupby("word", as_index=False)
                .agg(n_occ=("user_id", "size"))
                .merge(cnt_fd.reset_index(), on="word")
                .merge(cnt_t.reset_index(), on="word"))
    agg["fd_class"]      = agg[config.FD_LEVELS].idxmax(axis=1)
    agg["dat_class"]     = agg[LABELS_DAT].idxmax(axis=1)
    # Add embedding
    emb_df = embeddings.load_word_embeddings()
    agg = agg.merge(emb_df, on="word")
    return agg


def _draw_map(ax, agg: pd.DataFrame, group_col: str,
              order: list[str], palette: dict, legend_title: str):
    for g in order:
        sub = agg[agg[group_col] == g]
        ax.scatter(sub["x"], sub["y"], s=sub["dot_size"],
                   alpha=0.7, color=palette[g], edgecolor="none")
        kde_patch(ax, sub, palette[g])
    texts = []
    big = agg[agg["n_occ"] >= LABEL_THRESHOLD]
    for _, r in big.iterrows():
        texts.append(ax.text(r["x"], r["y"], r["word"],
                             fontsize=6, weight="bold",
                             ha="center", va="center",
                             color="black", zorder=3))
    if texts:
        adjust_text(texts, ax=ax,
                    expand_text=(1.05, 1.10),
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="gray"))
    ax.set_xticks([]); ax.set_yticks([])
    handles = [Patch(facecolor=palette[g], edgecolor="none", label=g)
               for g in order]
    ax.legend(handles=handles, title=legend_title, frameon=False,
              loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=7)


def main(show: bool = True):
    style.apply()
    agg = _build_words()
    print(f"{len(agg):,} unique words")

    # Shared t-SNE projection
    perplexity = int(np.clip(len(agg) / 50, 5, 50))
    X = np.vstack(agg["embedding"].apply(np.asarray).to_numpy())
    coords = TSNE(2, metric="cosine", perplexity=perplexity,
                   init="random", random_state=RNG_SEED,
                   n_jobs=-1).fit_transform(X)
    agg[["x", "y"]] = coords
    agg["dot_size"] = np.interp(
        agg["n_occ"], (agg["n_occ"].min(), agg["n_occ"].max()),
        DOT_SIZE_RANGE,
    )

    fig, axes = plt.subplots(
        2, 1, figsize=(style.COL_1_5, style.COL_1_5 * 1.9),
        gridspec_kw=dict(hspace=0.1),
    )
    palette_dat = {k: style.COLORS[k] for k in LABELS_DAT}
    palette_fd  = {k: style.COLORS[k] for k in config.FD_LEVELS}
    _draw_map(axes[0], agg, "dat_class",  LABELS_DAT,
               palette_dat, "Creativity tertile")
    axes[0].set_title("(a) by creativity tertile", loc="left", fontsize=8)
    _draw_map(axes[1], agg, "fd_class",   config.FD_LEVELS,
               palette_fd,  "Stimulus FD")
    axes[1].set_title("(b) by stimulus FD", loc="left", fontsize=8)
    fig.suptitle("Semantic territory of collective pareidolia", y=0.99)
    if show: plt.show()
    style.savefig(fig, "fig3_semantic_territory")
    plt.close(fig)
    agg.drop(columns=["embedding"]).to_csv(
        config.OUTPUTS_DIR / "fig3_semantic_territory.csv", index=False,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    main(show=not args.no_show)
