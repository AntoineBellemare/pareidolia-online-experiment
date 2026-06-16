"""Companion to fig_semantic_territory — coloured by **FD level** instead of
DAT tertile. Each unique word is assigned to its majority-FD bucket
(the FD level where it was produced most often) and projected with t-SNE.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib.patches import Patch
from sklearn.manifold import TSNE

from .. import config, dat_helper, embeddings, parse_events, style
from .fig_diversity_vs_dat import long_words_table
from .fig_semantic_territory import (
    DOT_SIZE_RANGE, KDE_LEVELS, LABEL_THRESHOLD, RNG_SEED, kde_patch,
)


def per_word_by_fd(participants: pd.DataFrame) -> pd.DataFrame:
    """One row per unique word with per-FD counts and a majority-FD label."""
    import re as _re
    ASCII = _re.compile(r"^[A-Za-z]+$")
    edict = embeddings.embedding_dict()
    rows = []
    for _, r in participants.iterrows():
        for fd in config.FD_LEVELS:
            ws = r.get(f"{fd}_words")
            if ws is None: continue
            for w in list(ws):
                if (ASCII.fullmatch(w or "") and w in edict
                        and w not in config.SEMANTIC_STOPWORDS):
                    rows.append({"user_id": r["user_id"], "word": w,
                                 "fd_level": fd})
    df = pd.DataFrame(rows).drop_duplicates(["user_id", "word", "fd_level"])
    cnt = (df.pivot_table(index="word", columns="fd_level",
                          values="user_id", aggfunc="size", fill_value=0)
             .rename_axis(columns=None).reset_index())
    emb = embeddings.load_word_embeddings()
    agg = (df.groupby("word", as_index=False).agg(n_occ=("user_id", "size"))
             .merge(cnt, on="word").merge(emb, on="word"))
    agg["fd_class"] = agg[config.FD_LEVELS].idxmax(axis=1)
    return agg


def plot(agg: pd.DataFrame, out_name: str, show: bool = True):
    style.apply()
    fig = plt.figure(figsize=(style.COL_1_5, style.COL_1_5 * 0.95))
    ax = plt.gca()
    for fd in config.FD_LEVELS:
        sub = agg[agg["fd_class"] == fd]
        ax.scatter(sub["x"], sub["y"], s=sub["dot_size"],
                   alpha=0.7, color=style.COLORS[fd], edgecolor="none")
        kde_patch(ax, sub, style.COLORS[fd])

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
    handles = [Patch(facecolor=style.COLORS[fd], edgecolor="none", label=fd)
               for fd in config.FD_LEVELS]
    ax.legend(handles=handles, title="Stimulus FD", frameon=False,
              loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=7)
    fig.suptitle("Semantic territory by stimulus FD", y=0.98)
    if show: plt.show()
    style.savefig(fig, out_name)
    plt.close(fig)


def main(show: bool = True):
    p = dat_helper.notebook_cohort()
    agg = per_word_by_fd(p)
    print(f"{len(agg):,} unique words")
    perplexity = int(np.clip(len(agg) / 50, 5, 50))
    X = np.vstack(agg["embedding"].apply(np.asarray).to_numpy())
    coords = TSNE(2, metric="cosine", perplexity=perplexity,
                  init="random", random_state=RNG_SEED, n_jobs=-1).fit_transform(X)
    agg[["x", "y"]] = coords
    agg["dot_size"] = np.interp(agg["n_occ"],
                                (agg["n_occ"].min(), agg["n_occ"].max()),
                                DOT_SIZE_RANGE)
    plot(agg, "semantic_territory_by_fd", show=show)
    agg.drop(columns=["embedding"]).to_csv(
        config.OUTPUTS_DIR / "semantic_territory_by_fd.csv", index=False,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    main(show=not args.no_show)
