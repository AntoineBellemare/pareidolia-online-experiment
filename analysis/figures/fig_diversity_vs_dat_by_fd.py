"""Perceptual diversity vs DAT, computed *separately within each FD level*.

For each (participant, FD) cell, compute the semantic spread of the unique
words spoken at that FD, then correlate with the participant's DAT score.
Produces three scatter+regression panels (one per FD level) so we can see
whether the diversity↔creativity link is consistent across stimulus types.

Usage:
    python -m analysis.figures.fig_diversity_vs_dat_by_fd
    python -m analysis.figures.fig_diversity_vs_dat_by_fd --metric pairwise_mean --dat after
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr

import re as _re

from .. import config, dat_helper, embeddings, style
from .fig_diversity_vs_dat import METRICS, METRIC_LABELS

PALETTE = style.COLORS

# IMPORTANT: the BERT embeddings parquet contains 677 multi-word phrases
# (e.g. "world map", "baby hand reach") that were embedded as sentences.
# Sentence-vs-word vectors live in different regions of the space and
# inflate within-participant spread, diluting the DAT correlation. The
# notebook drops them via this same regex — do the same here for parity.
_ASCII_LETTERS = _re.compile(r"^[A-Za-z]+$")


def per_participant_per_fd_spread(
    participants: pd.DataFrame,
    dat_col: str,
    metric: str,
) -> pd.DataFrame:
    edict = embeddings.embedding_dict()
    fn = METRICS[metric]
    rows = []
    for _, r in participants.iterrows():
        if pd.isna(r.get(dat_col)):
            continue
        for fd in config.FD_LEVELS:
            words = r.get(f"{fd}_words")
            if words is None:
                continue
            words = [w for w in list(words)
                     if _ASCII_LETTERS.fullmatch(w or "")
                     and w in edict
                     and w not in config.SEMANTIC_STOPWORDS]
            if len(words) < 2:
                continue
            V = np.vstack([edict[w] for w in words])
            rows.append({
                "user_id": r["user_id"],
                "fd_level": fd,
                dat_col: r[dat_col],
                "spread": fn(V),
                "n_words": len(words),
            })
    return pd.DataFrame(rows)


def _drop_outliers(df: pd.DataFrame, col: str, sd: float = 3.0) -> pd.DataFrame:
    """Per-FD outlier removal — matches notebook's z <= sd convention."""
    out = []
    for _, sub in df.groupby("fd_level"):
        s = sub[col]
        if s.std() == 0 or len(sub) < 5:
            out.append(sub); continue
        z = (s - s.mean()).abs() / s.std()
        out.append(sub[z <= sd])
    return pd.concat(out, ignore_index=True)


def plot(df: pd.DataFrame, dat_col: str, metric: str,
         out_name: str = "diversity_vs_dat_by_fd",
         show: bool = True) -> pd.DataFrame:
    style.apply()
    fig, axes = plt.subplots(
        1, 3, figsize=(style.COL_1_5, 2.2), sharey=True, sharex=True,
        gridspec_kw=dict(wspace=0.18, top=0.78),
    )
    stats_rows = []
    xlabel = "DAT" if dat_col in {"ref_dat_score", "dat_before_glove"} else "DAT post"
    for ax, fd in zip(axes, config.FD_LEVELS):
        sub = df[df["fd_level"] == fd]
        if len(sub) < 5:
            ax.set_title(f"{fd}  (n={len(sub)} – too small)")
            continue
        ax.scatter(sub[dat_col], sub["spread"],
                   s=3.5, alpha=0.55, color=PALETTE[fd], linewidths=0)
        sns.regplot(data=sub, x=dat_col, y="spread", scatter=False,
                    color=style.COLORS["fit"], ci=95,
                    line_kws={"linewidth": 1.2}, ax=ax)
        r, p = pearsonr(sub[dat_col], sub["spread"])
        style.style_axis(ax, title=fd, xlabel=xlabel, ylabel="")
        ax.set_ylabel("")  # seaborn re-applies the column name; strip it
        style.small_stat_annotation(ax, r, p, loc="upper left")
        stats_rows.append({"fd_level": fd, "n": len(sub),
                           "pearson_r": r, "pearson_p": p})
    axes[0].set_ylabel(f"Perceptual diversity\n({METRIC_LABELS[metric]})")
    fig.suptitle("Perceptual diversity vs DAT, split by FD", y=1.02)
    if show:
        plt.show()
    style.savefig(fig, out_name)
    plt.close(fig)
    return pd.DataFrame(stats_rows)


def main(metric: str = "pairwise_median", dat_when: str = "before",
         show: bool = True, cohort: str = "notebook"):
    participants = dat_helper.select_cohort(cohort, dat_when)
    dat_col = "ref_dat_score" if dat_when == "before" else dat_helper.dat_column(dat_when)
    participants = participants.dropna(subset=[dat_col])

    df = per_participant_per_fd_spread(participants, dat_col, metric)
    df = _drop_outliers(df, "spread")
    print(f"Per-(participant, FD) rows: {len(df):,}  "
          f"(cohort={cohort}, dat={dat_when})")

    tag = f"{cohort}_{dat_when}_{metric}"
    stats = plot(df, dat_col, metric,
                 out_name=f"diversity_vs_dat_by_fd_{tag}", show=show)

    df.to_csv(
        config.OUTPUTS_DIR / f"diversity_vs_dat_by_fd_{tag}.csv", index=False,
    )
    print(stats.to_string(index=False))
    return df, stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="pairwise_median", choices=list(METRICS))
    ap.add_argument("--dat", dest="dat_when", default="before",
                    choices=["before", "after", "raw_before"])
    ap.add_argument("--cohort", default="notebook", choices=["notebook", "main"])
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    main(metric=args.metric, dat_when=args.dat_when,
         cohort=args.cohort, show=not args.no_show)
