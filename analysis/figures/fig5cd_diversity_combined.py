"""Paper Fig 5cd, two-panel composite.

  (a) Perceptual diversity vs DAT, pooled across FD
  (b) Same, split across the three FD panels

Reuses the canonical metric (median pair-wise cosine distance of a
participant's unique percept embeddings) and the standard
±3 SD bivariate outlier trim.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr

from .. import config, dat_helper, embeddings, style
from .fig_diversity_vs_dat import (
    METRICS, METRIC_LABELS, long_words_table, per_participant_spread,
)
from .fig_diversity_vs_dat_by_fd import (
    per_participant_per_fd_spread, _drop_outliers,
)


def main(metric: str = "pairwise_median", show: bool = True):
    style.apply()
    participants = dat_helper.notebook_cohort().dropna(subset=["ref_dat_score"])
    dat_col = "ref_dat_score"

    # Panel (a): pooled
    words_df = long_words_table(participants)
    pooled = per_participant_spread(words_df, metric=metric, dat_col=dat_col)
    r_p, p_p = pearsonr(pooled[dat_col], pooled["spread"])

    # Panel (b): per-FD
    perfd = per_participant_per_fd_spread(participants, dat_col, metric)
    perfd = _drop_outliers(perfd, "spread")

    # Single row, 4 panels: (a) Pooled · FD12 · FD14 · FD16. All same size
    # so the slope is directly comparable between the pooled view and each FD.
    fig, axes = plt.subplots(
        1, 4, figsize=(style.COL_2, 2.6), sharey=True,
        gridspec_kw=dict(wspace=0.15),
    )

    # (a) pooled
    ax_a = axes[0]
    ax_a.scatter(pooled[dat_col], pooled["spread"],
                 s=3.5, alpha=0.55, color=style.COLORS["dot"], linewidths=0)
    sns.regplot(data=pooled, x=dat_col, y="spread", scatter=False,
                color=style.COLORS["fit"], ci=95,
                line_kws=dict(linewidth=1.2), ax=ax_a)
    style.style_axis(
        ax_a, title="(a) all FDs", xlabel="DAT",
        ylabel=f"Perceptual diversity\n({METRIC_LABELS[metric]})",
    )
    style.small_stat_annotation(ax_a, r_p, p_p, loc="upper left")

    # (b) per FD
    for i, fd in enumerate(config.FD_LEVELS):
        ax = axes[i + 1]
        sub = perfd[perfd["fd_level"] == fd]
        ax.scatter(sub[dat_col], sub["spread"],
                   s=3.5, alpha=0.55, color=style.COLORS[fd], linewidths=0)
        sns.regplot(data=sub, x=dat_col, y="spread", scatter=False,
                    color=style.COLORS["fit"], ci=95,
                    line_kws=dict(linewidth=1.2), ax=ax)
        r, p = pearsonr(sub[dat_col], sub["spread"])
        prefix = "(b) " if i == 0 else ""
        style.style_axis(ax, title=f"{prefix}{fd}", xlabel="DAT",
                          ylabel="")
        ax.set_ylabel("")
        style.small_stat_annotation(ax, r, p, loc="upper left")
    fig.suptitle("Perceptual diversity vs DAT", y=1.04)
    if show: plt.show()
    style.savefig(fig, "fig5cd_diversity")
    plt.close(fig)
    print(f"Pooled: n={len(pooled)} r={r_p:+.3f} p={p_p:.3g}")
    print("Per-FD rows:", len(perfd))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="pairwise_median", choices=list(METRICS))
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    main(metric=args.metric, show=not args.no_show)
