"""Zoom-in on the gaze metrics whose Kruskal-Wallis tested significant by DAT
tertile but whose Pearson/Spearman correlations were null — i.e. non-monotonic
effects:

    * gaze_dispersion_norm
    * gaze_entropy
    * prop_on_stim

For each metric: bar with 95 % CI per tertile + pairwise Mann–Whitney U
contrasts (Low–Mid, Mid–High, Low–High) so we can see *which* tertile is
driving the difference.

Usage:
    python -m analysis.figures.gaze_tertile_zoom
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

from .. import config, dat_helper, parse_events
from .eyetracking_by_creativity import (
    LABELS, PALETTE, _load_or_build_gaze, per_participant,
)

METRICS = [
    ("gaze_dispersion_norm", "Gaze dispersion (norm RMS)"),
    ("gaze_entropy",         "Gaze entropy"),
    ("prop_on_stim",         "Proportion of gaze on stimulus"),
]


def main(show: bool = True, dat_when: str = "before"):
    participants = dat_helper.main_cohort_scored()
    dat_col = "ref_dat_score" if dat_when == "before" else dat_helper.dat_column(dat_when)
    participants = participants.dropna(subset=[dat_col])
    gaze = _load_or_build_gaze()
    per_user = per_participant(gaze, participants, dat_col=dat_col)
    per_user = per_user.dropna(subset=["dat_tertile"])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    summary_rows = []
    for ax, (col, label) in zip(axes, METRICS):
        sub = per_user.dropna(subset=[col])
        means = sub.groupby("dat_tertile")[col].mean().reindex(LABELS)
        sems  = sub.groupby("dat_tertile")[col].sem().reindex(LABELS)
        ns    = sub.groupby("dat_tertile")[col].size().reindex(LABELS)
        xs = np.arange(len(LABELS))
        ax.bar(xs, means.values, yerr=1.96 * sems.values,
               color=[PALETTE[l] for l in LABELS],
               capsize=6, alpha=0.85, edgecolor="black", lw=0.5)
        # individual points (jittered)
        for i, lab in enumerate(LABELS):
            vals = sub[sub["dat_tertile"] == lab][col]
            jitter = (np.random.RandomState(i).rand(len(vals)) - 0.5) * 0.25
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       s=6, alpha=0.35, color="grey", zorder=1)
        ax.set_xticks(xs); ax.set_xticklabels([f"{l}\n(n={ns[l]})" for l in LABELS])
        ax.set_ylabel(label)
        # KW + per-pair MWU
        groups = [sub[sub["dat_tertile"] == lab][col].values for lab in LABELS]
        H, p_kw = kruskal(*groups)
        pairs = []
        for (a_i, a), (b_i, b) in [((0, "Low"), (1, "Mid")),
                                     ((1, "Mid"), (2, "High")),
                                     ((0, "Low"), (2, "High"))]:
            _, p_mwu = mannwhitneyu(groups[a_i], groups[b_i], alternative="two-sided")
            pairs.append((a, b, p_mwu))
            stars = "***" if p_mwu < .001 else "**" if p_mwu < .01 else "*" if p_mwu < .05 else ""
            if stars:
                # bracket between the two bars
                y_top = (means + 1.96 * sems).max()
                bump = 0.04 * (means.max() - means.min() + 1e-6)
                y_h = y_top + bump * (1 + pairs.index((a, b, p_mwu)))
                ax.plot([a_i, b_i], [y_h, y_h], color="black", lw=0.7)
                ax.text((a_i + b_i) / 2, y_h, stars,
                        ha="center", va="bottom", fontsize=11)
        ax.set_title(f"{label}\nKW p={p_kw:.2g}", fontsize=10)
        summary_rows.append({
            "metric": col, "kw_p": p_kw,
            **{f"{a}_vs_{b}_mwu_p": pp for a, b, pp in pairs},
            **{f"mean_{lab}": means[lab] for lab in LABELS},
        })

    plt.suptitle(f"Non-monotonic gaze metrics by DAT-{dat_when} tertile",
                 y=1.03, fontsize=14, weight="bold")
    plt.tight_layout()
    out = config.OUTPUTS_DIR / f"fig_gaze_tertile_zoom_{dat_when}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")
    if show:
        plt.show()
    plt.close()

    stats = pd.DataFrame(summary_rows)
    stats.to_csv(config.OUTPUTS_DIR / f"gaze_tertile_zoom_{dat_when}.csv", index=False)
    print(stats.to_string(index=False))
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dat", dest="dat_when", default="before",
                    choices=["before", "after", "raw_before"])
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    main(show=not args.no_show, dat_when=args.dat_when)
