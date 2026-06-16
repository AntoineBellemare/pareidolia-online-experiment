"""Standard eye-tracking metrics contrasted across three axes:

    (A) pareidolia vs no-pareidolia trials — within-subject paired
    (B) stimulus FD level — within-subject FD12 / FD14 / FD16
    (C) trait creativity (DAT) — between-subject Pearson

Only trials surviving the tracking-failure filter are used. Outlier
trimming (±3 SD bivariate) is applied to the DAT correlations.

A *pareidolia* trial = participant submitted ≥ 1 word; a *no-pareidolia*
trial = 0 words. The pareidolia/no-pareidolia contrast is restricted to
the participants who have ≥ 2 trials of each kind.

Metrics shown (chosen as the most interpretable + non-collinear subset):
    * mean fixation duration (ms)
    * fixations per trial
    * scanpath length (norm)
    * gaze dispersion (norm RMS)
    * gaze entropy (bits)
    * proportion of gaze on stimulus ROI
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare, pearsonr, wilcoxon

from .. import config, dat_helper, outliers, parse_events, style
from .eyetracking_by_creativity import _load_or_build_gaze


# Mix of well-known (top half) and cutting-edge (bottom half) ET metrics:
#   * fixation duration / count : classical
#   * fix-duration CV : within-trial variability
#   * saccade direction entropy : how varied the scan directions are
#   * scanpath length / gaze dispersion / gaze entropy / Gini : spread
#   * recurrence rate : RQA-style revisits
# Paper-ready set, mixing classical and cutting-edge metrics
METRICS = [
    ("n_fixations",            "Fixations / trial"),
    ("mean_fix_dur_ms",        "Fixation dur. (ms)"),
    ("scanpath_length_norm",   "Scanpath length"),
    ("gaze_entropy",           "Gaze entropy (bits)"),
    ("gini_gaze",              "Gini concentration"),
    ("recurrence_rate",        "Recurrence rate"),
]
PAIR_X = {"FD12": 0, "FD14": 1, "FD16": 2}
LABELS_DAT = ["Low", "Mid", "High"]


# ─── shared trial-level table ────────────────────────────────────────────────

def _trial_table() -> pd.DataFrame:
    gaze = _load_or_build_gaze()
    gaze = gaze[~gaze["tracking_failed"]]
    trials = parse_events.cached_trials()
    trials = trials[["user_id", "trial_index", "fd_level", "n_words"]]
    df = gaze.merge(trials, on=["user_id", "trial_index", "fd_level"],
                    how="inner")
    df["pareidolia"] = (df["n_words"] > 0).astype(int)
    return df


# ─── plotting helpers ────────────────────────────────────────────────────────

def _bar_with_brackets(ax, pivot: pd.DataFrame, colors: list[str],
                       comparisons: list[tuple[int, int, float]]) -> None:
    means = pivot.mean(); sems = pivot.sem()
    xs = np.arange(len(pivot.columns))
    ax.bar(xs, means.values, yerr=1.96 * sems.values,
           color=colors, capsize=2.5, alpha=0.85, edgecolor="black", lw=0.4)
    ax.set_xticks(xs); ax.set_xticklabels(pivot.columns)
    ci = 1.96 * sems
    lo = float((means - ci).min()); hi = float((means + ci).max())
    pad = (hi - lo) * 0.25 if hi > lo else 0.1
    ax.set_ylim(lo - pad, hi + pad)
    style.sig_brackets(ax, comparisons)


# ─── three contrasts ─────────────────────────────────────────────────────────

def contrast_pareidolia(df: pd.DataFrame, axes_row,
                         min_trials_each: int = 2) -> pd.DataFrame:
    """Per-participant mean per metric for pareidolia vs no-pareidolia trials.

    Only participants with at least `min_trials_each` of each kind are kept.
    """
    rows = []
    grp = df.groupby(["user_id", "pareidolia"])
    counts = grp.size().unstack(fill_value=0)
    counts.columns = ["no_pareidolia", "pareidolia"]
    keep = counts[(counts["no_pareidolia"] >= min_trials_each) &
                  (counts["pareidolia"]   >= min_trials_each)].index
    sub = df[df["user_id"].isin(keep)]
    print(f"  pareidolia vs no-pareidolia: {len(keep)} participants with "
          f"≥{min_trials_each} of each")

    for ax, (col, label) in zip(axes_row, METRICS):
        per_user = (sub.groupby(["user_id", "pareidolia"])[col]
                       .mean().unstack())
        per_user.columns = ["no", "yes"]
        per_user = per_user.dropna()
        if len(per_user) < 5:
            ax.set_title(f"{label}\n(n={len(per_user)})")
            continue
        try:
            stat, p = wilcoxon(per_user["yes"], per_user["no"])
        except ValueError:
            stat, p = np.nan, np.nan
        pivot = per_user.rename(columns={"no": "no perc.", "yes": "perc."})
        pivot = pivot[["no perc.", "perc."]]
        _bar_with_brackets(
            ax, pivot,
            colors=["#bbbbbb", style.COLORS["High"]],
            comparisons=[(0, 1, p)],
        )
        style.style_axis(ax, title="", ylabel=label, xlabel="")
        rows.append({"metric": col, "n_subjects": len(per_user),
                      "mean_no": float(per_user["no"].mean()),
                      "mean_yes": float(per_user["yes"].mean()),
                      "wilcoxon_p": float(p)})
    return pd.DataFrame(rows)


def contrast_fd(df: pd.DataFrame, axes_row) -> pd.DataFrame:
    rows = []
    for ax, (col, label) in zip(axes_row, METRICS):
        per_user = (df.groupby(["user_id", "fd_level"])[col]
                      .mean().unstack().dropna())
        per_user = per_user[[fd for fd in config.FD_LEVELS if fd in per_user.columns]]
        if len(per_user) < 5 or per_user.shape[1] < 2:
            ax.set_title(f"{label}\n(n={len(per_user)})")
            continue
        chi, p_fr = friedmanchisquare(*[per_user[c] for c in per_user.columns])
        # Pairwise
        comparisons = []
        for a, b in [("FD12", "FD14"), ("FD14", "FD16"), ("FD12", "FD16")]:
            if a in per_user and b in per_user:
                _, pw = wilcoxon(per_user[a], per_user[b])
                comparisons.append((PAIR_X[a], PAIR_X[b], float(pw)))
        _bar_with_brackets(
            ax, per_user,
            colors=[style.COLORS[fd] for fd in per_user.columns],
            comparisons=comparisons,
        )
        style.style_axis(ax, title="", ylabel=label, xlabel="")
        rows.append({"metric": col, "n_subjects": len(per_user),
                      "friedman_p": float(p_fr),
                      **{f"mean_{fd}": float(per_user[fd].mean())
                         for fd in per_user.columns}})
    return pd.DataFrame(rows)


def contrast_dat(df: pd.DataFrame, axes_row,
                  participants: pd.DataFrame) -> pd.DataFrame:
    dat_col = "ref_dat_score"
    per_user = (df.groupby("user_id")
                   [[c for c, _ in METRICS]].mean()
                   .reset_index())
    per_user = per_user.merge(
        participants[["user_id", dat_col]], on="user_id", how="inner",
    )
    rows = []
    for ax, (col, label) in zip(axes_row, METRICS):
        sub, _ = outliers.trim_sd(per_user, [col, dat_col], sd=3.0)
        if len(sub) < 10:
            ax.set_title(f"{label}\n(n={len(sub)})")
            continue
        r, p = pearsonr(sub[dat_col], sub[col])
        ax.scatter(sub[dat_col], sub[col], s=3.5, alpha=0.55,
                   color=style.COLORS["dot"], linewidths=0)
        sns.regplot(data=sub, x=dat_col, y=col, scatter=False,
                    color=style.COLORS["fit"], ci=95,
                    line_kws=dict(linewidth=1.0), ax=ax)
        style.style_axis(ax, title="", ylabel=label, xlabel="DAT")
        style.small_stat_annotation(ax, r, p, loc="upper left")
        rows.append({"metric": col, "n_subjects": len(sub),
                      "pearson_r": r, "pearson_p": p})
    return pd.DataFrame(rows)


# ─── runner ──────────────────────────────────────────────────────────────────

def main(show: bool = True):
    df = _trial_table()
    print(f"Total valid trials (tracking OK): {len(df):,}")
    print(f"  pareidolia (n_words ≥ 1): {(df['pareidolia'] == 1).sum():,}")
    print(f"  no-pareidolia (n_words 0): {(df['pareidolia'] == 0).sum():,}")

    p = dat_helper.notebook_cohort()

    style.apply()
    # Layout: 3 rows of metric panels with extra width on the left for
    # row-label panels (dedicated narrow axes that hold the (A)/(B)/(C)
    # captions, ensuring they never overlap the data axes).
    n_m = len(METRICS)
    fig = plt.figure(figsize=(style.COL_2, 5.4))
    # Wider label column so the (A)/(B)/(C) text sits cleanly to the LEFT
    # of the first plot's y-axis ticks; explicit gap between label column
    # and the metric columns prevents any overlap on the leftmost panel.
    gs = fig.add_gridspec(
        3, n_m + 1,
        width_ratios=[0.45] + [1.0] * n_m,
        hspace=0.45, wspace=0.42,
        left=0.02, right=0.985, top=0.94, bottom=0.08,
    )
    label_axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    metric_axes = [
        [fig.add_subplot(gs[i, j + 1]) for j in range(n_m)]
        for i in range(3)
    ]
    for ax, txt in zip(label_axes,
                        ["(A) pareidolia\nvs no-pareidolia",
                         "(B) stimulus FD",
                         "(C) DAT (continuous)"]):
        # Right-aligned at the right edge of the label column so the text
        # ends just before the gap to the first plot.
        ax.text(0.95, 0.5, txt, transform=ax.transAxes,
                ha="right", va="center", fontsize=8, weight="bold")
        ax.set_axis_off()

    print("\n=== A. Pareidolia vs no-pareidolia (within-subject) ===")
    rows_p = contrast_pareidolia(df, metric_axes[0])
    print(rows_p.to_string(index=False))

    print("\n=== B. Stimulus FD (within-subject) ===")
    rows_fd = contrast_fd(df, metric_axes[1])
    print(rows_fd.to_string(index=False))

    print("\n=== C. DAT (between-subject) ===")
    rows_d = contrast_dat(df, metric_axes[2], p)
    print(rows_d.to_string(index=False))

    fig.suptitle("Eye-tracking metrics across three contrasts", y=0.98)
    if show: plt.show()
    style.savefig(fig, "eyetracking_contrasts")
    plt.close(fig)

    # CSV summaries
    rows_p.to_csv(config.OUTPUTS_DIR / "et_contrast_pareidolia.csv", index=False)
    rows_fd.to_csv(config.OUTPUTS_DIR / "et_contrast_fd.csv", index=False)
    rows_d.to_csv(config.OUTPUTS_DIR / "et_contrast_dat.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    main(show=not args.no_show)
