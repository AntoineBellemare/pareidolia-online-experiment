"""How does the stimulus fractal dimension change *how people look*?

Per (participant, FD) we average each gaze metric across that FD's trials,
then test the within-subject FD effect (Friedman + per-pair Wilcoxon).
Paper-ready figure: bar with 95 % CI on top, paired Δ scatter below.

Usage:
    python -m analysis.figures.fd_effect_eyetracking
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from .. import config, parse_events, style
from .eyetracking_by_creativity import _load_or_build_gaze

METRICS = [
    ("mean_fix_dur_ms",        "Fixation\nduration (ms)"),
    ("n_fixations",            "Fixations\nper trial"),
    ("fix_rate_hz",            "Fixation\nrate (Hz)"),
    ("mean_saccade_amp_norm",  "Saccade\namplitude"),
    ("scanpath_length_norm",   "Scanpath\nlength"),
    ("revisit_ratio",          "Revisit\nratio"),
    ("gaze_dispersion_norm",   "Gaze\ndispersion"),
    ("gaze_entropy",           "Gaze\nentropy"),
    ("prop_on_stim",           "Gaze on\nstimulus"),
    ("central_bias_norm",      "Distance\nfrom centre"),
]


def per_participant_per_fd(gaze: pd.DataFrame,
                           participants: pd.DataFrame) -> pd.DataFrame:
    g = gaze[~gaze["tracking_failed"]]
    g = g[g["user_id"].isin(participants["user_id"])]
    cols = [c for c, _ in METRICS]
    return (
        g.groupby(["user_id", "fd_level"])[cols]
         .mean()
         .reset_index()
    )


def within_subject(df: pd.DataFrame, col: str) -> dict:
    pivot = df.pivot(index="user_id", columns="fd_level",
                     values=col).dropna()
    pivot = pivot[[fd for fd in config.FD_LEVELS if fd in pivot.columns]]
    if len(pivot) < 5 or pivot.shape[1] < 2:
        return {"friedman_p": np.nan, "n": len(pivot)}
    chi, p = friedmanchisquare(*[pivot[c] for c in pivot.columns])
    pairwise = {}
    for a, b in [("FD12", "FD14"), ("FD12", "FD16"), ("FD14", "FD16")]:
        if a in pivot and b in pivot:
            _, pp = wilcoxon(pivot[a], pivot[b])
            pairwise[f"{a}_vs_{b}"] = float(pp)
    return {"friedman_chi2": float(chi), "friedman_p": float(p),
            "n": len(pivot), "pairwise": pairwise,
            "means": pivot.mean().to_dict(),
            "sems": pivot.sem().to_dict()}


def plot(df: pd.DataFrame, show: bool = True) -> pd.DataFrame:
    style.apply()
    # 10 metrics → 2 rows × 5 cols, each ~1 column wide.
    fig, axes = plt.subplots(
        2, 5, figsize=(style.COL_2, 3.4),
        gridspec_kw=dict(hspace=0.85, wspace=0.55),
    )
    rows = []
    fd_pal = [style.COLORS[fd] for fd in config.FD_LEVELS]
    for ax, (col, label) in zip(axes.flat, METRICS):
        s = within_subject(df, col)
        means = s.get("means", {})
        sems = s.get("sems", {})
        if not means:
            style.style_axis(ax, title=f"{label}\n(no data)")
            continue
        xs = np.arange(len(config.FD_LEVELS))
        ax.bar(xs,
               [means[fd] for fd in config.FD_LEVELS],
               yerr=[1.96 * sems[fd] for fd in config.FD_LEVELS],
               color=fd_pal, edgecolor="black", lw=0.4,
               capsize=2.5, alpha=0.85)
        ax.set_xticks(xs); ax.set_xticklabels(config.FD_LEVELS)
        # zoom y to span ±2.5 SE around means
        lo = min(means[fd] - 2.0 * sems[fd] for fd in config.FD_LEVELS)
        hi = max(means[fd] + 2.0 * sems[fd] for fd in config.FD_LEVELS)
        if hi > lo:
            ax.set_ylim(lo, hi)
        p = s["friedman_p"]
        style.style_axis(ax, title=label, ylabel="")
        pair_x = {"FD12": 0, "FD14": 1, "FD16": 2}
        comparisons = []
        for k, pw in s.get("pairwise", {}).items():
            a, _, b = k.partition("_vs_")
            if a in pair_x and b in pair_x and pw is not None:
                comparisons.append((pair_x[a], pair_x[b], pw))
        style.sig_brackets(ax, comparisons, fontsize=8)
        rows.append({"metric": col, **{f"mean_{fd}": means[fd] for fd in config.FD_LEVELS},
                     "friedman_p": p,
                     **{f"wilcox_{k}_p": v for k, v in s["pairwise"].items()},
                     "n": s["n"]})
    fig.suptitle("Stimulus FD shapes gaze dynamics", y=1.02)
    if show:
        plt.show()
    style.savefig(fig, "fd_effect_eyetracking")
    plt.close(fig)
    return pd.DataFrame(rows)


def main(show: bool = True):
    participants = parse_events.main_cohort()
    gaze = _load_or_build_gaze()
    df = per_participant_per_fd(gaze, participants)
    print(f"per-(user, FD) rows: {len(df):,}")
    stats = plot(df, show=show)
    stats.to_csv(config.OUTPUTS_DIR / "fd_effect_eyetracking.csv", index=False)
    print(stats.to_string(index=False))
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    main(show=not args.no_show)
