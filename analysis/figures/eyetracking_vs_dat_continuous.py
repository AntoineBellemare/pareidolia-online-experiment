"""Eye-tracking metrics × creativity (DAT score) — continuous, Nature-style.

Per participant we average each gaze metric across their trials (rejecting
WebGazer "failed-tracking" trials). We then:
  * scatter each metric vs continuous DAT score, regression line + 95 % CI
  * report Pearson + Spearman (and Pearson partial-correlation controlling for
    n_trials_with_gaze, screen aspect)
  * compute the same correlations within each FD level — if eye behaviour
    differs by complexity, that may unmask creativity effects too

Usage:
    python -m analysis.figures.eyetracking_vs_dat_continuous
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

from .. import config, dat_helper, outliers, parse_events, style
from .eyetracking_by_creativity import _load_or_build_gaze


METRICS = [
    ("mean_fix_dur_ms",        "Fixation duration (ms)"),
    ("fix_rate_hz",            "Fixation rate (Hz)"),
    ("n_fixations",            "Fixations per trial"),
    ("mean_saccade_amp_norm",  "Saccade amplitude (norm)"),
    ("scanpath_length_norm",   "Scanpath length (norm)"),
    ("revisit_ratio",          "Revisit ratio"),
    ("gaze_dispersion_norm",   "Gaze dispersion (norm)"),
    ("gaze_entropy",           "Gaze entropy (bits)"),
    ("prop_on_stim",           "Gaze on stimulus"),
    ("central_bias_norm",      "Distance from centre (norm)"),
]


def _per_user(gaze: pd.DataFrame, participants: pd.DataFrame,
              dat_col: str) -> pd.DataFrame:
    g = gaze[gaze["user_id"].isin(participants["user_id"])].copy()
    g = g[~g["tracking_failed"]]
    cols = [c for c, _ in METRICS]
    out = g.groupby("user_id")[cols].mean().reset_index()
    n_trials = g.groupby("user_id").size().rename("n_trials_with_gaze")
    aspect   = g.groupby("user_id").apply(
        lambda d: (d["screen_w_px"] / d["screen_h_px"]).mean()
    ).rename("screen_aspect")
    out = out.merge(n_trials, on="user_id").merge(aspect, on="user_id")
    out = out.merge(participants[["user_id", dat_col]], on="user_id")
    return out.dropna(subset=[dat_col])


def _partial_pearson(x: pd.Series, y: pd.Series,
                     controls: pd.DataFrame) -> tuple[float, float]:
    """Residualise x and y on `controls`, then Pearson r between residuals."""
    from numpy.linalg import lstsq
    df = pd.concat([x, y, controls], axis=1).dropna()
    if len(df) < 10:
        return float("nan"), float("nan")
    X = np.column_stack([np.ones(len(df)), df[controls.columns].values])
    bx, *_ = lstsq(X, df[x.name].values, rcond=None)
    by, *_ = lstsq(X, df[y.name].values, rcond=None)
    res_x = df[x.name].values - X @ bx
    res_y = df[y.name].values - X @ by
    r, p = pearsonr(res_x, res_y)
    return float(r), float(p)


def plot(per_user: pd.DataFrame, dat_col: str, show: bool = True):
    style.apply()
    fig, axes = plt.subplots(
        2, 5, figsize=(style.COL_2, 3.6),
        gridspec_kw=dict(hspace=0.85, wspace=0.55),
    )
    stats_rows = []
    for ax, (col, label) in zip(axes.flat, METRICS):
        sub, n_drop = outliers.trim_sd(per_user, [col, dat_col], sd=3.0)
        if n_drop:
            print(f"  [{col}] dropped {n_drop} ±3 SD outliers")
        if len(sub) < 10:
            style.style_axis(ax, title=f"{label}\nn={len(sub)} (too small)")
            continue
        ax.scatter(sub[dat_col], sub[col], s=3.5, alpha=0.55,
                   color=style.COLORS["dot"], linewidths=0)
        sns.regplot(data=sub, x=dat_col, y=col, scatter=False,
                    color=style.COLORS["fit"], ci=95,
                    line_kws={"linewidth": 1.0}, ax=ax)
        r, p = pearsonr(sub[dat_col], sub[col])
        rho, p_rho = spearmanr(sub[dat_col], sub[col])  # CSV only
        pr, pp = _partial_pearson(sub[dat_col].rename(dat_col),
                                  sub[col].rename(col),
                                  sub[["n_trials_with_gaze", "screen_aspect"]])
        title = f"{label}\n{style.short_stats(r, p)}"
        style.style_axis(ax, title=title, xlabel="DAT (pre)", ylabel="")
        stats_rows.append({
            "metric": col, "n": len(sub),
            "pearson_r": r, "pearson_p": p,
            "spearman_rho": rho, "spearman_p": p_rho,
            "partial_r": pr, "partial_p": pp,
        })
    fig.suptitle("Eye tracking metrics vs creativity (DAT, pre)", y=1.00)
    if show:
        plt.show()
    style.savefig(fig, "eyetracking_vs_dat_continuous")
    plt.close(fig)
    return pd.DataFrame(stats_rows)


def per_fd_correlations(gaze: pd.DataFrame, participants: pd.DataFrame,
                        dat_col: str) -> pd.DataFrame:
    """For each FD level, compute per-user means then Pearson vs DAT."""
    g = gaze[~gaze["tracking_failed"]]
    g = g[g["user_id"].isin(participants["user_id"])]
    cols = [c for c, _ in METRICS]
    rows = []
    for fd in config.FD_LEVELS:
        sub = g[g["fd_level"] == fd]
        per_u = sub.groupby("user_id")[cols].mean().reset_index().merge(
            participants[["user_id", dat_col]], on="user_id"
        ).dropna(subset=[dat_col])
        for col in cols:
            d, _ = outliers.trim_sd(per_u, [col, dat_col], sd=3.0)
            if len(d) < 10:
                continue
            r, p = pearsonr(d[dat_col], d[col])
            rows.append({"fd_level": fd, "metric": col, "n": len(d),
                         "pearson_r": r, "pearson_p": p})
    return pd.DataFrame(rows)


def plot_per_fd_heatmap(stats: pd.DataFrame, show: bool = True):
    """Heatmap of Pearson r for each (metric, FD) cell."""
    style.apply()
    if stats.empty:
        return
    mat = stats.pivot(index="metric", columns="fd_level", values="pearson_r")
    pmat = stats.pivot(index="metric", columns="fd_level", values="pearson_p")
    # Order rows by the average |r|, descending.
    mat = mat.reindex([m for m, _ in METRICS])
    pmat = pmat.reindex(mat.index)

    fig, ax = plt.subplots(figsize=(style.COL_1, 3.0))
    vmax = max(0.1, np.nanmax(np.abs(mat.values)))
    sns.heatmap(mat, annot=False, cmap="RdBu_r", center=0,
                vmin=-vmax, vmax=vmax, cbar_kws=dict(label="Pearson r"),
                ax=ax, linewidths=0.4, linecolor="white")
    # Add stars for significance
    for i, m in enumerate(mat.index):
        for j, fd in enumerate(mat.columns):
            r = mat.iloc[i, j]
            p = pmat.iloc[i, j]
            if pd.isna(r): continue
            stars = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
            txt = f"{r:+.2f}{stars}"
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center",
                    fontsize=5.5, color="black" if abs(r) < vmax * 0.6 else "white")
    nice = {c: lab for c, lab in METRICS}
    ax.set_yticklabels([nice[c] for c in mat.index], rotation=0)
    ax.set_xlabel(""); ax.set_ylabel("")
    fig.suptitle("Gaze × DAT, per FD level", y=0.98)
    if show:
        plt.show()
    style.savefig(fig, "eyetracking_vs_dat_per_fd_heatmap")
    plt.close(fig)


def main(show: bool = True):
    participants = parse_events.main_cohort()
    gaze = _load_or_build_gaze()
    per_user = _per_user(gaze, participants, "ref_dat_score")
    print(f"n participants with gaze: {len(per_user):,}")
    stats = plot(per_user, "ref_dat_score", show=show)
    stats.to_csv(config.OUTPUTS_DIR / "eyetracking_vs_dat_continuous.csv",
                 index=False)
    print(stats.to_string(index=False))

    fd_stats = per_fd_correlations(gaze, participants, "ref_dat_score")
    fd_stats.to_csv(config.OUTPUTS_DIR / "eyetracking_vs_dat_per_fd.csv",
                    index=False)
    print("\n=== per-FD ===")
    print(fd_stats.to_string(index=False))
    plot_per_fd_heatmap(fd_stats, show=show)
    return stats, fd_stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    main(show=not args.no_show)
