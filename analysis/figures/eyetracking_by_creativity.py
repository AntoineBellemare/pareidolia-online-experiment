"""How creative participants move their eyes during pareidolia.

For each trial we extract gaze metrics (see analysis/eyetracking.py for the
exact derivations and the geometry-normalisation logic). We then:

  1. Aggregate trial metrics to per-participant means (averaged within FD too).
  2. Split participants into Low / Mid / High creativity tertiles by DAT.
  3. Boxplot + Kruskal-Wallis test for each metric.
  4. Pearson/Spearman correlations between continuous DAT and each metric.

Caveats (see eyetracking.py for full notes):
  - Stimulus bounding box is approximate (gaze_target.width is 0 in the
    raw data). `prop_on_stim` should be read relatively, not absolutely.
  - Screen extent is inferred from the per-session calibration sweep.

Usage:
    python -m analysis.figures.eyetracking_by_creativity
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal, pearsonr, spearmanr

from .. import config, dat_helper, eyetracking, parse_events

PALETTE = {"Low": "#66D1F4", "Mid": "#9FCA1E", "High": "#922B21"}
LABELS = ["Low", "Mid", "High"]

ET_METRICS = [
    ("mean_fix_dur_ms",       "Mean fixation\nduration (ms)"),
    ("n_fixations",           "Fixations / trial"),
    ("n_saccades",            "Saccades / trial"),
    ("mean_saccade_amp_norm", "Mean saccade\namplitude (norm)"),
    ("gaze_dispersion_norm",  "Gaze dispersion\n(norm RMS)"),
    ("gaze_entropy",          "Gaze entropy\n(2D histogram)"),
    ("prop_on_stim",          "Prop. gaze\non stimulus"),
]


def _load_or_build_gaze() -> pd.DataFrame:
    read_path = config.cached_parquet("gaze_metrics.parquet")
    if read_path.exists():
        print(f"Loading cached gaze metrics: {read_path}")
        return pd.read_parquet(read_path)
    print("Building gaze metrics (slow, walks all sessions)…")
    df = eyetracking.build_gaze_metrics()
    df.to_parquet(config.CACHE_DIR / "gaze_metrics.parquet")
    return df


def per_participant(gaze: pd.DataFrame,
                    participants: pd.DataFrame,
                    dat_col: str = "ref_dat_score") -> pd.DataFrame:
    """Mean across trials, per participant."""
    g = gaze[gaze["user_id"].isin(participants["user_id"])].copy()
    cols = [c for c, _ in ET_METRICS]
    out = (
        g.groupby("user_id")[cols].mean().reset_index()
          .merge(participants[["user_id", dat_col]], on="user_id", how="left")
    )
    out["n_trials_with_gaze"] = (
        g.groupby("user_id").size().rename("n_trials_with_gaze").reindex(out["user_id"]).values
    )
    out = out.dropna(subset=[dat_col])
    out["dat_tertile"] = pd.qcut(
        out[dat_col], 3, labels=LABELS, duplicates="drop"
    )
    return out


def plot(per_user: pd.DataFrame, out_path=None, show: bool = True,
         dat_col: str = "ref_dat_score") -> pd.DataFrame:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    stats_rows = []
    for ax, (col, label) in zip(axes.flat, ET_METRICS):
        sub = per_user.dropna(subset=[col, dat_col, "dat_tertile"])
        if len(sub) < 10:
            ax.set_title(f"{label}: n too small ({len(sub)})")
            continue
        sns.boxplot(data=sub, x="dat_tertile", y=col, order=LABELS,
                    palette=[PALETTE[l] for l in LABELS], ax=ax)
        groups = [sub[sub["dat_tertile"] == lab][col].dropna() for lab in LABELS]
        if all(len(g) >= 3 for g in groups):
            H, p_k = kruskal(*groups)
        else:
            H, p_k = np.nan, np.nan
        r, p_r = pearsonr(sub[dat_col], sub[col])
        rho, p_rho = spearmanr(sub[dat_col], sub[col])  # CSV only
        ax.set_title(f"{label}\nKW p={p_k:.2g}  r={r:+.2f}", fontsize=10)
        ax.set_xlabel("Creativity tertile")
        ax.set_ylabel("")
        stats_rows.append({
            "metric": col, "n": len(sub),
            "kw_H": H, "kw_p": p_k,
            "pearson_r": r, "pearson_p": p_r,
            "spearman_rho": rho, "spearman_p": p_rho,
        })
    # 8th panel — n per tertile
    ax = axes.flat[-1]
    counts = per_user["dat_tertile"].value_counts().reindex(LABELS).fillna(0)
    ax.bar(counts.index, counts.values,
           color=[PALETTE[l] for l in LABELS])
    ax.set_ylabel("Participants with gaze data")
    ax.set_title("Cohort size by tertile")

    plt.suptitle("Eye movements during pareidolia by creativity tertile",
                 y=1.02, fontsize=14, weight="bold")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {out_path}")
    if show:
        plt.show()
    plt.close()
    return pd.DataFrame(stats_rows)


def main(show: bool = True, rebuild: bool = False, dat_when: str = "before"):
    participants = dat_helper.main_cohort_scored()
    dat_col = dat_helper.dat_column(dat_when)
    participants = participants.dropna(subset=[dat_col])

    if rebuild:
        cache = config.CACHE_DIR / "gaze_metrics.parquet"
        if cache.exists():
            cache.unlink()
    gaze = _load_or_build_gaze()
    print(f"Total trials with gaze metrics: {len(gaze):,}")

    per_user = per_participant(gaze, participants, dat_col=dat_col)
    print(f"Participants with gaze + DAT-{dat_when}: {len(per_user):,}")

    out_png = config.OUTPUTS_DIR / f"fig_eyetracking_by_creativity_{dat_when}.png"
    stats = plot(per_user, out_path=out_png, show=show, dat_col=dat_col)

    per_user.to_csv(
        config.OUTPUTS_DIR / f"eyetracking_per_user_{dat_when}.csv", index=False
    )
    stats.to_csv(
        config.OUTPUTS_DIR / f"eyetracking_correlations_{dat_when}.csv", index=False
    )
    print(f"\n=== Eye tracking vs DAT-{dat_when} ===")
    print(stats.to_string(index=False))
    return per_user, stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--rebuild", action="store_true",
                    help="recompute gaze metrics from scratch")
    ap.add_argument("--dat", dest="dat_when", default="before",
                    choices=["before", "after", "raw_before"])
    args = ap.parse_args()
    main(show=not args.no_show, rebuild=args.rebuild, dat_when=args.dat_when)
