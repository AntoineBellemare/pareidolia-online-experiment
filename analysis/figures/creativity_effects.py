"""Creativity (DAT) effects on pareidolia beyond semantic diversity.

For each participant we summarise:
    * task_completion      : fraction of the 30 trials with at least one word
    * mean_n_words         : mean number of words submitted per trial
    * mean_n_desc          : mean number of descriptions
    * mean_rt              : mean reaction time
    * total_unique_words   : pooled unique words across all FDs
    * mean_word_rarity     : mean log-inverse-frequency of the words a
                             participant produces — high values = uncommon
                             percepts (using the corpus-wide word counts).

Each is correlated with DAT score (Pearson + Spearman), with FD-interaction
checks via OLS  metric ~ DAT * FD  on the per-(participant,FD) table.

Usage:
    python -m analysis.figures.creativity_effects
"""
from __future__ import annotations

import argparse
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

from .. import config, dat_helper, outliers, parse_events, style


# ─── feature engineering ───────────────────────────────────────────────────────

def _flatten_words(participants: pd.DataFrame) -> Counter:
    """Corpus-wide word frequency across all participants in the cohort."""
    cnt: Counter = Counter()
    for _, r in participants.iterrows():
        for fd in config.FD_LEVELS:
            v = r.get(f"{fd}_words")
            if v is None:
                continue
            cnt.update(list(v))
    return cnt


def participant_features(participants: pd.DataFrame,
                         trials: pd.DataFrame,
                         DAT_COL: str = "ref_dat_score") -> pd.DataFrame:
    corpus = _flatten_words(participants)
    total = sum(corpus.values()) or 1

    trials = trials[trials["user_id"].isin(participants["user_id"])].copy()
    trials["n_words"] = trials["words"].apply(lambda v: len(list(v)) if v is not None else 0)
    trials["n_desc"]  = trials["descriptions"].apply(lambda v: len(list(v)) if v is not None else 0)
    trials["any_word"] = trials["n_words"] > 0

    per_user = (
        trials.groupby("user_id")
              .agg(n_trials_seen=("trial_index", "count"),
                   n_trials_with_word=("any_word", "sum"),
                   mean_n_words=("n_words", "mean"),
                   mean_n_desc=("n_desc", "mean"),
                   mean_rt=("reaction_time", "mean"))
              .reset_index()
    )
    per_user["task_completion"] = per_user["n_trials_with_word"] / 30.0
    per_user["total_unique_words"] = per_user["user_id"].map(
        lambda u: sum(1 for w in set(_words_of(participants, u)))
    )

    def rarity(uid: str) -> float:
        ws = _words_of(participants, uid)
        if not ws:
            return np.nan
        rarities = [-np.log((corpus.get(w, 1) / total) + 1e-12) for w in ws]
        return float(np.mean(rarities))

    per_user["mean_word_rarity"] = per_user["user_id"].apply(rarity)
    per_user = per_user.merge(
        participants[["user_id", DAT_COL]], on="user_id", how="left",
    )
    return per_user.dropna(subset=[DAT_COL])


def _words_of(participants: pd.DataFrame, uid: str) -> list[str]:
    rows = participants[participants["user_id"] == uid]
    if rows.empty:
        return []
    r = rows.iloc[0]
    out = []
    for fd in config.FD_LEVELS:
        v = r.get(f"{fd}_words")
        if v is None:
            continue
        out.extend(list(v))
    return out


# ─── correlations & plots ──────────────────────────────────────────────────────

# Slimmed for the paper figure: just the two cleanest behavioural axes.
# The full 6-metric version remains available in MAIN_METRICS for the CSV /
# supplements.
METRICS = [
    ("mean_n_words", "Mean words / trial"),
    ("mean_rt",      "Mean RT (ms)"),
]
MAIN_METRICS = [
    ("task_completion",   "% task completed"),
    ("mean_n_words",      "Mean words / trial"),
    ("mean_n_desc",       "Mean descriptions / trial"),
    ("mean_rt",           "Mean RT (ms)"),
    ("total_unique_words", "Total unique words"),
    ("mean_word_rarity",  "Mean word rarity\n(-log p)"),
]


def _outliers_dropped(df: pd.DataFrame, col: str,
                      dat_col: str = "DAT_PLACEHOLDER",
                      sd: float = 3.0) -> pd.DataFrame:
    """Bivariate ±sd trim on both metric and DAT score."""
    kept, _ = outliers.trim_sd(df, [c for c in [col, dat_col] if c in df.columns],
                                sd=sd)
    return kept


def plot(per_user: pd.DataFrame, out_path=None, show: bool = True,
         DAT_COL: str = "ref_dat_score", xlabel: str = "DAT") -> pd.DataFrame:
    style.apply()
    fig, axes = plt.subplots(
        1, len(METRICS), figsize=(style.COL_1, 1.9),
        gridspec_kw=dict(wspace=0.5),
    )
    if len(METRICS) == 1:
        axes = [axes]
    stats_rows = []
    for ax, (col, label) in zip(axes, METRICS):
        sub = per_user.dropna(subset=[col, DAT_COL])
        sub = _outliers_dropped(sub, col, dat_col=DAT_COL, sd=3.0)
        if len(sub) < 5:
            ax.set_title(f"{label}: n too small ({len(sub)})")
            continue
        ax.scatter(sub[DAT_COL], sub[col], s=3.5, alpha=0.55,
                   color=style.COLORS["dot"], linewidths=0)
        sns.regplot(data=sub, x=DAT_COL, y=col, scatter=False,
                    color=style.COLORS["fit"], ci=95,
                    line_kws=dict(linewidth=1.0), ax=ax)
        r, p = pearsonr(sub[DAT_COL], sub[col])
        rho, p_rho = spearmanr(sub[DAT_COL], sub[col])  # CSV only
        style.style_axis(ax, title="", xlabel=xlabel, ylabel=label)
        style.small_stat_annotation(ax, r, p, loc="upper left")
        stats_rows.append({
            "metric": col, "n": len(sub),
            "pearson_r": r, "pearson_p": p,
            "spearman_rho": rho, "spearman_p": p_rho,
        })
    fig.suptitle("Creativity (DAT) vs general pareidolia behaviour", y=1.04)
    # out_path is preserved for backward compatibility but we always
    # also write the dual (titled + untitled) versions via style.savefig.
    stem = "creativity_effects"
    if out_path is not None:
        # extract dat_when from filename if present
        from pathlib import Path as _P
        nm = _P(out_path).stem
        # e.g. fig_creativity_effects_before
        stem = nm[4:] if nm.startswith("fig_") else nm
    style.savefig(fig, stem)
    if show:
        plt.show()
    plt.close(fig)
    return pd.DataFrame(stats_rows)


DAT_LABELS = {
    "ref_dat_score":    "DAT before (raw frontend score)",
    "dat_before_glove": "DAT before (GloVe re-score)",
    "dat_after_score":  "DAT after (GloVe)",
}


def main(show: bool = True, dat_when: str = "before"):
    participants = dat_helper.main_cohort_scored()
    dat_col = dat_helper.dat_column(dat_when)
    participants = participants.dropna(subset=[dat_col])
    trials = parse_events.cached_trials()
    per_user = participant_features(participants, trials, DAT_COL=dat_col)

    print(f"Per-user features ({dat_when} DAT): {len(per_user):,}")
    print(per_user.describe().T[["mean", "std", "min", "max"]])

    out_png = config.OUTPUTS_DIR / f"fig_creativity_effects_{dat_when}.png"
    stats = plot(
        per_user, out_path=out_png, show=show,
        DAT_COL=dat_col, xlabel="DAT",
    )

    per_user.to_csv(
        config.OUTPUTS_DIR / f"creativity_features_per_user_{dat_when}.csv",
        index=False,
    )
    stats.to_csv(
        config.OUTPUTS_DIR / f"creativity_correlations_{dat_when}.csv",
        index=False,
    )
    print(f"\n=== DAT-{dat_when} vs pareidolia metrics ===")
    print(stats.to_string(index=False))
    return per_user, stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--dat", dest="dat_when", default="before",
                    choices=["before", "after", "raw_before"])
    args = ap.parse_args()
    main(show=not args.no_show, dat_when=args.dat_when)
