"""Effect of fractal dimension (FD) of the stimulus on pareidolia.

Each participant viewed 30 stimuli split across three FD levels (FD12, FD14,
FD16 — perceived complexity increases with FD). For each (participant, FD)
cell we compute:
    * mean reaction time
    * mean number of words submitted per trial
    * mean number of descriptions
    * **within-FD semantic spread** (median pairwise cosine distance across
      the participant's unique words at that FD)

We then test whether each metric varies systematically with FD, using
within-subject repeated-measures (Friedman test) plus per-pair Wilcoxon
contrasts.

Usage:
    python -m analysis.figures.fd_effect
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist
from scipy.stats import friedmanchisquare, wilcoxon

from .. import config, embeddings, parse_events


# ─── per (participant, FD) aggregates ──────────────────────────────────────────

def per_participant_fd(trials: pd.DataFrame,
                       participants: pd.DataFrame) -> pd.DataFrame:
    """Trial-level table -> one row per (user_id, fd_level)."""
    trials = trials[trials["fd_level"].isin(config.FD_LEVELS)].copy()
    trials["n_words"] = trials["words"].apply(lambda v: len(list(v)) if v is not None else 0)
    trials["n_descriptions"] = trials["descriptions"].apply(
        lambda v: len(list(v)) if v is not None else 0
    )
    agg = (
        trials.groupby(["user_id", "fd_level"])
              .agg(rt_mean=("reaction_time", "mean"),
                   n_words_mean=("n_words", "mean"),
                   n_desc_mean=("n_descriptions", "mean"),
                   n_trials=("trial_index", "count"))
              .reset_index()
    )
    agg = agg.merge(
        participants[["user_id", "ref_dat_score"]], on="user_id", how="left"
    )
    return agg


def semantic_spread_per_fd(participants: pd.DataFrame) -> pd.DataFrame:
    """Median pairwise cosine distance of each (participant, FD)'s words.

    Note: drops multi-word phrases (see fig_diversity_vs_dat_by_fd for the why).
    """
    import re as _re
    ASCII = _re.compile(r"^[A-Za-z]+$")
    edict = embeddings.embedding_dict()
    rows = []
    for _, r in participants.iterrows():
        for fd in config.FD_LEVELS:
            words = r.get(f"{fd}_words")
            if words is None:
                continue
            words = [w for w in list(words)
                     if ASCII.fullmatch(w or "") and w in edict
                     and w not in config.SEMANTIC_STOPWORDS]
            if len(words) < 3:
                continue
            V = np.vstack([edict[w] for w in words])
            d = float(np.median(pdist(V, "cosine")))
            rows.append({
                "user_id": r["user_id"],
                "fd_level": fd,
                "n_unique_words": len(words),
                "spread": d,
            })
    return pd.DataFrame(rows)


# ─── stats ─────────────────────────────────────────────────────────────────────

def within_subject_stats(df: pd.DataFrame, value_col: str) -> dict:
    """Return Friedman test result + per-pair Wilcoxon contrasts."""
    pivot = df.pivot(index="user_id", columns="fd_level", values=value_col).dropna()
    if len(pivot) < 5 or pivot.shape[1] < 2:
        return {"friedman_p": None, "n_subjects": len(pivot)}
    chi, p = friedmanchisquare(*[pivot[c] for c in config.FD_LEVELS if c in pivot.columns])
    pairwise = {}
    for a, b in [("FD12", "FD14"), ("FD12", "FD16"), ("FD14", "FD16")]:
        if a in pivot and b in pivot:
            _, pp = wilcoxon(pivot[a], pivot[b])
            pairwise[f"{a}_vs_{b}"] = float(pp)
    return {
        "friedman_chi2": float(chi),
        "friedman_p": float(p),
        "n_subjects": len(pivot),
        "pairwise": pairwise,
        "means": pivot.mean().to_dict(),
        "medians": pivot.median().to_dict(),
    }


# ─── plotting ──────────────────────────────────────────────────────────────────

def plot(fd_df: pd.DataFrame, spread_df: pd.DataFrame,
         out_path=None, show: bool = True) -> dict:
    """Single-row figure: mean ± 95 % CI per FD for each behavioural metric.

    Within-subject paired contrasts (formerly the bottom row) are computed
    and saved to CSV but no longer plotted — the bar+CI panel already
    conveys the effect.
    """
    from .. import style as _style
    _style.apply()
    metrics = [
        ("rt_mean",       "Reaction time (ms)",  fd_df),
        ("n_words_mean",  "Words / trial",       fd_df),
        ("n_desc_mean",   "Descriptions / trial", fd_df),
        ("spread",        "Semantic spread", spread_df),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(_style.COL_2, 2.4),
                             gridspec_kw=dict(wspace=0.45))
    palette = [_style.COLORS[fd] for fd in config.FD_LEVELS]
    stats = {}

    for ax, (col, label, src) in zip(axes, metrics):
        sub = src.dropna(subset=[col])
        pivot = sub.pivot(index="user_id", columns="fd_level", values=col).dropna()
        if pivot.empty or pivot.shape[1] < 2:
            ax.set_title(f"{label}  (too few data)")
            continue
        pivot = pivot[[fd for fd in config.FD_LEVELS if fd in pivot.columns]]
        means = pivot.mean()
        sems  = pivot.sem()
        ci95  = 1.96 * sems
        xs = np.arange(len(pivot.columns))
        ax.bar(xs, [means[fd] for fd in pivot.columns],
               yerr=[ci95[fd] for fd in pivot.columns],
               color=palette[:len(pivot.columns)],
               capsize=3, alpha=0.85, edgecolor="black", lw=0.5)
        ax.set_xticks(xs); ax.set_xticklabels(pivot.columns)
        # Zoom y to show the actual variation.
        lo = float((means - ci95).min()); hi = float((means + ci95).max())
        pad = (hi - lo) * 0.25 if hi > lo else 0.1
        ax.set_ylim(lo - pad, hi + pad)
        s = within_subject_stats(src, col)
        stats[col] = s
        _style.style_axis(ax, title="", ylabel=label, xlabel="")
        # Pairwise Wilcoxon → brackets above the bars
        pair_x = {"FD12": 0, "FD14": 1, "FD16": 2}
        comparisons = []
        for k, pw in s.get("pairwise", {}).items():
            a, _, b = k.partition("_vs_")
            if a in pair_x and b in pair_x and pw is not None:
                comparisons.append((pair_x[a], pair_x[b], pw))
        _style.sig_brackets(ax, comparisons)

    fig.suptitle("Stimulus FD effects on pareidolia behaviour", y=1.05)
    _style.savefig(fig, "fd_effect")
    if show: plt.show()
    plt.close(fig)
    return stats


# ─── entry-point ───────────────────────────────────────────────────────────────

def main(show: bool = True):
    participants = parse_events.main_cohort()
    trials = parse_events.cached_trials()
    trials = trials[trials["user_id"].isin(participants["user_id"])]

    fd_df = per_participant_fd(trials, participants)
    spread_df = semantic_spread_per_fd(participants)

    print(f"Per-(participant, FD) rows: {len(fd_df):,}")
    print(f"Semantic-spread rows:       {len(spread_df):,}")

    stats = plot(
        fd_df, spread_df,
        out_path=config.OUTPUTS_DIR / "fig_fd_effect.png",
        show=show,
    )

    fd_df.to_csv(config.OUTPUTS_DIR / "fd_effect_by_participant.csv", index=False)
    spread_df.to_csv(config.OUTPUTS_DIR / "fd_semantic_spread.csv", index=False)

    print("\n=== Within-subject statistics ===")
    for m, s in stats.items():
        print(f"\n[{m}]  n={s.get('n_subjects')}")
        if s.get("friedman_p") is not None:
            print(f"  Friedman chi2={s['friedman_chi2']:.2f}  p={s['friedman_p']:.3g}")
            print(f"  Means: {s['means']}")
            for k, v in s["pairwise"].items():
                print(f"    Wilcoxon {k}: p={v:.3g}")
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    main(show=not args.no_show)
