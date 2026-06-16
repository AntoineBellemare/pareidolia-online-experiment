"""Figure 1 — Perceptual semantic diversity vs DAT (creativity).

Per participant we:
  1. Pool all unique words spoken across FD12/14/16 trials.
  2. Embed them (BERT MiniLM, 384d, cached in parquet).
  3. Reduce them to a single spread metric (default: median pairwise cosine
     distance — i.e. how semantically scattered each participant's percepts are).
  4. Correlate that spread against the reference DAT score.

Reproduces the figure produced by the notebook cell that begins
`spread_metric = "pairwise_median"`.

Usage:
    python -m analysis.figures.fig_diversity_vs_dat
    python -m analysis.figures.fig_diversity_vs_dat --metric pairwise_mean --dat after
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist, pdist
from scipy.stats import chi2, pearsonr
from numpy.linalg import eigh

from .. import config, dat, dat_helper, embeddings, parse_events


# ─── spread metrics ─────────────────────────────────────────────────────────────

def _centroid_mean(v):   c = v.mean(0, keepdims=True); return cdist(v, c, "cosine").mean()
def _centroid_median(v): c = v.mean(0, keepdims=True); return float(np.median(cdist(v, c, "cosine")))
def _pairwise_mean(v):   return float(pdist(v, "cosine").mean())
def _pairwise_median(v): return float(np.median(pdist(v, "cosine")))
def _pairwise_90(v):     return float(np.percentile(pdist(v, "cosine"), 90))
def _ellipse(v):
    cov = np.cov(v, rowvar=False)
    lam = eigh(cov, eigvals_only=True)[::-1]
    return float(np.pi * np.prod(np.sqrt(lam * chi2.ppf(0.95, 2))))


METRICS = {
    "centroid_mean":   _centroid_mean,
    "centroid_median": _centroid_median,
    "pairwise_mean":   _pairwise_mean,
    "pairwise_median": _pairwise_median,
    "pairwise_90th":   _pairwise_90,
    "ellipse_area":    _ellipse,
}

METRIC_LABELS = {
    "centroid_mean":   "Mean dist.→centroid",
    "centroid_median": "Median dist.→centroid",
    "pairwise_mean":   "Mean pair-wise dist.",
    "pairwise_median": "Median pair-wise dist.",
    "pairwise_90th":   "90th-pct pair-wise dist.",
    "ellipse_area":    "95 % ellipse area",
}


# ─── data prep ──────────────────────────────────────────────────────────────────

_ASCII_LETTERS_POOL = __import__("re").compile(r"^[A-Za-z]+$")


def long_words_table(participants: pd.DataFrame,
                     fds: list[str] = config.FD_LEVELS,
                     drop_stopwords: bool = True,
                     ascii_only: bool = True) -> pd.DataFrame:
    """One row per (participant, word) with the participant's DAT score(s).

    `ascii_only=True` (default, matches notebook) drops multi-word phrases
    from the embedding table — see fig_diversity_vs_dat_by_fd for why this
    matters for the spread metric.
    """
    rows = []
    for _, r in participants.iterrows():
        all_words = []
        for fd in fds:
            v = r.get(f"{fd}_words")
            if v is None:
                continue
            all_words.extend(list(v))
        for w in dict.fromkeys(all_words):  # dedup, preserve order
            if ascii_only and not _ASCII_LETTERS_POOL.fullmatch(w or ""):
                continue
            rows.append({
                "user_id": r["user_id"],
                "word": w,
                "ref_dat_score": r["ref_dat_score"],
                "dat_after_score": r.get("dat_after_score"),
            })
    df = pd.DataFrame(rows)
    if drop_stopwords:
        df = df[~df["word"].isin(config.SEMANTIC_STOPWORDS)]
    return df.reset_index(drop=True)


def per_participant_spread(words_df: pd.DataFrame,
                           metric: str,
                           dat_col: str = "ref_dat_score",
                           sd_outlier: float = 3.0) -> pd.DataFrame:
    """Embed → spread metric → drop ±SD outliers."""
    words_df = embeddings.attach(words_df, drop_missing=True)
    fn = METRICS[metric]
    rows = []
    for uid, sub in words_df.groupby("user_id"):
        if len(sub) < 2:
            continue
        V = np.stack(sub["embedding"].to_numpy())
        rows.append({
            "user_id": uid,
            dat_col: sub[dat_col].iloc[0],
            "spread": fn(V),
            "n_words": len(sub),
        })
    out = pd.DataFrame(rows).dropna(subset=[dat_col, "spread"])
    if sd_outlier:
        z = np.abs((out["spread"] - out["spread"].mean()) / out["spread"].std())
        out = out[z <= sd_outlier].reset_index(drop=True)
    return out


# ─── plotting ───────────────────────────────────────────────────────────────────

def plot(df: pd.DataFrame, metric: str, dat_col: str,
         out_path=None, show: bool = True,
         out_name: str | None = None) -> tuple[float, float]:
    import analysis.style as _style
    _style.apply()
    r, p = pearsonr(df[dat_col], df["spread"])

    fig, ax = plt.subplots(figsize=(_style.COL_1, 2.6))
    ax.scatter(df[dat_col], df["spread"], s=3.5, alpha=0.55,
               color=_style.COLORS["dot"], linewidths=0)
    sns.regplot(data=df, x=dat_col, y="spread", scatter=False,
                color=_style.COLORS["fit"], ci=95,
                line_kws=dict(linewidth=1.0), ax=ax)
    xlabel = "DAT" if dat_col in {"ref_dat_score", "dat_before_glove"} else "DAT post"
    _style.style_axis(
        ax,
        title="Perceptual diversity vs DAT",
        ylabel=f"Perceptual diversity\n({METRIC_LABELS[metric]})",
        xlabel=xlabel,
    )
    _style.small_stat_annotation(ax, r, p, loc="upper left")
    if out_name:
        _style.savefig(fig, out_name)
    elif out_path:
        fig.savefig(out_path)
    if show: plt.show()
    plt.close(fig)
    return r, p


# ─── entry-point ────────────────────────────────────────────────────────────────

def main(metric: str = "pairwise_median", dat_when: str = "before",
         show: bool = True, cohort: str = "notebook"):
    participants = dat_helper.select_cohort(cohort, dat_when)
    dat_col = "ref_dat_score" if dat_when == "before" else dat_helper.dat_column(dat_when)
    participants = participants.dropna(subset=[dat_col])

    words_df = long_words_table(participants)
    summary = per_participant_spread(words_df, metric=metric, dat_col=dat_col)
    tag = f"{cohort}_{dat_when}_{metric}"
    r, p = plot(summary, metric, dat_col,
                out_name=f"diversity_vs_dat_{tag}", show=show)

    summary.to_csv(
        config.OUTPUTS_DIR / f"diversity_vs_dat_{tag}.csv", index=False,
    )
    print(f"\nMetric: {metric} | DAT: {dat_when}")
    print(f"  n = {len(summary):,}   r = {r:.3f}   p = {p:.4g}")
    return summary, r, p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="pairwise_median", choices=list(METRICS))
    ap.add_argument("--dat", dest="dat_when", default="before",
                    choices=["before", "after", "raw_before"])
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--cohort", default="notebook", choices=["notebook", "main"])
    args = ap.parse_args()
    main(metric=args.metric, dat_when=args.dat_when,
         show=not args.no_show, cohort=args.cohort)
