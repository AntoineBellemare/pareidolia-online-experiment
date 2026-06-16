"""Does perceptual diversity during the pareidolia task predict DAT-post?

Two questions, on the same per-participant table:

  (a) Higher semantic spread of percepts -> higher DAT post-task score?
  (b) Higher semantic spread of percepts -> larger DAT delta (post - pre)?

Both are tested for all five spread metrics (centroid-mean, centroid-median,
pairwise-mean, pairwise-median, pairwise-90th-percentile); the paper
figure uses ``pairwise_median`` to stay consistent with the rest of the
manuscript. ±3 SD bivariate outlier trim before every correlation. For
the delta question, we also report the partial correlation controlling
for the baseline DAT-pre score (an explicit test of whether the spread
predicts the *change* over and above the baseline trait).

Usage:
    python -m analysis.figures.diversity_vs_post_dat
"""
from __future__ import annotations

import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist, pdist
from scipy.stats import pearsonr, spearmanr

from .. import config, dat_helper, embeddings, outliers, style


ASCII = re.compile(r"^[A-Za-z]+$")

METRICS = {
    "centroid_mean":   lambda v: float(cdist(v, v.mean(0, keepdims=True),
                                              "cosine").mean()),
    "centroid_median": lambda v: float(np.median(cdist(v, v.mean(0, keepdims=True),
                                                        "cosine"))),
    "pairwise_mean":   lambda v: float(pdist(v, "cosine").mean()),
    "pairwise_median": lambda v: float(np.median(pdist(v, "cosine"))),
    "pairwise_90th":   lambda v: float(np.percentile(pdist(v, "cosine"), 90)),
}
NICE = {
    "centroid_mean":   "Mean dist. to centroid",
    "centroid_median": "Median dist. to centroid",
    "pairwise_mean":   "Mean pair-wise dist.",
    "pairwise_median": "Median pair-wise dist.",
    "pairwise_90th":   "90th-pct pair-wise dist.",
}


# ─── build per-participant spread + DAT table ────────────────────────────────

def _build_table() -> pd.DataFrame:
    edict = embeddings.embedding_dict()
    p = dat_helper.main_cohort_scored().dropna(
        subset=["dat_before_glove", "dat_after_score"]
    ).copy()
    p["delta_dat"] = p["dat_after_score"] - p["dat_before_glove"]

    rows = []
    for _, r in p.iterrows():
        words = []
        for fd in config.FD_LEVELS:
            v = r.get(f"{fd}_words")
            if v is None: continue
            for w in list(v):
                w = (w or "").strip().lower()
                if (ASCII.fullmatch(w) and w in edict
                        and w not in config.SEMANTIC_STOPWORDS):
                    words.append(w)
        words = list(dict.fromkeys(words))
        if len(words) < 2: continue
        V = np.vstack([edict[w] for w in words])
        row = {name: fn(V) for name, fn in METRICS.items()}
        row.update({
            "user_id":  r["user_id"],
            "dat_pre":  r["dat_before_glove"],
            "dat_post": r["dat_after_score"],
            "delta_dat": r["delta_dat"],
            "n_unique_words": len(words),
        })
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f"Per-participant table: {len(df)} participants with diversity + DAT pre + post")
    return df


# ─── full stats: every metric × target × correction ─────────────────────────

def _partial_r(df: pd.DataFrame, x: str, y: str, z: str
               ) -> tuple[float, float, int]:
    """Partial correlation r(x, y | z): regress out z then correlate residuals."""
    from numpy.linalg import lstsq
    clean, _ = outliers.trim_sd(df, [x, y, z], sd=3.0)
    Z = np.column_stack([np.ones(len(clean)), clean[z].values])
    bx, *_ = lstsq(Z, clean[x].values, rcond=None)
    by, *_ = lstsq(Z, clean[y].values, rcond=None)
    rx = clean[x].values - Z @ bx
    ry = clean[y].values - Z @ by
    r, p = pearsonr(rx, ry)
    return float(r), float(p), len(clean)


def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name in METRICS:
        # (a) spread vs DAT post
        clean, _ = outliers.trim_sd(df, [name, "dat_post"], sd=3.0)
        r1,  p1  = pearsonr(clean[name],  clean["dat_post"])
        rho1, prho1 = spearmanr(clean[name], clean["dat_post"])
        # (a') partial controlling for DAT pre
        pr1, pp1, n_partial1 = _partial_r(df, name, "dat_post", "dat_pre")

        # (b) spread vs DAT delta
        clean2, _ = outliers.trim_sd(df, [name, "delta_dat"], sd=3.0)
        r2,  p2  = pearsonr(clean2[name],  clean2["delta_dat"])
        rho2, prho2 = spearmanr(clean2[name], clean2["delta_dat"])
        # (b') partial controlling for DAT pre
        pr2, pp2, n_partial2 = _partial_r(df, name, "delta_dat", "dat_pre")

        out.append({
            "metric": name,
            "n_post":            len(clean),
            "post_pearson_r":    r1,  "post_pearson_p":    p1,
            "post_spearman_rho": rho1, "post_spearman_p":  prho1,
            "post_partial_r":    pr1, "post_partial_p":    pp1,
            "n_delta":           len(clean2),
            "delta_pearson_r":   r2,  "delta_pearson_p":   p2,
            "delta_spearman_rho": rho2, "delta_spearman_p": prho2,
            "delta_partial_r":   pr2, "delta_partial_p":   pp2,
        })
    return pd.DataFrame(out)


# ─── paper figure ────────────────────────────────────────────────────────────

def figure_paper(df: pd.DataFrame, metric: str = "pairwise_median",
                  show: bool = True) -> dict:
    style.apply()
    fig, axes = plt.subplots(
        1, 2, figsize=(style.COL_1_5, 2.4),
        gridspec_kw=dict(wspace=0.45),
    )

    # ── (a) spread vs DAT post ────────────────────────────────────────────
    ax = axes[0]
    sub, _ = outliers.trim_sd(df, [metric, "dat_post"], sd=3.0)
    r1, p1 = pearsonr(sub[metric], sub["dat_post"])
    ax.scatter(sub[metric], sub["dat_post"], s=3.5, alpha=0.55,
                color=style.COLORS["dot"], linewidths=0)
    sns.regplot(data=sub, x=metric, y="dat_post", scatter=False,
                 color=style.COLORS["fit"], ci=95,
                 line_kws=dict(linewidth=1.0), ax=ax)
    style.style_axis(ax, title="",
                      xlabel=f"Perceptual diversity\n({NICE[metric]})",
                      ylabel="DAT post")
    style.small_stat_annotation(ax, r1, p1, loc="upper left")

    # ── (b) spread vs DAT delta ───────────────────────────────────────────
    ax = axes[1]
    sub2, _ = outliers.trim_sd(df, [metric, "delta_dat"], sd=3.0)
    r2, p2 = pearsonr(sub2[metric], sub2["delta_dat"])
    ax.axhline(0, color=style.COLORS["muted"], lw=0.5, ls="--")
    ax.scatter(sub2[metric], sub2["delta_dat"], s=3.5, alpha=0.55,
                color=style.COLORS["dot"], linewidths=0)
    sns.regplot(data=sub2, x=metric, y="delta_dat", scatter=False,
                 color=style.COLORS["fit"], ci=95,
                 line_kws=dict(linewidth=1.0), ax=ax)
    style.style_axis(ax, title="",
                      xlabel=f"Perceptual diversity\n({NICE[metric]})",
                      ylabel="Δ DAT  (post − pre)")
    style.small_stat_annotation(ax, r2, p2, loc="upper left")

    fig.suptitle("Does perceptual diversity predict DAT after the task?",
                  y=1.05)
    if show: plt.show()
    style.savefig(fig, "diversity_vs_post_and_delta")
    plt.close(fig)
    return {"r_post": r1, "p_post": p1, "n_post": len(sub),
            "r_delta": r2, "p_delta": p2, "n_delta": len(sub2)}


# ─── runner ─────────────────────────────────────────────────────────────────

def main(metric: str = "pairwise_median", show: bool = True) -> None:
    df = _build_table()
    stats = compute_stats(df)
    out_csv = config.OUTPUTS_DIR / "diversity_vs_post_and_delta.csv"
    stats.to_csv(out_csv, index=False)
    print(f"Wrote stats: {out_csv}\n")
    print(stats.round(4).to_string(index=False))
    figure_paper(df, metric=metric, show=show)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="pairwise_median",
                    choices=list(METRICS))
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    main(metric=args.metric, show=not args.no_show)
