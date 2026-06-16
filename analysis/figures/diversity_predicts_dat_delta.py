"""Does perceptual diversity predict the DAT post − pre delta?

Tests all 5 spread metrics; outputs a single panel + CSV.
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
    "centroid_mean":   lambda v: float(cdist(v, v.mean(0, keepdims=True), "cosine").mean()),
    "centroid_median": lambda v: float(np.median(cdist(v, v.mean(0, keepdims=True), "cosine"))),
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


def main(metric: str = "pairwise_median", show: bool = True):
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
        row.update({"user_id": r["user_id"],
                    "delta_dat": r["delta_dat"],
                    "dat_pre":   r["dat_before_glove"]})
        rows.append(row)
    df = pd.DataFrame(rows)

    # Full stats table for every metric (Pearson + Spearman + partial-on-pre)
    stat_rows = []
    for name in METRICS:
        clean, _ = outliers.trim_sd(df, [name, "delta_dat"], sd=3.0)
        r, p_r = pearsonr(clean[name], clean["delta_dat"])
        rho, p_rho = spearmanr(clean[name], clean["delta_dat"])
        # Partial controlling for DAT-pre
        clean2, _ = outliers.trim_sd(df, [name, "delta_dat", "dat_pre"], sd=3.0)
        from numpy.linalg import lstsq
        X = np.column_stack([np.ones(len(clean2)), clean2["dat_pre"].values])
        bx, *_ = lstsq(X, clean2[name].values, rcond=None)
        by, *_ = lstsq(X, clean2["delta_dat"].values, rcond=None)
        pr, p_pr = pearsonr(clean2[name].values - X @ bx,
                              clean2["delta_dat"].values - X @ by)
        stat_rows.append({"metric": name, "n": len(clean),
                          "pearson_r": r, "pearson_p": p_r,
                          "spearman_rho": rho, "spearman_p": p_rho,
                          "partial_r": pr, "partial_p": p_pr})
    stats = pd.DataFrame(stat_rows)
    stats.to_csv(config.OUTPUTS_DIR / "diversity_vs_dat_delta.csv", index=False)
    print(stats.to_string(index=False))

    # Single-panel scatter for the chosen metric
    style.apply()
    fig, ax = plt.subplots(figsize=(style.COL_1, 2.6))
    sub, _ = outliers.trim_sd(df, [metric, "delta_dat"], sd=3.0)
    r, p = pearsonr(sub[metric], sub["delta_dat"])
    ax.axhline(0, color=style.COLORS["muted"], lw=0.5, ls="--")
    ax.scatter(sub[metric], sub["delta_dat"], s=3.5, alpha=0.55,
               color=style.COLORS["dot"], linewidths=0)
    sns.regplot(data=sub, x=metric, y="delta_dat", scatter=False,
                color=style.COLORS["fit"], ci=95,
                line_kws=dict(linewidth=1.0), ax=ax)
    style.style_axis(ax, title="", xlabel=f"Perceptual diversity\n({NICE[metric]})",
                     ylabel="Δ DAT  (post, pre)")
    style.small_stat_annotation(ax, r, p, loc="upper left")
    fig.suptitle("Does percept diversity predict the DAT delta?", y=1.02)
    if show: plt.show()
    style.savefig(fig, f"diversity_vs_dat_delta_{metric}")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="pairwise_median",
                    choices=list(METRICS))
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    main(metric=args.metric, show=not args.no_show)
