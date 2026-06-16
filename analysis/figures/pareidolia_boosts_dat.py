"""Does pareidolia engagement boost DAT scores?

Test whether participants who engaged more with the pareidolia task
showed bigger DAT improvements from pre to post.

Strategy:
  1. Compute DAT-after − DAT-before (Δ) using GloVe-scored values.
  2. For each engagement metric (n trials with words, total unique words,
     mean words/trial, semantic diversity, mean word rarity), correlate
     with Δ. A positive correlation = "more pareidolia → more DAT gain".
  3. Also fit OLS  DAT-after ~ DAT-before + engagement  and report the
     engagement coefficient (controls for regression to the mean).
  4. Plot:
       row 1 — Δ DAT vs each engagement metric (scatter + reg)
       row 2 — DAT-after vs DAT-before, coloured by tertile of engagement
                so one can see whether the high-engagement group lies above
                the y=x line.

Usage:
    python -m analysis.figures.pareidolia_boosts_dat
"""
from __future__ import annotations

import argparse
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr

from .. import config, dat_helper, embeddings, outliers, parse_events, style
from .creativity_effects import participant_features


ASCII = __import__("re").compile(r"^[A-Za-z]+$")


def semantic_diversity_per_user(participants: pd.DataFrame) -> pd.Series:
    edict = embeddings.embedding_dict()
    out = {}
    for _, r in participants.iterrows():
        words = []
        for fd in config.FD_LEVELS:
            v = r.get(f"{fd}_words")
            if v is None: continue
            words.extend(list(v))
        words = list(dict.fromkeys(words))
        words = [w for w in words
                 if ASCII.fullmatch(w or "") and w in edict
                 and w not in config.SEMANTIC_STOPWORDS]
        if len(words) < 2:
            continue
        V = np.vstack([edict[w] for w in words])
        out[r["user_id"]] = float(np.median(pdist(V, "cosine")))
    return pd.Series(out, name="diversity")


def build_table() -> pd.DataFrame:
    p = dat_helper.main_cohort_scored().copy()
    p = p.dropna(subset=["dat_before_glove", "dat_after_score"])

    trials = parse_events.cached_trials()
    feat = participant_features(p[["user_id"] + [c for c in p.columns if c != "user_id"]],
                                trials, DAT_COL="dat_before_glove")
    # `participant_features` drops user_id if no rows; merge keeping all p rows
    keep_cols = ["user_id", "task_completion", "mean_n_words", "mean_n_desc",
                 "mean_rt", "total_unique_words", "mean_word_rarity"]
    feat = feat[keep_cols]
    div = semantic_diversity_per_user(p).rename("diversity").reset_index()\
              .rename(columns={"index": "user_id"})
    tbl = p[["user_id", "ref_dat_score", "dat_before_glove",
             "dat_after_score"]].merge(feat, on="user_id", how="left") \
                                .merge(div, on="user_id", how="left")
    tbl["delta_dat"] = tbl["dat_after_score"] - tbl["dat_before_glove"]
    return tbl


ENGAGEMENT = [
    ("task_completion",   "% task completed"),
    ("mean_n_words",      "Words / trial"),
    ("total_unique_words","Total unique words"),
    ("diversity",         "Semantic diversity"),
    ("mean_word_rarity",  "Word rarity"),
]


def plot(tbl: pd.DataFrame, show: bool = True) -> pd.DataFrame:
    style.apply()

    # ── row 1: Δ DAT vs each engagement metric ───────────────────────────────
    fig, axes = plt.subplots(
        2, 5, figsize=(style.COL_2, 3.8),
        gridspec_kw=dict(hspace=0.85, wspace=0.55),
    )
    stats_rows = []
    from numpy.linalg import lstsq
    for ax, (col, label) in zip(axes[0], ENGAGEMENT):
        sub, n_drop = outliers.trim_sd(tbl, [col, "delta_dat"], sd=3.0)
        if n_drop:
            print(f"  [{col}] dropped {n_drop} ±3 SD outliers")
        if len(sub) < 10:
            style.style_axis(ax, title=f"{label} (n={len(sub)})")
            continue
        ax.axhline(0, color=style.COLORS["muted"], lw=0.5, ls="--")
        ax.scatter(sub[col], sub["delta_dat"], s=3.5, alpha=0.55,
                   color=style.COLORS["dot"], linewidths=0)
        sns.regplot(data=sub, x=col, y="delta_dat", scatter=False,
                    color=style.COLORS["fit"], ci=95,
                    line_kws={"linewidth": 1.0}, ax=ax)
        r, p = pearsonr(sub[col], sub["delta_dat"])
        rho, p_rho = spearmanr(sub[col], sub["delta_dat"])
        X = np.column_stack([np.ones(len(sub)), sub["dat_before_glove"], sub[col]])
        coefs, *_ = lstsq(X, sub["dat_after_score"].values, rcond=None)
        beta_eng = float(coefs[2])
        title = f"{label} {style.sig_marker(p)}"
        # Stat string as small annotation in upper-left corner instead of title
        ax.text(0.03, 0.97, style.short_stats(r, p),
                transform=ax.transAxes, ha="left", va="top",
                fontsize=5.5, color=style.COLORS["muted"])
        style.style_axis(
            ax, title=title, xlabel=label,
            ylabel="Δ DAT (post − pre)" if col == ENGAGEMENT[0][0] else "",
        )
        stats_rows.append({"metric": col, "n": len(sub),
                           "pearson_r": r, "pearson_p": p,
                           "spearman_rho": rho, "spearman_p": p_rho,
                           "beta_engagement_controlling_pre": beta_eng})

    # ── row 2: post vs pre, coloured by engagement tertile ────────────────────
    palette = {1: style.COLORS["Low"], 2: style.COLORS["Mid"], 3: style.COLORS["High"]}
    for ax, (col, label) in zip(axes[1], ENGAGEMENT):
        sub = tbl.dropna(subset=[col, "dat_before_glove", "dat_after_score"]).copy()
        if len(sub) < 10:
            style.style_axis(ax, title="")
            continue
        sub["tertile"] = pd.qcut(sub[col], 3, labels=[1, 2, 3], duplicates="drop")
        for t in [1, 2, 3]:
            s = sub[sub["tertile"] == t]
            ax.scatter(s["dat_before_glove"], s["dat_after_score"],
                       s=3.5, alpha=0.55, color=palette[t], linewidths=0,
                       label=f"T{t}")
        lo = min(sub["dat_before_glove"].min(), sub["dat_after_score"].min())
        hi = max(sub["dat_before_glove"].max(), sub["dat_after_score"].max())
        ax.plot([lo, hi], [lo, hi], color="black", lw=0.6, ls="--")
        style.style_axis(ax, title="",
                         xlabel="DAT pre",
                         ylabel="DAT post" if col == ENGAGEMENT[0][0] else "")
    fig.suptitle("Does pareidolia engagement boost DAT?", y=1.00)
    if show:
        plt.show()
    style.savefig(fig, "pareidolia_boosts_dat")
    plt.close(fig)
    return pd.DataFrame(stats_rows)


def main(show: bool = True):
    tbl = build_table()
    print(f"Participants with pre & post DAT + engagement: {len(tbl):,}")
    stats = plot(tbl, show=show)
    tbl.to_csv(config.OUTPUTS_DIR / "pareidolia_boost_table.csv", index=False)
    stats.to_csv(config.OUTPUTS_DIR / "pareidolia_boost_stats.csv", index=False)
    print(stats.to_string(index=False))
    return tbl, stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    main(show=not args.no_show)
