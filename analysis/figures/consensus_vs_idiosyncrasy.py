"""Do high-creativity participants give the rare percepts, and low-creativity
the consensus percepts?

Two related questions:

(Q1) **Within-image idiosyncrasy** — for each (participant, image), how
     uncommon are the words they typed compared to the rest of the cohort's
     answers for that *same image*? Operationalised as the mean
     −log relative frequency (mean surprisal) of the participant's words
     conditioned on the image. Per-participant mean across images is the
     trait-level idiosyncrasy index.

(Q2) **Consensus-image vs ambiguous-image preference** — split stimuli into
     "high-consensus" (low entropy of percept distribution) and
     "low-consensus / ambiguous" (high entropy). Do high-creativity
     participants disproportionately submit *any* word on ambiguous images
     while low-creativity stick to high-consensus ones?

Both are tested with bivariate ±3 SD outlier trim before Pearson.

Usage:
    python -m analysis.figures.consensus_vs_idiosyncrasy
"""
from __future__ import annotations

import argparse
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

from .. import config, dat_helper, embeddings, outliers, parse_events, style


ASCII = re.compile(r"^[A-Za-z]+$")
LABELS = ["Low", "Mid", "High"]


# ─── trials → long words ─────────────────────────────────────────────────────

def _trials_long_with_dat() -> pd.DataFrame:
    """One row per (participant, trial, word) with ASCII filter and DAT score."""
    trials = parse_events.cached_trials()
    edict = embeddings.embedding_dict()
    nb = dat_helper.notebook_cohort()
    keep = set(nb["user_id"])
    dat_map = dict(zip(nb["user_id"], nb["ref_dat_score"]))
    rows = []
    for _, t in trials.iterrows():
        if t["user_id"] not in keep: continue
        ws = t.get("words")
        if ws is None: continue
        for w in list(ws):
            w = str(w).strip().lower()
            if not (ASCII.fullmatch(w) and w in edict
                    and w not in config.SEMANTIC_STOPWORDS):
                continue
            rows.append({"user_id": t["user_id"], "trial_index": t["trial_index"],
                         "fd_level": t["fd_level"],
                         "url": t["url_stimulus"], "word": w,
                         "ref_dat_score": dat_map.get(t["user_id"])})
    return pd.DataFrame(rows).dropna(subset=["ref_dat_score"])


# ─── Q1: per-(participant, image) idiosyncrasy ───────────────────────────────

def per_user_idiosyncrasy(long: pd.DataFrame) -> pd.DataFrame:
    """For each image, compute the word-frequency distribution across the
    cohort. For each (participant, image), compute the mean −log relative
    frequency of their words on that image. Average across images per user.
    """
    long = long.dropna(subset=["url", "word"])
    # Per-image word counts (all participants).
    img_word_cnt = long.groupby(["url", "word"]).size().rename("n").reset_index()
    img_total = img_word_cnt.groupby("url")["n"].sum().rename("img_total")
    img_word_cnt = img_word_cnt.join(img_total, on="url")
    img_word_cnt["neg_log_p"] = -np.log(img_word_cnt["n"] / img_word_cnt["img_total"])
    # Lookup table (url, word) -> neg_log_p
    lookup = img_word_cnt.set_index(["url", "word"])["neg_log_p"]
    long = long.copy()
    long["neg_log_p"] = long.apply(
        lambda r: lookup.get((r["url"], r["word"]), np.nan), axis=1
    )
    # mean per (user, image), then mean across images
    per_user_img = (
        long.groupby(["user_id", "url"])["neg_log_p"]
            .mean().rename("img_mean_neg_log_p").reset_index()
    )
    per_user = (
        per_user_img.groupby("user_id")["img_mean_neg_log_p"]
                    .mean().rename("idiosyncrasy").reset_index()
                    .merge(long.groupby("user_id")["ref_dat_score"].first()
                                .reset_index(), on="user_id")
    )
    return per_user, per_user_img


# Fraction of percepts that are *unique* (only that participant said this
# word for this image) — alternative metric, more interpretable.
def per_user_unique_fraction(long: pd.DataFrame) -> pd.DataFrame:
    long = long.dropna(subset=["url", "word"])
    pic = long.groupby(["url", "word"])["user_id"].nunique().rename("n_users")
    long = long.join(pic, on=["url", "word"])
    long["is_unique"] = long["n_users"] == 1
    per_user = (
        long.groupby("user_id")
            .agg(frac_unique=("is_unique", "mean"),
                 n_words=("word", "size"),
                 dat=("ref_dat_score", "first"))
            .reset_index()
    )
    return per_user


# ─── Q2: image consensus split ───────────────────────────────────────────────

def image_consensus_table(long: pd.DataFrame) -> pd.DataFrame:
    """Per-image entropy of percept distribution + n_respondents."""
    cnt = long.groupby(["url", "word"]).size().rename("n").reset_index()
    out = []
    for url, g in cnt.groupby("url"):
        p = g["n"] / g["n"].sum()
        H = float(-(p * np.log2(p)).sum())
        out.append({"url": url, "vocab_entropy": H,
                    "n_words": int(g["n"].sum()),
                    "n_unique_words": int(len(g))})
    return pd.DataFrame(out)


def per_user_engagement_by_image_class(long: pd.DataFrame,
                                        img_table: pd.DataFrame) -> pd.DataFrame:
    """For each participant: how many words they gave on "consensus" images
    (low-entropy) vs "ambiguous" images (high-entropy)."""
    # Median split on image entropy
    med = img_table["vocab_entropy"].median()
    img_class = dict(zip(
        img_table["url"],
        np.where(img_table["vocab_entropy"] >= med, "ambiguous", "consensus"),
    ))
    long = long.copy()
    long["img_class"] = long["url"].map(img_class)
    long = long.dropna(subset=["img_class"])
    g = (long.groupby(["user_id", "img_class"]).size()
              .unstack(fill_value=0))
    g.columns = ["n_words_ambiguous" if c == "ambiguous"
                  else "n_words_consensus" for c in g.columns]
    for c in ("n_words_ambiguous", "n_words_consensus"):
        if c not in g.columns: g[c] = 0
    g["ambig_minus_cons"] = g["n_words_ambiguous"] - g["n_words_consensus"]
    total = g["n_words_ambiguous"] + g["n_words_consensus"]
    g["ambig_share"] = np.where(total > 0,
                                  g["n_words_ambiguous"] / total, np.nan)
    return g.reset_index().merge(
        long.groupby("user_id")["ref_dat_score"].first().reset_index(),
        on="user_id",
    )


# ─── plotting ─────────────────────────────────────────────────────────────────

def figure_idiosyncrasy(per_user: pd.DataFrame, per_user_uniq: pd.DataFrame,
                         show: bool = True):
    style.apply()
    fig, axes = plt.subplots(
        1, 2, figsize=(style.COL_1_5, 2.6),
        gridspec_kw=dict(wspace=0.45),
    )

    # Panel 1: per-image surprisal vs DAT
    sub, n_drop = outliers.trim_sd(per_user, ["idiosyncrasy", "ref_dat_score"], sd=3.0)
    r, p = pearsonr(sub["ref_dat_score"], sub["idiosyncrasy"])
    rho, p_rho = spearmanr(sub["ref_dat_score"], sub["idiosyncrasy"])
    print(f"  idiosyncrasy: r={r:+.3f} p={p:.3g}  rho={rho:+.3f} p={p_rho:.3g}")
    axes[0].scatter(sub["ref_dat_score"], sub["idiosyncrasy"],
                    s=3.5, alpha=0.55, color=style.COLORS["dot"], linewidths=0)
    sns.regplot(data=sub, x="ref_dat_score", y="idiosyncrasy",
                scatter=False, color=style.COLORS["fit"],
                line_kws=dict(linewidth=1.0), ax=axes[0])
    style.style_axis(axes[0], title="",
                     xlabel="DAT",
                     ylabel="Idiosyncrasy\n(per-image surprisal of percepts)")
    style.small_stat_annotation(axes[0], r, p, loc="upper left")

    # Panel 2: fraction of fully-unique percepts vs DAT
    sub, n_drop = outliers.trim_sd(per_user_uniq, ["frac_unique", "dat"], sd=3.0)
    r2, p2 = pearsonr(sub["dat"], sub["frac_unique"])
    rho2, p_rho2 = spearmanr(sub["dat"], sub["frac_unique"])
    print(f"  frac_unique:  r={r2:+.3f} p={p2:.3g}  rho={rho2:+.3f} p={p_rho2:.3g}")
    axes[1].scatter(sub["dat"], sub["frac_unique"],
                    s=3.5, alpha=0.55, color=style.COLORS["dot"], linewidths=0)
    sns.regplot(data=sub, x="dat", y="frac_unique",
                scatter=False, color=style.COLORS["fit"],
                line_kws=dict(linewidth=1.0), ax=axes[1])
    style.style_axis(axes[1], title="",
                     xlabel="DAT",
                     ylabel="Fraction of percepts\nthat are unique to this user")
    style.small_stat_annotation(axes[1], r2, p2, loc="upper left")

    fig.suptitle("Per-image idiosyncrasy of percepts vs creativity", y=1.04)
    if show: plt.show()
    style.savefig(fig, "consensus_idiosyncrasy")
    plt.close(fig)
    return r, p, r2, p2


def figure_image_class(per_user_class: pd.DataFrame, show: bool = True):
    """Does high-DAT mean MORE words on ambiguous images and FEWER on
    consensus ones? Plot two correlations."""
    style.apply()
    fig, axes = plt.subplots(
        1, 3, figsize=(style.COL_2, 2.6),
        gridspec_kw=dict(wspace=0.45),
    )
    for ax, col, label in zip(
        axes,
        ["n_words_consensus", "n_words_ambiguous", "ambig_share"],
        ["Words on high-consensus images",
         "Words on ambiguous images",
         "Share of words on ambiguous images"],
    ):
        sub, _ = outliers.trim_sd(per_user_class, [col, "ref_dat_score"], sd=3.0)
        r, p = pearsonr(sub["ref_dat_score"], sub[col])
        rho, p_rho = spearmanr(sub["ref_dat_score"], sub[col])
        print(f"  {col}: r={r:+.3f} p={p:.3g}  rho={rho:+.3f} p={p_rho:.3g}")
        ax.scatter(sub["ref_dat_score"], sub[col],
                   s=3.5, alpha=0.55, color=style.COLORS["dot"], linewidths=0)
        sns.regplot(data=sub, x="ref_dat_score", y=col, scatter=False,
                    color=style.COLORS["fit"],
                    line_kws=dict(linewidth=1.0), ax=ax)
        style.style_axis(ax, title="", xlabel="DAT", ylabel=label)
        style.small_stat_annotation(ax, r, p, loc="upper left")
    fig.suptitle("Engagement with consensus vs ambiguous images, by creativity",
                 y=1.04)
    if show: plt.show()
    style.savefig(fig, "consensus_image_engagement")
    plt.close(fig)


# ─── runner ──────────────────────────────────────────────────────────────────

def main(show: bool = True):
    long = _trials_long_with_dat()
    print(f"long: {len(long):,} (user, trial, word) rows, "
          f"{long['user_id'].nunique()} users, "
          f"{long['url'].nunique()} images")

    per_user, per_user_img = per_user_idiosyncrasy(long)
    per_user_uniq = per_user_unique_fraction(long)
    print("\n=== Per-image idiosyncrasy ===")
    figure_idiosyncrasy(per_user, per_user_uniq, show=show)

    img_table = image_consensus_table(long)
    per_user_class = per_user_engagement_by_image_class(long, img_table)
    print("\n=== Engagement by image class (median split on entropy) ===")
    figure_image_class(per_user_class, show=show)

    per_user.to_csv(config.OUTPUTS_DIR / "idiosyncrasy_per_user.csv",
                    index=False)
    per_user_uniq.to_csv(config.OUTPUTS_DIR / "unique_fraction_per_user.csv",
                          index=False)
    per_user_class.to_csv(config.OUTPUTS_DIR / "consensus_engagement_per_user.csv",
                          index=False)
    img_table.to_csv(config.OUTPUTS_DIR / "image_consensus_table.csv",
                     index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    main(show=not args.no_show)
