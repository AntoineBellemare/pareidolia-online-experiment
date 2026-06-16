"""How much do observers agree on what's in each image?

Three orthogonal questions, all at the image level:

  (A) For each of the 30 stimuli, is there a canonical percept (high
      consensus) or do observers report wildly different things
      (low consensus)?

  (B) Does consensus vary with stimulus FD?

  (C) Does consensus vary with observer creativity? Specifically, do
      low-DAT participants converge on similar percepts for the same
      image while high-DAT participants disperse?

Metrics (all approximately sample-size invariant):

  * vocab_entropy   = Shannon entropy of the per-image percept-word
                      distribution (bits). Lower = more consensus.
  * modal_share     = fraction of all percept words on this image that
                      equal the single most common word. Higher = more
                      consensus.
  * mean_pair_sim   = mean cosine similarity between two random percept
                      embeddings from *different* participants for this
                      image. Higher = more semantic consensus
                      (synonym-tolerant; the right metric across tertiles
                      because it does not depend on raw count).

Usage:
    python -m analysis.figures.image_consensus
"""
from __future__ import annotations

import argparse
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare, kruskal, mannwhitneyu, wilcoxon

from .. import config, dat_helper, embeddings, parse_events, style


ASCII = re.compile(r"^[A-Za-z]+$")
TERTILES = ["Low", "Mid", "High"]


# ─── data ───────────────────────────────────────────────────────────────────

def _trials_long_with_dat() -> pd.DataFrame:
    """One row per (participant, image, word), restricted to the main
    cohort + ASCII single-word filter + DAT-pre attached.
    """
    trials = parse_events.cached_trials()
    edict = embeddings.embedding_dict()
    nb = dat_helper.notebook_cohort()
    keep = set(nb["user_id"])
    dat_map = dict(zip(nb["user_id"], nb["ref_dat_score"]))
    rows = []
    for _, t in trials.iterrows():
        if t["user_id"] not in keep:
            continue
        ws = t.get("words")
        if ws is None:
            continue
        for w in list(ws):
            w = str(w).strip().lower()
            if not (ASCII.fullmatch(w) and w in edict
                    and w not in config.SEMANTIC_STOPWORDS):
                continue
            rows.append({
                "user_id": t["user_id"], "url": t["url_stimulus"],
                "fd_level": t["fd_level"], "word": w,
                "ref_dat_score": dat_map.get(t["user_id"]),
            })
    df = pd.DataFrame(rows).dropna(subset=["ref_dat_score", "url"])
    return df


def _tertile(score: pd.Series) -> pd.Categorical:
    return pd.qcut(score, 3, labels=TERTILES, duplicates="drop")


# ─── per-(image, tertile) metrics ───────────────────────────────────────────

def _entropy(words: list[str]) -> float:
    if len(words) < 2:
        return np.nan
    cnt = Counter(words)
    p = np.array(list(cnt.values()), dtype=float) / sum(cnt.values())
    return float(-(p * np.log2(p)).sum())


def _modal_share(words: list[str]) -> float:
    if not words:
        return np.nan
    cnt = Counter(words)
    return cnt.most_common(1)[0][1] / sum(cnt.values())


def _mean_pair_sim(words: list[str], user_ids: list[str],
                    edict, n_pairs: int = 400,
                    rng: np.random.Generator | None = None) -> float:
    """Mean cosine similarity between random *cross-participant* word pairs.

    Words from the same participant on the same image are excluded so the
    metric estimates inter-observer consensus rather than within-observer
    self-similarity.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if len(words) < 2:
        return np.nan
    V = np.vstack([edict[w] for w in words])
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    uids = np.array(user_ids)
    n = len(words)
    sims = []
    tries = 0
    max_tries = n_pairs * 20
    while len(sims) < n_pairs and tries < max_tries:
        i, j = rng.integers(0, n, size=2)
        tries += 1
        if i == j or uids[i] == uids[j]:
            continue
        sims.append(float(Vn[i] @ Vn[j]))
    if not sims:
        return np.nan
    return float(np.mean(sims))


def per_image_overall(long: pd.DataFrame, edict) -> pd.DataFrame:
    """One row per image with overall consensus metrics."""
    rng = np.random.default_rng(0)
    rows = []
    for url, g in long.groupby("url"):
        words = g["word"].tolist()
        uids = g["user_id"].tolist()
        rows.append({
            "url": url,
            "fd_level": g["fd_level"].iloc[0],
            "n_words": len(words),
            "n_users": g["user_id"].nunique(),
            "vocab_entropy": _entropy(words),
            "modal_share": _modal_share(words),
            "mean_pair_sim": _mean_pair_sim(words, uids, edict, rng=rng),
            "top_word": Counter(words).most_common(1)[0][0],
            "top_word_n": Counter(words).most_common(1)[0][1],
        })
    return pd.DataFrame(rows)


def per_image_by_tertile(long: pd.DataFrame, edict,
                          n_pairs: int = 300) -> pd.DataFrame:
    """One row per (image, DAT tertile) with the same consensus metrics."""
    long = long.copy()
    long["tertile"] = _tertile(long["ref_dat_score"])
    rng = np.random.default_rng(1)
    rows = []
    for (url, tert), g in long.groupby(["url", "tertile"]):
        words = g["word"].tolist()
        uids = g["user_id"].tolist()
        rows.append({
            "url": url, "tertile": str(tert),
            "fd_level": g["fd_level"].iloc[0],
            "n_words": len(words),
            "n_users": g["user_id"].nunique(),
            "vocab_entropy": _entropy(words),
            "modal_share": _modal_share(words),
            "mean_pair_sim": _mean_pair_sim(words, uids, edict,
                                             n_pairs=n_pairs, rng=rng),
        })
    return pd.DataFrame(rows)


# ─── figures ────────────────────────────────────────────────────────────────

def figure_overall(per_img: pd.DataFrame, show: bool) -> None:
    """Three panels:
      (a) per-image consensus ranked, coloured by FD
      (b) consensus by FD (boxplot)
      (c) consensus by FD using the modal-share metric (sanity check)
    """
    style.apply()
    fig, axes = plt.subplots(
        1, 3, figsize=(style.COL_2, 2.6),
        gridspec_kw=dict(wspace=0.45, width_ratios=[1.4, 0.85, 0.85]),
    )

    # Panel a: ranked per-image bar chart (mean_pair_sim, 30 dots)
    s = per_img.sort_values("mean_pair_sim").reset_index(drop=True)
    xs = np.arange(len(s))
    cols = [style.COLORS[fd] for fd in s["fd_level"]]
    axes[0].bar(xs, s["mean_pair_sim"].values, color=cols, edgecolor="black",
                lw=0.3, alpha=0.9)
    axes[0].set_xticks([])
    style.style_axis(axes[0],
                     title="",
                     ylabel="Inter-observer similarity\n(mean cross-user cos)",
                     xlabel="30 images, ranked by consensus")
    # FD legend
    from matplotlib.patches import Patch
    handles = [Patch(color=style.COLORS[f], label=f) for f in config.FD_LEVELS]
    axes[0].legend(handles=handles, frameon=False, fontsize=6,
                    loc="upper left", title="FD level", title_fontsize=6)

    # Panel b: box of mean_pair_sim by FD
    sns.boxplot(data=per_img, x="fd_level", y="mean_pair_sim",
                 order=config.FD_LEVELS,
                 palette=[style.COLORS[fd] for fd in config.FD_LEVELS],
                 ax=axes[1], width=0.55, fliersize=0, linewidth=0.5)
    sns.stripplot(data=per_img, x="fd_level", y="mean_pair_sim",
                   order=config.FD_LEVELS, color="black",
                   size=2.5, alpha=0.6, jitter=0.15, ax=axes[1])
    groups_b = [per_img[per_img["fd_level"] == f]["mean_pair_sim"].values
                for f in config.FD_LEVELS]
    H_b, p_b = kruskal(*groups_b)
    style.style_axis(axes[1],
                     title="", ylabel="Mean pair sim", xlabel="")
    # pair-wise Mann-Whitney brackets
    pair_x = {"FD12": 0, "FD14": 1, "FD16": 2}
    comps = []
    for a, b in [("FD12", "FD14"), ("FD14", "FD16"), ("FD12", "FD16")]:
        ga = per_img[per_img["fd_level"] == a]["mean_pair_sim"].values
        gb = per_img[per_img["fd_level"] == b]["mean_pair_sim"].values
        _, pw = mannwhitneyu(ga, gb, alternative="two-sided")
        comps.append((pair_x[a], pair_x[b], float(pw)))
    style.sig_brackets(axes[1], comps)

    # Panel c: same but with modal_share
    sns.boxplot(data=per_img, x="fd_level", y="modal_share",
                 order=config.FD_LEVELS,
                 palette=[style.COLORS[fd] for fd in config.FD_LEVELS],
                 ax=axes[2], width=0.55, fliersize=0, linewidth=0.5)
    sns.stripplot(data=per_img, x="fd_level", y="modal_share",
                   order=config.FD_LEVELS, color="black",
                   size=2.5, alpha=0.6, jitter=0.15, ax=axes[2])
    style.style_axis(axes[2],
                     title="", ylabel="Modal-word share", xlabel="")
    comps2 = []
    for a, b in [("FD12", "FD14"), ("FD14", "FD16"), ("FD12", "FD16")]:
        ga = per_img[per_img["fd_level"] == a]["modal_share"].values
        gb = per_img[per_img["fd_level"] == b]["modal_share"].values
        _, pw = mannwhitneyu(ga, gb, alternative="two-sided")
        comps2.append((pair_x[a], pair_x[b], float(pw)))
    style.sig_brackets(axes[2], comps2)

    fig.suptitle("Per-image consensus and its modulation by stimulus FD",
                 y=1.04)
    if show:
        plt.show()
    style.savefig(fig, "image_consensus_fd")
    plt.close(fig)
    print(f"  KW on mean_pair_sim by FD: H={H_b:.2f} p={p_b:.3g}")


def figure_fd_paper(per_img: pd.DataFrame, show: bool) -> None:
    """2-panel paper figure: per-FD boxplots of mean_pair_sim + modal_share.

    Tighter y-limits than the exploratory figure so the FD shift is
    visible rather than dwarfed by a few outliers.
    """
    style.apply()
    fig, axes = plt.subplots(
        1, 2, figsize=(style.COL_1_5, 2.4),
        gridspec_kw=dict(wspace=0.45),
    )
    pair_x = {"FD12": 0, "FD14": 1, "FD16": 2}

    # ── panel a: mean cross-observer cosine ────────────────────────────────
    ax = axes[0]
    sns.boxplot(data=per_img, x="fd_level", y="mean_pair_sim",
                 order=config.FD_LEVELS,
                 palette=[style.COLORS[fd] for fd in config.FD_LEVELS],
                 ax=ax, width=0.55, fliersize=0, linewidth=0.5)
    sns.stripplot(data=per_img, x="fd_level", y="mean_pair_sim",
                   order=config.FD_LEVELS, color="black",
                   size=2, alpha=0.55, jitter=0.15, ax=ax)
    # Clip y around the bulk of the distribution.
    vals = per_img["mean_pair_sim"].values
    lo, hi = float(np.percentile(vals, 2)), float(np.percentile(vals, 98))
    pad = (hi - lo) * 0.10
    ax.set_ylim(lo - pad, hi + pad)
    style.style_axis(ax, title="", xlabel="",
                     ylabel="Mean cross-observer\ncosine similarity")
    comps = []
    for a, b in [("FD12", "FD14"), ("FD14", "FD16"), ("FD12", "FD16")]:
        ga = per_img[per_img["fd_level"] == a]["mean_pair_sim"].values
        gb = per_img[per_img["fd_level"] == b]["mean_pair_sim"].values
        _, pw = mannwhitneyu(ga, gb, alternative="two-sided")
        comps.append((pair_x[a], pair_x[b], float(pw)))
    style.sig_brackets(ax, comps)

    # ── panel b: modal-word share, with tighter y to see the box ──────────
    ax = axes[1]
    sns.boxplot(data=per_img, x="fd_level", y="modal_share",
                 order=config.FD_LEVELS,
                 palette=[style.COLORS[fd] for fd in config.FD_LEVELS],
                 ax=ax, width=0.55, fliersize=0, linewidth=0.5)
    sns.stripplot(data=per_img, x="fd_level", y="modal_share",
                   order=config.FD_LEVELS, color="black",
                   size=2, alpha=0.55, jitter=0.15, ax=ax)
    # Cap y above the 95th percentile so the box-and-whisker is readable.
    vals = per_img["modal_share"].values
    y_top = float(np.percentile(vals, 95))
    ax.set_ylim(0, y_top * 1.05)
    style.style_axis(ax, title="", xlabel="",
                     ylabel="Modal-word share\n(top word / all words)")
    comps2 = []
    for a, b in [("FD12", "FD14"), ("FD14", "FD16"), ("FD12", "FD16")]:
        ga = per_img[per_img["fd_level"] == a]["modal_share"].values
        gb = per_img[per_img["fd_level"] == b]["modal_share"].values
        _, pw = mannwhitneyu(ga, gb, alternative="two-sided")
        comps2.append((pair_x[a], pair_x[b], float(pw)))
    style.sig_brackets(ax, comps2)

    fig.suptitle("Inter-observer agreement decreases with image complexity",
                 y=1.04)
    if show: plt.show()
    style.savefig(fig, "image_consensus_fd_paper")
    plt.close(fig)


def figure_tertile(per_img_tert: pd.DataFrame, per_img: pd.DataFrame,
                    show: bool) -> None:
    """Do the same 30 images elicit different consensus across DAT tertiles?

    Within-image paired design: same image rated by Low / Mid / High DAT
    sub-cohorts. We compare per-tertile consensus on the same image with
    Friedman + pair-wise Wilcoxon.
    """
    style.apply()
    fig, axes = plt.subplots(
        1, 3, figsize=(style.COL_2, 2.8),
        gridspec_kw=dict(wspace=0.45, width_ratios=[1.0, 1.0, 1.3]),
    )

    # Panel a: paired lines (one line per image) across the 3 tertiles,
    # metric = mean_pair_sim.
    pivot = per_img_tert.pivot(index="url", columns="tertile",
                                 values="mean_pair_sim")
    pivot = pivot[TERTILES].dropna()
    xs = np.arange(3)
    for url, row in pivot.iterrows():
        fd = per_img.set_index("url").loc[url, "fd_level"]
        axes[0].plot(xs, row.values, marker="o", ms=2, lw=0.5, alpha=0.55,
                      color=style.COLORS[fd])
    means = pivot.mean()
    axes[0].plot(xs, means.values, marker="s", ms=5, lw=1.6, color="black",
                  zorder=5, label="mean")
    axes[0].set_xticks(xs); axes[0].set_xticklabels(TERTILES)
    style.style_axis(axes[0],
                     title="", ylabel="Mean pair sim",
                     xlabel="DAT tertile")
    # Friedman + Wilcoxon
    chi, p_fr = friedmanchisquare(*[pivot[c] for c in TERTILES])
    print(f"  Friedman on mean_pair_sim by tertile: chi2={chi:.2f}, "
          f"p={p_fr:.3g}, n_images={len(pivot)}")
    pair_x = {"Low": 0, "Mid": 1, "High": 2}
    comps = []
    for a, b in [("Low", "Mid"), ("Mid", "High"), ("Low", "High")]:
        _, pw = wilcoxon(pivot[a], pivot[b])
        comps.append((pair_x[a], pair_x[b], float(pw)))
        print(f"    Wilcoxon {a} vs {b}: p={pw:.3g}")
    style.sig_brackets(axes[0], comps)

    # Panel b: same plot but using vocab_entropy (lower=more consensus).
    pivot_e = per_img_tert.pivot(index="url", columns="tertile",
                                   values="vocab_entropy")
    pivot_e = pivot_e[TERTILES].dropna()
    for url, row in pivot_e.iterrows():
        fd = per_img.set_index("url").loc[url, "fd_level"]
        axes[1].plot(xs, row.values, marker="o", ms=2, lw=0.5, alpha=0.55,
                      color=style.COLORS[fd])
    means_e = pivot_e.mean()
    axes[1].plot(xs, means_e.values, marker="s", ms=5, lw=1.6, color="black",
                  zorder=5)
    axes[1].set_xticks(xs); axes[1].set_xticklabels(TERTILES)
    style.style_axis(axes[1],
                     title="", ylabel="Vocab entropy (bits)\n← more consensus",
                     xlabel="DAT tertile")
    comps_e = []
    chi_e, p_fr_e = friedmanchisquare(*[pivot_e[c] for c in TERTILES])
    print(f"  Friedman on vocab_entropy by tertile: chi2={chi_e:.2f}, "
          f"p={p_fr_e:.3g}, n_images={len(pivot_e)}")
    for a, b in [("Low", "Mid"), ("Mid", "High"), ("Low", "High")]:
        _, pw = wilcoxon(pivot_e[a], pivot_e[b])
        comps_e.append((pair_x[a], pair_x[b], float(pw)))
    style.sig_brackets(axes[1], comps_e)

    # Panel c: heatmap of pivot (rows=images sorted by overall consensus,
    # cols=tertile, value=mean_pair_sim) — gives a visual sense of whether
    # the rank order of images is stable across tertiles, and where the
    # tertile differences concentrate.
    img_order = (per_img.sort_values("mean_pair_sim")["url"].tolist())
    img_order = [u for u in img_order if u in pivot.index]
    M = pivot.loc[img_order, TERTILES].values
    im = axes[2].imshow(M, aspect="auto", cmap="viridis",
                         interpolation="nearest")
    axes[2].set_xticks(np.arange(3)); axes[2].set_xticklabels(TERTILES)
    axes[2].set_yticks([])
    axes[2].set_xlabel("DAT tertile")
    axes[2].set_ylabel("Images sorted by overall consensus\n(bottom = low)")
    # Add a colorbar
    cax = axes[2].inset_axes([1.05, 0.05, 0.05, 0.9])
    plt.colorbar(im, cax=cax)
    cax.set_ylabel("Mean pair sim", fontsize=6)
    cax.tick_params(labelsize=6)

    fig.suptitle("Same image, different observers: consensus by DAT tertile",
                 y=1.04)
    if show:
        plt.show()
    style.savefig(fig, "image_consensus_dat")
    plt.close(fig)


def figure_examples(per_img: pd.DataFrame, long: pd.DataFrame,
                     show: bool, k: int = 5) -> None:
    """Top-k highest- and lowest-consensus images with their dominant words.

    Renders a small text-table style figure (no thumbnails since URLs are
    just slugs in the dataset).
    """
    style.apply()
    top = per_img.nlargest(k, "mean_pair_sim")
    bot = per_img.nsmallest(k, "mean_pair_sim")

    fig, axes = plt.subplots(1, 2, figsize=(style.COL_2, 3.0),
                              gridspec_kw=dict(wspace=0.05))
    for ax, tbl, title in [
        (axes[0], top, "Most consensual images"),
        (axes[1], bot, "Most ambiguous images"),
    ]:
        ax.axis("off")
        ax.set_title(title, fontsize=9, weight="bold", color="#222222")
        rows_text = []
        for _, r in tbl.iterrows():
            url = r["url"]
            words = long[long["url"] == url]["word"].tolist()
            top_words = [w for w, _ in Counter(words).most_common(8)]
            stem = str(url).rsplit("/", 1)[-1][:24]
            rows_text.append(
                (f"[{r['fd_level']}]  {stem}",
                 r["mean_pair_sim"],
                 ", ".join(top_words))
            )
        for i, (lbl, sim, words) in enumerate(rows_text):
            y = 0.95 - i * 0.18
            fd = lbl[1:5]
            ax.text(0.0, y, lbl, transform=ax.transAxes,
                     fontsize=6.5, family="monospace",
                     color=style.COLORS.get(fd, "#444"))
            ax.text(0.0, y - 0.05, f"sim = {sim:.3f}",
                     transform=ax.transAxes, fontsize=6.0,
                     color=style.COLORS["muted"])
            ax.text(0.0, y - 0.10, words, transform=ax.transAxes,
                     fontsize=7, color="#111", wrap=True)

    fig.suptitle("Examples: top words for the highest- and lowest-consensus "
                 "images", y=1.02)
    if show:
        plt.show()
    style.savefig(fig, "image_consensus_examples")
    plt.close(fig)


# ─── runner ─────────────────────────────────────────────────────────────────

def main(show: bool = True) -> None:
    long = _trials_long_with_dat()
    edict = embeddings.embedding_dict()
    print(f"Long table: {len(long):,} (user, image, word) rows; "
          f"{long['user_id'].nunique()} users; "
          f"{long['url'].nunique()} images")

    per_img = per_image_overall(long, edict)
    per_img.to_csv(config.OUTPUTS_DIR / "image_consensus_overall.csv",
                    index=False)
    print(f"  Per-image table: {len(per_img)} images")
    print(per_img.describe()[["vocab_entropy", "modal_share",
                                "mean_pair_sim"]].T)

    print("\n=== Per-image consensus by FD ===")
    figure_overall(per_img, show=show)
    figure_fd_paper(per_img, show=show)

    per_img_tert = per_image_by_tertile(long, edict)
    per_img_tert.to_csv(config.OUTPUTS_DIR / "image_consensus_by_tertile.csv",
                         index=False)
    print(f"  Per-(image, tertile) table: {len(per_img_tert)} rows")

    print("\n=== Per-image consensus by DAT tertile ===")
    figure_tertile(per_img_tert, per_img, show=show)

    print("\n=== Example images ===")
    figure_examples(per_img, long, show=show)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    main(show=not args.no_show)
