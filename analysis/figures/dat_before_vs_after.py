"""Compute DAT-after, compare to DAT-before.

Pipeline:
  1. Load main cohort (~500 participants with DAT-after responses).
  2. Score DAT-after with the canonical GloVe-840B method.
  3. (Optional) Re-score DAT-before from the stored `ref_dat_words` with the
     SAME GloVe model — gives an apples-to-apples comparison since the
     `ref_dat_score` in the DB was computed by the production frontend (which
     may use a different embedding source / cleaning rules).
  4. Plot:
        a) Scatter of pre vs post DAT (with y=x line, paired test).
        b) Histogram of (post - pre) deltas.
  5. Save tabular CSV: user_id, ref_dat_score_db, dat_before_glove,
     dat_after_glove, delta.

Usage:
    python -m analysis.figures.dat_before_vs_after
    python -m analysis.figures.dat_before_vs_after --no-rescore-before
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon

from .. import config, dat, loader, parse_events


def _parse_ref_dat_words(s) -> list[str]:
    if isinstance(s, list):
        return s
    if not isinstance(s, str) or not s.strip():
        return []
    try:
        return list(json.loads(s))
    except Exception:
        return []


def main(rescore_before: bool = True, show: bool = True):
    # Prefer the shipped enriched parquet (has dat_before_glove + dat_after_score
    # already computed) so the figure rebuilds without the SQLite or GloVe.
    shipped = config.cached_parquet("participants_with_dat_after.parquet")
    if shipped.exists():
        participants = pd.read_parquet(shipped)
        print(f"Loaded enriched DAT cache: {shipped} ({len(participants):,} rows)")
    else:
        participants = parse_events.main_cohort().copy()
        model = dat.load_model()
        print(f"DAT model: {model.source}  ({len(model.vectors):,} words)")

        # DAT-after — score from the 10 Q-responses captured AFTER the task.
        after_scores = []
        after_used_n = []
        for words in participants["dat_after_words"]:
            words = list(words) if words is not None else []
            s, used = dat.score(words, model=model, return_words=True)
            after_scores.append(s)
            after_used_n.append(len(used))
        participants["dat_after_score"] = after_scores
        participants["dat_after_n_used"] = after_used_n

        # DAT-before — optionally re-score for consistency.
        if rescore_before:
            sessions = loader.load_sessions()
            ref_map = dict(zip(sessions["user_id"], sessions["ref_dat_words"]))
            before_scores = []
            for uid in participants["user_id"]:
                words = _parse_ref_dat_words(ref_map.get(uid))
                s = dat.score(words, model=model)
                before_scores.append(s)
            participants["dat_before_glove"] = before_scores
        else:
            participants["dat_before_glove"] = participants["ref_dat_score"]

        # Cache the enriched participants table for downstream scripts.
        enriched = config.CACHE_DIR / "participants_with_dat_after.parquet"
        participants.to_parquet(enriched)
        print(f"Cached -> {enriched}")

    keep = participants.dropna(subset=["dat_before_glove", "dat_after_score"]).copy()
    keep["delta"] = keep["dat_after_score"] - keep["dat_before_glove"]
    print(f"\nValid pre & post DAT scores: n = {len(keep):,}")

    # ── stats ─────────────────────────────────────────────────────────────────
    r, p = pearsonr(keep["dat_before_glove"], keep["dat_after_score"])
    w_stat, w_p = wilcoxon(keep["delta"])
    print(f"  Pearson  pre vs post:  r = {r:.3f}   p = {p:.3g}")
    print(f"  Wilcoxon  on delta:    W = {w_stat:.1f}   p = {w_p:.3g}")
    print(f"  Mean delta:           {keep['delta'].mean():+.2f}  (sd {keep['delta'].std():.2f})")

    # ── plot ──────────────────────────────────────────────────────────────────
    from .. import style
    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(style.COL_1, 2.0),
                             gridspec_kw=dict(wspace=0.55))

    ax = axes[0]
    ax.scatter(keep["dat_before_glove"], keep["dat_after_score"],
               s=3.5, alpha=0.55, color=style.COLORS["dot"], linewidths=0)
    lo, hi = float(min(keep[["dat_before_glove", "dat_after_score"]].min())), \
             float(max(keep[["dat_before_glove", "dat_after_score"]].max()))
    ax.plot([lo, hi], [lo, hi], "--", color=style.COLORS["muted"], lw=0.7)
    style.style_axis(ax, title="DAT post vs DAT pre",
                     xlabel="DAT pre", ylabel="DAT post")
    style.small_stat_annotation(ax, r, p, loc="upper left")

    ax = axes[1]
    ax.hist(keep["delta"], bins=30, color=style.COLORS["fit"],
            edgecolor="white", alpha=0.9)
    ax.axvline(0, color=style.COLORS["muted"], ls="--", lw=0.7)
    style.style_axis(ax, title="Δ DAT  (post − pre)",
                     xlabel="Δ DAT", ylabel="Participants")
    style.stars_in_axes(ax, w_p, loc="upper right")

    fig.suptitle("Test–retest reliability of DAT (pre vs post)", y=1.02)
    out = style.savefig(fig, "dat_before_vs_after")
    if show:
        plt.show()
    plt.close(fig)

    keep[[
        "user_id", "ref_dat_score", "dat_before_glove",
        "dat_after_score", "delta", "dat_after_n_used",
    ]].to_csv(config.OUTPUTS_DIR / "dat_before_vs_after.csv", index=False)
    return keep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rescore-before", dest="rescore_before",
                    action="store_false")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    main(rescore_before=args.rescore_before, show=not args.no_show)
