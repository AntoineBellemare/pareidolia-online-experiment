"""Animals vs human-body percepts across FD and creativity (DAT).

Builds a fine-grained classification of every percept word into one of:

    HUMAN buckets        ANIMAL buckets        OTHER
    ─────────────        ──────────────        ─────
    face                 mammal                creature   (dragon, monster, …)
    body-part            bird                  object
    person               fish                  landscape
                         insect-bug            weather
                                               food
                                               abstract / texture

Each word is assigned to its nearest seed-centroid in BERT space, with a
minimum cosine threshold (otherwise → "other"). Then we compute, for each
participant and each FD:

  * fraction of percepts that are ANIMAL
  * fraction of percepts that are HUMAN
  * animal–human bias index  =  (A − H) / (A + H)

and run within-subject FD effects + DAT correlations on each, plus a paired
tradeoff scatter (animal share vs human share, coloured by DAT tertile).

Per-image (300 images): we also compute the bias index per image and look
at the distribution by FD.

Usage:
    python -m analysis.figures.animal_vs_body
"""
from __future__ import annotations

import argparse
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare, kruskal, pearsonr, spearmanr, wilcoxon

from .. import config, dat_helper, embeddings, outliers, parse_events, style
from .semantic_deep_dive import _long_words, _trials_long


# ─── fine-grained seed sets ─────────────────────────────────────────────────

SEEDS = {
    # HUMAN
    # Seeds expanded (v2) to lock in the high-frequency basics so they cannot
    # drift to a nearby non-human bucket: head, skull, body, hands, smile,
    # silhouette, profile, figure, soldier, warrior.
    "face":      ["face", "eye", "nose", "mouth", "ear", "cheek", "eyebrow",
                  "eyes", "lips", "chin", "forehead", "teeth", "tongue",
                  "beard", "hair", "smile", "jaw", "head", "skull"],
    "body_part": ["hand", "arm", "leg", "foot", "finger", "torso", "shoulder",
                  "elbow", "knee", "stomach", "back", "neck", "knees",
                  "fingers", "hands", "legs", "arms", "feet", "body",
                  "breast", "hip", "toe"],
    "person":    ["man", "woman", "person", "human", "child", "baby", "girl",
                  "boy", "lady", "people", "men", "women", "guy", "kid",
                  "children", "silhouette", "profile", "figure", "dancer",
                  "soldier", "warrior"],

    # ANIMAL — five zoological buckets (added reptile, v2). The reptile
    # bucket captures lizard / alligator / crocodile / dinosaur, which v1
    # leaked to fish_aquatic, insect_bug, or creature.
    "mammal":    ["dog", "cat", "horse", "elephant", "lion", "wolf", "rabbit",
                  "bear", "pig", "cow", "fox", "deer", "monkey", "mouse",
                  "sheep", "goat", "camel", "donkey", "squirrel", "tiger",
                  "giraffe", "kangaroo", "panda", "rhino", "hippo",
                  "lamb", "rat", "bunny", "puppy", "kitten"],
    "bird":      ["bird", "eagle", "owl", "duck", "chicken", "parrot", "crow",
                  "sparrow", "rooster", "swan", "penguin", "dove", "goose",
                  "turkey", "hawk", "falcon", "flamingo", "peacock"],
    "fish_aquatic": ["fish", "shark", "dolphin", "whale", "octopus", "seahorse",
                  "turtle", "jellyfish", "starfish", "crab", "snail",
                  "shrimp", "lobster", "squid", "eel", "stingray",
                  "salmon", "goldfish", "seal"],
    "insect_bug": ["spider", "ant", "bee", "butterfly", "beetle", "moth",
                  "wasp", "worm", "snake", "scorpion", "grasshopper",
                  "centipede", "fly", "mosquito", "caterpillar"],
    "reptile":   ["lizard", "alligator", "crocodile", "gecko", "iguana",
                  "dinosaur", "tadpole", "frog", "toad", "newt"],

    # OTHER (kept to give "other" a sensible neighbourhood)
    "creature":  ["dragon", "alien", "monster", "ghost", "demon", "creature",
                  "angel", "fairy", "elf", "goblin", "witch", "gremlin"],
    "object":    ["bottle", "chair", "table", "car", "tool", "lamp", "cup",
                  "key", "book", "hammer", "hat", "gun", "cross", "ball",
                  "bat", "vase"],
    "landscape": ["mountain", "river", "ocean", "valley", "cliff", "forest",
                  "beach", "desert", "sea", "lake", "cave"],
    "weather":   ["cloud", "rain", "snow", "fog", "lightning", "storm"],
    "abstract":  ["shape", "pattern", "form", "blob", "swirl", "smear",
                  "texture", "noise"],
    "food":      ["bread", "fruit", "apple", "meat", "cake", "vegetable"],
    "action":    ["kiss", "kissing", "dancing", "hugging", "running",
                  "jumping", "fighting"],
}

HUMAN_CATS  = {"face", "body_part", "person"}
ANIMAL_CATS = {"mammal", "bird", "fish_aquatic", "insect_bug", "reptile"}

# Hard reject list: words that must NEVER be classified as human or animal,
# regardless of how close they sit to a HUMAN/ANIMAL centroid in BERT
# space. These are the most expensive v1 leaks identified by the audit
# (e.g. 'hat'->face, 'cross'->body_part, 'hammer'->body_part, 'gun'->mammal).
HARD_REJECT_FROM_HA = {
    # objects that drifted into face/body_part/mammal
    "hat", "cross", "hammer", "gun", "ball", "bat", "vase", "cob",
    # actions that drifted into face/body_part
    "kiss", "kissing", "dancing", "dance", "hugging",
    # landscape items that drifted into fish_aquatic
    "sea", "lake", "ocean", "river", "shore", "coast",
}

PALETTE = {
    "face":         "#1f77b4",
    "body_part":    "#aec7e8",
    "person":       "#ff7f0e",
    "mammal":       "#2ca02c",
    "bird":         "#98df8a",
    "fish_aquatic": "#17becf",
    "insect_bug":   "#bcbd22",
    "reptile":      "#3a6e3f",
    "creature":     "#d62728",
    "object":       "#9467bd",
    "landscape":    "#8c564b",
    "weather":      "#e377c2",
    "abstract":     "#7f7f7f",
    "food":         "#c49c94",
    "action":       "#bdbdbd",
    "other":        "#cccccc",
}


def classify_words(words: list[str], min_sim: float = 0.30) -> dict[str, str]:
    """Assign each word to its nearest seed-centroid category.

    Words in ``HARD_REJECT_FROM_HA`` are routed to the *second-best*
    non-(human/animal) category instead. This protects against high-level
    BERT confusions like 'hat' clustering with face features or 'gun'
    clustering with animal nouns, which inflate human/animal share with
    obvious objects.
    """
    edict = embeddings.embedding_dict()
    cats = list(SEEDS.keys())
    centroids = {}
    for c in cats:
        V = np.vstack([edict[w] for w in SEEDS[c] if w in edict])
        v = V.mean(0); centroids[c] = v / (np.linalg.norm(v) or 1)
    S = np.vstack([centroids[c] for c in cats])
    W = np.vstack([edict[w] for w in words])
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    sims = Wn @ S.T
    out: dict[str, str] = {}
    ha_cats = HUMAN_CATS | ANIMAL_CATS
    non_ha_idx = [i for i, c in enumerate(cats) if c not in ha_cats]
    for w, row in zip(words, sims):
        best_i = int(row.argmax())
        best_sim = float(row[best_i])
        chosen = cats[best_i] if best_sim >= min_sim else "other"
        if w in HARD_REJECT_FROM_HA and chosen in ha_cats:
            # pick the best non-(human/animal) category that still clears
            # the similarity threshold; otherwise fall through to 'other'.
            sub = row[non_ha_idx]
            j = int(sub.argmax())
            if float(sub[j]) >= min_sim:
                chosen = cats[non_ha_idx[j]]
            else:
                chosen = "other"
        out[w] = chosen
    return out


# ─── per-participant features ───────────────────────────────────────────────

def per_participant_per_fd(df_long: pd.DataFrame,
                           word2cat: dict[str, str]) -> pd.DataFrame:
    """One row per (user, FD) with fractions of each bucket and bias index."""
    df = df_long.assign(category=df_long["word"].map(word2cat))
    df["is_human"]  = df["category"].isin(HUMAN_CATS)
    df["is_animal"] = df["category"].isin(ANIMAL_CATS)
    g = (df.groupby(["user_id", "fd_level"])
           .agg(n_words=("word", "size"),
                n_human=("is_human", "sum"),
                n_animal=("is_animal", "sum"))
           .reset_index())
    g["frac_human"]  = g["n_human"]  / g["n_words"]
    g["frac_animal"] = g["n_animal"] / g["n_words"]
    # bias index: (A - H) / (A + H)  in [-1, +1]; nan if both 0
    denom = g["n_human"] + g["n_animal"]
    g["bias_AmH"] = np.where(denom > 0,
                              (g["n_animal"] - g["n_human"]) / denom,
                              np.nan)
    dat = df_long.groupby("user_id")["ref_dat_score"].first().reset_index()
    g = g.merge(dat, on="user_id")
    return g


# ─── plots ─────────────────────────────────────────────────────────────────

def _stat_pair_text(p_pearson, p_spearman):  # legacy, kept for CSV
    return style.sig_marker(min(p_pearson, p_spearman))


def _per_image_entropy() -> pd.DataFrame:
    """Per-image vocabulary entropy by FD (used as panel d)."""
    from .semantic_deep_dive import _trials_long
    long = _trials_long().dropna(subset=["url_stimulus"])
    rows = []
    for url, g in long.groupby("url_stimulus"):
        words = g["word"].tolist()
        if len(set(words)) < 2: continue
        cnt = Counter(words)
        p = np.array(list(cnt.values())) / sum(cnt.values())
        H = float(-(p * np.log2(p)).sum())
        rows.append({"url": url, "fd_level": g["fd_level"].iloc[0],
                      "vocab_entropy": H})
    return pd.DataFrame(rows)


def figure_fd_effect(perfd: pd.DataFrame, show: bool):
    """Four-panel FD figure:
        (a) human share         — bar
        (b) animal share        — bar
        (c) animal − human bias — bar
        (d) per-image vocab entropy — violin (image-level)
    All panels: per-FD x-axis, bracket+star contrasts, y-axis label =
    metric (titles removed for cleaner reading).
    """
    style.apply()
    fig, axes = plt.subplots(
        1, 4, figsize=(style.COL_2, 2.6),
        gridspec_kw=dict(wspace=0.55),
    )
    stats = {}
    pair_x = {"FD12": 0, "FD14": 1, "FD16": 2}

    # ── panels a-c: participant-level bar+CI ────────────────────────────────
    for ax, col, label in [
        (axes[0], "frac_human",  "Human share"),
        (axes[1], "frac_animal", "Animal share"),
        (axes[2], "bias_AmH",    "Animal − Human bias index"),
    ]:
        pivot = perfd.pivot(index="user_id", columns="fd_level",
                            values=col).dropna()
        pivot = pivot[[fd for fd in config.FD_LEVELS if fd in pivot.columns]]
        chi, p_fr = friedmanchisquare(*[pivot[c] for c in pivot.columns])
        means = pivot.mean(); sems = pivot.sem()
        xs = np.arange(len(pivot.columns))
        ax.bar(xs, means.values, yerr=1.96 * sems.values,
               color=[style.COLORS[fd] for fd in pivot.columns],
               capsize=2.5, alpha=0.85, edgecolor="black", lw=0.4)
        ax.set_xticks(xs); ax.set_xticklabels(pivot.columns)
        ci = 1.96 * sems
        lo = float((means - ci).min()); hi = float((means + ci).max())
        pad = (hi - lo) * 0.25
        ax.set_ylim(lo - pad, hi + pad)
        if col == "bias_AmH":
            ax.axhline(0, color=style.COLORS["muted"], lw=0.4, ls="--",
                        zorder=0)
        style.style_axis(ax, title="", ylabel=label, xlabel="")
        pairwise = {}
        comparisons = []
        for a, b in [("FD12", "FD14"), ("FD14", "FD16"), ("FD12", "FD16")]:
            if a in pivot and b in pivot:
                _, pw = wilcoxon(pivot[a], pivot[b])
                pairwise[f"{a}_vs_{b}"] = float(pw)
                comparisons.append((pair_x[a], pair_x[b], pw))
        style.sig_brackets(ax, comparisons)
        stats[col] = {"friedman_p": float(p_fr), "means": means.to_dict(),
                      "pairwise": pairwise, "n": len(pivot)}

    # ── panel d: per-image vocabulary entropy — mean ± 95 % CI bar ──────────
    img = _per_image_entropy()
    ax = axes[3]
    means = img.groupby("fd_level")["vocab_entropy"].mean().reindex(config.FD_LEVELS)
    sems  = img.groupby("fd_level")["vocab_entropy"].sem().reindex(config.FD_LEVELS)
    xs = np.arange(len(config.FD_LEVELS))
    ax.bar(xs, means.values, yerr=1.96 * sems.values,
           color=[style.COLORS[fd] for fd in config.FD_LEVELS],
           capsize=2.5, alpha=0.85, edgecolor="black", lw=0.4)
    ax.set_xticks(xs); ax.set_xticklabels(config.FD_LEVELS)
    ci = 1.96 * sems
    lo = float((means - ci).min()); hi = float((means + ci).max())
    pad = (hi - lo) * 0.25
    ax.set_ylim(lo - pad, hi + pad)
    from scipy.stats import mannwhitneyu
    comparisons = []
    for a, b in [("FD12", "FD14"), ("FD14", "FD16"), ("FD12", "FD16")]:
        ga = img[img["fd_level"] == a]["vocab_entropy"].values
        gb = img[img["fd_level"] == b]["vocab_entropy"].values
        if len(ga) >= 3 and len(gb) >= 3:
            _, pw = mannwhitneyu(ga, gb, alternative="two-sided")
            comparisons.append((pair_x[a], pair_x[b], float(pw)))
    style.style_axis(ax, title="", ylabel="Per-image vocab\nentropy (bits)",
                     xlabel="")
    style.sig_brackets(ax, comparisons)
    stats["vocab_entropy_per_image"] = {"means": means.to_dict(),
                                          "n": len(img)}

    fig.suptitle("Stimulus FD shapes what people see", y=1.04)
    if show: plt.show()
    style.savefig(fig, "animal_vs_body_fd")
    plt.close(fig)
    return stats


def figure_dat_effect(perfd: pd.DataFrame, show: bool):
    """Per-user mean across FDs vs DAT (Pearson + Spearman)."""
    style.apply()
    per_user = perfd.groupby("user_id").agg(
        frac_human=("frac_human", "mean"),
        frac_animal=("frac_animal", "mean"),
        bias_AmH=("bias_AmH", "mean"),
        dat=("ref_dat_score", "first"),
    ).dropna()
    fig, axes = plt.subplots(
        1, 3, figsize=(style.COL_1, 2.0),
        gridspec_kw=dict(wspace=0.65),
    )
    stats = {}
    for ax, col, label in [
        (axes[0], "frac_human",  "Human share"),
        (axes[1], "frac_animal", "Animal share"),
        (axes[2], "bias_AmH",    "Animal − Human bias"),
    ]:
        sub, n_drop = outliers.trim_sd(per_user, [col, "dat"], sd=3.0)
        if n_drop:
            print(f"  [{col}] dropped {n_drop} ±3 SD outliers")
        r, p = pearsonr(sub["dat"], sub[col])
        rho, p_rho = spearmanr(sub["dat"], sub[col])
        ax.scatter(sub["dat"], sub[col], s=3.5, alpha=0.55,
                   color=style.COLORS["dot"], linewidths=0)
        sns.regplot(data=sub, x="dat", y=col, scatter=False,
                    color=style.COLORS["fit"],
                    line_kws=dict(linewidth=1.0), ax=ax)
        if col == "bias_AmH":
            ax.axhline(0, color=style.COLORS["muted"], lw=0.4, ls="--")
        style.style_axis(ax, title="", xlabel="DAT", ylabel=label)
        style.small_stat_annotation(ax, r, p, loc="upper left")
        stats[col] = {"pearson_r": r, "pearson_p": p,
                       "spearman_rho": rho, "spearman_p": p_rho,
                       "n": len(sub)}
    fig.suptitle("Creativity (DAT) vs human / animal pareidolia", y=1.04)
    if show: plt.show()
    style.savefig(fig, "animal_vs_body_dat")
    plt.close(fig)
    return per_user, stats


def figure_tradeoff(per_user: pd.DataFrame, show: bool):
    """Animal share vs Human share, coloured by DAT tertile."""
    style.apply()
    pu = per_user.dropna(subset=["frac_human", "frac_animal", "dat"]).copy()
    pu["tertile"] = pd.qcut(pu["dat"], 3, labels=["Low", "Mid", "High"],
                             duplicates="drop")
    fig, ax = plt.subplots(figsize=(style.COL_1_5, 3.0))
    for t in ["Low", "Mid", "High"]:
        s = pu[pu["tertile"] == t]
        ax.scatter(s["frac_human"], s["frac_animal"],
                   s=8, alpha=0.65, color=style.COLORS[t],
                   linewidths=0, label=f"{t} DAT (n={len(s)})")
    # diagonal y = x
    mx = max(pu["frac_human"].max(), pu["frac_animal"].max())
    ax.plot([0, mx], [0, mx], color="black", lw=0.4, ls="--", zorder=0)
    ax.set_xlim(0); ax.set_ylim(0)
    style.style_axis(ax, title="Per-user mean (across FDs)",
                     xlabel="Human share", ylabel="Animal share")
    ax.legend(frameon=False, fontsize=6)
    fig.suptitle("Animal vs Human percept tradeoff", y=1.02)
    if show: plt.show()
    style.savefig(fig, "animal_vs_body_tradeoff")
    plt.close(fig)


def figure_subcategories(df_long: pd.DataFrame, word2cat: dict[str, str],
                         show: bool):
    """Sub-bucket FD effect — face vs body-part vs person; mammal vs bird vs fish vs insect."""
    style.apply()
    df = df_long.assign(category=df_long["word"].map(word2cat))
    df["is_human"]  = df["category"].isin(HUMAN_CATS)
    df["is_animal"] = df["category"].isin(ANIMAL_CATS)

    sub_cats = list(HUMAN_CATS) + list(ANIMAL_CATS)
    rows = []
    for cat in sub_cats:
        g = (df.assign(hit=df["category"] == cat)
               .groupby(["user_id", "fd_level"])
               .agg(frac=("hit", "mean"),
                    n_words=("word", "size"))
               .reset_index())
        for fd in config.FD_LEVELS:
            s = g[g["fd_level"] == fd]
            rows.append({"category": cat, "fd_level": fd,
                         "mean": float(s["frac"].mean()),
                         "sem":  float(s["frac"].sem()),
                         "n":    len(s)})
    sumtbl = pd.DataFrame(rows)

    fig, axes = plt.subplots(
        1, 2, figsize=(style.COL_2, 2.8),
        gridspec_kw=dict(wspace=0.35),
    )

    def _grouped_bars(ax, cats, title):
        xs = np.arange(len(config.FD_LEVELS))
        w = 0.9 / len(cats)
        for i, cat in enumerate(cats):
            s = sumtbl[sumtbl["category"] == cat]
            means = s.set_index("fd_level")["mean"].reindex(config.FD_LEVELS)
            sems  = s.set_index("fd_level")["sem"].reindex(config.FD_LEVELS)
            ax.bar(xs - 0.45 + (i + 0.5) * w,
                   means.values, w * 0.95,
                   yerr=1.96 * sems.values,
                   color=PALETTE[cat], label=cat,
                   capsize=1.5, edgecolor="black", lw=0.3)
        ax.set_xticks(xs); ax.set_xticklabels(config.FD_LEVELS)
        style.style_axis(ax, title=title, ylabel="Share of percepts",
                         xlabel="")
        ax.legend(frameon=False, fontsize=6)

    _grouped_bars(axes[0], list(HUMAN_CATS),  "Human sub-types")
    _grouped_bars(axes[1], list(ANIMAL_CATS), "Animal sub-types")
    fig.suptitle("Sub-category FD breakdown", y=1.05)
    if show: plt.show()
    style.savefig(fig, "animal_vs_body_subcats")
    plt.close(fig)

    sumtbl.to_csv(config.OUTPUTS_DIR / "animal_vs_body_subcategory_means.csv",
                  index=False)
    return sumtbl


def figure_per_image_bias(trials_long: pd.DataFrame,
                          word2cat: dict[str, str],
                          show: bool):
    """Per-image animal vs human bias. Distribution by FD."""
    style.apply()
    tl = trials_long.dropna(subset=["url_stimulus"]).copy()
    tl["category"] = tl["word"].map(word2cat)
    tl["is_human"]  = tl["category"].isin(HUMAN_CATS)
    tl["is_animal"] = tl["category"].isin(ANIMAL_CATS)
    g = (tl.groupby("url_stimulus")
            .agg(fd_level=("fd_level", "first"),
                 n=("word", "size"),
                 n_h=("is_human", "sum"),
                 n_a=("is_animal", "sum"))
            .reset_index())
    g = g[g["n"] >= 5]
    g["bias"] = (g["n_a"] - g["n_h"]) / (g["n_a"] + g["n_h"]).replace(0, np.nan)
    g = g.dropna(subset=["bias"])

    fig, ax = plt.subplots(figsize=(style.COL_1_5, 2.6))
    sns.violinplot(
        data=g, x="fd_level", y="bias",
        order=config.FD_LEVELS, inner=None, cut=0,
        palette=[style.COLORS[fd] for fd in config.FD_LEVELS], ax=ax,
        linewidth=0.6,
    )
    sns.stripplot(data=g, x="fd_level", y="bias",
                  order=config.FD_LEVELS, color="black",
                  size=2, alpha=0.5, jitter=0.2, ax=ax)
    ax.axhline(0, color=style.COLORS["muted"], lw=0.4, ls="--")
    groups = [g[g["fd_level"] == fd]["bias"].values for fd in config.FD_LEVELS]
    H, p_kw = kruskal(*groups)
    style.style_axis(
        ax, title=f"Per-image bias  ·  KW p = {p_kw:.2g}{style.sig_marker(p_kw)}",
        xlabel="", ylabel="Animal − Human bias",
    )
    fig.suptitle("Each image's animal vs human bias by FD", y=1.05)
    if show: plt.show()
    style.savefig(fig, "animal_vs_body_per_image")
    plt.close(fig)
    g.to_csv(config.OUTPUTS_DIR / "animal_vs_body_per_image.csv", index=False)
    return g


# ─── entry-point ───────────────────────────────────────────────────────────

def main(show: bool = True, min_sim: float = 0.30):
    df, _ = _long_words()
    print(f"Long words: {len(df):,}")

    # Classify every unique word once.
    unique = df["word"].unique().tolist()
    word2cat = classify_words(unique, min_sim=min_sim)
    cnt = Counter(word2cat.values())
    print("Category counts (unique words):")
    for c, n in sorted(cnt.items(), key=lambda x: -x[1]):
        print(f"  {c:14s} {n}")

    perfd = per_participant_per_fd(df, word2cat)
    print(f"\nPer-(user, FD) rows: {len(perfd):,}")

    fd_stats = figure_fd_effect(perfd, show=show)
    per_user, dat_stats = figure_dat_effect(perfd, show=show)
    figure_tradeoff(per_user, show=show)
    figure_subcategories(df, word2cat, show=show)

    figure_per_image_bias(_trials_long(), word2cat, show=show)

    print("\n=== FD effects ===")
    for k, v in fd_stats.items():
        if "friedman_p" not in v:
            # image-level panel (no within-subject test); just report means
            print(f"  {k}: image-level means = "
                  f"{dict((fd, round(m, 3)) for fd, m in v['means'].items())}")
            continue
        print(f"  {k}: Friedman p = {v['friedman_p']:.3g}, n = {v['n']}, "
              f"means = {dict((fd, round(m, 3)) for fd, m in v['means'].items())}")
    print("\n=== DAT effects ===")
    for k, v in dat_stats.items():
        print(f"  {k}: r = {v['pearson_r']:+.3f} (p={v['pearson_p']:.3g}), "
              f"rho = {v['spearman_rho']:+.3f} (p={v['spearman_p']:.3g}), "
              f"n = {v['n']}")

    perfd.to_csv(config.OUTPUTS_DIR / "animal_vs_body_per_user_per_fd.csv",
                 index=False)
    pd.DataFrame([
        {"category": w, "label": c}
        for w, c in word2cat.items()
    ]).to_csv(config.OUTPUTS_DIR / "animal_vs_body_word_labels.csv",
              index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--min-sim", type=float, default=0.30,
                    help="cosine threshold for category assignment")
    args = ap.parse_args()
    main(show=not args.no_show, min_sim=args.min_sim)
