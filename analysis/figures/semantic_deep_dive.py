"""Deep dive on the semantic space — directions picked by the team.

Implemented sub-commands
------------------------
  A1   per-participant semantic cohesion (within-participant compactness)
  A3   PCA/UMAP of (participant, FD) centroid trajectories
  B1   vocabulary specialisation index (rare-word fraction)
  B2   type-token ratio per (participant, FD)
  C1   richer category taxonomy (15 seed-based categories)
  C2   face / body / person bias
  D2   per-participant co-occurrence-graph statistics (PMI weighted)
  E1   per-image consensus (entropy / semantic spread of percepts)
  E2   image clustering by percept distribution

Usage:
    python -m analysis.figures.semantic_deep_dive            # runs all
    python -m analysis.figures.semantic_deep_dive --only A1 B1
"""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from math import log

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import entropy, kruskal, pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .. import config, dat_helper, embeddings, outliers, parse_events, style


ASCII = re.compile(r"^[A-Za-z]+$")
LABELS = ["Low", "Mid", "High"]


# ─── shared infrastructure ────────────────────────────────────────────────────

def _long_words(use_main_cohort: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort = "main" if use_main_cohort else "notebook"
    p = dat_helper.select_cohort(cohort, "before").dropna(subset=["ref_dat_score"])
    edict = embeddings.embedding_dict()
    rows = []
    for _, r in p.iterrows():
        for fd in config.FD_LEVELS:
            words = r.get(f"{fd}_words")
            if words is None: continue
            for w in list(words):
                if not (ASCII.fullmatch(w or "") and w in edict
                        and w not in config.SEMANTIC_STOPWORDS):
                    continue
                rows.append({"user_id": r["user_id"], "word": w, "fd_level": fd,
                             "ref_dat_score": r["ref_dat_score"]})
    df = pd.DataFrame(rows)
    df["dat_tertile"] = pd.qcut(df["ref_dat_score"], 3, labels=LABELS,
                                duplicates="drop")
    return df, p


def _trials_long() -> pd.DataFrame:
    """One row per (trial, word) with ASCII filter and embedding presence."""
    trials = parse_events.cached_trials()
    edict = embeddings.embedding_dict()
    rows = []
    for _, t in trials.iterrows():
        words = t.get("words")
        if words is None: continue
        for w in list(words):
            w = str(w).strip().lower()
            if not (ASCII.fullmatch(w) and w in edict
                    and w not in config.SEMANTIC_STOPWORDS):
                continue
            rows.append({"user_id": t["user_id"], "trial_index": t["trial_index"],
                         "fd_level": t["fd_level"], "image_id": t["image_id"],
                         "url_stimulus": t["url_stimulus"], "word": w,
                         "ref_dat_score": t.get("ref_dat_score")})
    return pd.DataFrame(rows)


def _summary_scatter(ax, x, y, *, color=None, **kwargs):
    ax.scatter(x, y, s=4, alpha=0.55,
               color=color or style.COLORS["dot"], linewidths=0, **kwargs)


# ═════════════════════════════════════════════════════════════════════════════
# A1 — per-participant semantic cohesion
# ═════════════════════════════════════════════════════════════════════════════

def A1_cohesion(df: pd.DataFrame, show: bool = True):
    edict = embeddings.embedding_dict()
    rows = []
    for uid, sub in df.groupby("user_id"):
        if len(sub) < 3:
            continue
        V = np.vstack([edict[w] for w in sub["word"]])
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        c = V.mean(0); c = c / (np.linalg.norm(c) or 1)
        cohesion = float(1 - (V @ c).mean())     # 0 = perfectly cohesive
        rows.append({"user_id": uid, "n_words": len(sub),
                     "ref_dat_score": sub["ref_dat_score"].iloc[0],
                     "cohesion": cohesion})
    res = pd.DataFrame(rows)
    z = (res["cohesion"] - res["cohesion"].mean()).abs() / res["cohesion"].std()
    res = res[z <= 3]
    r, p = pearsonr(res["ref_dat_score"], res["cohesion"])

    style.apply()
    fig, ax = plt.subplots(figsize=(style.COL_1, 2.4))
    _summary_scatter(ax, res["ref_dat_score"], res["cohesion"])
    sns.regplot(data=res, x="ref_dat_score", y="cohesion",
                scatter=False, color=style.COLORS["fit"], ci=95,
                line_kws=dict(linewidth=1.0), ax=ax)
    style.style_axis(
        ax, title="A1 · semantic cohesion",
        xlabel="DAT", ylabel="Within-user cohesion\n(1 − mean cos to centroid)",
    )
    style.small_stat_annotation(ax, r, p, loc="upper left")
    if show: plt.show()
    style.savefig(fig, "semantic_A1_cohesion")
    plt.close(fig)

    res.to_csv(config.OUTPUTS_DIR / "semantic_A1_cohesion.csv", index=False)
    print(f"  A1: n={len(res)}  r={r:+.3f}  p={p:.3g}")
    return res


# ═════════════════════════════════════════════════════════════════════════════
# A3 — PCA/UMAP of (participant, FD) centroids
# ═════════════════════════════════════════════════════════════════════════════

def A3_centroid_pca(df: pd.DataFrame, show: bool = True):
    edict = embeddings.embedding_dict()
    rows = []
    for (uid, fd), sub in df.groupby(["user_id", "fd_level"]):
        if len(sub) < 3:
            continue
        V = np.vstack([edict[w] for w in sub["word"]])
        c = V.mean(0); c = c / (np.linalg.norm(c) or 1)
        rows.append({"user_id": uid, "fd_level": fd,
                     "ref_dat_score": sub["ref_dat_score"].iloc[0],
                     "centroid": c, "n_words": len(sub)})
    C = pd.DataFrame(rows)
    if C.empty:
        print("  A3: too few rows")
        return None
    C["dat_tertile"] = pd.qcut(C["ref_dat_score"], 3, labels=LABELS,
                               duplicates="drop")
    X = np.vstack(C["centroid"].to_numpy())
    pca = PCA(n_components=2).fit(X)
    C[["pc1", "pc2"]] = pca.transform(X)

    style.apply()
    fig, axes = plt.subplots(
        1, 2, figsize=(style.COL_1_5, 2.4),
        gridspec_kw=dict(wspace=0.4),
    )
    # left: colour by FD
    for fd in config.FD_LEVELS:
        s = C[C["fd_level"] == fd]
        axes[0].scatter(s["pc1"], s["pc2"], s=4, alpha=0.45,
                        color=style.COLORS[fd], label=fd, linewidths=0)
        m = s[["pc1", "pc2"]].mean()
        axes[0].scatter(m["pc1"], m["pc2"], s=40, marker="X",
                        color=style.COLORS[fd], edgecolor="black", linewidths=0.6,
                        zorder=5)
    axes[0].legend(frameon=False, fontsize=5)
    style.style_axis(axes[0], title="centroids · coloured by FD",
                     xlabel=f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)",
                     ylabel=f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)")

    # right: colour by tertile
    for t in LABELS:
        s = C[C["dat_tertile"] == t]
        axes[1].scatter(s["pc1"], s["pc2"], s=4, alpha=0.45,
                        color=style.COLORS[t], label=t, linewidths=0)
        m = s[["pc1", "pc2"]].mean()
        axes[1].scatter(m["pc1"], m["pc2"], s=40, marker="X",
                        color=style.COLORS[t], edgecolor="black", linewidths=0.6,
                        zorder=5)
    axes[1].legend(frameon=False, fontsize=5)
    style.style_axis(axes[1], title="centroids · coloured by DAT tertile",
                     xlabel=f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)",
                     ylabel="")
    fig.suptitle("A3 · PCA of (participant × FD) percept centroids", y=1.02)
    if show: plt.show()
    style.savefig(fig, "semantic_A3_centroid_pca")
    plt.close(fig)

    # FD-trajectory length per participant
    traj_rows = []
    for uid, sub in C.groupby("user_id"):
        if set(config.FD_LEVELS) - set(sub["fd_level"]):
            continue
        pts = {row["fd_level"]: (row["pc1"], row["pc2"]) for _, row in sub.iterrows()}
        d12_14 = np.hypot(pts["FD12"][0] - pts["FD14"][0], pts["FD12"][1] - pts["FD14"][1])
        d14_16 = np.hypot(pts["FD14"][0] - pts["FD16"][0], pts["FD14"][1] - pts["FD16"][1])
        d12_16 = np.hypot(pts["FD12"][0] - pts["FD16"][0], pts["FD12"][1] - pts["FD16"][1])
        traj_rows.append({"user_id": uid, "dat": sub["ref_dat_score"].iloc[0],
                          "traj_total": d12_14 + d14_16, "d12_16": d12_16})
    T = pd.DataFrame(traj_rows)
    if not T.empty:
        r, p = pearsonr(T["dat"], T["traj_total"])
        print(f"  A3: n_trajectories={len(T)}  traj_total ↔ DAT r={r:+.3f} p={p:.3g}")
    C.drop(columns=["centroid"]).to_csv(
        config.OUTPUTS_DIR / "semantic_A3_centroids.csv", index=False)
    T.to_csv(config.OUTPUTS_DIR / "semantic_A3_trajectories.csv", index=False)
    return C, T


# ═════════════════════════════════════════════════════════════════════════════
# B1 — vocabulary specialisation index
# ═════════════════════════════════════════════════════════════════════════════

def B1_specialisation(df: pd.DataFrame, percentile: float = 25.0,
                      show: bool = True):
    corpus = Counter(df["word"])
    cutoff = np.percentile(list(corpus.values()), percentile)
    rare = {w for w, n in corpus.items() if n <= cutoff}

    rows = []
    for uid, sub in df.groupby("user_id"):
        words = sub["word"].unique()
        if len(words) < 3:
            continue
        rows.append({
            "user_id": uid,
            "ref_dat_score": sub["ref_dat_score"].iloc[0],
            "n_unique": len(words),
            "rare_fraction": float(np.mean([w in rare for w in words])),
            "mean_neg_log_p": float(np.mean(
                [-log((corpus[w] / sum(corpus.values())) + 1e-12) for w in words]
            )),
        })
    res = pd.DataFrame(rows)
    res1, n1 = outliers.trim_sd(res, ["rare_fraction", "ref_dat_score"], sd=3.0)
    res2, n2 = outliers.trim_sd(res, ["mean_neg_log_p", "ref_dat_score"], sd=3.0)
    if n1 or n2:
        print(f"  B1: dropped {n1} outliers (rare), {n2} outliers (surprisal)")
    r1, p1 = pearsonr(res1["ref_dat_score"], res1["rare_fraction"])
    r2, p2 = pearsonr(res2["ref_dat_score"], res2["mean_neg_log_p"])

    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(style.COL_1_5, 2.4),
                             gridspec_kw=dict(wspace=0.4))
    for ax, (col, lab, r, p, src) in zip(
        axes,
        [("rare_fraction", f"Frac. rare words\n(bottom {percentile:.0f}% of corpus)", r1, p1, res1),
         ("mean_neg_log_p", "Word surprisal\n(mean −log p)", r2, p2, res2)],
    ):
        _summary_scatter(ax, src["ref_dat_score"], src[col])
        sns.regplot(data=src, x="ref_dat_score", y=col,
                    scatter=False, color=style.COLORS["fit"],
                    line_kws=dict(linewidth=1.0), ax=ax)
        style.style_axis(ax, title="", xlabel="DAT", ylabel=lab)
        style.small_stat_annotation(ax, r, p, loc="upper left")
    fig.suptitle("Vocabulary specialisation vs DAT", y=1.02)
    if show: plt.show()
    style.savefig(fig, "semantic_B1_specialisation")
    plt.close(fig)
    res.to_csv(config.OUTPUTS_DIR / "semantic_B1_specialisation.csv", index=False)
    print(f"  B1: rare_fraction r={r1:+.3f} p={p1:.3g} | "
          f"−log p r={r2:+.3f} p={p2:.3g}")
    return res


# ═════════════════════════════════════════════════════════════════════════════
# B2 — type-token ratio per (participant, FD)
# ═════════════════════════════════════════════════════════════════════════════

def B2_ttr(trials_long: pd.DataFrame, participants: pd.DataFrame,
           show: bool = True):
    """TTR = unique words / total words within each (user, FD)."""
    keep = trials_long[trials_long["user_id"].isin(participants["user_id"])]
    grouped = (
        keep.groupby(["user_id", "fd_level"])
            .agg(n_unique=("word", "nunique"),
                 n_total=("word", "size"))
            .reset_index()
    )
    grouped["ttr"] = grouped["n_unique"] / grouped["n_total"]
    grouped = grouped.merge(
        participants[["user_id", "ref_dat_score"]], on="user_id"
    )

    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(style.COL_1_5, 2.4),
                             gridspec_kw=dict(wspace=0.4))
    # panel 1: TTR by FD (within-subject)
    pivot = grouped.pivot(index="user_id", columns="fd_level",
                          values="ttr").dropna()
    pivot = pivot[[fd for fd in config.FD_LEVELS if fd in pivot.columns]]
    from scipy.stats import friedmanchisquare, wilcoxon
    chi, p_fr = friedmanchisquare(*[pivot[c] for c in pivot.columns])
    means = pivot.mean(); sems = pivot.sem()
    xs = np.arange(len(pivot.columns))
    axes[0].bar(xs, means.values, yerr=1.96 * sems.values,
                color=[style.COLORS[fd] for fd in pivot.columns],
                capsize=2.5, alpha=0.85, edgecolor="black", lw=0.4)
    axes[0].set_xticks(xs); axes[0].set_xticklabels(pivot.columns)
    style.style_axis(axes[0], title=f"TTR by FD · p = {p_fr:.2g}{style.sig_marker(p_fr)}",
                     ylabel="Type-token ratio", xlabel="")
    # panel 2: pooled TTR vs DAT
    per_user = grouped.groupby("user_id").agg(
        ttr=("ttr", "mean"), dat=("ref_dat_score", "first")
    ).dropna()
    per_user, n_drop = outliers.trim_sd(per_user, ["ttr", "dat"], sd=3.0)
    if n_drop:
        print(f"  B2: dropped {n_drop} ±3 SD outliers")
    r, p = pearsonr(per_user["dat"], per_user["ttr"])
    _summary_scatter(axes[1], per_user["dat"], per_user["ttr"])
    sns.regplot(data=per_user, x="dat", y="ttr", scatter=False,
                color=style.COLORS["fit"], line_kws=dict(linewidth=1.0),
                ax=axes[1])
    style.style_axis(axes[1], title=f"TTR vs DAT · {style.short_stats(r, p)}",
                     xlabel="DAT", ylabel="")
    fig.suptitle("B2 · type-token ratio", y=1.02)
    if show: plt.show()
    style.savefig(fig, "semantic_B2_ttr")
    plt.close(fig)
    grouped.to_csv(config.OUTPUTS_DIR / "semantic_B2_ttr.csv", index=False)
    print(f"  B2: Friedman FD p={p_fr:.3g} | TTR↔DAT r={r:+.3f} p={p:.3g}")
    return grouped


# ═════════════════════════════════════════════════════════════════════════════
# C1 — richer category taxonomy (15 seed sets)
# ═════════════════════════════════════════════════════════════════════════════

SEEDS_15 = {
    "face":     ["face", "eye", "nose", "mouth", "ear", "cheek"],
    "body":     ["hand", "arm", "leg", "foot", "finger", "torso", "shoulder", "head"],
    "person":   ["man", "woman", "person", "human", "child", "baby", "girl", "boy"],
    "animal":   ["dog", "cat", "horse", "fish", "elephant", "lion", "wolf", "rabbit"],
    "bird":     ["bird", "eagle", "owl", "duck", "chicken", "parrot", "crow", "sparrow"],
    "plant":    ["tree", "flower", "leaf", "branch", "grass", "bush"],
    "landscape":["mountain", "river", "ocean", "valley", "island", "cliff", "forest"],
    "weather":  ["cloud", "rain", "snow", "fog", "lightning", "storm"],
    "tool":     ["hammer", "knife", "saw", "wrench", "drill", "axe"],
    "vehicle":  ["car", "boat", "plane", "bicycle", "ship", "truck"],
    "food":     ["bread", "fruit", "apple", "meat", "cheese", "vegetable"],
    "structure":["house", "building", "tower", "bridge", "wall", "temple"],
    "creature": ["dragon", "alien", "monster", "ghost", "demon", "creature"],
    "abstract": ["shape", "pattern", "symbol", "figure", "form", "blob", "swirl"],
    "texture":  ["smear", "noise", "scatter", "fuzz", "marble", "stain"],
}


def C1_categories(df: pd.DataFrame, show: bool = True):
    edict = embeddings.embedding_dict()
    seed_vecs = {}
    for cat, seeds in SEEDS_15.items():
        V = np.vstack([edict[w] for w in seeds if w in edict])
        c = V.mean(0); seed_vecs[cat] = c / (np.linalg.norm(c) or 1)
    cats = list(seed_vecs.keys())
    words = df["word"].unique().tolist()
    W = np.vstack([edict[w] for w in words])
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    S = np.vstack([seed_vecs[c] for c in cats])
    sims = W @ S.T
    idx = sims.argmax(axis=1)
    word2cat = dict(zip(words, [cats[i] for i in idx]))
    df = df.assign(category=df["word"].map(word2cat))

    # Plot stacked bars per tertile and per FD
    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(style.COL_2, 3.0),
                             gridspec_kw=dict(wspace=0.3))
    palette = sns.color_palette("tab20", len(cats))

    for ax, group_col in zip(axes, ["dat_tertile", "fd_level"]):
        levels = LABELS if group_col == "dat_tertile" else config.FD_LEVELS
        pivot = (df.groupby([group_col, "category"]).size()
                   .unstack(fill_value=0).reindex(levels))
        prop = pivot.div(pivot.sum(axis=1), axis=0)[cats]
        bottom = np.zeros(len(prop))
        xs = np.arange(len(prop))
        for cat, col in zip(cats, palette):
            ax.bar(xs, prop[cat], bottom=bottom, color=col,
                   edgecolor="white", lw=0.3, label=cat)
            bottom += prop[cat].values
        ax.set_xticks(xs); ax.set_xticklabels(prop.index)
        style.style_axis(ax, title=group_col.replace("_", " ").title(),
                         ylabel="Proportion", xlabel="")
        ax.set_ylim(0, 1)
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                    frameon=False, fontsize=5, ncol=1)
    fig.suptitle("C1 · 15-category percept mix", y=1.02)
    if show: plt.show()
    style.savefig(fig, "semantic_C1_categories")
    plt.close(fig)

    cnt_t = df.groupby(["dat_tertile", "category"]).size().unstack(fill_value=0)
    cnt_fd = df.groupby(["fd_level", "category"]).size().unstack(fill_value=0)
    cnt_t.to_csv(config.OUTPUTS_DIR / "semantic_C1_by_tertile.csv")
    cnt_fd.to_csv(config.OUTPUTS_DIR / "semantic_C1_by_fd.csv")
    print(f"  C1: {len(cats)} categories assigned to {len(words)} unique words")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# C2 — face / body / person bias
# ═════════════════════════════════════════════════════════════════════════════

FACE_BODY_CATS = {"face", "body", "person"}


def C2_face_body(df: pd.DataFrame, show: bool = True):
    edict = embeddings.embedding_dict()
    seed_vecs = {cat: np.vstack([edict[w] for w in SEEDS_15[cat] if w in edict]).mean(0)
                 for cat in SEEDS_15}
    for cat, v in seed_vecs.items():
        seed_vecs[cat] = v / (np.linalg.norm(v) or 1)
    cats = list(seed_vecs.keys())
    words = df["word"].unique().tolist()
    W = np.vstack([edict[w] for w in words])
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    S = np.vstack([seed_vecs[c] for c in cats])
    idx = (W @ S.T).argmax(axis=1)
    word2cat = dict(zip(words, [cats[i] for i in idx]))
    df = df.assign(category=df["word"].map(word2cat))
    df["is_face_body"] = df["category"].isin(FACE_BODY_CATS)

    # Per (user, FD) face/body fraction
    grouped = (df.groupby(["user_id", "fd_level"])["is_face_body"]
                 .mean().rename("fb_fraction").reset_index()
                 .merge(df.groupby("user_id")["ref_dat_score"].first().reset_index(),
                        on="user_id"))
    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(style.COL_1_5, 2.4),
                             gridspec_kw=dict(wspace=0.4))
    # FD effect
    pivot = grouped.pivot(index="user_id", columns="fd_level",
                          values="fb_fraction").dropna()
    pivot = pivot[[fd for fd in config.FD_LEVELS if fd in pivot.columns]]
    from scipy.stats import friedmanchisquare
    chi, p_fr = friedmanchisquare(*[pivot[c] for c in pivot.columns])
    means = pivot.mean(); sems = pivot.sem()
    xs = np.arange(len(pivot.columns))
    axes[0].bar(xs, means.values, yerr=1.96 * sems.values,
                color=[style.COLORS[fd] for fd in pivot.columns],
                capsize=2.5, alpha=0.85, edgecolor="black", lw=0.4)
    axes[0].set_xticks(xs); axes[0].set_xticklabels(pivot.columns)
    style.style_axis(axes[0],
                     title=f"face/body share by FD · p = {p_fr:.2g}{style.sig_marker(p_fr)}",
                     ylabel="Proportion face/body/person", xlabel="")

    # Per-user mean vs DAT
    per_u = grouped.groupby("user_id").agg(
        fb=("fb_fraction", "mean"), dat=("ref_dat_score", "first")
    ).dropna()
    per_u, n_drop = outliers.trim_sd(per_u, ["fb", "dat"], sd=3.0)
    if n_drop:
        print(f"  C2: dropped {n_drop} ±3 SD outliers")
    r, p = pearsonr(per_u["dat"], per_u["fb"])
    _summary_scatter(axes[1], per_u["dat"], per_u["fb"])
    sns.regplot(data=per_u, x="dat", y="fb", scatter=False,
                color=style.COLORS["fit"], line_kws=dict(linewidth=1.0),
                ax=axes[1])
    style.style_axis(axes[1],
                     title=f"vs DAT · {style.short_stats(r, p)}",
                     xlabel="DAT",
                     ylabel="Proportion face/body/person")
    fig.suptitle("C2 · face / body / person bias", y=1.02)
    if show: plt.show()
    style.savefig(fig, "semantic_C2_face_body")
    plt.close(fig)
    grouped.to_csv(config.OUTPUTS_DIR / "semantic_C2_face_body.csv", index=False)
    print(f"  C2: face/body Friedman FD p={p_fr:.3g} | DAT r={r:+.3f} p={p:.3g}")
    return grouped


# ═════════════════════════════════════════════════════════════════════════════
# D2 — per-participant co-occurrence graph (PMI)
# ═════════════════════════════════════════════════════════════════════════════

def D2_cooccurrence(trials_long: pd.DataFrame, participants: pd.DataFrame,
                    show: bool = True):
    """For each participant, build word-trial bipartite graph -> word co-occurrence
    (two words co-occur if they were typed in the same trial). Compute density,
    average clustering coefficient and modularity of the per-user graph.
    Avoids networkx for clustering; uses simple numpy stats.
    """
    keep = trials_long[trials_long["user_id"].isin(participants["user_id"])]
    rows = []
    for uid, sub in keep.groupby("user_id"):
        # word x trial matrix
        bag = sub.groupby(["trial_index", "word"]).size().unstack(fill_value=0)
        if bag.shape[1] < 4 or bag.shape[0] < 4:
            continue
        # binary co-occurrence
        M = (bag > 0).astype(int).values
        co = M.T @ M  # word x word counts
        np.fill_diagonal(co, 0)
        n = co.shape[0]
        density = float(co.sum() / (n * (n - 1) + 1e-9))
        deg = co.sum(axis=1)
        avg_deg = float(deg.mean())
        # local clustering coefficient (binary)
        A = (co > 0).astype(int); np.fill_diagonal(A, 0)
        # triangles per node: count for node i = sum_j sum_k A[i,j] * A[j,k] * A[k,i] / 2
        triangles = np.einsum("ij,jk,ki->i", A, A, A) // 2
        deg_a = A.sum(axis=1)
        possible = deg_a * (deg_a - 1) / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            local_cc = np.where(possible > 0, triangles / possible, 0)
        avg_cc = float(local_cc[deg_a > 1].mean()) if (deg_a > 1).any() else 0
        rows.append({"user_id": uid, "n_words": n,
                     "n_trials_with_words": int((bag.sum(axis=1) > 0).sum()),
                     "density": density, "avg_degree": avg_deg,
                     "avg_clustering": avg_cc,
                     "ref_dat_score": sub["ref_dat_score"].iloc[0]})
    res = pd.DataFrame(rows).dropna(subset=["ref_dat_score"])

    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(style.COL_2, 2.4),
                             gridspec_kw=dict(wspace=0.4))
    for ax, col, lab in zip(
        axes, ["density", "avg_degree", "avg_clustering"],
        ["Graph density", "Avg degree", "Avg clustering coef"],
    ):
        sub, n_drop = outliers.trim_sd(res, [col, "ref_dat_score"], sd=3.0)
        if n_drop:
            print(f"  D2[{col}]: dropped {n_drop} ±3 SD outliers")
        if len(sub) < 10:
            style.style_axis(ax, title=f"{lab} · n too small")
            continue
        r, p = pearsonr(sub["ref_dat_score"], sub[col])
        _summary_scatter(ax, sub["ref_dat_score"], sub[col])
        sns.regplot(data=sub, x="ref_dat_score", y=col,
                    scatter=False, color=style.COLORS["fit"],
                    line_kws=dict(linewidth=1.0), ax=ax)
        style.style_axis(ax, title=f"{lab} · {style.short_stats(r, p)}",
                         xlabel="DAT", ylabel="")
    fig.suptitle("D2 · per-participant co-occurrence graph stats", y=1.02)
    if show: plt.show()
    style.savefig(fig, "semantic_D2_cooccurrence")
    plt.close(fig)
    res.to_csv(config.OUTPUTS_DIR / "semantic_D2_cooccurrence.csv", index=False)
    print(f"  D2: n={len(res)}  density mean={res['density'].mean():.3f}")
    return res


# ═════════════════════════════════════════════════════════════════════════════
# E1 — per-image consensus
# ═════════════════════════════════════════════════════════════════════════════

def E1_image_consensus(trials_long: pd.DataFrame, show: bool = True):
    """For each unique stimulus image, compute consensus metrics:
        * n_participants seen
        * vocabulary entropy of submitted words (low = consensus)
        * mean pairwise cosine distance of submitted word embeddings
    """
    edict = embeddings.embedding_dict()
    rows = []
    for url, sub in trials_long.groupby("url_stimulus"):
        if url is None or pd.isna(url):
            continue
        words = sub["word"].tolist()
        if len(set(words)) < 2:
            continue
        cnt = Counter(words)
        n = sum(cnt.values())
        p = np.array(list(cnt.values())) / n
        H = float(-(p * np.log2(p)).sum())
        V = np.vstack([edict[w] for w in set(words)])
        sem = float(np.median(pdist(V, "cosine")))
        rows.append({
            "url": url,
            "fd_level": sub["fd_level"].iloc[0],
            "n_responses": n,
            "n_unique_words": len(cnt),
            "vocab_entropy": H,
            "semantic_spread": sem,
        })
    res = pd.DataFrame(rows)

    style.apply()
    # Paper figure: single-panel boxplot of vocab entropy; the semantic-spread
    # panel was redundant (we already show within-subject spread elsewhere).
    fig, ax = plt.subplots(figsize=(style.COL_1, 2.6))
    col, lab = "vocab_entropy", "Vocab entropy (bits)"
    sns.violinplot(
        data=res, x="fd_level", y=col, order=config.FD_LEVELS,
        palette=[style.COLORS[fd] for fd in config.FD_LEVELS],
        ax=ax, inner="box", cut=0, linewidth=0.5, saturation=0.9, width=0.9,
    )
    # Soften the violin edges
    for body in ax.collections:
        try: body.set_alpha(0.75); body.set_edgecolor("white")
        except Exception: pass
    from scipy.stats import mannwhitneyu
    pair_x = {"FD12": 0, "FD14": 1, "FD16": 2}
    comparisons = []
    for a, b in [("FD12", "FD14"), ("FD14", "FD16"), ("FD12", "FD16")]:
        ga = res[res["fd_level"] == a][col].values
        gb = res[res["fd_level"] == b][col].values
        if len(ga) >= 3 and len(gb) >= 3:
            _, pw = mannwhitneyu(ga, gb, alternative="two-sided")
            comparisons.append((pair_x[a], pair_x[b], float(pw)))
    style.style_axis(ax, title="", xlabel="", ylabel=lab)
    ax.set_xlabel("")  # seaborn re-applies "fd_level" — strip it
    style.sig_brackets(ax, comparisons)
    fig.suptitle("Per-image vocabulary entropy by FD", y=1.02)
    if show: plt.show()
    style.savefig(fig, "semantic_E1_image_consensus")
    plt.close(fig)
    res.to_csv(config.OUTPUTS_DIR / "semantic_E1_image_consensus.csv",
               index=False)
    print(f"  E1: {len(res)} images analysed; "
          f"entropy means by FD: "
          f"{res.groupby('fd_level')['vocab_entropy'].mean().to_dict()}")
    return res


# ═════════════════════════════════════════════════════════════════════════════
# E2 — image clustering by percept distribution
# ═════════════════════════════════════════════════════════════════════════════

# Surface-level repeats that don't carry a perceptual category — these
# previously polluted multiple clusters (geo names, generic adjectives).
E2_EXTRA_STOPWORDS = {
    "italy", "europe", "australia", "africa", "asia", "america", "country",
    "land", "island", "islands", "world", "earth", "globe", "continent",
    "thing", "stuff", "something", "object", "image", "picture",
    "old", "young", "big", "small", "large", "many", "few",
    "head",  # appears in nearly every cluster — too generic
}


def _image_embedding_matrix(trials_long: pd.DataFrame,
                            min_responses: int = 5
                            ) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Mean BERT embedding per image, weighted by word frequency in that image.

    Returns (img_meta, embedding_matrix, per-image word counts).
    """
    edict = embeddings.embedding_dict()
    tl = trials_long.dropna(subset=["url_stimulus"]).copy()
    tl = tl[~tl["word"].isin(E2_EXTRA_STOPWORDS)]
    grouped = tl.groupby(["url_stimulus", "word"]).size().rename("n").reset_index()

    img_meta_rows, X_rows, words_rows = [], [], []
    for url, sub in grouped.groupby("url_stimulus"):
        sub = sub[sub["word"].isin(edict)]
        n_responses = int(sub["n"].sum())
        if n_responses < min_responses or sub["word"].nunique() < 3:
            continue
        V = np.vstack([edict[w] for w in sub["word"]])
        weights = sub["n"].to_numpy(dtype=float)
        v = (V * weights[:, None]).sum(0) / weights.sum()
        v = v / (np.linalg.norm(v) or 1)
        fd = tl.loc[tl["url_stimulus"] == url, "fd_level"].iloc[0]
        img_meta_rows.append({"url": url, "fd_level": fd, "n_responses": n_responses})
        X_rows.append(v)
        words_rows.append((url, dict(zip(sub["word"], sub["n"]))))

    img_meta = pd.DataFrame(img_meta_rows).reset_index(drop=True)
    X = np.vstack(X_rows)
    counts = pd.DataFrame(words_rows, columns=["url", "word_counts"])
    return img_meta, X, counts


def _top_words_per_cluster(counts: pd.DataFrame, labels: np.ndarray,
                           top_k: int = 8,
                           min_count: int = 4,
                           log_prior: float = 0.5) -> dict[int, list[str]]:
    """For each cluster, words that are *distinctive* to it.

    Score = log((c+α)/(N_c+α)) − log((g − c + α)/(N_g − N_c + α)),
    i.e. log-odds of seeing the word inside vs outside this cluster.
    This filters out ubiquitous words like 'face' that appear everywhere.
    """
    counts = counts.assign(cluster=labels).reset_index(drop=True)

    # Global word -> total count
    global_counter: Counter = Counter()
    for d in counts["word_counts"]:
        global_counter.update(d)
    g_total = sum(global_counter.values())

    out = {}
    for c in sorted(set(labels)):
        in_counter: Counter = Counter()
        for d in counts.loc[counts["cluster"] == c, "word_counts"]:
            in_counter.update(d)
        n_in = sum(in_counter.values())
        if n_in == 0:
            out[c] = []
            continue
        scored = []
        for w, c_in in in_counter.items():
            if c_in < min_count:
                continue
            c_out = global_counter[w] - c_in
            n_out = g_total - n_in
            p_in = (c_in + log_prior) / (n_in + log_prior)
            p_out = (c_out + log_prior) / (n_out + log_prior)
            scored.append((w, np.log(p_in) - np.log(p_out)))
        scored.sort(key=lambda x: x[1], reverse=True)
        out[c] = [w for w, _ in scored[:top_k]]
    return out


def _pick_k_by_silhouette(X: np.ndarray, k_range=range(3, 13)) -> tuple[int, dict[int, float]]:
    """Pick k that maximises silhouette score under cosine distance."""
    from sklearn.metrics import silhouette_score
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(X)
        try:
            scores[k] = float(silhouette_score(X, km.labels_, metric="cosine"))
        except Exception:
            scores[k] = float("nan")
    best_k = max(scores, key=lambda k: scores[k])
    return best_k, scores


def E2_image_clusters(trials_long: pd.DataFrame, show: bool = True,
                      method: str = "hdbscan", n_clusters: int | None = None,
                      min_responses: int = 5,
                      umap_dim: int = 8):
    """Cluster images by the *mean BERT embedding* of their percept words.

    method='hdbscan': UMAP(8d, cosine) → HDBSCAN. No k to set; outputs noise
                     as cluster -1.
    method='kmeans':  cosine-normalised vectors → K-means. k picked by
                     silhouette unless `n_clusters` given.
    method='agglo':   ward agglomerative on UMAP space (interpretable trees).
    """
    import umap
    import hdbscan
    from sklearn.cluster import AgglomerativeClustering

    img_meta, X, counts = _image_embedding_matrix(trials_long, min_responses)
    print(f"  E2: {len(img_meta)} images × {X.shape[1]}-d mean-BERT embedding")

    # UMAP to a low-d space using cosine — preserves semantic neighbourhoods.
    reducer = umap.UMAP(n_components=umap_dim, metric="cosine",
                         random_state=0, n_neighbors=15, min_dist=0.0)
    X_low = reducer.fit_transform(X)
    # A second UMAP to 2D for plotting only.
    XY = umap.UMAP(n_components=2, metric="cosine", random_state=0,
                    n_neighbors=15, min_dist=0.1).fit_transform(X)
    img_meta[["x", "y"]] = XY

    sil_scores = None
    if method == "hdbscan":
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5,
                                     cluster_selection_method="eom")
        labels = clusterer.fit_predict(X_low)
    elif method == "kmeans":
        if n_clusters is None:
            n_clusters, sil_scores = _pick_k_by_silhouette(X_low)
            print(f"  k by silhouette: {n_clusters}  scores={ {k: round(s, 3) for k, s in sil_scores.items()} }")
        labels = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0).fit_predict(X_low)
    elif method == "agglo":
        if n_clusters is None:
            n_clusters, sil_scores = _pick_k_by_silhouette(X_low)
        labels = AgglomerativeClustering(n_clusters=n_clusters,
                                          linkage="ward").fit_predict(X_low)
    else:
        raise ValueError(method)
    img_meta = img_meta.assign(cluster=labels)
    top_words_per = _top_words_per_cluster(counts.loc[img_meta.index], labels)

    n_real = sum(1 for c in top_words_per if c != -1)
    print(f"  {method}: {n_real} clusters"
          + (f" + noise(n={int((labels == -1).sum())})" if -1 in labels else ""))
    for c in sorted(top_words_per):
        n = int((labels == c).sum())
        tag = "noise" if c == -1 else f"C{c}"
        print(f"    {tag} (n={n}):  {', '.join(top_words_per[c])}")

    # ── plot ──
    style.apply()
    fig, axes = plt.subplots(
        1, 2, figsize=(style.COL_2, 3.4),
        gridspec_kw=dict(wspace=0.35, width_ratios=[1.0, 1.6]),
    )
    real = sorted(c for c in set(labels) if c != -1)
    palette = sns.color_palette("tab10", len(real))
    if -1 in labels:
        s = img_meta[img_meta["cluster"] == -1]
        axes[0].scatter(s["x"], s["y"], s=6, alpha=0.3,
                        color="lightgrey", linewidths=0, label="noise")
    for idx, c in enumerate(real):
        s = img_meta[img_meta["cluster"] == c]
        axes[0].scatter(s["x"], s["y"], s=10, alpha=0.8,
                        color=palette[idx], linewidths=0,
                        label=f"C{c}: {', '.join(top_words_per[c][:2])}")
    style.style_axis(axes[0], title=f"{method.upper()} clusters · UMAP 2-D",
                     xlabel="UMAP-1", ylabel="UMAP-2")
    axes[0].legend(frameon=False, fontsize=5, loc="lower center",
                   bbox_to_anchor=(0.5, -0.65), ncol=2)

    # heatmap with labelled rows
    ct = (img_meta[img_meta["cluster"] != -1]
          .groupby(["cluster", "fd_level"]).size()
          .unstack(fill_value=0)
          .div(img_meta[img_meta["cluster"] != -1].groupby("fd_level").size(), axis=1))
    ct = ct.reindex(real)
    n_imgs = img_meta[img_meta["cluster"] != -1].groupby("cluster").size().reindex(real)
    yticklabels = [
        f"C{c} (n={int(n_imgs[c])})\n{', '.join(top_words_per[c][:5])}"
        for c in real
    ]
    sns.heatmap(ct, annot=True, fmt=".2f", cmap="rocket_r",
                ax=axes[1], cbar_kws=dict(label="Share of FD's images"),
                annot_kws=dict(fontsize=6),
                yticklabels=yticklabels,
                linewidths=0.5, linecolor="white")
    axes[1].set_yticklabels(yticklabels, rotation=0, fontsize=5.5)
    style.style_axis(axes[1], title="Cluster × FD", xlabel="", ylabel="")
    fig.suptitle(f"E2 · image clustering ({method}, BERT mean embedding)", y=1.02)
    if show: plt.show()
    style.savefig(fig, f"semantic_E2_image_clusters_{method}")
    plt.close(fig)

    # silhouette curve when relevant
    if sil_scores:
        sfig, sax = plt.subplots(figsize=(style.COL_1, 1.8))
        sax.plot(list(sil_scores), list(sil_scores.values()), "o-",
                 color=style.COLORS["fit"])
        sax.axvline(n_clusters, ls="--", color="grey", lw=0.6)
        style.style_axis(sax, title="Silhouette vs k",
                         xlabel="k", ylabel="silhouette (cosine)")
        style.savefig(sfig, f"semantic_E2_silhouette_{method}")
        plt.close(sfig)

    rows = [
        {"cluster": c, "n_images": int((labels == c).sum()),
         "top_words": ", ".join(top_words_per[c]),
         **{f"share_{fd}": float(ct.loc[c, fd]) for fd in config.FD_LEVELS
            if c in ct.index and fd in ct.columns}}
        for c in real
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(config.OUTPUTS_DIR / f"semantic_E2_clusters_{method}.csv",
                   index=False)
    img_meta.reset_index().to_csv(
        config.OUTPUTS_DIR / f"semantic_E2_image_clusters_{method}_full.csv",
        index=False,
    )
    return summary


# ─── runner ──────────────────────────────────────────────────────────────────

REGISTRY = {
    "A1": A1_cohesion,
    "A3": A3_centroid_pca,
    "B1": B1_specialisation,
    "B2": lambda df, show: B2_ttr(_trials_long(),
                                   df.drop_duplicates("user_id")[["user_id", "ref_dat_score"]],
                                   show=show),
    "C1": C1_categories,
    "C2": C2_face_body,
    "D2": lambda df, show: D2_cooccurrence(_trials_long(),
                                            df.drop_duplicates("user_id")[["user_id", "ref_dat_score"]],
                                            show=show),
    "E1": lambda _df, show: E1_image_consensus(_trials_long(), show=show),
    "E2": lambda _df, show: E2_image_clusters(_trials_long(), method="hdbscan", show=show),
    "E2_kmeans": lambda _df, show: E2_image_clusters(_trials_long(), method="kmeans", show=show),
    "E2_agglo":  lambda _df, show: E2_image_clusters(_trials_long(), method="agglo", show=show),
}


def main(only: list[str] | None = None, show: bool = True):
    df, _ = _long_words()
    print(f"Long words table: {len(df):,} rows, "
          f"{df['user_id'].nunique()} users, {df['word'].nunique()} unique words")
    selected = only or list(REGISTRY.keys())
    for name in selected:
        print(f"\n--- {name} ---")
        REGISTRY[name](df, show=show)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(REGISTRY))
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    main(only=args.only, show=not args.no_show)
