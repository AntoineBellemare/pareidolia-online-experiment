"""Quantify semantic spread of percepts by DAT tertile and by FD level.

Five complementary metrics, all bootstrap-subsampled to equal N so that
between-group comparisons are not confounded by sample size:

  M1  median pair-wise cosine distance      (simple, interpretable)
  M2  area of the 95 % covariance ellipse in t-SNE 2-D
  M3  participation ratio of the embedding covariance
      = (Σ λ)² / Σ λ²  ; effective number of dimensions occupied
  M4  differential entropy estimated via 10-bin 2-D histogram in
      shared t-SNE space  (KDE-free, bandwidth-agnostic)
  M5  number of HDBSCAN clusters that the group has ≥ 4 words in
      (categorical diversity)

Each metric is computed B = 500 times on a bootstrap subsample of size
n_sub = min(group_n) − 5 (with replacement); we report the mean and
95 % bootstrap CI per group.

Usage:
    python -m analysis.figures.group_spread_metrics
"""
from __future__ import annotations

import argparse
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import eigvalsh
from scipy.spatial.distance import pdist
from scipy.stats import chi2
from sklearn.manifold import TSNE

from .. import config, dat_helper, embeddings, parse_events, style

ASCII = re.compile(r"^[A-Za-z]+$")
LABELS_DAT = ["Low", "Mid", "High"]
LABELS_FD  = config.FD_LEVELS
N_BOOT  = 200          # bootstrap iterations per group
N_SUB_CAP = 600        # cap subsample size (pdist is O(n²); 600² = 360k pairs)
RNG = np.random.default_rng(0)


# ─── data assembly ───────────────────────────────────────────────────────────

def _long_words() -> pd.DataFrame:
    p = dat_helper.notebook_cohort().dropna(subset=["ref_dat_score"])
    edict = embeddings.embedding_dict()
    rows = []
    for _, r in p.iterrows():
        for fd in config.FD_LEVELS:
            ws = r.get(f"{fd}_words")
            if ws is None: continue
            for w in list(ws):
                if (ASCII.fullmatch(w or "") and w in edict
                        and w not in config.SEMANTIC_STOPWORDS):
                    rows.append({"user_id": r["user_id"], "word": w,
                                  "fd_level": fd,
                                  "ref_dat_score": r["ref_dat_score"]})
    df = pd.DataFrame(rows)
    df["dat_tertile"] = pd.qcut(df["ref_dat_score"], 3, labels=LABELS_DAT,
                                duplicates="drop")
    return df


# ─── shared 2-D embedding for ellipse & entropy ──────────────────────────────

def _shared_tsne(words: list[str]) -> dict[str, tuple[float, float]]:
    edict = embeddings.embedding_dict()
    V = np.vstack([edict[w] for w in words])
    coords = TSNE(2, metric="cosine", perplexity=30, init="random",
                   random_state=0, n_jobs=-1).fit_transform(V)
    return dict(zip(words, map(tuple, coords)))


# ─── HDBSCAN clusters for M5 ─────────────────────────────────────────────────

def _hdbscan_assignment(words: list[str], min_count: int = 3) -> dict[str, int]:
    import umap, hdbscan
    edict = embeddings.embedding_dict()
    V = np.vstack([edict[w] for w in words])
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    X = umap.UMAP(n_components=10, metric="cosine",
                   random_state=0, n_neighbors=15,
                   min_dist=0.0).fit_transform(Vn)
    labels = hdbscan.HDBSCAN(min_cluster_size=12, min_samples=3,
                              cluster_selection_method="eom").fit_predict(X)
    return dict(zip(words, labels.tolist()))


# ─── metric implementations ──────────────────────────────────────────────────

def m1_pairwise(V: np.ndarray) -> float:
    if len(V) < 2: return np.nan
    return float(np.median(pdist(V, "cosine")))


def m2_ellipse(xy: np.ndarray) -> float:
    if len(xy) < 5: return np.nan
    cov = np.cov(xy, rowvar=False)
    lam = eigvalsh(cov)[::-1]
    return float(np.pi * np.prod(np.sqrt(lam * chi2.ppf(0.95, 2))))


def m3_participation(V: np.ndarray) -> float:
    if len(V) < 3: return np.nan
    cov = np.cov(V, rowvar=False)
    lam = eigvalsh(cov)
    lam = lam[lam > 0]
    if lam.size == 0: return np.nan
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def m4_entropy_2d(xy: np.ndarray, bins: int = 12) -> float:
    if len(xy) < 5: return np.nan
    h, *_ = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins)
    p = h.ravel(); p = p[p > 0]; p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def m5_cluster_diversity(words: list[str],
                          word2cluster: dict[str, int],
                          min_words: int = 4) -> int:
    cnt: Counter = Counter()
    for w in words:
        c = word2cluster.get(w, -1)
        if c == -1: continue
        cnt[c] += 1
    return int(sum(1 for v in cnt.values() if v >= min_words))


# ─── bootstrap engine ────────────────────────────────────────────────────────

def _compute_all_metrics(words: list[str],
                          coords: dict[str, tuple[float, float]],
                          word2cluster: dict[str, int]) -> dict[str, float]:
    edict = embeddings.embedding_dict()
    V = np.vstack([edict[w] for w in words])
    xy = np.array([coords[w] for w in words])
    return {
        "pairwise_cos": m1_pairwise(V),
        "ellipse_area_2d": m2_ellipse(xy),
        "participation_ratio": m3_participation(V),
        "entropy_2d_hist": m4_entropy_2d(xy),
        "cluster_diversity": m5_cluster_diversity(words, word2cluster),
    }


def bootstrap_group(words: list[str], n_sub: int,
                     coords: dict[str, tuple[float, float]],
                     word2cluster: dict[str, int],
                     n_boot: int = N_BOOT,
                     label: str = "") -> pd.DataFrame:
    import time
    rows = []
    n_sub = min(n_sub, len(words), N_SUB_CAP)
    arr = np.array(words)
    t0 = time.time()
    for b in range(n_boot):
        sample = arr[RNG.integers(0, len(arr), size=n_sub)].tolist()
        rows.append(_compute_all_metrics(sample, coords, word2cluster))
        if (b + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"    [{label}] {b + 1}/{n_boot}  ({elapsed:.1f}s, "
                  f"eta {elapsed / (b + 1) * (n_boot - b - 1):.0f}s)",
                  flush=True)
    return pd.DataFrame(rows)


# ─── plotting ────────────────────────────────────────────────────────────────

METRIC_LABELS = {
    "pairwise_cos":         "Median pair-wise\ncosine distance",
    "ellipse_area_2d":      "95 % ellipse area\nin t-SNE",
    "participation_ratio":  "Participation ratio\n(effective dim.)",
    "entropy_2d_hist":      "2-D histogram\nentropy (nats)",
    "cluster_diversity":    "# HDBSCAN clusters\nused (≥ 4 words)",
}

# The three metrics promoted to the paper figure; the other two are in
# supplementary table only.
PAPER_METRICS = ["pairwise_cos", "ellipse_area_2d", "participation_ratio"]


def plot_per_group(boots_dat: dict, boots_fd: dict,
                    show: bool = True,
                    metrics: list[str] | None = None,
                    out_name: str = "group_spread_metrics") -> None:
    style.apply()
    metrics = metrics or list(METRIC_LABELS.keys())
    n = len(metrics)
    fig, axes = plt.subplots(
        2, n, figsize=(min(style.COL_2, 1.8 * n + 0.8), 4.2),
        gridspec_kw=dict(hspace=0.7, wspace=0.55),
    )
    if n == 1: axes = axes.reshape(2, 1)
    for ax, m in zip(axes[0], metrics):
        _draw_bar(ax, boots_dat, m, LABELS_DAT,
                   [style.COLORS[t] for t in LABELS_DAT])
        ax.set_title(METRIC_LABELS[m], fontsize=7.5)
        ax.tick_params(labelsize=6)
    for ax, m in zip(axes[1], metrics):
        _draw_bar(ax, boots_fd, m, LABELS_FD,
                   [style.COLORS[t] for t in LABELS_FD])
        ax.tick_params(labelsize=6)
    axes[0, 0].set_ylabel("DAT tertile", fontsize=7.5)
    axes[1, 0].set_ylabel("FD level",    fontsize=7.5)
    fig.suptitle("Pooled semantic spread per group "
                  "(bootstrap-equalised N)", y=1.02)
    if show: plt.show()
    style.savefig(fig, out_name)
    plt.close(fig)


def _draw_bar(ax, boots: dict, metric: str, levels: list[str],
              colors: list[str]) -> None:
    means = np.array([boots[t][metric].mean() for t in levels])
    lo = np.array([np.percentile(boots[t][metric], 2.5) for t in levels])
    hi = np.array([np.percentile(boots[t][metric], 97.5) for t in levels])
    yerr = np.vstack([means - lo, hi - means])
    xs = np.arange(len(levels))
    ax.bar(xs, means, yerr=yerr, color=colors,
           capsize=2.5, alpha=0.85, edgecolor="black", lw=0.4)
    ax.set_xticks(xs); ax.set_xticklabels(levels, fontsize=6)
    # Zoom y so the variation is visible
    span = (hi - lo).max() if (hi - lo).max() > 0 else 1
    ax.set_ylim(lo.min() - 0.25 * span, hi.max() + 0.25 * span)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ─── runner ──────────────────────────────────────────────────────────────────

def main(show: bool = True):
    df = _long_words()

    # Shared 2-D embedding once for ellipse + entropy comparability
    print("Building shared t-SNE for M2, M4…")
    all_unique = sorted(df["word"].unique())
    coords = _shared_tsne(all_unique)

    print("Building shared HDBSCAN clustering for M5…")
    word2cluster = _hdbscan_assignment(all_unique)

    # Run bootstrap per group
    print("Bootstrapping DAT tertile groups…")
    boots_dat = {}
    grp_sizes_d = {}
    for t in LABELS_DAT:
        words = df.loc[df["dat_tertile"] == t, "word"].tolist()
        grp_sizes_d[t] = len(words)
    n_sub_d = min(grp_sizes_d.values()) - 5
    for t in LABELS_DAT:
        words = df.loc[df["dat_tertile"] == t, "word"].tolist()
        boots_dat[t] = bootstrap_group(words, n_sub_d, coords, word2cluster, label=f"DAT={t}")
        print(f"  {t}: N={grp_sizes_d[t]} → n_sub={n_sub_d}, "
              f"means={ {m: round(boots_dat[t][m].mean(), 3) for m in METRIC_LABELS} }")

    print("Bootstrapping FD groups…")
    boots_fd = {}
    grp_sizes_f = {}
    for fd in LABELS_FD:
        words = df.loc[df["fd_level"] == fd, "word"].tolist()
        grp_sizes_f[fd] = len(words)
    n_sub_f = min(grp_sizes_f.values()) - 5
    for fd in LABELS_FD:
        words = df.loc[df["fd_level"] == fd, "word"].tolist()
        boots_fd[fd] = bootstrap_group(words, n_sub_f, coords, word2cluster, label=f"FD={fd}")
        print(f"  {fd}: N={grp_sizes_f[fd]} → n_sub={n_sub_f}, "
              f"means={ {m: round(boots_fd[fd][m].mean(), 3) for m in METRIC_LABELS} }")

    # Save summary table
    rows = []
    for grouping, boots, levels in [("DAT", boots_dat, LABELS_DAT),
                                     ("FD",  boots_fd,  LABELS_FD)]:
        for level in levels:
            row = {"grouping": grouping, "level": level}
            for m in METRIC_LABELS:
                vals = boots[level][m]
                row[f"{m}_mean"]  = float(vals.mean())
                row[f"{m}_ci_lo"] = float(np.percentile(vals, 2.5))
                row[f"{m}_ci_hi"] = float(np.percentile(vals, 97.5))
            rows.append(row)
    pd.DataFrame(rows).to_csv(
        config.OUTPUTS_DIR / "group_spread_metrics.csv", index=False
    )

    # Paper version (3 metrics, narrower) and full version (5 metrics).
    plot_per_group(boots_dat, boots_fd, show=show,
                   metrics=PAPER_METRICS,
                   out_name="group_spread_metrics_paper")
    plot_per_group(boots_dat, boots_fd, show=show,
                   out_name="group_spread_metrics")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    main(show=not args.no_show)
