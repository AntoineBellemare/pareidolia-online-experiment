"""Per-image spectral / spatial features that predict pareidolia.

Ports the image-feature extraction from
``MachinePareidolia/scripts/human_pareidolia_analysis.py`` and re-runs it on
the current cohort + 300-image stimulus bank, producing paper-ready
figures (untitled + titled variants via ``style.savefig``).

For each of the 300 stimulus images we extract approximately 24 spectral
and spatial features: radially-averaged power-spectrum slope, low / mid /
high frequency band shares, residual after the 1/fβ fit, Sobel edge
magnitude, local-contrast mean and SD, connected-component statistics
under a mean-intensity binarisation, pixel entropy, four symmetry
measures (left-right pixel, top-bottom pixel, gradient-orientation,
coarse-downsampled and Fourier-magnitude), and basic intensity
statistics. We then ask which of these features predict per-image
pareidolia rate, mean words per trial, agreement (modal-word share) and
word diversity, both via Pearson correlations and via cross-validated
Random Forest regression with permutation feature importance.

Cache: ``analysis_cache/image_features.parquet`` — 300 × ~25 columns.
Re-run with ``--rebuild`` to regenerate.

Usage:
    python -m analysis.figures.image_features
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .. import config, dat_helper, parse_events, style

ASCII = re.compile(r"^[A-Za-z]+$")
REMOVED = config.SEMANTIC_STOPWORDS  # same set used elsewhere in the paper

# Where the 300 stimuli live in this repo. Accepts both the manuscript
# layout ("stimuli/FDxx/...") and the legacy website layout
# ("images/FDxx/..."); both URL forms are resolved against STIMULI_ROOT.
IMAGE_ROOT = config.STIMULI_DIR


def _resolve_image(image_url: str) -> Path:
    rel = str(image_url)
    for prefix in ("stimuli/", "images/"):
        if rel.startswith(prefix):
            rel = rel[len(prefix):]
            break
    return IMAGE_ROOT / rel


# Cache the extracted features. The shipped variant lives in data/ and
# is preferred via config.cached_parquet when present.
CACHE_PATH = config.cached_parquet("image_features.parquet")
WRITE_CACHE_PATH = config.CACHE_DIR / "image_features.parquet"


# ─── per-image pareidolia metrics ────────────────────────────────────────────

def _per_image_stats() -> pd.DataFrame:
    """Per-image: pareidolia rate, mean #words, agreement, word diversity."""
    trials = parse_events.cached_trials()
    nb = dat_helper.notebook_cohort()
    trials = trials[trials["user_id"].isin(nb["user_id"])].copy()

    def _first(ws):
        if ws is None: return ""
        for w in list(ws):
            w = str(w).strip().lower()
            if ASCII.fullmatch(w) and w not in REMOVED:
                return w
        return ""

    def _clean_n(ws):
        if ws is None: return 0
        return sum(1 for w in list(ws)
                   if ASCII.fullmatch(str(w).strip().lower())
                   and str(w).strip().lower() not in REMOVED)

    trials["n_clean"] = trials["words"].apply(_clean_n)
    trials["has_pareidolia"] = trials["n_clean"] > 0
    trials["first_word"] = trials["words"].apply(_first)

    img_stats = (
        trials.groupby("url_stimulus")
              .agg(fd_level=("fd_level", "first"),
                   n_trials=("user_id", "count"),
                   n_pareidolia=("has_pareidolia", "sum"),
                   mean_n_words=("n_clean", "mean"),
                   unique_first=("first_word",
                                  lambda x: len({w for w in x if w})))
              .reset_index()
              .rename(columns={"url_stimulus": "image_url"})
    )
    img_stats["pareidolia_rate"] = (img_stats["n_pareidolia"]
                                     / img_stats["n_trials"])
    img_stats["word_diversity"] = (img_stats["unique_first"]
                                     / img_stats["n_trials"])

    def _agreement(group):
        words = [w for w in group["first_word"] if w]
        if not words: return 0.0
        return Counter(words).most_common(1)[0][1] / len(words)
    agg = (trials.groupby("url_stimulus")
                 .apply(_agreement).reset_index(name="agreement")
                 .rename(columns={"url_stimulus": "image_url"}))
    img_stats = img_stats.merge(agg, on="image_url")
    return img_stats.sort_values("pareidolia_rate", ascending=False)\
                    .reset_index(drop=True)


# ─── feature extraction (port of the MachinePareidolia helper) ───────────────

def _extract_features(img_path: Path) -> dict:
    from PIL import Image
    from scipy import ndimage
    from scipy.ndimage import uniform_filter, zoom
    from scipy.stats import entropy as sp_entropy, kurtosis, skew

    img = Image.open(img_path).convert("L")
    px = np.array(img, dtype=np.float64) / 255.0
    h, w = px.shape
    f = {}

    # 1. intensity stats
    f["mean"] = px.mean(); f["std"] = px.std()
    f["skewness"] = float(skew(px.ravel()))
    f["kurtosis"] = float(kurtosis(px.ravel()))

    # 2. power spectrum
    F = np.fft.fftshift(np.fft.fft2(px))
    power = np.abs(F) ** 2
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    max_r = min(cx, cy)
    radial_power = np.zeros(max_r)
    for r in range(1, max_r):
        m = R == r
        if m.any():
            radial_power[r] = power[m].mean()
    freqs = np.arange(1, max_r)
    valid = radial_power[1:] > 0
    if valid.sum() > 5:
        log_f = np.log(freqs[valid]); log_p = np.log(radial_power[1:][valid])
        beta, intercept = np.polyfit(log_f, log_p, 1)
        f["spectral_slope"] = -beta
        f["spectral_residual"] = float(np.std(log_p - (beta * log_f + intercept)))
    else:
        f["spectral_slope"] = 0.0; f["spectral_residual"] = 0.0
    tot = radial_power[1:].sum()
    if tot > 0:
        third = max_r // 3
        f["power_low"]  = radial_power[1:third].sum() / tot
        f["power_mid"]  = radial_power[third:2*third].sum() / tot
        f["power_high"] = radial_power[2*third:max_r].sum() / tot
    else:
        f["power_low"] = f["power_mid"] = f["power_high"] = 0.0

    # 3. spatial — edges + local contrast
    sx = ndimage.sobel(px, axis=1); sy = ndimage.sobel(px, axis=0)
    edges = np.hypot(sx, sy)
    f["edge_mean"] = edges.mean(); f["edge_std"] = edges.std()
    local_mean = uniform_filter(px, size=16)
    local_sq   = uniform_filter(px ** 2, size=16)
    local_std  = np.sqrt(np.maximum(local_sq - local_mean ** 2, 0))
    f["local_contrast_mean"] = local_std.mean()
    f["local_contrast_std"]  = local_std.std()

    # 4. connected components (binarised at mean px)
    binary = px > px.mean()
    labeled, n = ndimage.label(binary)
    f["n_components"] = int(n)
    if n > 0:
        sizes = ndimage.sum(binary, labeled, range(1, n + 1))
        f["largest_component"]   = float(np.max(sizes)) / (h * w)
        f["mean_component_size"] = float(np.mean(sizes)) / (h * w)
        f["component_size_std"]  = float(np.std(sizes)) / (h * w)
    else:
        f["largest_component"] = f["mean_component_size"] = \
            f["component_size_std"] = 0.0

    # 5. entropy
    hist, _ = np.histogram(px, bins=64, range=(0, 1))
    hist = hist / hist.sum()
    f["pixel_entropy"] = float(sp_entropy(hist + 1e-10))

    # 6. symmetry
    left = px[:, :w//2]; right = px[:, w//2:2*(w//2)][:, ::-1]
    f["lr_symmetry"] = float(np.corrcoef(left.ravel(), right.ravel())[0, 1])
    top = px[:h//2, :]; bot = px[h//2:2*(h//2), :][::-1, :]
    f["tb_symmetry"] = float(np.corrcoef(top.ravel(), bot.ravel())[0, 1])
    gx = ndimage.sobel(px, axis=1); gy = ndimage.sobel(px, axis=0)
    angles = np.arctan2(gy, gx)
    lh, _ = np.histogram(angles[:, :w//2].ravel(),
                          bins=36, range=(-np.pi, np.pi))
    rh, _ = np.histogram(angles[:, w//2:].ravel(),
                          bins=36, range=(-np.pi, np.pi))
    f["grad_symmetry_lr"] = float(np.corrcoef(lh, rh)[0, 1])
    px_s = zoom(px, 0.25, order=1)
    lf = px_s[:, :px_s.shape[1]//2]
    rf = px_s[:, px_s.shape[1]//2:][:, ::-1]
    f["coarse_symmetry_lr"] = float(np.corrcoef(lf.ravel(), rf.ravel())[0, 1])
    Fmag = np.abs(np.fft.fftshift(np.fft.fft2(px)))
    f["fourier_symmetry_lr"] = float(np.corrcoef(
        Fmag[:, :w//2].ravel(), Fmag[:, w//2:][:, ::-1].ravel())[0, 1])
    return f


def _extract_all(img_stats: pd.DataFrame, rebuild: bool = False) -> pd.DataFrame:
    if not rebuild and CACHE_PATH.exists():
        feats = pd.read_parquet(CACHE_PATH)
        print(f"Loaded cached features: {CACHE_PATH} ({len(feats)} rows)")
        return feats

    print(f"Extracting features from {len(img_stats)} images (slow, ~30s)...")
    rows = []
    for i, r in enumerate(img_stats.itertuples()):
        p = _resolve_image(r.image_url)
        if not p.exists():
            print(f"  MISSING: {p}")
            continue
        f = _extract_features(p)
        f["image_url"] = r.image_url
        rows.append(f)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(img_stats)} done")
    feats = pd.DataFrame(rows)
    WRITE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(WRITE_CACHE_PATH)
    print(f"Cached: {WRITE_CACHE_PATH}")
    return feats


# ─── paper figures ───────────────────────────────────────────────────────────

TARGETS = ["pareidolia_rate", "mean_n_words", "agreement", "word_diversity"]
TARGET_LABELS = {
    "pareidolia_rate": "pareidolia\nrate",
    "mean_n_words":    "mean\nn words",
    "agreement":       "agreement",
    "word_diversity":  "word\ndiversity",
}


def _human_label(name: str) -> str:
    """Render feature names in a slightly more readable form for the paper."""
    return name.replace("_", " ").replace("lr", "L-R").replace("tb", "T-B")


def figure_heatmap(img_full: pd.DataFrame, feat_cols: list[str],
                    show: bool = True) -> pd.DataFrame:
    """Paper-ready heatmap: 24 features × 4 pareidolia metrics, signed
    Pearson r with significance stars overlaid in each cell.
    """
    from scipy.stats import pearsonr

    style.apply()

    # Compute r and p for every (feature, target) cell.
    r_mat = pd.DataFrame(index=feat_cols, columns=TARGETS, dtype=float)
    p_mat = pd.DataFrame(index=feat_cols, columns=TARGETS, dtype=float)
    for f in feat_cols:
        for t in TARGETS:
            r, p = pearsonr(img_full[f].values, img_full[t].values)
            r_mat.loc[f, t] = r
            p_mat.loc[f, t] = p
    # Sort rows by r against pareidolia_rate (most positive at top).
    r_mat = r_mat.sort_values("pareidolia_rate", ascending=False)
    p_mat = p_mat.loc[r_mat.index]

    # Build annotation strings: "<r>\n<stars>"
    annot = pd.DataFrame(index=r_mat.index, columns=r_mat.columns, dtype=object)
    for f in r_mat.index:
        for t in r_mat.columns:
            r = r_mat.loc[f, t]; p = p_mat.loc[f, t]
            stars = style.sig_marker(p)
            annot.loc[f, t] = f"{r:+.2f}{stars}" if stars else f"{r:+.2f}"

    fig, ax = plt.subplots(figsize=(style.COL_1_5, 6.4))
    sns.heatmap(
        r_mat.astype(float), annot=annot.values, fmt="",
        cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
        cbar_kws=dict(label="Pearson r", shrink=0.6, aspect=14),
        linewidths=0.3, linecolor="white",
        ax=ax, annot_kws=dict(size=6.5),
    )
    ax.set_yticklabels([_human_label(f) for f in r_mat.index],
                        fontsize=7, rotation=0)
    ax.set_xticklabels([TARGET_LABELS[t] for t in r_mat.columns],
                        fontsize=7, rotation=0)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(length=0)
    # Caption-friendly key for the stars.
    fig.text(0.99, 0.01,
              "* p<0.05    ** p<0.01    *** p<0.001",
              ha="right", va="bottom", fontsize=5.5, color="#666")
    fig.suptitle("Image features vs pareidolia metrics  (Pearson r, n = 300)",
                  y=1.005)
    if show: plt.show()
    style.savefig(fig, "image_features_heatmap")
    plt.close(fig)
    return r_mat


def figure_importance(img_full: pd.DataFrame, feat_cols: list[str],
                       show: bool = True) -> dict:
    """Paper-ready 2-panel feature importance for pareidolia rate:
    (a) Random Forest Gini importance and (b) permutation importance.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.preprocessing import StandardScaler

    style.apply()
    y = img_full["pareidolia_rate"].values
    X_raw = img_full[feat_cols].values
    X = StandardScaler().fit_transform(X_raw)
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=4, min_samples_leaf=10, random_state=42,
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv = cross_val_score(rf, X, y, cv=kf, scoring="r2")
    r2_mean, r2_std = float(cv.mean()), float(cv.std())
    print(f"RF 5-fold CV R² on pareidolia_rate: {r2_mean:+.3f} ± {r2_std:.3f}")
    rf.fit(X, y)
    perm = permutation_importance(rf, X, y, n_repeats=30, random_state=42,
                                    n_jobs=-1)

    gini = pd.Series(rf.feature_importances_, index=feat_cols).sort_values()
    pimp = pd.Series(perm.importances_mean, index=feat_cols).sort_values()

    fig, axes = plt.subplots(
        1, 2, figsize=(style.COL_2, 5.4),
        gridspec_kw=dict(wspace=0.55),
    )

    # ── (a) Gini importance ────────────────────────────────────────────────
    ax = axes[0]
    ys = np.arange(len(gini))
    ax.barh(ys, gini.values, color=style.COLORS["FD12"], alpha=0.9,
             edgecolor="black", lw=0.3, height=0.7)
    ax.set_yticks(ys)
    ax.set_yticklabels([_human_label(f) for f in gini.index], fontsize=6.5)
    style.style_axis(ax, title="(a) Random Forest importance (Gini)",
                      xlabel="Importance", ylabel="")

    # ── (b) Permutation importance ─────────────────────────────────────────
    ax = axes[1]
    ys = np.arange(len(pimp))
    ax.barh(ys, pimp.values, color=style.COLORS["dot"], alpha=0.9,
             edgecolor="black", lw=0.3, height=0.7)
    ax.set_yticks(ys)
    ax.set_yticklabels([_human_label(f) for f in pimp.index], fontsize=6.5)
    style.style_axis(ax, title="(b) Permutation importance",
                      xlabel="Mean R² decrease", ylabel="")
    ax.text(0.98, 0.04,
             f"RF 5-fold CV\nR² = {r2_mean:+.2f} ± {r2_std:.2f}",
             ha="right", va="bottom", transform=ax.transAxes,
             fontsize=6, color="#444",
             bbox=dict(facecolor="white", edgecolor="#aaa",
                        boxstyle="round,pad=0.3", linewidth=0.4))

    fig.suptitle("Which image features predict pareidolia rate?", y=1.01)
    if show: plt.show()
    style.savefig(fig, "image_features_importance")
    plt.close(fig)
    return {"gini": gini, "permutation": pimp,
            "r2_mean": r2_mean, "r2_std": r2_std}


# ─── supplementary: high vs low pareidolia image gallery ───────────────────

def figure_gallery(img_full: pd.DataFrame, show: bool = True,
                    n_per_row: int = 6) -> None:
    """Paper-ready 2-row image gallery: highest- vs lowest-pareidolia images.

    Each thumbnail is labelled with FD level, pareidolia rate, and the
    three most common percept words (from the trial-level table).
    """
    from PIL import Image as PILImage

    style.apply()
    trials = parse_events.cached_trials()
    nb = dat_helper.notebook_cohort()
    trials = trials[trials["user_id"].isin(nb["user_id"])].copy()

    def _top_words(url: str, k: int = 3) -> str:
        sub = trials[trials["url_stimulus"] == url]
        ws = []
        for v in sub["words"]:
            if v is None: continue
            for w in list(v):
                w = str(w).strip().lower()
                if ASCII.fullmatch(w) and w not in REMOVED:
                    ws.append(w)
        if not ws: return ""
        top = Counter(ws).most_common(k)
        return ", ".join(w for w, _ in top)

    high = img_full.nlargest(n_per_row, "pareidolia_rate")
    low  = img_full.nsmallest(n_per_row, "pareidolia_rate")

    fig, axes = plt.subplots(
        2, n_per_row, figsize=(style.COL_2, 3.4),
        gridspec_kw=dict(wspace=0.05, hspace=0.35),
    )
    for row, (label, tbl) in enumerate([
            ("Most pareidoligenic", high),
            ("Least pareidoligenic", low)]):
        for col, (_, r) in enumerate(tbl.iterrows()):
            ax = axes[row, col]
            try:
                im = PILImage.open(_resolve_image(r["image_url"])).convert("L")
                ax.imshow(im, cmap="gray")
            except FileNotFoundError:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ("top", "right", "left", "bottom"):
                ax.spines[s].set_edgecolor(style.COLORS[r["fd_level"]])
                ax.spines[s].set_linewidth(1.2)
            tw = _top_words(r["image_url"])
            ax.set_title(
                f"{r['fd_level']}  ·  rate={r['pareidolia_rate']:.2f}\n{tw}",
                fontsize=5.5, color="#222", pad=2,
            )
        # row label on the left
        axes[row, 0].set_ylabel(label, fontsize=8, weight="bold",
                                  labelpad=10)

    fig.suptitle("Per-image pareidolia rate, high vs low", y=1.02)
    if show: plt.show()
    style.savefig(fig, "image_features_gallery")
    plt.close(fig)


# ─── runner ─────────────────────────────────────────────────────────────────

def main(show: bool = True, rebuild: bool = False) -> None:
    img_stats = _per_image_stats()
    print(f"Per-image stats: {len(img_stats)} images")
    print(img_stats[["pareidolia_rate", "mean_n_words", "agreement",
                       "word_diversity"]].describe().round(3))

    feats = _extract_all(img_stats, rebuild=rebuild)
    feat_cols = [c for c in feats.columns if c != "image_url"]
    img_full = img_stats.merge(feats, on="image_url", how="inner")
    print(f"Joined: {len(img_full)} images × {len(feat_cols)} features")

    out_csv = config.OUTPUTS_DIR / "image_features_per_image.csv"
    img_full.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")

    r_mat = figure_heatmap(img_full, feat_cols, show=show)
    res = figure_importance(img_full, feat_cols, show=show)
    figure_gallery(img_full, show=show)
    r_mat.to_csv(config.OUTPUTS_DIR / "image_features_correlations.csv")

    print("\n=== Top features by |Pearson r| with pareidolia_rate ===")
    pr_col = r_mat["pareidolia_rate"]
    top = pr_col.reindex(pr_col.abs().sort_values(ascending=False).index).head(10)
    print(top.round(3).to_string())
    print("\n=== Top features by permutation importance ===")
    print(res["permutation"].tail(10).round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--rebuild", action="store_true",
                    help="ignore the on-disk feature cache")
    args = ap.parse_args()
    main(show=not args.no_show, rebuild=args.rebuild)
