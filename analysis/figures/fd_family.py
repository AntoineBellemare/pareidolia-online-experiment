"""FD × percept-family preference: humans match FD to a family's spatial scale.

Ported from ``MachinePareidolia/scripts/human_fd_family_figure.py`` and
re-implemented over the paper's analysis pipeline (cached parquets,
``style`` helpers, paper-ready figsizes, untitled + ``_titled`` reference
variants via ``style.savefig``).

For each fractal-dimension condition (FD12 / FD14 / FD16) we compute
the relative rate at which each percept *family* is reported, z-scored
across FD within each family. The qualitative pattern that emerges is
that observers match fractal dimension to a family's spatial structure:

  * low FD  (smooth, large blobs)  →  faces, body parts, persons,
                                       landscapes, objects
  * mid FD                         →  animals
  * high FD (fine, busy texture)   →  plants, mythical creatures, abstract
                                       textures

We also compute each family's continuous **preferred FD** (rate-weighted
mean FD), which orders the families along a global-form to fine-texture
axis, plus a per-percept preferred FD (mean FD of the images on which
each word was reported, restricted to words with ≥ 30 reports).

**Taxonomy.** The eight families here are a hand-curated, coarse-grained
view independent of the BERT-centroid classifier in
:mod:`analysis.figures.animal_vs_body` (which uses 15 finer buckets
plus an explicit hard-reject list to keep objects out of the human and
animal categories). The two taxonomies overlap but disagree on edge
cases (e.g. *kiss / dance* are person-actions here but rejected from
animal_vs_body's human bucket); the FD-family view privileges
*coverage* and a continuous spatial-scale interpretation, while
animal_vs_body privileges *purity* for the human-vs-animal contrast.

**Cohort.** Uses :func:`analysis.dat_helper.notebook_cohort` (English
sessions with feedback supplied), consistent with the preferred-categories
and image-consensus figures.

Outputs in ``analysis_outputs/``:

  * ``fig_fd_family_heatmap.{png,pdf}`` + ``_titled`` variants
       heatmap only (families × FD, z-scored)
  * ``fig_fd_family_paper.{png,pdf}`` + ``_titled`` variants
       paper-ready 2-panel composite (heatmap + per-percept strip)
  * ``fig_fd_family_word_strip.{png,pdf}`` + ``_titled`` variants
       per-percept preferred-FD strip plot only
  * ``fd_family_word_preferred_fd.csv``
       word, family, preferred_fd, n_reports
  * ``fd_family_matrix.csv``
       families × FD raw rate + z-score matrices

Usage:
    python -m analysis.figures.fd_family
"""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec

from .. import config, dat_helper, parse_events, style


ASCII = re.compile(r"^[A-Za-z]+$")
REMOVED = config.SEMANTIC_STOPWORDS

FD_ORDER = list(config.FD_LEVELS)             # ["FD12", "FD14", "FD16"]

# Realised fractal dimensions and display labels: defined centrally in
# config so all FD-related figures stay in sync if the calibration ever
# changes. Using the measured values (rather than the nominal generation
# targets 1.2 / 1.4 / 1.6) keeps the rate-weighted "preferred FD"
# comparable to the rest of the manuscript, which reports FD ≈ 1.3,
# 1.5, 1.7.
FD_VALUE = config.FD_REALISED
FD_LABEL = config.FD_DISPLAY_LABEL


# ─── percept-family taxonomy ────────────────────────────────────────────────
# Hand-curated families covering the most frequent percept vocabulary in the
# corpus. Words not in any family fall into "other" and do not contribute to
# the heatmap rates; coverage is reported in main() so the lists can be
# extended if the residual is large.

FAMILIES: dict[str, set[str]] = {
    "face / body part": {
        "face", "faces", "head", "heads", "nose", "skull", "mouth", "eyes",
        "eye", "hair", "profile", "foot", "feet", "lips", "arm", "arms",
        "hand", "hands", "finger", "fingers", "fist", "penis", "fetus",
        "ear", "ears", "tongue", "teeth", "tooth", "claw", "claws", "tail",
        "bone", "brain", "leg", "legs", "neck", "chin", "beard", "breast",
        "belly", "jaw", "cheek", "butt", "skeleton", "smile", "smiling",
        "laughing", "scream", "frown", "body", "blood", "stomach", "kidney",
        "uterus", "bust",
    },
    "person / people": {
        "man", "men", "woman", "women", "person", "people", "baby", "lady",
        "girl", "boy", "child", "children", "human", "dancer", "dancing",
        "dance", "angel", "guy", "kid", "soldier", "king", "queen", "clown",
        "chef", "figure", "silhouette", "couple", "kiss", "kissing",
        "mother", "father", "cartoon", "puppet", "ninja", "pirate",
        "superhero", "cowboy",
    },
    "animal": {
        "dog", "dogs", "bird", "birds", "fish", "cat", "cats", "bear",
        "horse", "elephant", "rabbit", "pig", "dinosaur", "dino", "chicken",
        "seahorse", "lion", "snake", "duck", "cow", "bat", "turtle",
        "mouse", "frog", "wolf", "butterfly", "monkey", "bunny", "whale",
        "crab", "goat", "camel", "rat", "rhino", "sheep", "gorilla",
        "squirrel", "hippo", "alligator", "shark", "deer", "fox", "penguin",
        "owl", "eagle", "dolphin", "octopus", "spider", "worm", "snail",
        "lizard", "crocodile", "giraffe", "zebra", "tiger", "animal",
        "animals", "insect", "bug", "poodle", "puppy", "caterpillar",
        "bull", "boar", "rooster", "seal", "peacock", "parrot", "hen",
        "donkey", "llama", "hedgehog", "goose", "swan", "moth", "bee",
        "jellyfish", "toad", "panda", "koala", "otter", "kangaroo",
        "dragonfly", "starfish",
    },
    "mythical creature": {
        "dragon", "ghost", "ghosts", "monster", "monsters", "alien",
        "aliens", "demon", "creature", "zombie", "troll", "devil", "fairy",
        "gargoyle", "gremlin", "goblin", "ogre", "elf", "vampire", "reaper",
        "spirit", "genie", "mermaid", "unicorn", "yeti", "grinch", "witch",
        "wizard", "imp",
    },
    "plant": {
        "tree", "trees", "forest", "mushroom", "mushrooms", "flower",
        "flowers", "leaf", "leaves", "plant", "plants", "branch", "bush",
        "grass", "vine", "fern", "cactus", "seaweed", "palm",
    },
    "place / landscape": {
        "island", "islands", "ocean", "oceans", "continent", "continents",
        "land", "water", "lake", "lakes", "river", "rivers", "italy",
        "australia", "country", "countries", "africa", "europe", "mountain",
        "mountains", "sea", "cave", "caves", "sky", "coastline", "coast",
        "cliff", "world", "earth", "space", "spain", "russia", "beach",
        "greenland", "asia", "america", "china", "france", "england",
        "india", "turkey", "isthmus", "peninsula", "archipelago", "bay",
        "valley", "desert", "volcano", "waterfall", "pond", "shore",
        "swamp", "lagoon", "hill", "canyon", "glacier",
    },
    "object / symbol": {
        "boot", "boots", "hat", "heart", "cross", "gun", "dress", "ship",
        "sword", "key", "plus", "mask", "shoe", "bottle", "cup", "knife",
        "axe", "hammer", "chair", "bell", "crown", "arrow", "star",
        "letter", "sign", "clock", "flag", "anchor", "boat", "car", "vase",
        "lamp", "pipe", "apple", "pacman", "rocket", "umbrella", "guitar",
        "teapot", "spoon", "book", "phone", "house", "building", "castle",
        "tower", "bridge", "tent",
    },
    "abstract / texture": {
        "ink", "paint", "smoke", "blob", "blobs", "splatter", "blot",
        "stain", "spill", "fire", "explosion", "tornado", "chaos", "mess",
        "pattern", "scribble", "shape", "dots", "outline", "noise", "swirl",
        "fog", "smear",
    },
}


# ─── data ───────────────────────────────────────────────────────────────────

def _word2family() -> dict[str, str]:
    return {w: f for f, ws in FAMILIES.items() for w in ws}


def _load_trial_words() -> pd.DataFrame:
    """Return per-trial valid percept words, restricted to the main cohort
    and the same ASCII single-token filter + stopword list used elsewhere.

    One row per trial; the ``words`` column is the cleaned list of strings.
    """
    trials = parse_events.cached_trials()
    nb = dat_helper.notebook_cohort()
    trials = trials[trials["user_id"].isin(nb["user_id"])].copy()

    def _clean(ws) -> list[str]:
        if ws is None:
            return []
        out = []
        for w in list(ws):
            w = str(w).strip().lower()
            if ASCII.fullmatch(w) and w not in REMOVED:
                out.append(w)
        return out

    trials["words"] = trials["words"].apply(_clean)
    return trials[["user_id", "fd_level", "url_stimulus", "words"]]


# ─── analysis ───────────────────────────────────────────────────────────────

def family_fd_matrix(trials: pd.DataFrame
                      ) -> tuple[list[str], np.ndarray, np.ndarray, float]:
    """Compute (families, raw-rate matrix M, z-scored matrix Z, coverage).

    M[i, j] = fraction of all valid percept tokens at FD j that belong to
    family i. Z is M z-scored along axis=1 (across FD) per family, so
    Z[i, j] > 0 means family i over-reports at FD j relative to its own
    cross-FD average.
    """
    w2f = _word2family()
    fams = list(FAMILIES)

    M = np.zeros((len(fams), len(FD_ORDER)))
    overall: Counter[str] = Counter()
    for j, fd in enumerate(FD_ORDER):
        cnt: Counter[str] = Counter()
        for wl in trials.loc[trials["fd_level"] == fd, "words"]:
            cnt.update(wl)
        overall.update(cnt)
        tot = sum(cnt.values())
        if tot == 0:
            continue
        for i, f in enumerate(fams):
            M[i, j] = sum(cnt[w] for w in FAMILIES[f]) / tot

    Z = (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)
    coverage = (sum(n for w, n in overall.items() if w in w2f)
                / max(sum(overall.values()), 1))
    return fams, M, Z, coverage


def preferred_fd(fams: list[str], M: np.ndarray) -> dict[str, float]:
    """Rate-weighted mean FD per family (continuous "preferred fractal dimension")."""
    fdv = np.array([FD_VALUE[f] for f in FD_ORDER])
    out: dict[str, float] = {}
    for i, f in enumerate(fams):
        s = M[i].sum()
        out[f] = float((M[i] * fdv).sum() / s) if s > 0 else float("nan")
    return out


def word_preferred_fd(trials: pd.DataFrame, min_reports: int = 30
                       ) -> pd.DataFrame:
    """Per-percept preferred FD = mean FD of the images where it was reported.

    Restricted to words that (i) appear at least ``min_reports`` times and
    (ii) belong to one of the families above.
    """
    w2f = _word2family()
    fd_sum: dict[str, float] = defaultdict(float)
    n_of: Counter[str] = Counter()
    for fd, ws in zip(trials["fd_level"], trials["words"]):
        v = FD_VALUE.get(fd)
        if v is None:
            continue
        for w in ws:
            fd_sum[w] += v
            n_of[w] += 1
    rows = [
        {"word": w, "family": w2f[w],
         "preferred_fd": fd_sum[w] / n_of[w], "n_reports": n_of[w]}
        for w in n_of
        if n_of[w] >= min_reports and w in w2f
    ]
    return pd.DataFrame(rows).sort_values("preferred_fd").reset_index(drop=True)


# ─── plotting ───────────────────────────────────────────────────────────────

def _heatmap(ax, Z: np.ndarray, fams_in_order: list[str]) -> None:
    """Render the families × FD z-score heatmap onto ``ax``."""
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(len(FD_ORDER)))
    ax.set_xticklabels([FD_LABEL[f] for f in FD_ORDER])
    ax.set_yticks(range(len(fams_in_order)))
    ax.set_yticklabels(fams_in_order)
    for i in range(len(fams_in_order)):
        for j in range(len(FD_ORDER)):
            ax.text(j, i, f"{Z[i, j]:+.1f}", ha="center", va="center",
                    fontsize=6, color="black")
    ax.tick_params(length=0)
    return im


def _strip(ax, wdf: pd.DataFrame, family_order: list[str]) -> None:
    """Render the per-percept preferred-FD strip plot onto ``ax``."""
    rng = np.random.default_rng(0)
    for i, fam in enumerate(family_order):
        sub = wdf[wdf["family"] == fam]
        ax.scatter(i + (rng.random(len(sub)) - 0.5) * 0.34,
                    sub["preferred_fd"], s=12, alpha=0.6,
                    color=style.COLORS["dot"], edgecolors="none")
        m = float(sub["preferred_fd"].mean())
        ax.plot([i - 0.30, i + 0.30], [m, m], color="black", lw=1.4)
    ax.set_xticks(range(len(family_order)))
    ax.set_xticklabels(family_order, rotation=30, ha="right")
    ax.margins(x=0.04)


def figure_heatmap(fams: list[str], Z: np.ndarray,
                    family_order: list[str], show: bool) -> None:
    """Standalone heatmap (families × FD, z-scored)."""
    style.apply()
    idx = [fams.index(f) for f in family_order]
    Zo = Z[idx]

    fig, ax = plt.subplots(figsize=(style.COL_1_5, 3.4))
    im = _heatmap(ax, Zo, family_order)
    cb = plt.colorbar(im, ax=ax, fraction=0.042, pad=0.04)
    cb.set_label("Relative report rate (z)")
    style.style_axis(ax, title="", xlabel="", ylabel="")
    fig.suptitle("Reported percept family by fractal dimension",
                 y=1.02)
    if show: plt.show()
    style.savefig(fig, "fd_family_heatmap")
    plt.close(fig)


def figure_word_strip(wdf: pd.DataFrame, family_order: list[str],
                       show: bool) -> None:
    """Standalone per-percept preferred-FD strip plot."""
    style.apply()
    fig, ax = plt.subplots(figsize=(style.COL_1_5, 3.0))
    _strip(ax, wdf, family_order)
    style.style_axis(ax, title="", xlabel="", ylabel="Preferred FD")
    fig.suptitle("Per-percept preferred fractal dimension, by family",
                 y=1.02)
    if show: plt.show()
    style.savefig(fig, "fd_family_word_strip")
    plt.close(fig)


def figure_paper(fams: list[str], Z: np.ndarray, wdf: pd.DataFrame,
                  family_order: list[str], show: bool) -> None:
    """Paper-ready 2-panel composite: (a) heatmap + (b) per-percept strip,
    both ordered smooth-to-rough by the rate-weighted preferred FD.
    """
    style.apply()
    idx = [fams.index(f) for f in family_order]
    Zo = Z[idx]

    fig = plt.figure(figsize=(style.COL_2, 3.6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.35], wspace=0.50)

    ax_a = fig.add_subplot(gs[0])
    im = _heatmap(ax_a, Zo, family_order)
    cb = plt.colorbar(im, ax=ax_a, fraction=0.042, pad=0.04)
    cb.set_label("Report rate (z)")
    style.style_axis(ax_a, title="(a) Reported family × FD",
                      xlabel="", ylabel="")

    ax_b = fig.add_subplot(gs[1])
    _strip(ax_b, wdf, family_order)
    style.style_axis(ax_b, title="(b) Per-percept preferred FD",
                      xlabel="", ylabel="Preferred FD")

    fig.suptitle("Humans match fractal dimension to percept-family scale",
                  y=1.04)
    if show: plt.show()
    style.savefig(fig, "fd_family_paper")
    plt.close(fig)


# ─── runner ────────────────────────────────────────────────────────────────

def main(show: bool = True, min_reports: int = 30) -> None:
    trials = _load_trial_words()
    print(f"Trials in cohort: {len(trials):,} "
          f"({trials['user_id'].nunique()} participants, "
          f"{trials['url_stimulus'].nunique()} images)")

    fams, M, Z, coverage = family_fd_matrix(trials)
    print(f"Family coverage: {coverage:.1%} of all reported word tokens\n")

    pref = preferred_fd(fams, M)
    family_order = sorted(fams, key=lambda f: pref[f])
    print("Preferred FD per family (rate-weighted, smooth → rough):")
    for f in family_order:
        print(f"  {pref[f]:.3f}  {f}")

    print("\nZ-scored per-family report rate across FD "
          "(the heatmap values, ordered smooth → rough):")
    Zt = pd.DataFrame(Z.round(2), index=fams,
                      columns=[FD_LABEL[f] for f in FD_ORDER])
    print(Zt.loc[family_order].to_string())

    wdf = word_preferred_fd(trials, min_reports=min_reports)
    print(f"\nPer-percept preferred FD ({len(wdf)} percepts, "
          f"≥{min_reports} reports): "
          f"range [{wdf['preferred_fd'].min():.2f}, "
          f"{wdf['preferred_fd'].max():.2f}]")

    # Save tables.
    raw = pd.DataFrame(M, index=fams,
                       columns=[f"raw_rate_{f}" for f in FD_ORDER])
    zsc = pd.DataFrame(Z, index=fams,
                       columns=[f"z_{f}" for f in FD_ORDER])
    raw.join(zsc).assign(preferred_fd=lambda d: d.index.map(pref)) \
       .to_csv(config.OUTPUTS_DIR / "fd_family_matrix.csv")
    wdf.to_csv(config.OUTPUTS_DIR / "fd_family_word_preferred_fd.csv",
                index=False)
    print(f"Wrote: {config.OUTPUTS_DIR / 'fd_family_matrix.csv'}")
    print(f"Wrote: {config.OUTPUTS_DIR / 'fd_family_word_preferred_fd.csv'}")

    figure_heatmap(fams, Z, family_order, show=show)
    figure_word_strip(wdf, family_order, show=show)
    figure_paper(fams, Z, wdf, family_order, show=show)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--min-reports", type=int, default=30,
                    help="word-level: minimum reports for a percept to "
                         "appear in the strip plot")
    args = ap.parse_args()
    main(show=not args.no_show, min_reports=args.min_reports)
