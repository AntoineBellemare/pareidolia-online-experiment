# `analysis/` — clean pipeline for the Pareidolia experiment

Replaces the ad-hoc notebook (`data_export/analyze_pareidolia_offline.ipynb`)
with a small, deterministic Python package. Every figure is a CLI script that
re-runs end-to-end from the raw SQLite dump and writes both a PNG and a CSV
into `analysis_outputs/`.

## Layout

```
analysis/
  config.py           paths, FD levels, stop-words, DAT constants
  loader.py           SQLite -> deduplicated sessions DataFrame
  parse_events.py     data_json blob -> trials / participants / DAT-after
  embeddings.py       cached BERT word embeddings
  dat.py              canonical Olson-et-al. DAT scoring (GloVe 840B)
  eyetracking.py      per-trial gaze metrics (I-DT fixations, etc.)
  figures/
    fig_diversity_vs_dat.py        Fig 1 — perceptual diversity vs DAT
    fig_semantic_territory.py      Fig 2 — t-SNE map by creativity tertile
    dat_before_vs_after.py         post-task DAT comparison
    fd_effect.py                   stimulus FD vs RT / words / spread
    creativity_effects.py          DAT vs behaviour (beyond diversity)
    eyetracking_by_creativity.py   gaze metrics × creativity tertile
```

## Where the data is expected

By default the loader reads from
`C:\Users\skite\Documents\Github\pareidolia-website\data_export\` (parent
checkout). Override via env vars:

```
PAREIDOLIA_DATA_DIR        path to the data_export/ folder (sqlite, parquet)
PAREIDOLIA_BERT_PARQUET    path to word_embeddings.parquet
PAREIDOLIA_GLOVE_PATH      path to glove.840B.300d.txt (or cached .pkl)
```

## First-time setup

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn pyarrow adjustText

# Score DAT canonically (Olson et al. 2021) — needs GloVe 840B 300d (~2 GB
# download, ~5 GB extracted). One-time:
python -m analysis.dat download
```

If GloVe is unavailable, the DAT module silently falls back to MiniLM —
useful for prototyping but **not comparable** to the published DAT scale.
A `RuntimeWarning` is emitted whenever this happens.

## Common entry-points

```bash
# Run everything (will prompt before downloading GloVe if missing):
python -m analysis.run_all
python -m analysis.run_all --skip-glove --only fd_effect creativity_effects
python -m analysis.run_all --rebuild       # wipe analysis_cache/*.parquet first

# Reproduce the notebook figures (Nature-style, PNG + PDF)
python -m analysis.figures.fig_diversity_vs_dat
python -m analysis.figures.fig_diversity_vs_dat_by_fd
python -m analysis.figures.fig_semantic_territory

# Stimulus-level effects
python -m analysis.figures.fd_effect                   # words / RT / spread
python -m analysis.figures.fd_effect_eyetracking       # gaze dynamics

# Creativity effects
python -m analysis.figures.creativity_effects
python -m analysis.figures.eyetracking_vs_dat_continuous   # continuous DAT, per-FD heatmap
python -m analysis.figures.eyetracking_by_creativity        # tertile view

# DAT-after (needs GloVe)
python -m analysis.figures.dat_before_vs_after
python -m analysis.figures.pareidolia_boosts_dat            # engagement → Δ DAT

# Semantic deep dive
python -m analysis.figures.semantic_exploration             # centroids, distinctive words, categories, JSD
python -m analysis.figures.semantic_exploration --only distinctive_tertile distinctive_fd

# Diagnostics
python -m analysis.figures.gaze_geometry_check              # sanity-check ET geometry
```

Add `--no-show` to suppress `plt.show()` for headless runs.
Every figure script writes a PDF alongside the PNG into `analysis_outputs/`.

## Figure style

All figures use `analysis/style.py` — Nature single (89 mm) / 1.5 / double
(183 mm) column widths, Arial 7 pt body / 6 pt ticks, top + right spines
hidden, 300 dpi PNG + PDF export.

## Caching

Parsed tables are cached in `analysis_cache/`:

* `participants.parquet`             — one row per participant
* `trials.parquet`                   — one row per pareidolia trial
* `gaze_metrics.parquet`             — one row per trial with gaze
* `participants_with_dat_after.parquet` — main cohort + computed DAT-after
* `glove_dat.pkl`                    — first-load pickle of GloVe vectors

Delete a file to force its rebuild, or call
`analysis.parse_events.rebuild_cache()`.

## Cohorts

| name                                       | how                                                       | n     |
| ------------------------------------------ | --------------------------------------------------------- | ----- |
| `loader.load_sessions()`                   | English, deduplicated, DAT-before present                 | 3,954 |
| `dat_helper.notebook_cohort()` (default)   | English **with feedback** — matches the notebook variable | 579   |
| `parse_events.main_cohort()` / `…scored()` | `dat_after_words` present **and** ≥ 5 trials              | 508   |

Diversity figures default to the **notebook cohort** (so r/p match the
notebook exactly). Anything that depends on DAT-after silently switches to
the main cohort. Pass `--cohort main` to force the DAT-after cohort
anywhere.

## Key numbers reproduced (GloVe + ASCII-letter filter)

| analysis                                              | DAT-before                                | DAT-after                                |
| ----------------------------------------------------- | ----------------------------------------- | ---------------------------------------- |
| **Diversity (pooled, pairwise median)**               | **r = 0.144, p = 0.001, n = 500**         | r = 0.028, p = 0.53, n = 495             |
| **Diversity FD12**                                    | r = 0.094, p = 0.037                      | r = −0.01, p = 0.80                      |
| **Diversity FD14**                                    | **r = 0.167, p = 0.00027**                | r = 0.052, p = 0.27                      |
| **Diversity FD16**                                    | r = −0.004, p = 0.93                      | r = 0.017, p = 0.72                      |
| **Pre vs post DAT correlation**                       | —                                         | **r = 0.193, p = 1e-5, n = 507**         |
| **Mean Δ (post − pre)**                               | —                                         | −0.02 (Wilcoxon p = 0.99 — no shift)     |
| FD effect on words / trial                            | Friedman p < 1e-10 — monotonic ↓ with FD  |                                          |
| FD effect on descriptions / trial                     | Friedman p < 1e-9 — monotonic ↓ with FD   |                                          |
| FD effect on semantic spread (within-subject)         | Friedman p = 0.02 — slight ↓ with FD      |                                          |
| Creativity → mean word rarity                         | r = 0.09, p = 0.03                        | r ≈ 0,  NS                               |
| Creativity → eye tracking (after rejecting bad trials) | KW p > 0.92 for every metric (n=226)     | KW p > 0.74 for every metric             |

### Headline reading

* **Diversity ↔ creativity lives at FD14**: r=0.17, p=0.0003 at mid-complexity,
  r≈0 at FD12 and FD16. The link is only present with the pre-task DAT.
* **Distinctive vocabulary by creativity tertile** (`fig_distinctive_words_dat_tertile`):
  - Low: texture words — *cold, movement, abstract, tired, smears, splash, color*
  - High: unusual specifics — *anglerfish, chimpanzee, drill, trumpet, tapir, dodo, tophat*
* **Distinctive vocabulary by FD** (`fig_distinctive_words_fd_level`):
  - FD12 (simplest): discrete figures — *vase, handshake, elbow, horn, ape, cave*
  - FD14: animals — *ox, bunnies, warthog, salamander, llama, ducks, mice*
  - FD16 (most complex): textural, not figural — *canopy, stars, fuzzy, marble, rain, scatter, noise, swamp*
* **FD shapes gaze dynamics**: scanpath gets shorter (Friedman p = 0.006,
  FD12→FD16 Wilcoxon p = 0.0007) and fixations longer at higher FD — people
  look more carefully when stimuli are noisier.
* **One reliable creativity × gaze effect**: at FD16, more creative people make
  smaller saccades (r = −0.16, p = 0.029) — possibly searching more carefully
  in the most ambiguous condition.
* **Pareidolia engagement ≈ small DAT boost** (n = 507): mean_n_words → Δ DAT
  Spearman ρ = +0.115, p = 0.010; total_unique_words ρ = +0.099, p = 0.026.
  Diversity and word rarity have no boost effect.
* **Eye tracking otherwise null**: 9.4 % of trials rejected for WebGazer
  diagonal-line failure; all global gaze metrics ≈ 0 vs DAT after the fix.

## Notes & caveats

* **Multi-word phrases in the embedding parquet.** The BERT word_embeddings
  parquet contains 677 phrases ("ocean straight", "baby hand reach", …) that
  were embedded as sentences. Sentence vs single-word vectors live in
  different parts of the space and inflate within-subject spread, diluting
  the DAT correlation (~0.17 → ~0.11 if you don't filter). Both diversity
  scripts and the t-SNE figure drop them via `^[A-Za-z]+$`.
* **Eye tracking geometry.** Validation events are empty;
  `gaze_target.width/height` are 0; per-session screen size varies. We infer
  the screen extent from each session's calibration sweep (1st–99th
  percentile) and normalise gaze to [0, 1]. The stimulus ROI is a
  per-session ±0.275 × (±0.275 × W/H) box centred on `gaze_target.x/y`,
  based on the `max-width: 55 %` CSS rule in
  [`pareidolia_website/experiment.html:563`](../pareidolia_website/experiment.html#L563).
* **WebGazer failed-tracking rejection.** Some trials show a near-perfect
  diagonal line of "gaze" predictions (face-mesh tracker lost the eyes).
  Trials with `|corr(x, y)| > 0.97` *and* both spans > 0.6 are rejected
  (`tracking_failed = True`). 530 / 5,660 trials (9.4 %) flagged.
  See [`fig_gaze_geometry_check.py`](figures/gaze_geometry_check.py) and
  [`fig_gaze_tertile_zoom.py`](figures/gaze_tertile_zoom.py).
* **DAT-after** sample sizes will look tiny until GloVe is downloaded — the
  MiniLM fallback only knows the pareidolia vocabulary, so most DAT words
  are missing. With GloVe, n_after ≈ 507.
