# Image statistics and trait creativity jointly shape visual pareidolia

Code and data accompanying the manuscript *Image statistics and trait
creativity jointly shape visual pareidolia* (Bellemare-Pepin et al., in
preparation). Approximately 580 online participants viewed 30
fractal-noise patches at three controlled fractal-dimension levels
(FD ≈ 1.3, 1.5, 1.7) and freely typed every percept they saw, before
and after completing the Divergent Association Task (DAT) and, on
opt-in, with webcam eye-tracking.

## Repository layout

```
analysis/         Python package (data pipeline + figure scripts)
  figures/          one script per figure
                    - 11 paper-final scripts (Figs 1-11)
                    - exploratory scripts referenced as supplementary
paper/            Manuscript build pipeline
  build_preview_pdf.py    HTML/PDF preview via WeasyPrint
  build_word.js           Word .docx via docx-js
  sync_figures.py         Copy analysis_outputs/ → paper/figures/
  references.bib          BibTeX references
  draft.{tex,docx}        Manuscripts
  figures/                Final manuscript figures (PNG + PDF)
stimuli/          300 fractal-noise images (100 per FD level)
data/             Parquet caches needed to rebuild every figure
                    - trials, participants_*, gaze_metrics
                    - word_embeddings (MiniLM 384-d, ~9 MB)
                    - image_features
                  raw/  source SQLite + per-trial CSV exports
outputs/          Summary tables referenced in the paper
```

## Installation

```bash
# Python (3.10+)
pip install -r requirements.txt

# Node.js (for the Word manuscript builder)
npm install
```

## Rebuilding the figures

The Python package reads the parquet caches in `data/` directly — no
SQLite database needed. Each script writes a paper-ready PNG (untitled,
for the manuscript) and a `_titled` reference variant.

```bash
# A single figure
python -m analysis.figures.fd_effect --no-show

# Every paper figure, end-to-end
python -m analysis.run_all
```

GloVe 840B (~2 GB) is required to score the DAT. The first time you
run a DAT-related script, `analysis/dat.py` downloads it into
`analysis_cache/` automatically (set `PAREIDOLIA_GLOVE_PATH` to point at
an existing copy).

## Rebuilding the manuscript

```bash
python paper/build_preview_pdf.py   # → paper/draft_preview.{html,pdf}
node   paper/build_word.js          # → paper/draft.docx
python paper/sync_figures.py        # refresh paper/figures/ from analysis_outputs/
```

## Data provenance

Raw data lives in `data/raw/` (`pareidolia_2025-oct.sqlite`,
`pareidolia_trials_2025.csv`, `participants_words.csv`); see
`data/RAW.md` for the schema and the regeneration steps. The parquet
caches in `data/` are the output of `analysis.parse_events` and contain
no PII beyond what's in the public manuscript.

## Citation

If you use this code or data, please cite:

> Bellemare-Pepin et al. (in preparation). Image statistics and trait
> creativity jointly shape visual pareidolia.

And the foundational divergent-perception papers:

> Bellemare-Pepin, A., Harel, Y., O'Byrne, J., Mageau, G., Dietrich, A.,
> & Jerbi, K. (2022). Processing visual ambiguity in fractal patterns:
> Pareidolia as a sign of creativity. *iScience*, 25(10), 105103.
>
> Bellemare-Pepin, A., & Jerbi, K. (2024). Divergent perception:
> framing creative cognition through the lens of sensory flexibility.
> *Journal of Creative Behavior*.
