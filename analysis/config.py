"""Central paths and constants for the pareidolia analysis pipeline."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Raw data ships in data/raw/. Override with PAREIDOLIA_DATA_DIR if needed.
DATA_DIR = Path(os.environ.get(
    "PAREIDOLIA_DATA_DIR",
    str(REPO_ROOT / "data" / "raw"),
))

SQLITE_PATH = DATA_DIR / "pareidolia_2025-oct.sqlite"
PARTICIPANTS_WORDS_CSV = DATA_DIR / "participants_words.csv"
TRIALS_CSV = DATA_DIR / "pareidolia_trials_2025.csv"

# Stimuli ship in stimuli/<FD>/<FD>_<i>.png.
STIMULI_DIR = REPO_ROOT / "stimuli"

# BERT embeddings of every pareidolia word (sentence-transformers
# all-MiniLM-L6-v2, 384-d). Shipped under data/ so analyses run without
# downloading anything; override with PAREIDOLIA_BERT_PARQUET to recompute.
BERT_EMBEDDINGS_PARQUET = Path(os.environ.get(
    "PAREIDOLIA_BERT_PARQUET",
    str(REPO_ROOT / "data" / "word_embeddings.parquet"),
))

# GloVe 840B 300d vectors for canonical DAT scoring (Olson et al. 2021).
# Either the original glove.840B.300d.txt or a derived .pkl mapping word->np.array.
GLOVE_PATH = Path(os.environ.get(
    "PAREIDOLIA_GLOVE_PATH",
    str(REPO_ROOT / "analysis_cache" / "glove.840B.300d.txt"),
))
GLOVE_CACHE_PKL = REPO_ROOT / "analysis_cache" / "glove_dat.pkl"

# Parquet caches shipped with the repo live in data/. Heavy/transient caches
# (GloVe, BERT embeddings) live in analysis_cache/.
DATA_CACHE_DIR = REPO_ROOT / "data"
CACHE_DIR = REPO_ROOT / "analysis_cache"
OUTPUTS_DIR = REPO_ROOT / "analysis_outputs"
CACHE_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

def cached_parquet(name: str) -> Path:
    """Resolve a parquet cache by name.

    Prefers ``data/<name>`` (shipped with the paper repo, read-only) when
    it exists; otherwise falls back to ``analysis_cache/<name>`` so that
    a fresh rebuild via ``parse_events`` lands in the writable cache.
    """
    shipped = DATA_CACHE_DIR / name
    if shipped.exists():
        return shipped
    return CACHE_DIR / name


# FD levels used in the experiment.
FD_LEVELS = ["FD12", "FD14", "FD16"]

# Realised fractal dimensions per nominal level, from the standard 2D
# box-counting method (Liebovitch & Toth 1989) over the 100 generated
# images per level. Reported in the manuscript as FD ≈ 1.3, 1.5, 1.7;
# see paper/draft.tex Materials and Methods (Stimuli).
FD_REALISED = {"FD12": 1.27, "FD14": 1.49, "FD16": 1.67}
FD_REALISED_SD = {"FD12": 0.012, "FD14": 0.016, "FD16": 0.011}
# Short display labels for FD axis ticks.
FD_DISPLAY_LABEL = {"FD12": "FD ≈ 1.3", "FD14": "FD ≈ 1.5", "FD16": "FD ≈ 1.7"}

# Stop-words to drop in semantic analyses (kept from notebook).
SEMANTIC_STOPWORDS = {
    "nothing", "nothingness", "black", "white", "map", "cloud", "clouds",
}

# DAT scoring uses 7 of 10 words (canonical).
DAT_N_WORDS = 10
DAT_N_USED = 7
