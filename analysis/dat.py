"""Canonical DAT scoring (Olson, Nahas, Chmoulevitch, Cropper, Webb 2021).

Reference: https://www.datcreativity.com/  /  https://github.com/jayolson/divergent-association-task

Algorithm:
  1. Lowercase each of the 10 words, strip non-letters, drop dupes.
  2. Lookup GloVe 840B 300d vectors.
  3. Take the first `n_used = 7` valid words (in submission order).
  4. Compute cosine distance between every pair  -> 21 pairwise distances.
  5. DAT score = mean(distances) * 100.

We provide:
  * `DatModel`   : wraps the vector lookup (lazy on-disk pickle cache).
  * `score`      : score a single list of words → float (or None if <7 valid).
  * `score_many` : vectorised over a Series/DataFrame.
  * `download_glove` : helper that fetches glove.840B.300d.zip.

If GloVe is unavailable we fall back to the sentence-transformers MiniLM
vectors (which are already present for the pareidolia vocabulary) — useful
for prototyping but NOT directly comparable to the published DAT scale.
"""
from __future__ import annotations

import io
import pickle
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import config

GLOVE_URL = "https://nlp.stanford.edu/data/glove.840B.300d.zip"  # ~2.2 GB

WORD_RE = re.compile(r"[A-Za-z]+")


def _clean(word: str) -> str | None:
    if not isinstance(word, str):
        return None
    # The DAT manual says: lowercase, strip, take first word, only letters.
    word = word.strip().lower()
    m = WORD_RE.match(word)
    return m.group(0) if m else None


# ─── vector store ─────────────────────────────────────────────────────────────

@dataclass
class DatModel:
    """Word → vector lookup. Vectors are L2-normalised."""
    vectors: dict[str, np.ndarray]
    source: str  # "glove" or "minilm"

    def get(self, word: str) -> np.ndarray | None:
        return self.vectors.get(word)

    @classmethod
    def from_glove_txt(cls, txt_path: Path | str) -> "DatModel":
        txt_path = Path(txt_path)
        vecs: dict[str, np.ndarray] = {}
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) != 301:
                    continue
                w = parts[0]
                if not WORD_RE.fullmatch(w):
                    continue
                v = np.asarray(parts[1:], dtype=np.float32)
                n = np.linalg.norm(v)
                if n == 0:
                    continue
                vecs[w.lower()] = v / n
        return cls(vectors=vecs, source="glove")

    @classmethod
    def from_pickle(cls, pkl_path: Path | str) -> "DatModel":
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            return cls(vectors=obj, source="glove")
        return obj  # already a DatModel

    def to_pickle(self, pkl_path: Path | str) -> None:
        with open(pkl_path, "wb") as f:
            pickle.dump(self.vectors, f, protocol=pickle.HIGHEST_PROTOCOL)


@lru_cache(maxsize=1)
def load_model() -> DatModel:
    """Resolve vectors in this order:
        1. cached pickle  (analysis_cache/glove_dat.pkl)
        2. raw GloVe txt  (config.GLOVE_PATH)
        3. MiniLM fallback (analysis.embeddings — pareidolia-only vocab)
    """
    if config.GLOVE_CACHE_PKL.exists():
        return DatModel.from_pickle(config.GLOVE_CACHE_PKL)

    if config.GLOVE_PATH.exists():
        m = DatModel.from_glove_txt(config.GLOVE_PATH)
        m.to_pickle(config.GLOVE_CACHE_PKL)
        return m

    # Fallback — pareidolia BERT vocab only. Warn loudly.
    import warnings
    from . import embeddings
    warnings.warn(
        f"GloVe vectors not found at {config.GLOVE_PATH}. Falling back to MiniLM "
        f"(pareidolia vocab only — most DAT words will be missing). "
        f"Run `python -m analysis.dat download` to fetch GloVe.",
        RuntimeWarning,
    )
    edict = embeddings.embedding_dict()
    normed = {w: v / (np.linalg.norm(v) or 1) for w, v in edict.items()}
    return DatModel(vectors=normed, source="minilm")


def download_glove(target: Path | str = config.GLOVE_PATH) -> None:
    """Download glove.840B.300d.zip and extract the .txt next to `target`.

    ~2.2 GB download, ~5 GB extracted. Idempotent: skipped if target exists.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"Already present: {target}")
        return
    zip_path = target.parent / "glove.840B.300d.zip"
    if not zip_path.exists():
        print(f"Downloading {GLOVE_URL} → {zip_path} (~2.2 GB)…")
        urllib.request.urlretrieve(GLOVE_URL, zip_path)
    print(f"Extracting → {target} (~5 GB)…")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target.parent)
    print("Done.")


# ─── scoring ───────────────────────────────────────────────────────────────────

def score(words: Iterable[str | None],
          model: DatModel | None = None,
          n_used: int = config.DAT_N_USED,
          return_words: bool = False):
    """Compute the DAT score for a list of words.

    Returns the score in [0, 100] (mean cosine distance × 100) or None if
    fewer than `n_used` valid embeddings are found. With return_words=True
    additionally returns the list of words that were actually scored.
    """
    model = model or load_model()
    used: list[str] = []
    used_vecs: list[np.ndarray] = []
    seen: set[str] = set()
    for w in words:
        cw = _clean(w or "")
        if not cw or cw in seen:
            continue
        v = model.get(cw)
        if v is None:
            continue
        seen.add(cw)
        used.append(cw)
        used_vecs.append(v)
        if len(used) == n_used:
            break

    if len(used) < n_used:
        return (None, used) if return_words else None

    V = np.stack(used_vecs)
    sim = V @ V.T          # vectors are L2-normalised → cosine similarity
    iu = np.triu_indices_from(sim, k=1)
    dist = 1.0 - sim[iu]
    s = float(dist.mean() * 100.0)
    return (s, used) if return_words else s


def score_many(word_lists: pd.Series, **kw) -> pd.Series:
    model = kw.pop("model", None) or load_model()
    return word_lists.apply(lambda ws: score(ws or [], model=model, **kw))


# ─── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "download":
        download_glove()
        sys.exit(0)

    m = load_model()
    print(f"Model source: {m.source}  ({len(m.vectors):,} words)")
    # Quick sanity check using the example from the DAT site.
    demo = ["arm", "eels", "money", "looking", "ear", "tarp", "rosebush"]
    s = score(demo, m)
    print(f"Demo score for {demo!r}: {s}")
