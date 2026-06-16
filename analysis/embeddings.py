"""Loader for the BERT word embeddings used in the semantic analyses.

The notebook treats `bert_outputs/word_embeddings.parquet` as the source of
truth: 384-d sentence-transformers vectors for every unique pareidolia word.
We expose a convenience function returning a {word: np.array} dict and a
helper to merge embeddings onto a long-format word DataFrame.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


@lru_cache(maxsize=1)
def load_word_embeddings(path: Path | str = config.BERT_EMBEDDINGS_PARQUET) -> pd.DataFrame:
    """Return DataFrame with columns ['word', 'embedding'] (embedding = np.ndarray)."""
    df = pd.read_parquet(path)
    # Ensure embeddings are numpy arrays (parquet sometimes returns lists).
    df["embedding"] = df["embedding"].apply(lambda v: np.asarray(v, dtype=np.float32))
    return df


@lru_cache(maxsize=1)
def embedding_dict() -> dict[str, np.ndarray]:
    df = load_word_embeddings()
    return dict(zip(df["word"].str.lower(), df["embedding"]))


def attach(words_df: pd.DataFrame, word_col: str = "word",
           drop_missing: bool = True) -> pd.DataFrame:
    """Left-join the embedding column onto a long-format word DataFrame."""
    emb = load_word_embeddings().rename(columns={"word": word_col})
    out = words_df.merge(emb, on=word_col, how="left")
    if drop_missing:
        out = out.dropna(subset=["embedding"]).reset_index(drop=True)
    return out


if __name__ == "__main__":
    df = load_word_embeddings()
    print(f"{len(df):,} unique words × {len(df.iloc[0]['embedding'])}d")
