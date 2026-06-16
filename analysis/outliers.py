"""Shared outlier-removal utilities.

Convention used across all DAT-correlation figures:
    * **Bivariate ±3 SD** trim — drop a row if EITHER the metric OR the DAT
      score is more than 3 standard deviations from its mean.
    * Standardisation uses the **unbiased** sample SD (pandas default).
    * Always applied AFTER dropna on both columns.
    * The helper returns (kept_df, n_dropped) so the caller can log it.

For per-condition analyses (e.g. per-FD spread, per-tertile cohesion) the
helper can be called per-group via groupby().apply(...).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def trim_sd(df: pd.DataFrame, cols: str | Iterable[str], sd: float = 3.0,
            verbose: bool = False) -> tuple[pd.DataFrame, int]:
    """Drop rows whose absolute z-score on ANY of `cols` exceeds `sd`.

    Returns the trimmed dataframe and the number of rows dropped.
    """
    if isinstance(cols, str):
        cols = [cols]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df, 0
    sub = df.dropna(subset=cols).copy()
    mask = np.ones(len(sub), dtype=bool)
    for c in cols:
        x = sub[c].astype(float)
        if x.std(ddof=1) == 0 or len(x) < 5:
            continue
        z = np.abs((x - x.mean()) / x.std(ddof=1))
        mask &= (z <= sd)
    out = sub[mask].copy()
    dropped = len(sub) - len(out)
    if verbose and dropped:
        print(f"  trim_sd({cols}, sd={sd}): dropped {dropped} / {len(sub)} rows")
    return out, dropped


def per_group_trim(df: pd.DataFrame, group_col: str, value_col: str,
                   sd: float = 3.0) -> pd.DataFrame:
    """Per-group ±sd trim on a single value column.

    Useful for per-FD or per-tertile analyses where outliers in one group
    shouldn't drag the threshold in another.
    """
    out = []
    for _, sub in df.groupby(group_col):
        kept, _ = trim_sd(sub, value_col, sd=sd)
        out.append(kept)
    return pd.concat(out, ignore_index=True) if out else df
