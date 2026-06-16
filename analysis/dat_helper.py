"""Shared helper: load the main cohort with both DAT-before and DAT-after
GloVe scores already attached. Cached so the GloVe model is loaded once."""
from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd

from . import config, dat, loader, parse_events


def _parse_ref_words(s):
    if isinstance(s, list):
        return s
    if not isinstance(s, str) or not s.strip():
        return []
    try:
        return list(json.loads(s))
    except Exception:
        return []


@lru_cache(maxsize=4)
def main_cohort_scored() -> pd.DataFrame:
    """Main cohort + dat_before_glove + dat_after_score (both GloVe-scored).

    Cached to disk so repeated figure runs don't re-load GloVe.
    """
    read_path = config.cached_parquet("participants_dat_scored.parquet")
    if read_path.exists():
        return pd.read_parquet(read_path)
    cache = config.CACHE_DIR / "participants_dat_scored.parquet"

    participants = parse_events.main_cohort().copy()
    model = dat.load_model()

    sessions = loader.load_sessions()
    ref_map = dict(zip(sessions["user_id"], sessions["ref_dat_words"]))

    participants["dat_before_glove"] = participants["user_id"].map(
        lambda u: dat.score(_parse_ref_words(ref_map.get(u)), model=model)
    )
    participants["dat_after_score"] = participants["dat_after_words"].apply(
        lambda ws: dat.score(list(ws) if ws is not None else [], model=model)
    )

    participants.to_parquet(cache)
    return participants


@lru_cache(maxsize=4)
def notebook_cohort() -> pd.DataFrame:
    """The cohort used in `analyze_pareidolia_offline.ipynb`:
    English sessions with non-empty feedback. Matches the notebook's
    `participants_words` table (~579 sessions, ~500 with DAT before).

    Use this for any analysis that re-uses or extends the notebook's
    figures; reach for `main_cohort_scored()` when DAT-after is needed.
    """
    read_path = config.cached_parquet("participants_notebook.parquet")
    if read_path.exists():
        return pd.read_parquet(read_path)
    sessions = loader.load_sessions(
        english_only=True, require_dat=False, require_feedback=True,
    )
    df = parse_events.build_participants(sessions=sessions)
    df.to_parquet(config.CACHE_DIR / "participants_notebook.parquet")
    return df


def select_cohort(name: str, dat_when: str = "before") -> pd.DataFrame:
    """Resolve a cohort name to a DataFrame, validating it has the required DAT col."""
    name = (name or "notebook").lower()
    if dat_when == "after":
        # DAT-after only exists in main_cohort.
        return main_cohort_scored()
    if name == "notebook":
        return notebook_cohort()
    if name == "main":
        return main_cohort_scored()
    raise ValueError(f"Unknown cohort: {name}")


def dat_column(when: str) -> str:
    """Return the column name in main_cohort_scored() for a given timepoint."""
    when = (when or "before").lower()
    if when == "before":
        return "dat_before_glove"
    if when == "after":
        return "dat_after_score"
    if when == "raw_before":
        return "ref_dat_score"
    raise ValueError(f"Unknown DAT timepoint: {when}")


def cohort_for(when: str, min_n_words_for_metric: int = 0) -> pd.DataFrame:
    """Main cohort with rows missing the requested DAT score dropped."""
    p = main_cohort_scored()
    col = dat_column(when)
    return p.dropna(subset=[col]).reset_index(drop=True)
