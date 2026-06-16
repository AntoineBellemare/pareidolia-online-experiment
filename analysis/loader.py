"""Load the raw SQLite dump into a tidy DataFrame.

The logs table stores one row per session with a JSON blob (`data_json`)
containing all of that session's events. The same blob is occasionally
re-uploaded by the client which is why the notebook deduplicates based on
ref_dat_words. We keep that behaviour.
"""
from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import config


@lru_cache(maxsize=1)
def load_sessions(
    sqlite_path: Path | str = config.SQLITE_PATH,
    english_only: bool = True,
    require_feedback: bool = False,
    require_dat: bool = True,
) -> pd.DataFrame:
    """Return one row per (deduplicated) session.

    Parameters mirror the filters used in the notebook so downstream analyses
    can opt in or out of them.
    """
    con = sqlite3.connect(str(sqlite_path))
    df = pd.read_sql_query("SELECT * FROM logs", con)
    con.close()

    df = df.reset_index(drop=True)

    # Dedup: drop rows whose ref_dat_words equal one of the previous two rows.
    # The client sometimes re-POSTs a session, producing exact dupes.
    dupe1 = df["ref_dat_words"] == df["ref_dat_words"].shift(1)
    dupe2 = df["ref_dat_words"] == df["ref_dat_words"].shift(2)
    df = df.loc[~(dupe1 | dupe2)].copy()

    if english_only:
        df = df[df["primary_language"] == "English"]
    if require_feedback:
        df = df[df["feedback"].notna() & (df["feedback"] != "")]
    if require_dat:
        df = df[df["ref_dat_score"].notna()]

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    return df.reset_index(drop=True)


def n_sessions(**kwargs) -> int:
    return len(load_sessions(**kwargs))


if __name__ == "__main__":
    df = load_sessions()
    print(f"Sessions (English, with DAT): {len(df):,}")
    print(df[["user_id", "age", "ref_dat_score"]].head())
