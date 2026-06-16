"""Parse the per-session data_json blob into structured per-trial tables.

A session's data_json is a flat list of event dicts in chronological order:
    demographic        : age, primary_language, ...
    calibration        : gaze_data (list[{x,y,t}]) during the 9-pt routine
    validation         : gaze_data, accuracy, precision, percent_in_roi
    experiment         : per-trial — url_stimulus, reaction_time, offset_time,
                         gaze_target, gaze_data_realtime
    words              : per-trial — word1..5, description1..5
    DAT_test           : 10 word responses (Q0..Q9) given AFTER the pareidolia

A `words` event always follows the most recent `experiment` event for the same
trial. We pair them by walking the event list in order and tracking the
"current" stimulus.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from . import config, loader

FD_PATTERN = re.compile(r"(FD\d{2})/FD_\d{2}_(\d+)\.png")
WORD_KEYS = [f"word{i}" for i in range(1, 6)]
DESC_KEYS = [f"description{i}" for i in range(1, 6)]


# ─── per-event helpers ─────────────────────────────────────────────────────────

def _events(blob: str) -> list[dict]:
    if not blob:
        return []
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return []


def _fd_of(stimulus: str | None) -> tuple[str | None, str | None]:
    """Return (fd_level, image_id) e.g. ('FD14', '50')."""
    if not stimulus:
        return None, None
    m = FD_PATTERN.search(stimulus)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def _strip_list(vals: Iterable) -> list[str]:
    out = []
    for v in vals:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        s = str(v).strip().lower()
        if s:
            out.append(s)
    return out


# ─── public extractors ─────────────────────────────────────────────────────────

def iter_trial_events(blob: str) -> Iterator[dict]:
    """Yield one dict per pareidolia trial.

    Pairs each `experiment` event with the next `words` event having the same
    trial_index. Includes gaze samples.
    """
    evs = _events(blob)
    pending: dict | None = None
    for e in evs:
        name = e.get("name")
        if name == "experiment":
            if pending is not None:
                yield pending  # un-paired experiment (no words submitted)
            fd, img_id = _fd_of(e.get("url_stimulus"))
            pending = {
                "trial_index": e.get("trial_index"),
                "url_stimulus": e.get("url_stimulus"),
                "fd_level": fd,
                "image_id": img_id,
                "reaction_time": e.get("reaction_time"),
                "offset_time": e.get("offset_time"),
                "saved_at_experiment": e.get("saved_at"),
                "gaze_target": e.get("gaze_target") or {},
                "gaze_data_realtime": e.get("gaze_data_realtime") or [],
                "words": [],
                "descriptions": [],
            }
        elif name == "words" and pending is not None and \
             e.get("trial_index") == pending["trial_index"]:
            pending["words"] = _strip_list(e.get(k) for k in WORD_KEYS)
            pending["descriptions"] = _strip_list(e.get(k) for k in DESC_KEYS)
            pending["saved_at_words"] = e.get("saved_at")
            yield pending
            pending = None
    if pending is not None:
        yield pending


def extract_dat_after(blob: str) -> list[str] | None:
    """Return the 10 DAT words submitted AFTER the pareidolia task, or None."""
    for e in _events(blob):
        if e.get("name") == "DAT_test":
            resp = e.get("responses") or {}
            words = [resp.get(f"Q{i}") for i in range(config.DAT_N_WORDS)]
            words = [w.strip().lower() for w in words if isinstance(w, str) and w.strip()]
            return words or None
    return None


def extract_calibration_gaze(blob: str) -> list[dict]:
    for e in _events(blob):
        if e.get("name") == "calibration":
            return e.get("gaze_data") or []
    return []


def extract_validation(blob: str) -> dict:
    for e in _events(blob):
        if e.get("name") == "validation":
            return {
                "accuracy": e.get("accuracy"),
                "precision": e.get("precision"),
                "percent_in_roi": e.get("percent_in_roi"),
                "n_samples": len(e.get("gaze_data") or []),
            }
    return {"accuracy": None, "precision": None, "percent_in_roi": None, "n_samples": 0}


# ─── high-level tables ─────────────────────────────────────────────────────────

def build_trials(sessions: pd.DataFrame | None = None,
                 include_gaze: bool = False) -> pd.DataFrame:
    """One row per trial across all sessions."""
    if sessions is None:
        sessions = loader.load_sessions()
    rows = []
    for _, rec in sessions.iterrows():
        for t in iter_trial_events(rec["data_json"]):
            row = {
                "user_id": rec["user_id"],
                "ref_dat_score": rec["ref_dat_score"],
                "trial_index": t["trial_index"],
                "fd_level": t["fd_level"],
                "image_id": t["image_id"],
                "url_stimulus": t["url_stimulus"],
                "reaction_time": t["reaction_time"],
                "offset_time": t["offset_time"],
                "n_words": len(t["words"]),
                "n_descriptions": len(t["descriptions"]),
                "words": t["words"],
                "descriptions": t["descriptions"],
            }
            if include_gaze:
                row["gaze_data_realtime"] = t["gaze_data_realtime"]
                row["gaze_target"] = t["gaze_target"]
            rows.append(row)
    return pd.DataFrame(rows)


def build_participants(sessions: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per participant with aggregated words per FD and DAT words.

    Reproduces `participants_words.csv` but additionally exposes:
      * DAT words *after* the pareidolia task (`dat_after_words`)
      * Validation QC summary (`et_accuracy`, `et_precision`, `et_percent_in_roi`)
      * Counts (`n_trials`, `n_images_per_fd`)
    """
    if sessions is None:
        sessions = loader.load_sessions()
    rows = []
    for _, rec in sessions.iterrows():
        per_fd: dict[str, list[str]] = {fd: [] for fd in config.FD_LEVELS}
        n_imgs_per_fd: dict[str, int] = {fd: 0 for fd in config.FD_LEVELS}
        n_trials = 0
        for t in iter_trial_events(rec["data_json"]):
            n_trials += 1
            fd = t["fd_level"]
            if fd in per_fd:
                n_imgs_per_fd[fd] += 1
                per_fd[fd].extend(t["words"])

        # Dedup while preserving order within each FD.
        for fd in per_fd:
            per_fd[fd] = list(dict.fromkeys(per_fd[fd]))

        val = extract_validation(rec["data_json"])
        rows.append({
            "user_id": rec["user_id"],
            "ref_dat_score": rec["ref_dat_score"],
            "ref_dat_words": rec.get("ref_dat_words"),
            "age": rec.get("age"),
            "dat_after_words": extract_dat_after(rec["data_json"]),
            "n_trials": n_trials,
            **{f"{fd}_words": per_fd[fd] for fd in config.FD_LEVELS},
            **{f"{fd}_n_images": n_imgs_per_fd[fd] for fd in config.FD_LEVELS},
            "et_accuracy": val["accuracy"],
            "et_precision": val["precision"],
            "et_percent_in_roi": val["percent_in_roi"],
            "feedback": rec.get("feedback"),
        })
    df = pd.DataFrame(rows)
    df["all_words"] = df.apply(
        lambda r: list(dict.fromkeys(
            sum((r[f"{fd}_words"] for fd in config.FD_LEVELS), [])
        )),
        axis=1,
    )
    return df


# ─── on-disk cache for the parsed tables ───────────────────────────────────────

@lru_cache(maxsize=4)
def cached_participants() -> pd.DataFrame:
    read_path = config.cached_parquet("participants.parquet")
    if read_path.exists():
        return pd.read_parquet(read_path)
    df = build_participants()
    df.to_parquet(config.CACHE_DIR / "participants.parquet")
    return df


def main_cohort(participants: pd.DataFrame | None = None,
                min_trials: int = 5) -> pd.DataFrame:
    """The cohort used in the manuscript: participants who completed the DAT
    *after* the pareidolia task and have at least `min_trials` valid trials.
    This matches the notebook's `has_feedback & english` filter (~500 ppts).
    """
    if participants is None:
        participants = cached_participants()
    p = participants[participants["dat_after_words"].notna()].copy()
    n_imgs = sum(p[f"{fd}_n_images"] for fd in config.FD_LEVELS)
    p = p[n_imgs >= min_trials]
    return p.reset_index(drop=True)


@lru_cache(maxsize=4)
def cached_trials(include_gaze: bool = False) -> pd.DataFrame:
    name = "trials_with_gaze.parquet" if include_gaze else "trials.parquet"
    read_path = config.cached_parquet(name)
    if read_path.exists():
        return pd.read_parquet(read_path)
    df = build_trials(include_gaze=include_gaze)
    df.to_parquet(config.CACHE_DIR / name)
    return df


def rebuild_cache():
    """Force rebuild of cached parquet files."""
    for n in ("participants.parquet", "trials.parquet", "trials_with_gaze.parquet"):
        p = config.CACHE_DIR / n
        if p.exists():
            p.unlink()
    cached_participants()
    cached_trials(include_gaze=False)


if __name__ == "__main__":
    rebuild_cache()
    p = cached_participants()
    t = cached_trials()
    print(f"Participants: {len(p):,}")
    print(f"  with DAT-after: {p['dat_after_words'].notna().sum():,}")
    print(f"Trials:       {len(t):,}")
    print(p.head()[["user_id", "ref_dat_score", "n_trials", "FD14_n_images"]])
