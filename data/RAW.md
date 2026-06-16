# Raw data

The contents of `data/raw/` are the unprocessed exports from the online
experiment. The Python package in `analysis/` originally read the
SQLite event log and produced the small parquet caches one level up
(`data/*.parquet`) that every figure script reads. Those parquets
fully reproduce every figure in the manuscript, so the heavy
`pareidolia_2025-oct.sqlite` database (~1.1 GB) is **not** shipped in
this repo; only the flat CSV exports derived from it are.

## Files shipped in this folder

| File | Size | Description |
|---|---|---|
| `pareidolia_trials_2025.csv` | 1.8 MB | Flat per-trial export: one row per (participant, trial), with the percept words and descriptions parsed out. Used by the legacy notebook. |
| `participants_words.csv` | 280 KB | Per-participant concatenated word lists (one row per participant), columns: `FD12_words, FD14_words, FD16_words`. |
| `sessions.csv` | 4 KB | Per-session metadata (locale, completion, opt-in flags). |
| `participants.csv` | 1 KB | Demographics summary. |
| `dat_responses.csv` | 1 KB | Raw DAT word lists (before + after). |
| `trials.csv` | 140 KB | Per-trial summary (rt, n_words). |
| `feedback.csv` | 1 KB | Closing-questionnaire responses. |
| `eyetracking_qc.csv` | 8 KB | Per-session WebGazer calibration QC. |

## SQLite event log (not shipped)

`pareidolia_2025-oct.sqlite` (~1.1 GB) is the master event log containing
every jsPsych event for every session (DAT, calibration, 30 pareidolia
trials, post-DAT, feedback) plus the raw WebGazer (x, y, t) samples. It
exceeds GitHub's 100 MB per-file limit and is kept out of the repo (an
entry in `.gitignore` prevents accidental commits).

If you need it for a re-derivation that the parquet caches don't
support (e.g., re-running the gaze pipeline with new filtering), it can
be obtained from the authors and dropped into `data/raw/`; the Python
package will detect it automatically (`analysis.config.SQLITE_PATH`).
Long-term we recommend publishing the SQLite to Zenodo or OSF as a
versioned dataset.

## Regenerating the parquet caches from scratch

If `data/*.parquet` is missing or out of date:

```bash
# Build participant + trial caches (reads SQLite + CSVs in data/raw/)
python -m analysis.parse_events

# Build the gaze metrics cache (slow, ~1 min)
python -m analysis.eyetracking

# Re-extract image features (fast, ~30 s)
python -m analysis.figures.image_features --rebuild
```

The first run of any DAT-related script also downloads GloVe 840B
(~2 GB) into `analysis_cache/` and re-scores the DAT.

## Schema

The SQLite database has two tables:

- `events`: one row per jsPsych event (`session_id`, `trial_index`,
  `trial_type`, `data_json`). The `data_json` blob carries everything,
  including the WebGazer samples for stimulus-presentation events.
- `sessions`: one row per session (`session_id`, `locale`, `start_time`,
  `finished`, `opt_in_eye_tracking`).

The CSVs are derived flat views of the same data.

## PII

No direct identifiers (name, email, IP) ship with the database; we
strip them upstream. Sessions are identified by an opaque
`session_id` (UUID). Free-text feedback responses occasionally contain
indirect identifiers — review before public release.
