"""Pareidolia experiment — clean analysis pipeline.

Modules:
    config        : paths and constants
    loader        : load raw sqlite into a pandas DataFrame (with dedup)
    parse_events  : parse the per-session data_json blob into structured tables
    dat           : compute DAT scores (Olson et al. 2021) — before & after
    embeddings    : load BERT word embeddings for the pareidolia vocabulary
    eyetracking   : extract per-trial gaze metrics

The `figures/` subpackage contains scripts that reproduce the notebook figures
and the new analyses.
"""
