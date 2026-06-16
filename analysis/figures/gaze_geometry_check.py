"""Sanity-check the eye-tracking geometry inference.

For a sample of sessions, overlay:
  * calibration gaze samples (grey)        — what we use to estimate screen extent
  * experiment gaze samples (purple)       — across all trials, in normalised space
  * inferred stimulus ROI box (orange)
  * inferred screen extent box (dashed)

If the orange box doesn't sit roughly in the middle of the gaze cloud, or if
the calibration gaze doesn't span the full [0, 1] square, the inference is
unreliable for that session and any per-stimulus metric should be treated
with caution.

Usage:
    python -m analysis.figures.gaze_geometry_check                # 12 random
    python -m analysis.figures.gaze_geometry_check --n 24 --seed 1
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config, eyetracking, loader, parse_events


def _gather_experiment_gaze(blob_evs, screen):
    xs, ys = [], []
    for t in parse_events.iter_trial_events(_blob_to_str(blob_evs)) if False else parse_events.iter_trial_events(blob_evs):
        for s in t.get("gaze_data_realtime") or []:
            if "x" in s and "y" in s:
                xs.append(s["x"]); ys.append(s["y"])
    if not xs:
        return np.empty(0), np.empty(0)
    xs = np.asarray(xs, dtype=float); ys = np.asarray(ys, dtype=float)
    nx, ny = screen.normalise(xs, ys)
    return np.clip(nx, -.2, 1.2), np.clip(ny, -.2, 1.2)


def _blob_to_str(_):  # vestigial — iter_trial_events takes the JSON string, not parsed events.
    raise RuntimeError("unused")


def _stim_centre(blob: str):
    """Return mean (sx, sy) of all gaze_target positions in a session."""
    xs, ys = [], []
    for t in parse_events.iter_trial_events(blob):
        gt = t.get("gaze_target") or {}
        sel = next(iter(gt.values()), None)
        if sel and "x" in sel and "y" in sel:
            xs.append(sel["x"]); ys.append(sel["y"])
    if not xs:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def main(n: int = 12, seed: int = 0, show: bool = True):
    sessions = loader.load_sessions()
    cohort_ids = set(parse_events.main_cohort()["user_id"])
    sessions = sessions[sessions["user_id"].isin(cohort_ids)]

    rng = np.random.default_rng(seed)
    sample = sessions.iloc[rng.permutation(len(sessions))[: n * 3]]
    # iterate until we have n sessions with usable calibration
    picked = []
    for _, rec in sample.iterrows():
        cal = parse_events.extract_calibration_gaze(rec["data_json"])
        screen = eyetracking.estimate_screen(cal)
        if screen is None:
            continue
        picked.append((rec, cal, screen))
        if len(picked) == n:
            break
    if not picked:
        print("No sessions with usable calibration in the sample.")
        return

    cols = 4
    rows = (len(picked) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_2d(axes).flatten()

    for ax, (rec, cal, screen) in zip(axes, picked):
        cx = np.array([s["x"] for s in cal if "x" in s], dtype=float)
        cy = np.array([s["y"] for s in cal if "y" in s], dtype=float)
        cxn, cyn = screen.normalise(cx, cy)

        exn, eyn = _gather_experiment_gaze(rec["data_json"], screen)

        ax.scatter(cxn, cyn, s=4, color="grey", alpha=0.4, label="calibration")
        if exn.size:
            ax.scatter(exn, eyn, s=2, color="darkviolet", alpha=0.3,
                       label="experiment")

        stim = _stim_centre(rec["data_json"])
        if stim is not None:
            sxn, syn = screen.normalise(np.array([stim[0]]), np.array([stim[1]]))
            sxn, syn = float(sxn[0]), float(syn[0])
            if not (0.1 <= sxn <= 0.9 and 0.1 <= syn <= 0.9):
                sxn, syn = 0.5, 0.5
            hw = eyetracking.STIM_HALF_W_REL_W
            hh = hw * (screen.width / screen.height)
            ax.add_patch(plt.Rectangle(
                (sxn - hw, syn - hh), 2 * hw, 2 * hh,
                fill=False, edgecolor="darkorange", lw=1.5, label="stim ROI",
            ))

        ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False,
                                   edgecolor="black", lw=0.5, ls="--"))
        ax.set_xlim(-0.2, 1.2); ax.set_ylim(1.2, -0.2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{rec['user_id'][:8]}…\n"
                     f"screen ≈ {screen.width:.0f}×{screen.height:.0f}px",
                     fontsize=8)

    for ax in axes[len(picked):]:
        ax.set_visible(False)

    fig.suptitle("Per-session gaze geometry check\n"
                 "grey = calibration, violet = experiment, orange = assumed stim ROI",
                 y=1.01, fontsize=12)
    plt.tight_layout()
    out = config.OUTPUTS_DIR / "fig_gaze_geometry_check.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    main(n=args.n, seed=args.seed, show=not args.no_show)
