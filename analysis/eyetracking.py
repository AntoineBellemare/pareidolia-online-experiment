"""Per-trial gaze metrics.

Why this module is non-trivial
==============================
The dataset comes from a *web* experiment using WebGazer. There is no fixed
screen resolution: each session was captured at whatever the participant's
browser viewport happened to be. The recorded gaze samples are in raw page
pixels — we cannot pool sessions directly.

What's available per session
----------------------------
* `calibration.gaze_data`  : ~150–200 (x,y) samples taken WHILE the participant
  looks at calibration points spanning most of the viewport. We use the 1st–99th
  percentile of these samples as a robust estimate of the usable gaze area
  (i.e. screen extent), and normalise all subsequent gaze to [0, 1].
* `experiment.gaze_target` : the bounding rect of the stimulus image element.
  In this dataset the width/height fields are zero (the image rect is recorded
  before layout settles), but the `x` and `y` are the element's top-left in
  page pixels. We treat (gaze_target.x, gaze_target.y) as the *centre* of the
  stimulus, then assume a square stimulus of half-width = 0.20 × screen
  width (jsPsych default image-keyboard-response renders at ~40 % viewport).
* `validation.accuracy` / `percent_in_roi` : QC scalars only — kept verbatim.

Metrics returned per trial
--------------------------
* n_samples
* duration_ms  (sample timestamp span)
* mean_fix_dur_ms          via I-DT fixation detection (dispersion ≤ 0.05 norm, min 100 ms)
* n_fixations
* n_saccades
* mean_saccade_amp_norm    (Euclidean distance between successive fixations)
* gaze_dispersion_norm     (RMS distance from session-centroid in normalised space)
* gaze_entropy             Shannon entropy of a 10×10 grid heat-map
* prop_on_stim             fraction of samples inside the assumed stimulus ROI

Limitations
-----------
* Stimulus size is *assumed* (gaze_target.width was missing). The
  `prop_on_stim` metric should be read as a relative — not absolute — measure
  of central-region focus. Document this in any reported figure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from . import parse_events

# Stimulus geometry — confirmed from pareidolia_website/experiment.html:563
#     <img style="max-width: 55%;" ...>
# So the rendered stimulus is a square of side 0.55 × viewport_width.
# Half-width (norm of viewport width)  = 0.275.
# Half-height (norm of viewport height) = 0.275 × (viewport_width / viewport_height),
# which we compute per-session from the inferred screen frame.
STIM_HALF_W_REL_W = 0.275
MIN_FIX_DURATION_MS = 100
FIX_DISPERSION_NORM = 0.05


# ─── per-session calibration → screen extent ──────────────────────────────────

@dataclass
class ScreenFrame:
    """Estimated screen extent for normalising gaze coords."""
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(self.x1 - self.x0, 1.0)

    @property
    def height(self) -> float:
        return max(self.y1 - self.y0, 1.0)

    def normalise(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (x - self.x0) / self.width, (y - self.y0) / self.height


def estimate_screen(calibration_samples: list[dict],
                    pad: float = 0.05) -> ScreenFrame | None:
    """1st–99th percentile of calibration gaze, padded outward by `pad`."""
    if not calibration_samples:
        return None
    xs = np.array([s["x"] for s in calibration_samples if "x" in s], dtype=float)
    ys = np.array([s["y"] for s in calibration_samples if "y" in s], dtype=float)
    if len(xs) < 10:
        return None
    x0, x1 = np.percentile(xs, [1, 99])
    y0, y1 = np.percentile(ys, [1, 99])
    px = (x1 - x0) * pad
    py = (y1 - y0) * pad
    return ScreenFrame(x0 - px, y0 - py, x1 + px, y1 + py)


# ─── per-trial gaze processing ────────────────────────────────────────────────

def _to_arrays(samples: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.array([s.get("x", np.nan) for s in samples], dtype=float)
    ys = np.array([s.get("y", np.nan) for s in samples], dtype=float)
    ts = np.array([s.get("t", np.nan) for s in samples], dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    return xs[mask], ys[mask], ts[mask]


def detect_fixations_idt(xs_n: np.ndarray, ys_n: np.ndarray, ts: np.ndarray,
                         min_dur_ms: float = MIN_FIX_DURATION_MS,
                         disp_thr: float = FIX_DISPERSION_NORM) -> list[dict]:
    """I-DT (dispersion-threshold) fixation detection in normalised coords.

    Dispersion = (max(x) - min(x)) + (max(y) - min(y)).
    Returns a list of {start_idx, end_idx, cx, cy, duration_ms}.
    """
    n = len(xs_n)
    if n == 0 or not np.all(np.isfinite(ts)):
        return []
    fixations = []
    i = 0
    while i < n:
        j = i
        # Grow the window until either dispersion exceeds the threshold or
        # we hit the end of the samples.
        while j < n:
            wx = xs_n[i:j + 1]
            wy = ys_n[i:j + 1]
            disp = (wx.max() - wx.min()) + (wy.max() - wy.min())
            if disp > disp_thr:
                break
            j += 1
        # j is exclusive end of the candidate fixation window.
        if j > i:
            duration = ts[j - 1] - ts[i]
            if duration >= min_dur_ms:
                fixations.append({
                    "start_idx": i,
                    "end_idx": j - 1,
                    "cx": float(xs_n[i:j].mean()),
                    "cy": float(ys_n[i:j].mean()),
                    "duration_ms": float(duration),
                })
                i = j
                continue
        i += 1
    return fixations


def _shannon_2d(xs_n: np.ndarray, ys_n: np.ndarray, bins: int = 10) -> float:
    h, _, _ = np.histogram2d(xs_n, ys_n, bins=bins, range=[[0, 1], [0, 1]])
    p = h.ravel()
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def _looks_like_failed_tracking(xs_n: np.ndarray, ys_n: np.ndarray) -> bool:
    """WebGazer occasionally outputs a near-perfect diagonal line of predictions
    when the face-mesh tracker locks onto something silly (mouse cursor, edge of
    the screen). Detect that by checking whether x and y are almost perfectly
    correlated AND span the whole screen — a real gaze trajectory rarely does
    both at once. Also reject sessions with near-zero variance.
    """
    if len(xs_n) < 20:
        return False
    sx, sy = xs_n.std(), ys_n.std()
    if sx < 0.01 or sy < 0.01:
        return True
    r = float(np.corrcoef(xs_n, ys_n)[0, 1])
    span_x = xs_n.max() - xs_n.min()
    span_y = ys_n.max() - ys_n.min()
    if abs(r) > 0.97 and span_x > 0.6 and span_y > 0.6:
        return True
    return False


def trial_metrics(trial: dict, screen: ScreenFrame) -> dict:
    """Compute gaze metrics for one trial."""
    samples = trial.get("gaze_data_realtime") or []
    xs, ys, ts = _to_arrays(samples)
    out = {
        "n_samples": len(xs),
        "duration_ms": float(ts.max() - ts.min()) if len(ts) > 1 else np.nan,
        # Fixations / saccades
        "mean_fix_dur_ms": np.nan,
        "median_fix_dur_ms": np.nan,
        "fix_dur_cv": np.nan,              # CV of fixation durations within trial
        "n_fixations": 0,
        "n_saccades": 0,
        "fix_rate_hz": np.nan,
        "mean_saccade_amp_norm": np.nan,
        "median_saccade_amp_norm": np.nan,
        "saccade_dir_entropy": np.nan,     # Shannon entropy of saccade angle distribution
        # Spatial spread
        "gaze_dispersion_norm": np.nan,
        "gaze_entropy": np.nan,
        "gini_gaze": np.nan,               # Gini coef of 2D gaze histogram (high = concentrated)
        "scanpath_length_norm": np.nan,
        "revisit_ratio": np.nan,           # # of fixations revisiting a previously-fixated 5x5 grid cell
        "recurrence_rate": np.nan,         # RQA-style: % of fixation pairs within ε
        # ROI / central bias
        "prop_on_stim": np.nan,
        "central_bias_norm": np.nan,       # mean distance from screen centre, normed
        "gaze_drift_norm": np.nan,         # distance between 1st-half and 2nd-half centroids
        "tracking_failed": False,
    }
    if len(xs) < 10:
        return out

    xs_n, ys_n = screen.normalise(xs, ys)
    # Clip to [0, 1] (calibration extent isn't perfect).
    xs_n = np.clip(xs_n, 0, 1)
    ys_n = np.clip(ys_n, 0, 1)
    if _looks_like_failed_tracking(xs_n, ys_n):
        out["tracking_failed"] = True
        return out

    fixations = detect_fixations_idt(xs_n, ys_n, ts)
    if fixations:
        durs = np.array([f["duration_ms"] for f in fixations])
        out["mean_fix_dur_ms"] = float(durs.mean())
        out["median_fix_dur_ms"] = float(np.median(durs))
        # Coefficient of variation of fixation durations
        if durs.size >= 2 and durs.mean() > 0:
            out["fix_dur_cv"] = float(durs.std(ddof=1) / durs.mean())
        out["n_fixations"] = len(fixations)
        dur_total = out["duration_ms"] or 1.0
        out["fix_rate_hz"] = len(fixations) / (dur_total / 1000.0)
        if len(fixations) > 1:
            amps = []
            angles = []
            for a, b in zip(fixations, fixations[1:]):
                dx = b["cx"] - a["cx"]; dy = b["cy"] - a["cy"]
                amps.append(np.hypot(dx, dy))
                if dx != 0 or dy != 0:
                    angles.append(np.arctan2(dy, dx))
            out["n_saccades"] = len(amps)
            out["mean_saccade_amp_norm"] = float(np.mean(amps))
            out["median_saccade_amp_norm"] = float(np.median(amps))
            # Saccade direction entropy (8-bin Shannon on [-π, π])
            if angles:
                hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
                p = hist[hist > 0] / hist.sum()
                out["saccade_dir_entropy"] = float(-(p * np.log2(p)).sum())
            # Recurrence rate: fraction of fixation pairs within ε = 0.05
            from scipy.spatial.distance import pdist
            xy = np.array([[f["cx"], f["cy"]] for f in fixations])
            d = pdist(xy)
            out["recurrence_rate"] = float((d < 0.05).mean()) if d.size else np.nan

        # Revisit ratio: fraction of fixations whose 5x5-grid cell was
        # already visited by an earlier fixation.
        seen: set[tuple[int, int]] = set()
        revisits = 0
        for f in fixations:
            cell = (min(int(f["cx"] * 5), 4), min(int(f["cy"] * 5), 4))
            if cell in seen:
                revisits += 1
            seen.add(cell)
        out["revisit_ratio"] = revisits / len(fixations)

    cx, cy = float(xs_n.mean()), float(ys_n.mean())
    out["gaze_dispersion_norm"] = float(np.sqrt(((xs_n - cx) ** 2 + (ys_n - cy) ** 2).mean()))
    out["gaze_entropy"] = _shannon_2d(xs_n, ys_n)

    # Gini concentration of the 2-D gaze histogram (0 = uniform, 1 = all in one cell).
    h, _, _ = np.histogram2d(xs_n, ys_n, bins=10, range=[[0, 1], [0, 1]])
    p = np.sort(h.ravel())
    n = p.size
    if n and p.sum() > 0:
        cum = np.cumsum(p)
        out["gini_gaze"] = float((n + 1 - 2 * cum.sum() / cum[-1]) / n)

    # Scanpath length: sum of step-to-step euclidean distances (norm space).
    if len(xs_n) > 1:
        out["scanpath_length_norm"] = float(np.sum(np.hypot(np.diff(xs_n), np.diff(ys_n))))
    # Central bias: mean distance of gaze from screen centre.
    out["central_bias_norm"] = float(np.mean(np.hypot(xs_n - 0.5, ys_n - 0.5)))

    # Gaze drift: distance between centroid of first half and second half
    # of the trial (in normalised space). Higher = more re-orientation.
    if len(xs_n) >= 20:
        half = len(xs_n) // 2
        c1 = (xs_n[:half].mean(), ys_n[:half].mean())
        c2 = (xs_n[half:].mean(), ys_n[half:].mean())
        out["gaze_drift_norm"] = float(np.hypot(c2[0] - c1[0], c2[1] - c1[1]))

    # Stimulus ROI. The image is centred on the viewport (jsPsych default)
    # and is a square of side 0.55 × viewport_width. In normalised coords
    # the half-height therefore depends on the session's aspect ratio.
    gt = trial.get("gaze_target") or {}
    sel = next(iter(gt.values()), None) if gt else None
    if sel and "x" in sel and "y" in sel:
        sx_n, sy_n = screen.normalise(np.array([sel["x"]]), np.array([sel["y"]]))
        sx_n, sy_n = float(sx_n[0]), float(sy_n[0])
        # Sanitise: gaze_target was captured pre-layout so sometimes off-centre.
        # If it falls outside [0.1, 0.9] in either axis, fall back to viewport centre.
        if not (0.1 <= sx_n <= 0.9 and 0.1 <= sy_n <= 0.9):
            sx_n, sy_n = 0.5, 0.5
        half_w = STIM_HALF_W_REL_W
        half_h = STIM_HALF_W_REL_W * (screen.width / screen.height)
        in_x = (xs_n >= sx_n - half_w) & (xs_n <= sx_n + half_w)
        in_y = (ys_n >= sy_n - half_h) & (ys_n <= sy_n + half_h)
        out["prop_on_stim"] = float((in_x & in_y).mean())
    return out


# ─── batch driver ─────────────────────────────────────────────────────────────

def build_gaze_metrics(sessions=None, cohort: str = "main") -> pd.DataFrame:
    """Per-trial gaze metric table.

    cohort='main' (default) restricts to the ~500 participants who completed
    the post-task DAT (i.e. finished the experiment). cohort='all' walks every
    session — useful for diagnostics but mixes in dropouts and re-uploads.
    """
    from . import loader
    if sessions is None:
        if cohort == "main":
            mc_ids = set(parse_events.main_cohort()["user_id"])
            sessions = loader.load_sessions()
            sessions = sessions[sessions["user_id"].isin(mc_ids)]
        else:
            sessions = loader.load_sessions()

    rows = []
    skipped_no_screen = 0
    skipped_opt_out = 0
    for _, rec in sessions.iterrows():
        cal = parse_events.extract_calibration_gaze(rec["data_json"])
        if not cal:
            skipped_opt_out += 1
            continue
        screen = estimate_screen(cal)
        if screen is None:
            skipped_no_screen += 1
            continue
        for t in parse_events.iter_trial_events(rec["data_json"]):
            m = trial_metrics(t, screen)
            m.update({
                "user_id": rec["user_id"],
                "ref_dat_score": rec["ref_dat_score"],
                "trial_index": t["trial_index"],
                "fd_level": t["fd_level"],
                "screen_w_px": screen.width,
                "screen_h_px": screen.height,
            })
            rows.append(m)
    print(f"Sessions considered: {len(sessions):,}  "
          f"(skipped {skipped_opt_out:,} opt-out of eye tracking, "
          f"{skipped_no_screen:,} with unusable calibration)")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_gaze_metrics()
    out = config_outputs = parse_events.config.CACHE_DIR / "gaze_metrics.parquet"
    df.to_parquet(out)
    print(f"Wrote {len(df):,} rows → {out}")
    print(df.describe().T[["mean", "std", "min", "max"]])
