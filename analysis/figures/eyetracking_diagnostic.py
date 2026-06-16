"""Eye-tracking diagnostic and exploratory figure.

Produces a single multi-panel figure that audits:

  (a) Per-session calibration sample count       → opt-in quality
  (b) Per-trial sample count distribution        → tracking density
  (c) Tracking-failure rate per session          → bad-tracking distribution
  (d) Calibration gaze coverage of the screen    → did participants look
                                                   at all the calibration
                                                   points?
  (e) Per-session participation in valid trials  → drop-out across trials
  (f) Aggregate gaze heatmap during stimulus presentation
                                                   (all valid trials pooled)

A second figure shows per-FD aggregate heatmaps.

A third figure shows per-DAT-tertile aggregate heatmaps.

Usage:
    python -m analysis.figures.eyetracking_diagnostic
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from .. import config, dat_helper, loader, parse_events, style
from .. import eyetracking as ET

LABELS_DAT = ["Low", "Mid", "High"]


# ─── helpers ────────────────────────────────────────────────────────────────

def _normalised_samples(rec) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (xs_n, ys_n) for one session's full set of experiment-trial
    gaze samples, normalised to its own calibration screen extent."""
    blob = rec["data_json"]
    cal = parse_events.extract_calibration_gaze(blob)
    if not cal: return None
    screen = ET.estimate_screen(cal)
    if screen is None: return None
    xs, ys = [], []
    for t in parse_events.iter_trial_events(blob):
        gd = t.get("gaze_data_realtime") or []
        if not gd: continue
        x = np.array([s.get("x", np.nan) for s in gd], dtype=float)
        y = np.array([s.get("y", np.nan) for s in gd], dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if not m.any(): continue
        xn, yn = screen.normalise(x[m], y[m])
        xs.append(np.clip(xn, 0, 1))
        ys.append(np.clip(yn, 0, 1))
    if not xs: return None
    return np.concatenate(xs), np.concatenate(ys)


def _calibration_norm(rec) -> tuple[np.ndarray, np.ndarray] | None:
    blob = rec["data_json"]
    cal = parse_events.extract_calibration_gaze(blob)
    if not cal: return None
    screen = ET.estimate_screen(cal)
    if screen is None: return None
    x = np.array([s.get("x", np.nan) for s in cal], dtype=float)
    y = np.array([s.get("y", np.nan) for s in cal], dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    xn, yn = screen.normalise(x[m], y[m])
    return np.clip(xn, -0.1, 1.1), np.clip(yn, -0.1, 1.1)


def _heatmap_2d(xs: np.ndarray, ys: np.ndarray, bins: int = 40) -> np.ndarray:
    h, _, _ = np.histogram2d(xs, ys, bins=bins, range=[[0, 1], [0, 1]])
    return h.T  # transpose for image convention


# ─── per-session quality table ──────────────────────────────────────────────

def build_session_quality(sessions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, rec in sessions.iterrows():
        blob = rec["data_json"]
        cal = parse_events.extract_calibration_gaze(blob)
        n_cal = len(cal)
        screen = ET.estimate_screen(cal) if cal else None
        if screen is None:
            rows.append({"user_id": rec["user_id"], "n_cal_samples": n_cal,
                          "screen_w": np.nan, "screen_h": np.nan,
                          "n_trials_with_gaze": 0,
                          "n_failed_trials": 0,
                          "median_samples_per_trial": np.nan,
                          "valid_trial_frac": np.nan})
            continue
        n_with = 0; n_failed = 0; samples = []
        n_total = 0
        for t in parse_events.iter_trial_events(blob):
            n_total += 1
            m = ET.trial_metrics(t, screen)
            if m["n_samples"] > 10:
                n_with += 1
                samples.append(m["n_samples"])
                if m["tracking_failed"]:
                    n_failed += 1
        rows.append({"user_id": rec["user_id"], "n_cal_samples": n_cal,
                      "screen_w": screen.width, "screen_h": screen.height,
                      "n_trials_with_gaze": n_with,
                      "n_failed_trials": n_failed,
                      "median_samples_per_trial":
                          float(np.median(samples)) if samples else np.nan,
                      "valid_trial_frac": (n_with - n_failed) / max(n_total, 1)})
    return pd.DataFrame(rows)


# ─── diagnostic figure ──────────────────────────────────────────────────────

def figure_diagnostic(quality: pd.DataFrame, calib_xy: tuple[np.ndarray, np.ndarray],
                      exp_xy: tuple[np.ndarray, np.ndarray],
                      show: bool = True) -> None:
    style.apply()
    fig, axes = plt.subplots(
        2, 3, figsize=(style.COL_2, 5.2),
        gridspec_kw=dict(hspace=0.65, wspace=0.4),
    )

    # (a) per-session calibration sample count
    ax = axes[0, 0]
    has_cal = quality["n_cal_samples"] > 0
    ax.hist(quality.loc[has_cal, "n_cal_samples"],
            bins=40, color=style.COLORS["fit"], edgecolor="white", lw=0.4)
    style.style_axis(
        ax, title="(a) Calibration samples / session\n"
                  f"(n={has_cal.sum()} opt-in / {len(quality)} cohort)",
        xlabel="calibration samples",
        ylabel="sessions",
    )

    # (b) per-trial sample count distribution
    ax = axes[0, 1]
    ax.hist(quality["median_samples_per_trial"].dropna(),
            bins=30, color=style.COLORS["dot"], edgecolor="white", lw=0.4)
    style.style_axis(ax, title="(b) Median gaze samples / trial\n"
                                "per opt-in session",
                     xlabel="median samples per trial", ylabel="sessions")

    # (c) tracking-failure rate per session
    ax = axes[0, 2]
    rate = quality["n_failed_trials"] / quality["n_trials_with_gaze"].replace(0, np.nan)
    ax.hist(rate.dropna() * 100, bins=30,
            color="#c0392b", edgecolor="white", lw=0.4)
    style.style_axis(
        ax, title="(c) Tracking-failure rate\n"
                  f"(global = {100 * quality['n_failed_trials'].sum() / quality['n_trials_with_gaze'].sum():.1f}%)",
        xlabel="% of trials flagged failed", ylabel="sessions",
    )

    # (d) calibration coverage heatmap
    ax = axes[1, 0]
    cx, cy = calib_xy
    h = _heatmap_2d(cx, cy, bins=40)
    im = ax.imshow(h, origin="upper", extent=[0, 1, 1, 0], cmap="magma",
                    norm=LogNorm(vmin=max(h[h > 0].min(), 1)
                                  if (h > 0).any() else 1, vmax=h.max() or 1),
                    aspect="auto")
    style.style_axis(ax, title=f"(d) Calibration gaze coverage\n"
                                f"(pooled, n={len(cx):,} samples)",
                     xlabel="screen x (norm)", ylabel="screen y (norm)")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.01).ax.tick_params(labelsize=5)

    # (e) valid trial fraction per session
    ax = axes[1, 1]
    ax.hist(quality["valid_trial_frac"].dropna() * 100,
            bins=30, color="#2ecc71", edgecolor="white", lw=0.4)
    style.style_axis(ax, title="(e) Valid-trial fraction\n"
                                "(opt-in, tracking-OK)",
                     xlabel="% valid trials in session",
                     ylabel="sessions")

    # (f) experiment-time gaze heatmap (all valid)
    ax = axes[1, 2]
    ex, ey = exp_xy
    h = _heatmap_2d(ex, ey, bins=40)
    im = ax.imshow(h, origin="upper", extent=[0, 1, 1, 0], cmap="YlOrRd",
                    norm=LogNorm(vmin=max(h[h > 0].min(), 1)
                                  if (h > 0).any() else 1, vmax=h.max() or 1),
                    aspect="auto")
    # Overlay the assumed stimulus ROI box (centred, ±0.275 each)
    hw = 0.275
    ax.plot([0.5 - hw, 0.5 + hw, 0.5 + hw, 0.5 - hw, 0.5 - hw],
            [0.5 - hw, 0.5 - hw, 0.5 + hw, 0.5 + hw, 0.5 - hw],
            color="cyan", lw=1.0)
    style.style_axis(ax, title=f"(f) Experiment gaze heatmap\n"
                                f"(pooled, n={len(ex):,} samples)",
                     xlabel="screen x (norm)", ylabel="screen y (norm)")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.01).ax.tick_params(labelsize=5)

    fig.suptitle("Eye-tracking diagnostic (main cohort, opt-in)", y=1.01)
    if show: plt.show()
    style.savefig(fig, "eyetracking_diagnostic")
    plt.close(fig)


# ─── per-FD and per-tertile heatmaps ─────────────────────────────────────────

def figure_heatmap_by_group(group_xy: dict[str, tuple[np.ndarray, np.ndarray]],
                              title: str, name: str, palette_keys: list[str],
                              show: bool = True) -> None:
    style.apply()
    fig, axes = plt.subplots(
        1, len(palette_keys), figsize=(style.COL_2, 2.6),
        gridspec_kw=dict(wspace=0.25),
    )
    if len(palette_keys) == 1: axes = [axes]
    # Shared color scale across the row
    all_max = max(
        _heatmap_2d(group_xy[k][0], group_xy[k][1]).max()
        for k in palette_keys if k in group_xy and group_xy[k][0].size > 0
    )
    for ax, k in zip(axes, palette_keys):
        if k not in group_xy or group_xy[k][0].size == 0:
            ax.set_axis_off(); continue
        cx, cy = group_xy[k]
        h = _heatmap_2d(cx, cy, bins=40)
        im = ax.imshow(h, origin="upper", extent=[0, 1, 1, 0],
                        cmap="YlOrRd",
                        norm=LogNorm(vmin=max(h[h > 0].min(), 1), vmax=all_max),
                        aspect="auto")
        hw = 0.275
        ax.plot([0.5 - hw, 0.5 + hw, 0.5 + hw, 0.5 - hw, 0.5 - hw],
                [0.5 - hw, 0.5 - hw, 0.5 + hw, 0.5 + hw, 0.5 - hw],
                color="cyan", lw=0.8)
        style.style_axis(ax,
                         title=f"{k}  (n={len(cx):,})",
                         xlabel="x (norm)",
                         ylabel="y (norm)" if k == palette_keys[0] else "")
    fig.suptitle(title, y=1.04)
    cax = fig.add_axes([0.92, 0.18, 0.012, 0.62])
    plt.colorbar(im, cax=cax).ax.tick_params(labelsize=5)
    if show: plt.show()
    style.savefig(fig, name)
    plt.close(fig)


# ─── runner ──────────────────────────────────────────────────────────────────

def main(show: bool = True):
    mc = parse_events.main_cohort()
    sessions = loader.load_sessions()
    sessions = sessions[sessions["user_id"].isin(mc["user_id"])]

    print("Computing per-session quality…")
    quality = build_session_quality(sessions)
    quality.to_csv(config.OUTPUTS_DIR / "eyetracking_session_quality.csv",
                   index=False)

    # Pooled calibration + experiment gaze for diagnostic
    print("Pooling normalised gaze for diagnostic…")
    cal_xs, cal_ys, exp_xs, exp_ys = [], [], [], []
    for _, rec in sessions.iterrows():
        c = _calibration_norm(rec)
        if c:
            cal_xs.append(c[0]); cal_ys.append(c[1])
        e = _normalised_samples(rec)
        if e:
            exp_xs.append(e[0]); exp_ys.append(e[1])
    cal_xy = (np.concatenate(cal_xs) if cal_xs else np.array([]),
               np.concatenate(cal_ys) if cal_ys else np.array([]))
    exp_xy = (np.concatenate(exp_xs) if exp_xs else np.array([]),
               np.concatenate(exp_ys) if exp_ys else np.array([]))
    print(f"  pooled calibration: {cal_xy[0].size:,} samples")
    print(f"  pooled experiment:  {exp_xy[0].size:,} samples")

    figure_diagnostic(quality, cal_xy, exp_xy, show=show)

    # Per-FD pooled experiment gaze
    print("Pooling experiment gaze by FD…")
    fd_xy: dict[str, tuple[list, list]] = {fd: ([], []) for fd in config.FD_LEVELS}
    for _, rec in sessions.iterrows():
        blob = rec["data_json"]
        cal = parse_events.extract_calibration_gaze(blob)
        if not cal: continue
        screen = ET.estimate_screen(cal)
        if screen is None: continue
        for t in parse_events.iter_trial_events(blob):
            fd = t.get("fd_level")
            if fd not in fd_xy: continue
            m = ET.trial_metrics(t, screen)
            if m["tracking_failed"] or m["n_samples"] < 10: continue
            gd = t.get("gaze_data_realtime") or []
            x = np.array([s.get("x", np.nan) for s in gd], dtype=float)
            y = np.array([s.get("y", np.nan) for s in gd], dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if not mask.any(): continue
            xn, yn = screen.normalise(x[mask], y[mask])
            fd_xy[fd][0].append(np.clip(xn, 0, 1))
            fd_xy[fd][1].append(np.clip(yn, 0, 1))
    fd_xy_arr = {k: (np.concatenate(v[0]) if v[0] else np.array([]),
                      np.concatenate(v[1]) if v[1] else np.array([]))
                  for k, v in fd_xy.items()}
    figure_heatmap_by_group(fd_xy_arr,
                             "Aggregate gaze heatmap by FD",
                             "eyetracking_heatmap_by_fd",
                             config.FD_LEVELS, show=show)

    # Per-DAT-tertile pooled gaze
    print("Pooling experiment gaze by DAT tertile…")
    p = dat_helper.notebook_cohort()[["user_id", "ref_dat_score"]].dropna()
    p["dat_tertile"] = pd.qcut(p["ref_dat_score"], 3, labels=LABELS_DAT,
                                duplicates="drop")
    uid_to_t = dict(zip(p["user_id"], p["dat_tertile"]))
    t_xy: dict[str, tuple[list, list]] = {t: ([], []) for t in LABELS_DAT}
    for _, rec in sessions.iterrows():
        t = uid_to_t.get(rec["user_id"])
        if t is None or t not in t_xy: continue
        blob = rec["data_json"]
        cal = parse_events.extract_calibration_gaze(blob)
        if not cal: continue
        screen = ET.estimate_screen(cal)
        if screen is None: continue
        for tr in parse_events.iter_trial_events(blob):
            m = ET.trial_metrics(tr, screen)
            if m["tracking_failed"] or m["n_samples"] < 10: continue
            gd = tr.get("gaze_data_realtime") or []
            x = np.array([s.get("x", np.nan) for s in gd], dtype=float)
            y = np.array([s.get("y", np.nan) for s in gd], dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if not mask.any(): continue
            xn, yn = screen.normalise(x[mask], y[mask])
            t_xy[t][0].append(np.clip(xn, 0, 1))
            t_xy[t][1].append(np.clip(yn, 0, 1))
    t_xy_arr = {k: (np.concatenate(v[0]) if v[0] else np.array([]),
                     np.concatenate(v[1]) if v[1] else np.array([]))
                 for k, v in t_xy.items()}
    figure_heatmap_by_group(t_xy_arr,
                             "Aggregate gaze heatmap by DAT tertile",
                             "eyetracking_heatmap_by_tertile",
                             LABELS_DAT, show=show)

    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    main(show=not args.no_show)
