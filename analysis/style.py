"""Paper-ready figure defaults (Nature-style).

Nature single-column width  = 89 mm  ≈ 3.50 in
Nature double-column width  = 183 mm ≈ 7.20 in
Body-text font sizes        = 7 pt (axis labels) / 6 pt (ticks) / 8 pt (titles)
Sans-serif (Arial preferred, fallback DejaVu Sans)

Usage:
    from analysis.style import apply, savefig, COLORS, mm
    apply()
    fig, ax = plt.subplots(figsize=(mm(89), mm(60)))
    ...
    savefig(fig, "myplot")   # writes PNG (300 dpi) + PDF
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from . import config

# ─── geometry ─────────────────────────────────────────────────────────────────

COL_1 = 3.50    # in
COL_1_5 = 5.20  # in (1.5 column)
COL_2 = 7.20    # in (double column)


def mm(mm_val: float) -> float:
    """Convert a length in millimetres to inches (matplotlib uses inches)."""
    return mm_val / 25.4


# ─── palette ──────────────────────────────────────────────────────────────────

# Two distinct palettes:
#   FD levels: blue → green → red (perceptually distinct categorical)
#   DAT tertiles: sequential purple gradient (ordinal, low → high creativity)
# Keeping them visually distinct prevents misreading FD-coloured panels as
# DAT-coloured ones or vice-versa.
COLORS = {
    # DAT tertiles — sequential warm (yellow → orange → red)
    "Low":  "#d4a017",   # dark yellow / amber
    "Mid":  "#e67e22",   # orange
    "High": "#c0392b",   # dark red
    # FD levels — categorical complementary
    "FD12": "#66D1F4",
    "FD14": "#9FCA1E",
    "FD16": "#922B21",
    "dot":  "#5b2a86",
    "fit":  "#e67e22",
    "ci":   "#e67e2233",
    "muted":"#7f8c8d",
}


# ─── style toggle ─────────────────────────────────────────────────────────────

_RC = {
    # Type — modern, slightly lighter weight
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Inter", "Helvetica Neue", "Helvetica", "Arial",
                            "DejaVu Sans"],
    "font.size":          7,
    "font.weight":        "normal",
    "axes.titlesize":     8.5,
    "axes.titleweight":   "normal",
    "axes.labelsize":     7,
    "axes.labelweight":   "normal",
    "axes.labelcolor":    "#222222",
    "axes.titlepad":      6.0,
    "xtick.labelsize":    6,
    "ytick.labelsize":    6,
    "xtick.color":        "#444444",
    "ytick.color":        "#444444",
    "legend.fontsize":    6,
    "figure.titlesize":   10,
    # Lines / markers — slightly thinner, softer
    "axes.linewidth":     0.5,
    "axes.edgecolor":     "#444444",
    "lines.linewidth":    1.0,
    "patch.linewidth":    0.3,
    "xtick.major.width":  0.5,
    "ytick.major.width":  0.5,
    "xtick.major.size":   2.5,
    "ytick.major.size":   2.5,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    # Spines — keep left & bottom only
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    # Layout
    "figure.dpi":         140,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.04,
    "savefig.facecolor":  "white",
    "axes.labelpad":      3.0,
    "axes.grid":          False,
    "figure.facecolor":   "white",
    # Boxplot / violin defaults
    "boxplot.boxprops.linewidth": 0.5,
    "boxplot.whiskerprops.linewidth": 0.5,
    "boxplot.capprops.linewidth": 0.5,
    "boxplot.medianprops.linewidth": 0.8,
    "boxplot.medianprops.color": "black",
}


def apply() -> None:
    """Apply Nature-style defaults globally for this Python process."""
    mpl.rcParams.update(_RC)


def savefig(fig, name: str, out_dir: Path | None = None,
            also_pdf: bool = True) -> Path:
    """Save `fig` in two variants:
        * `fig_<name>.png/.pdf`         — untitled, for manuscript placement
        * `fig_<name>_titled.png/.pdf`  — keeps the suptitle, for reference

    If the figure has no suptitle, only the untitled variant is produced.
    Returns the untitled PNG path. Strips any leading 'fig_' so callers can
    pass either 'fd_effect' or 'fig_fd_effect'.
    """
    out_dir = out_dir or config.OUTPUTS_DIR
    out_dir.mkdir(exist_ok=True)
    stem = name[4:] if name.startswith("fig_") else name

    sup = getattr(fig, "_suptitle", None)
    has_title = sup is not None and bool((sup.get_text() or "").strip()) \
                and sup.get_visible()

    # 1) Save the titled reference version first (if any).
    if has_title:
        fig.savefig(out_dir / f"fig_{stem}_titled.png")
        if also_pdf:
            fig.savefig(out_dir / f"fig_{stem}_titled.pdf")
        sup.set_visible(False)

    # 2) Save the manuscript (untitled) version.
    png = out_dir / f"fig_{stem}.png"
    fig.savefig(png)
    if also_pdf:
        fig.savefig(out_dir / f"fig_{stem}.pdf")

    # 3) Restore title state so callers that re-use the fig still see it.
    if has_title:
        sup.set_visible(True)

    print(f"Saved -> {png}" + (" (+ _titled)" if has_title else ""))
    return png


def style_axis(ax, *, ylabel: str = "", xlabel: str = "",
               title: str = "", strip_xticks: bool = False,
               strip_yticks: bool = False) -> None:
    """One-call cleanup for any axis."""
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if strip_xticks:
        ax.set_xticks([])
    if strip_yticks:
        ax.set_yticks([])
    ax.tick_params(direction="out", pad=2)
    for s in ("top", "right"):
        if s in ax.spines:
            ax.spines[s].set_visible(False)


def sig_marker(p: float) -> str:
    """Conventional star marker for a p-value (Nature style)."""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def short_stats(r: float, p: float) -> str:
    """One-line, low-clutter stat label for a figure title."""
    sig = sig_marker(p)
    if p < 0.001:
        return f"r = {r:+.2f}{sig}"
    return f"r = {r:+.2f}, p = {p:.2f}{sig}"


def stars_only_title(label: str, p: float) -> str:
    """Paper-ready title: short label + significance stars only.

    Numerical r / p stay in the saved CSV and the figure caption.
    """
    s = sig_marker(p)
    return f"{label} {s}" if s else label


def small_stat_annotation(ax, r: float, p: float,
                          loc: str = "upper left") -> None:
    """In-axis stat label. Stars are large + black (visible at a glance);
    `r = …` is small + muted (for the curious reader).
    """
    x, y, ha, va = {
        "upper left":  (0.04, 0.97, "left",  "top"),
        "upper right": (0.97, 0.97, "right", "top"),
        "lower left":  (0.04, 0.04, "left",  "bottom"),
        "lower right": (0.97, 0.04, "right", "bottom"),
    }[loc]
    stars = sig_marker(p)
    if stars:
        ax.text(x, y, stars,
                transform=ax.transAxes, ha=ha, va=va,
                fontsize=12, color="black", weight="bold")
        # offset r below the stars
        dy = -0.10 if va == "top" else 0.10
        ax.text(x, y + dy, f"r = {r:+.2f}",
                transform=ax.transAxes, ha=ha, va=va,
                fontsize=6, color=COLORS["muted"])
    else:
        ax.text(x, y, f"r = {r:+.2f} (n.s.)",
                transform=ax.transAxes, ha=ha, va=va,
                fontsize=6, color=COLORS["muted"])


def stars_in_axes(ax, p: float, *, loc: str = "upper right",
                  fontsize: float = 12) -> None:
    """Big star marker pinned inside the axes (no r). For bar / box plots."""
    s = sig_marker(p)
    if not s: return
    x, y, ha, va = {
        "upper left":  (0.04, 0.97, "left",  "top"),
        "upper right": (0.97, 0.97, "right", "top"),
        "lower left":  (0.04, 0.04, "left",  "bottom"),
        "lower right": (0.97, 0.04, "right", "bottom"),
    }[loc]
    ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va,
            fontsize=fontsize, color="black", weight="bold")


def sig_brackets(ax, comparisons: list[tuple[int, int, float]], *,
                 y_offset_frac: float = 0.12,
                 step_frac: float = 0.16,
                 bar_lw: float = 0.6,
                 fontsize: float = 10) -> None:
    """Draw significance brackets between bars on `ax`.

    `comparisons` is a list of (x1, x2, p) , x positions are the bar
    indices (0, 1, 2, ...). Stars are drawn above each bracket using the
    standard scheme (*<0.05, **<0.01, ***<0.001). Non-significant
    contrasts are omitted entirely. Brackets are stacked from the top of
    the data upward; ylim is expanded to leave a clear margin above the
    topmost star.
    """
    sig_comparisons = [(a, b, p) for a, b, p in comparisons
                        if sig_marker(p)]
    if not sig_comparisons:
        return
    y_lo, y_hi = ax.get_ylim()
    span = y_hi - y_lo
    base_y = y_hi - span * 0.02
    bracket_y = base_y + span * y_offset_frac
    for i, (a, b, p) in enumerate(sig_comparisons):
        y = bracket_y + i * span * step_frac
        tick = span * 0.015
        ax.plot([a, a, b, b], [y - tick, y, y, y - tick],
                color="black", lw=bar_lw, clip_on=False)
        ax.text((a + b) / 2, y + span * 0.01, sig_marker(p),
                ha="center", va="bottom",
                fontsize=fontsize, color="black", weight="bold",
                clip_on=False)
    # Generous top margin so the highest star is not clipped or
    # crammed against the spine.
    needed_top = bracket_y + (len(sig_comparisons) - 1) * span * step_frac \
                 + span * 0.22
    if needed_top > y_hi:
        ax.set_ylim(y_lo, needed_top)
