"""End-to-end runner: rebuild caches, fetch GloVe if needed, run every figure.

Usage:
    python -m analysis.run_all                # run everything, prompt before GloVe DL
    python -m analysis.run_all --skip-glove   # use whatever DAT vectors are already on disk
    python -m analysis.run_all --rebuild      # wipe analysis_cache/ first
    python -m analysis.run_all --only fd_effect creativity_effects
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import sys
import time

from . import config, dat, parse_events


FIGURE_MODULES = [
    "fig_diversity_vs_dat",
    "fig_semantic_territory",
    "dat_before_vs_after",
    "fd_effect",
    "creativity_effects",
    "eyetracking_by_creativity",
    "gaze_geometry_check",
]


def _maybe_download_glove(interactive: bool = True) -> None:
    if config.GLOVE_CACHE_PKL.exists() or config.GLOVE_PATH.exists():
        return
    if interactive:
        ans = input(
            "GloVe 840B 300d is not on disk (~2 GB download, ~5 GB extracted).\n"
            "Download now? [y/N] "
        ).strip().lower()
        if ans != "y":
            print("Skipping. DAT analyses will use the MiniLM fallback.")
            return
    dat.download_glove()


def _rebuild_cache():
    if config.CACHE_DIR.exists():
        for p in config.CACHE_DIR.glob("*.parquet"):
            p.unlink()
    parse_events.rebuild_cache()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-glove", action="store_true")
    ap.add_argument("--rebuild", action="store_true",
                    help="wipe analysis_cache/*.parquet before running")
    ap.add_argument("--only", nargs="+", choices=FIGURE_MODULES,
                    help="run only these figure scripts")
    ap.add_argument("--no-show", action="store_true", default=True,
                    help="default: do not call plt.show()")
    ap.add_argument("--show", dest="no_show", action="store_false")
    args = ap.parse_args(argv)

    if args.rebuild:
        print("Rebuilding parquet caches…")
        _rebuild_cache()

    if not args.skip_glove:
        _maybe_download_glove()

    mods = args.only or FIGURE_MODULES
    failures = []
    for name in mods:
        print(f"\n{'='*60}\n  {name}\n{'='*60}")
        t0 = time.time()
        try:
            mod = importlib.import_module(f"analysis.figures.{name}")
            mod.main(show=not args.no_show)
        except Exception as e:
            print(f"  FAILED: {e!r}")
            failures.append(name)
        print(f"  ({time.time() - t0:.1f}s)")

    print(f"\nDone. {len(mods) - len(failures)}/{len(mods)} figures succeeded.")
    if failures:
        print("Failed:", ", ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
