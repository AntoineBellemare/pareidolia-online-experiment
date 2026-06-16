"""Copy the latest figures from analysis_outputs/ into paper/figures/.

Maps each manuscript figure name to its source file in `analysis_outputs/`
and copies both the untitled (manuscript) version and the `_titled`
(reference) version, in PNG and PDF.

Usage:
    python -m paper.sync_figures
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "analysis_outputs"
DST = ROOT / "paper" / "figures"

# manuscript-name -> source stem (without 'fig_' prefix)
MAP = {
    "fig1_fd_behaviour":            "fd_effect",
    "fig2_fd_perception":           "animal_vs_body_fd",
    "fig3_semantic_territory":      "fig3_semantic_territory",
    "fig4_image_consensus":         "image_consensus_fd_paper",
    "fig5a_dat_pre_post":           "dat_before_vs_after",
    "fig5b_dat_behaviour":          "creativity_effects_before",
    "fig5cd_diversity":             "fig5cd_diversity",
    "fig5f_dat_human_animal_bias":  "animal_vs_body_dat",
    "fig6_preferred_categories":    "preferred_categories_combined",
    "fig7_preferred_categories_fd": "preferred_categories_fd",
    "fig8_eye_tracking":            "eyetracking_contrasts",
}


def _copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        print(f"  MISSING: {src.name}")
        return False
    DST.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  {src.name}  ->  {dst.name}")
    return True


def main() -> None:
    print(f"Source: {SRC}")
    print(f"Dest:   {DST}\n")
    n_ok = n_total = 0
    for paper_name, src_stem in MAP.items():
        for ext in ("png", "pdf"):
            # Untitled (manuscript) version
            n_total += 1
            if _copy(SRC / f"fig_{src_stem}.{ext}",
                     DST / f"{paper_name}.{ext}"):
                n_ok += 1
            # Titled (reference) version
            n_total += 1
            if _copy(SRC / f"fig_{src_stem}_titled.{ext}",
                     DST / f"{paper_name}_titled.{ext}"):
                n_ok += 1
    print(f"\n{n_ok}/{n_total} files copied.")


if __name__ == "__main__":
    main()
