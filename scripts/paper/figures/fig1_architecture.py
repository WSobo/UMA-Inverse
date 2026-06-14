"""Figure 1: UMA-Inverse architecture overview (schematic, no data).

End-to-end pipeline: inputs -> featurizer -> PairMixer encoder (x6) -> pair tensor
Z -> {distogram head (auxiliary), ligand-attention decoder -> sequence}. The
internals of the PairMixer block are not shown here; see the PairMixer reference
cited in the caption.

Hand-tunable matplotlib schematic; refine in a vector editor if desired.
Output: outputs/preprint/figures/fig1_architecture.{pdf,png}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

BLUE = "#2C5F8E"
LBLUE = "#D6E2EF"
GREEN = "#3F7D5A"
LGREEN = "#D8E8DF"
GREY = "#888888"
LGREY = "#ECECEC"
GOLD = "#C9A227"
LGOLD = "#F3E9C6"


def box(ax, x, y, w, h, text, fc, ec, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.015",
                                linewidth=1.3, facecolor=fc, edgecolor=ec, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            zorder=3, fontweight="bold" if bold else "normal", color="black")


def arrow(ax, x1, y1, x2, y2, color="black", style="-|>", lw=1.7):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=15,
                                 linewidth=lw, color=color, zorder=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path,
                    default=PROJECT_ROOT / "outputs" / "preprint" / "figures")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 3.3))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ymid, h = 0.40, 0.30
    box(ax, 0.005, ymid, 0.15, h,
        "Inputs\nbackbone coords +\nligand / NA atoms", LGREY, GREY, fs=9)
    box(ax, 0.185, ymid, 0.145, h,
        "Featurize\nnode feats +\npair RBF feats", LGREY, GREY, fs=9)
    box(ax, 0.365, ymid - 0.02, 0.165, h + 0.04,
        "PairMixer encoder\n(×6 blocks)\nall residue–residue &\nresidue–ligand pairs",
        LBLUE, BLUE, fs=9, bold=True)
    box(ax, 0.565, ymid + 0.03, 0.105, h - 0.06,
        "pair tensor\n$Z\\in\\mathbb{R}^{N\\times N\\times d}$", "#FFFFFF", BLUE, fs=9)

    # two heads off Z
    box(ax, 0.71, ymid + 0.30, 0.28, 0.18,
        "Distogram head\n(auxiliary loss, training only)", LGOLD, GOLD, fs=9)
    box(ax, 0.71, ymid + 0.02, 0.28, 0.22,
        "Ligand-attention decoder\nposition-specific readout of $Z$\n→ autoregressive amino acids",
        LGREEN, GREEN, fs=9, bold=True)
    box(ax, 0.745, ymid - 0.26, 0.21, 0.18, "designed sequence\nM K T A Y …", "#FFFFFF", GREEN, fs=9.5)

    arrow(ax, 0.155, ymid + h / 2, 0.185, ymid + h / 2)
    arrow(ax, 0.330, ymid + h / 2, 0.365, ymid + h / 2)
    arrow(ax, 0.530, ymid + h / 2, 0.565, ymid + h / 2)
    arrow(ax, 0.670, ymid + h / 2, 0.71, ymid + 0.39, color=GOLD)     # -> distogram head
    arrow(ax, 0.670, ymid + h / 2, 0.71, ymid + 0.13, color=GREEN)    # -> decoder
    arrow(ax, 0.85, ymid + 0.02, 0.85, ymid - 0.08, color=GREEN)      # decoder -> sequence

    fig.suptitle("UMA-Inverse architecture", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    pdf = args.out_dir / "fig1_architecture.pdf"
    png = args.out_dir / "fig1_architecture.png"
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
