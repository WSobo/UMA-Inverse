"""Figure 1: UMA-Inverse architecture overview (polished schematic, no data).

End-to-end pipeline: inputs -> featurizer -> PairMixer encoder (x6) -> pair tensor
Z -> {distogram head (auxiliary), ligand-attention decoder -> sequence}. The
PairMixer block internals are not shown (cited in the caption).

Output: outputs/preprint/figures/fig1_architecture.{pdf,png}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

INK = "#1A1A1A"
BLUE, LBLUE = "#2C5F8E", "#DCE7F2"
GREEN, LGREEN = "#3F7D5A", "#DBEAE1"
GOLD, LGOLD = "#B8901F", "#F4ECCD"
GREY, LGREY = "#6E6E6E", "#EDEDED"

FIG_W, FIG_H = 12.0, 3.5
ASPECT = FIG_W / FIG_H  # to render grid cells square in data coords


def box(ax, x, y, w, h, text, fc, ec, fs=9.5, bold=False, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.004,rounding_size=0.018",
                                linewidth=1.4, facecolor=fc, edgecolor=ec, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            zorder=4, fontweight="bold" if bold else "normal", color=tc)


def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.8):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=15, linewidth=lw, color=color,
                                 shrinkA=2, shrinkB=2, zorder=2))


def grid_glyph(ax, cx, cy, gw=0.052, n=5):
    """A small N×N grid to depict the pair tensor Z (cells kept square)."""
    gh = gw * ASPECT
    cw, ch = gw / n, gh / n
    x0, y0 = cx - gw / 2, cy - gh / 2
    for i in range(n):
        for j in range(n):
            shade = LBLUE if (i + j) % 2 == 0 else "#EEF3F9"
            ax.add_patch(Rectangle((x0 + j * cw, y0 + i * ch), cw, ch,
                                   facecolor=shade, edgecolor=BLUE, linewidth=0.35, zorder=3))
    ax.add_patch(Rectangle((x0, y0), gw, gh, fill=False, edgecolor=BLUE, linewidth=1.6, zorder=4))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path,
                    default=PROJECT_ROOT / "outputs" / "preprint" / "figures")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    yc, h = 0.46, 0.30   # main row center / box height

    box(ax, 0.005, yc, 0.135, h, "Inputs\nbackbone +\nligand / NA atoms", LGREY, GREY)
    box(ax, 0.175, yc, 0.135, h, "Featurize\nnode feats +\npair RBF feats", LGREY, GREY)
    box(ax, 0.345, yc - 0.025, 0.165, h + 0.05,
        "PairMixer encoder\n($\\times$6 blocks)\nall residue–residue &\nresidue–ligand pairs",
        LBLUE, BLUE, bold=True)

    # pair tensor Z as a grid glyph + label
    zx = 0.595
    grid_glyph(ax, zx, yc + h / 2)
    ax.text(zx, yc - 0.045, "pair tensor\n$Z\\in\\mathbb{R}^{N\\times N\\times d}$",
            ha="center", va="top", fontsize=9, color=INK)

    # two heads
    box(ax, 0.70, yc + 0.33, 0.295, 0.155, "Distogram head", LGOLD, GOLD, fs=9.5)
    ax.text(0.8475, yc + 0.315, "auxiliary loss — training only",
            ha="center", va="top", fontsize=7.5, color=GOLD, style="italic")
    box(ax, 0.70, yc + 0.02, 0.295, 0.22,
        "Ligand-attention decoder\nposition-specific readout of $Z$\n$\\rightarrow$ autoregressive amino acids",
        LGREEN, GREEN, bold=True)
    box(ax, 0.745, yc - 0.27, 0.205, 0.155, "designed sequence\nM K T A Y …", "#FFFFFF", GREEN, fs=9.5)

    # flow arrows
    arrow(ax, 0.140, yc + h / 2, 0.175, yc + h / 2)
    arrow(ax, 0.310, yc + h / 2, 0.345, yc + h / 2)
    arrow(ax, 0.510, yc + h / 2, zx - 0.040, yc + h / 2)
    arrow(ax, zx + 0.042, yc + h / 2, 0.70, yc + 0.41, color=GOLD)     # -> distogram
    arrow(ax, zx + 0.042, yc + h / 2, 0.70, yc + 0.13, color=GREEN)    # -> decoder
    arrow(ax, 0.8475, yc + 0.02, 0.8475, yc - 0.115, color=GREEN)      # decoder -> sequence

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    pdf = args.out_dir / "fig1_architecture.pdf"
    png = args.out_dir / "fig1_architecture.png"
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, bbox_inches="tight", dpi=220)
    plt.close()
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
