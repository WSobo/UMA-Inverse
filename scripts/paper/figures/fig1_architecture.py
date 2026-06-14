"""Figure 1: UMA-Inverse architecture overview (schematic, no data).

Top row  : end-to-end pipeline — inputs -> featurizer -> PairMixer encoder (x6)
           -> pair tensor Z -> {distogram head (aux), ligand-attention decoder -> sequence}.
Bottom row: internals of one PairMixer block (incoming + outgoing triangle
           multiplication + pair transition, with residual connections).

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
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                linewidth=1.3, facecolor=fc, edgecolor=ec, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            zorder=3, fontweight="bold" if bold else "normal", color="black")


def arrow(ax, x1, y1, x2, y2, color="black", style="-|>", lw=1.6, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                                 linewidth=lw, color=color, linestyle=ls, zorder=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path,
                    default=PROJECT_ROOT / "outputs" / "preprint" / "figures")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ── Top row: pipeline (y ~ 0.62) ─────────────────────────────────────────
    yt, h = 0.66, 0.16
    box(ax, 0.01, yt, 0.15, h,
        "Inputs\nbackbone coords +\nligand / NA atoms", LGREY, GREY, fs=8.5)
    box(ax, 0.20, yt, 0.15, h,
        "Featurize\nnode feats +\npair RBF feats", LGREY, GREY, fs=8.5)
    box(ax, 0.39, yt, 0.17, h + 0.02,
        "PairMixer encoder\n(×6 blocks)\nall residue–residue &\nresidue–ligand pairs",
        LBLUE, BLUE, fs=8.5, bold=True)
    box(ax, 0.60, yt + 0.03, 0.12, h - 0.06,
        "pair tensor\n$Z\\in\\mathbb{R}^{N\\times N\\times d}$", "#FFFFFF", BLUE, fs=9)

    # two heads off Z
    box(ax, 0.79, yt + 0.11, 0.20, 0.10,
        "Distogram head\n(auxiliary loss)", LGOLD, GOLD, fs=8.5)
    box(ax, 0.79, yt - 0.06, 0.20, 0.13,
        "Ligand-attention decoder\nposition-specific readout\n→ autoregressive AA",
        LGREEN, GREEN, fs=8.5, bold=True)
    box(ax, 0.79, yt - 0.20, 0.20, 0.09, "designed sequence\nM K T A Y …", "#FFFFFF", GREEN, fs=9)

    arrow(ax, 0.16, yt + h / 2, 0.20, yt + h / 2)
    arrow(ax, 0.35, yt + h / 2, 0.39, yt + h / 2)
    arrow(ax, 0.56, yt + h / 2, 0.60, yt + h / 2)
    arrow(ax, 0.72, yt + h / 2, 0.79, yt + 0.16, color=GOLD)        # to distogram
    arrow(ax, 0.72, yt + h / 2, 0.79, yt + 0.005, color=GREEN)      # to decoder
    arrow(ax, 0.89, yt - 0.06, 0.89, yt - 0.11, color=GREEN)        # decoder -> seq
    ax.text(0.89, yt + 0.165, "(training only)", ha="center", va="bottom", fontsize=7,
            color=GOLD, style="italic")

    # ── Bottom row: PairMixer block internals (y ~ 0.12) ──────────────────────
    ax.text(0.5, 0.46, "One PairMixer block (residual; triangle multiplication only — no triangle attention)",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color=BLUE)
    yb, hb = 0.12, 0.16
    box(ax, 0.02, yb, 0.13, hb, "$Z_{in}$", "#FFFFFF", BLUE, fs=10)
    box(ax, 0.22, yb, 0.18, hb, "Triangle multiplication\n(incoming)", LBLUE, BLUE, fs=8.5)
    box(ax, 0.46, yb, 0.18, hb, "Triangle multiplication\n(outgoing)", LBLUE, BLUE, fs=8.5)
    box(ax, 0.70, yb, 0.15, hb, "Pair transition\n(GELU MLP)", LBLUE, BLUE, fs=8.5)
    box(ax, 0.89, yb, 0.09, hb, "$Z_{out}$", "#FFFFFF", BLUE, fs=10)
    for x1, x2 in [(0.15, 0.22), (0.40, 0.46), (0.64, 0.70), (0.85, 0.89)]:
        arrow(ax, x1, yb + hb / 2, x2, yb + hb / 2)
    # residual arcs (dashed)
    for x1, x2 in [(0.31, 0.55), (0.55, 0.775)]:
        arrow(ax, x1, yb + hb, x2, yb + hb, color=GREY, lw=1.0, ls="--", style="-|>")
    ax.text(0.43, yb + hb + 0.025, "residual", ha="center", fontsize=7, color=GREY, style="italic")

    fig.suptitle("UMA-Inverse architecture", fontsize=13, fontweight="bold", y=0.99)
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
