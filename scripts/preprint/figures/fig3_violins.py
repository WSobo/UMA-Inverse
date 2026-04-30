"""Figure 3: per-PDB recovery violin plots.

Two panels (metal, small_molecule). Two violins per panel: UMA-v2 and UMA-v1.
LigandMPNN/ProteinMPNN paper numbers can't be added as violins (we don't have
their per-PDB distributions), so they're shown as horizontal reference lines.

Sources:
    UMA-v2 per-PDB:  outputs/benchmark/interface_recovery/v2-ep19-test_<cls>/per_pdb.csv
    UMA-v1 per-PDB:  outputs/benchmark/interface_recovery/ep11-test_<cls>/per_pdb.csv

Output:
    outputs/preprint/figures/fig3_violins.{pdf,png}
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

LIGANDMPNN_REF = {"metal": 0.775, "small_molecule": 0.633}
PROTEINMPNN_REF = {"metal": 0.406, "small_molecule": 0.505}


def _load_pdb_medians(csv_path: Path) -> list[float]:
    if not csv_path.exists():
        return []
    out = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            try:
                v = float(row["median_recovery"])
                if v == v:
                    out.append(v)
            except (KeyError, ValueError):
                continue
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bench-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "benchmark" / "interface_recovery",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "figures",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    splits = ("metal", "small_molecule")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)

    for ax, split in zip(axes, splits):
        v2 = _load_pdb_medians(args.bench_dir / f"v2-ep19-test_{split}" / "per_pdb.csv")
        v1 = _load_pdb_medians(args.bench_dir / f"ep11-test_{split}" / "per_pdb.csv")

        positions = [1, 2]
        violin_data = [v2, v1]
        violin_colors = ["#2C5F8E", "#7CA9CD"]
        labels = ["UMA-v2", "UMA-v1"]

        parts = ax.violinplot(violin_data, positions=positions, widths=0.7,
                              showmeans=False, showmedians=False, showextrema=False)
        for pc, color in zip(parts["bodies"], violin_colors):
            pc.set_facecolor(color)
            pc.set_edgecolor("black")
            pc.set_alpha(0.65)
            pc.set_linewidth(0.7)

        # Box / median markers on top of the violin
        for pos, data, color in zip(positions, violin_data, violin_colors):
            if not data:
                continue
            ax.boxplot([data], positions=[pos], widths=0.18, showfliers=False,
                       patch_artist=True,
                       boxprops=dict(facecolor="white", edgecolor=color, linewidth=1.2),
                       medianprops=dict(color=color, linewidth=2.0),
                       whiskerprops=dict(color="black", linewidth=0.7),
                       capprops=dict(color="black", linewidth=0.7))

        # Reference lines for paper-reported numbers
        if split in LIGANDMPNN_REF:
            ax.axhline(LIGANDMPNN_REF[split], color="#C13C3C", linestyle="--",
                       linewidth=1.3, alpha=0.85, zorder=0,
                       label=f"LigandMPNN (paper): {LIGANDMPNN_REF[split]:.3f}")
            ax.axhline(PROTEINMPNN_REF[split], color="#888888", linestyle=":",
                       linewidth=1.3, alpha=0.85, zorder=0,
                       label=f"ProteinMPNN (paper): {PROTEINMPNN_REF[split]:.3f}")

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_xlim(0.5, 2.5)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        n_label = f" (N={len(v2)})"
        ax.set_title(f"{split.replace('_', ' ').title()}{n_label}", fontsize=11)
        if split == splits[0]:
            ax.set_ylabel("Per-PDB interface recovery (median over 10 designs)", fontsize=10)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.suptitle(
        "UMA-Inverse interface recovery distributions vs paper-reported references",
        fontsize=11, y=1.0,
    )
    plt.tight_layout()

    pdf = args.out_dir / "fig3_violins.pdf"
    png = args.out_dir / "fig3_violins.png"
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
