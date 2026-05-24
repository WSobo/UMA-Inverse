"""Figure 3: per-PDB recovery violin plots.

Two panels (metal, small_molecule). Three violins per panel: UMA-Inverse-1 (v3,
primary), UMA-v2 and UMA-v1 (supplemental). LigandMPNN/ProteinMPNN paper numbers
shown as horizontal reference lines.

Sources:
    UMA-Inverse-1 (v3): outputs/benchmark/interface_recovery/v3-ep23-test_<cls>/per_pdb.csv
    UMA-v2:             outputs/benchmark/interface_recovery/v2-ep19-test_<cls>/per_pdb.csv
    UMA-v1:             outputs/benchmark/interface_recovery/ep11-test_<cls>/per_pdb.csv

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
        v3 = _load_pdb_medians(args.bench_dir / f"v3-ep23-test_{split}" / "per_pdb.csv")

        data = v3 or [0.0]
        parts = ax.violinplot([data], positions=[1], widths=0.6,
                              showmeans=False, showmedians=False, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor("#2C5F8E")
            pc.set_edgecolor("black")
            pc.set_alpha(0.65)
            pc.set_linewidth(0.7)

        if data != [0.0]:
            ax.boxplot([data], positions=[1], widths=0.14, showfliers=False,
                       patch_artist=True,
                       boxprops=dict(facecolor="white", edgecolor="#2C5F8E", linewidth=1.2),
                       medianprops=dict(color="#2C5F8E", linewidth=2.0),
                       whiskerprops=dict(color="black", linewidth=0.7),
                       capprops=dict(color="black", linewidth=0.7))

        if split in LIGANDMPNN_REF:
            ax.axhline(LIGANDMPNN_REF[split], color="#C13C3C", linestyle="--",
                       linewidth=1.3, alpha=0.85, zorder=0,
                       label=f"LigandMPNN (paper): {LIGANDMPNN_REF[split]:.3f}")
            ax.axhline(PROTEINMPNN_REF[split], color="#888888", linestyle=":",
                       linewidth=1.3, alpha=0.85, zorder=0,
                       label=f"ProteinMPNN (paper): {PROTEINMPNN_REF[split]:.3f}")

        ax.set_xticks([1])
        ax.set_xticklabels(["UMA-Inverse-1"], fontsize=10)
        ax.set_xlim(0.4, 1.6)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        n_v3 = len(v3)
        n_label = f" (N={n_v3})" if n_v3 else ""
        ax.set_title(f"{split.replace('_', ' ').title()}{n_label}", fontsize=11)
        if split == splits[0]:
            ax.set_ylabel("Per-PDB interface recovery (median over 10 designs)", fontsize=10)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.suptitle(
        "UMA-Inverse-1 interface recovery distribution vs paper-reported references",
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
